"""Refactor engine baseado em libcst e regras declarativas.

Uso:
    # Refatorar projeto inteiro
    python -m flext_infra refactor --project ../projeto --dry-run

    # Refatorar arquivo específico
    python -m flext_infra refactor --file src/module.py --dry-run

    # Rodar regras específicas
    python -m flext_infra refactor --project ../projeto --rules legacy,import --dry-run

    # Listar regras disponíveis
    python -m flext_infra refactor --list-rules
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import libcst as cst
import yaml

from flext_core import FlextService, r
from flext_infra.output import output


class MethodCategory(Enum):
    """Categorias de métodos para ordenação."""

    MAGIC = auto()
    PROPERTY = auto()
    STATIC = auto()
    CLASS = auto()
    PUBLIC = auto()
    PROTECTED = auto()
    PRIVATE = auto()


@dataclass
class MethodInfo:
    """Informações sobre um método para ordenação."""

    name: str
    category: MethodCategory
    node: cst.FunctionDef
    decorators: list[str] = field(default_factory=list)


@dataclass
class RefactorResult:
    """Resultado da refatoração de um arquivo."""

    file_path: Path
    success: bool
    modified: bool
    error: str | None = None
    changes: list[str] = field(default_factory=list)


class RefactorRule:
    """Base para regras de refatoração."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rule_id = config.get("id", "unknown")
        self.name = config.get("name", self.rule_id)
        self.description = config.get("description", "")
        self.enabled = config.get("enabled", True)
        self.severity = config.get("severity", "warning")

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        """Aplica a regra na AST.

        Retorna:
            Tuple de (árvore modificada, lista de mudanças)
        """
        return tree, []

    def matches_filter(self, filter_pattern: str) -> bool:
        """Verifica se a regra corresponde ao filtro."""
        pattern_lower = filter_pattern.lower()
        return (
            pattern_lower in self.rule_id.lower()
            or pattern_lower in self.name.lower()
            or pattern_lower in self.description.lower()
        )


class LegacyRemovalRule(RefactorRule):
    """Remove código legado: aliases, deprecated."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        changes = []

        # Remover aliases simples (OldName = NewName)
        tree, alias_changes = self._remove_aliases(tree)
        changes.extend(alias_changes)

        # Remover classes deprecated
        tree, deprecated_changes = self._remove_deprecated(tree)
        changes.extend(deprecated_changes)

        return tree, changes

    def _remove_aliases(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        """Remove aliases de compatibilidade no nível do módulo."""
        changes = []

        class AliasRemover(cst.CSTTransformer):
            def leave_Assign(self, original_node, updated_node):
                # Verificar se é simples: Name = Name (alias)
                if (
                    len(original_node.targets) == 1
                    and isinstance(original_node.targets[0].target, cst.Name)
                    and isinstance(original_node.value, cst.Name)
                ):
                    target = original_node.targets[0].target.value
                    value = original_node.value.value
                    # Remover se for alias (nomes diferentes) e não for __version__, __all__
                    if target != value and target not in {"__version__", "__all__"}:
                        changes.append(f"Removed alias: {target} = {value}")
                        return cst.RemovalSentinel.REMOVE
                return updated_node

        return tree.visit(AliasRemover()), changes

    def _remove_deprecated(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        """Remove classes marcadas como deprecated."""
        changes = []

        class DeprecatedRemover(cst.CSTTransformer):
            def leave_ClassDef(self, original_node, updated_node):
                class_name = original_node.name.value

                # Verificar se nome contém Deprecated
                if "deprecated" in class_name.lower():
                    changes.append(f"Removed deprecated class: {class_name}")
                    return cst.RemovalSentinel.REMOVE

                # Verificar se tem warnings.warn com DeprecationWarning
                for stmt in original_node.body.body:
                    if (
                        isinstance(stmt, cst.FunctionDef)
                        and stmt.name.value == "__init__"
                    ):
                        for sub_stmt in stmt.body.body:
                            if isinstance(sub_stmt, cst.SimpleStatementLine):
                                for line in sub_stmt.body:
                                    if isinstance(line, cst.Expr):
                                        if isinstance(line.value, cst.Call):
                                            func = line.value.func
                                            if isinstance(func, cst.Attribute):
                                                if func.attr.value == "warn":
                                                    changes.append(
                                                        f"Removed deprecated class: {class_name}"
                                                    )
                                                    return cst.RemovalSentinel.REMOVE

                return updated_node

        return tree.visit(DeprecatedRemover()), changes


class ImportModernizerRule(RefactorRule):
    """Moderniza imports para usar runtime aliases explícitos."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        forbidden = self.config.get("forbidden_imports", [])

        if not forbidden:
            return tree, []

        # Mapear imports a substituir
        imports_to_remove = []
        symbols_to_replace: dict[str, str] = {}
        needed_aliases: set[str] = set()

        for rule in forbidden:
            module = rule.get("module", "")
            mapping = rule.get("symbol_mapping", {})

            imports_to_remove.append(module)

            for symbol, alias_path in mapping.items():
                symbols_to_replace[symbol] = alias_path
                # Extrair alias principal (c, m, r, t, u)
                alias = alias_path.split(".")[0]
                needed_aliases.add(alias)

        # Aplicar transformações
        changes = []

        class ImportModernizer(cst.CSTTransformer):
            def __init__(self):
                self.modified_imports = False

            def leave_ImportFrom(self, original_node, updated_node):
                module_name = self._get_module_name(original_node.module)

                for mod in imports_to_remove:
                    if mod in module_name:
                        self.modified_imports = True
                        changes.append(f"Removed import: from {module_name}")
                        return cst.RemovalSentinel.REMOVE

                return updated_node

            def leave_Name(self, original_node, updated_node):
                if original_node.value in symbols_to_replace:
                    alias_path = symbols_to_replace[original_node.value]
                    parts = alias_path.split(".")

                    # Construir atributo: c.System.PLATFORM
                    result = cst.Name(parts[0])
                    for part in parts[1:]:
                        result = cst.Attribute(value=result, attr=cst.Name(part))

                    changes.append(f"Replaced: {original_node.value} -> {alias_path}")
                    return result
                return updated_node

            def _get_module_name(self, module) -> str:
                if isinstance(module, cst.Name):
                    return module.value
                elif isinstance(module, cst.Attribute):
                    parts = []
                    current = module
                    while isinstance(current, cst.Attribute):
                        parts.append(current.attr.value)
                        current = current.value
                    if isinstance(current, cst.Name):
                        parts.append(current.value)
                    return ".".join(reversed(parts))
                return ""

            def leave_Module(self, original_node, updated_node):
                if self.modified_imports and needed_aliases:
                    # Adicionar import dos aliases
                    new_import = cst.SimpleStatementLine(
                        body=[
                            cst.ImportFrom(
                                module=cst.Name("flext_core"), names=cst.ImportStar()
                            )
                        ]
                    )

                    # Inserir após docstring e __future__
                    body = list(updated_node.body)
                    insert_idx = 0

                    # Pular docstring
                    if (
                        body
                        and isinstance(body[0], cst.SimpleStatementLine)
                        and len(body[0].body) == 1
                        and isinstance(body[0].body[0], cst.Expr)
                        and isinstance(body[0].body[0].value, cst.SimpleString)
                    ):
                        insert_idx = 1

                    # Pular __future__ imports
                    while insert_idx < len(body) and isinstance(
                        body[insert_idx], cst.SimpleStatementLine
                    ):
                        stmt = body[insert_idx]
                        if (
                            len(stmt.body) == 1
                            and isinstance(stmt.body[0], cst.ImportFrom)
                            and isinstance(stmt.body[0].module, cst.Name)
                            and stmt.body[0].module.value == "__future__"
                        ):
                            insert_idx += 1
                        else:
                            break

                    changes.append(f"Added: from flext_core import *")
                    new_body = body[:insert_idx] + [new_import] + body[insert_idx:]
                    return updated_node.with_changes(body=new_body)
                return updated_node

        modernizer = ImportModernizer()
        return tree.visit(modernizer), changes


class ClassReconstructorRule(RefactorRule):
    """Reconstrói classes: ordena métodos, organiza estrutura."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        order_config = self.config.get("order", [])

        if not order_config:
            return tree, []

        changes = []

        class ClassReconstructor(cst.CSTTransformer):
            def __init__(self):
                self.order_config = order_config

            def leave_ClassDef(self, original_node, updated_node):
                # Separar métodos de outros membros
                methods: list[MethodInfo] = []
                other_members = []

                for stmt in updated_node.body.body:
                    if isinstance(stmt, cst.FunctionDef):
                        info = self._analyze_method(stmt)
                        methods.append(info)
                    else:
                        other_members.append(stmt)

                if not methods:
                    return updated_node

                # Ordenar métodos
                sorted_methods = self._sort_methods(methods)

                # Reconstruir corpo
                new_body = other_members + [m.node for m in sorted_methods]

                changes.append(
                    f"Reordered {len(methods)} methods in class {original_node.name.value}"
                )

                return updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_body)
                )

            def _analyze_method(self, node: cst.FunctionDef) -> MethodInfo:
                """Analisa um método e determina sua categoria."""
                name = node.name.value
                decorators = []

                for dec in node.decorators.elements:
                    if isinstance(dec.decorator, cst.Name):
                        decorators.append(dec.decorator.value)
                    elif isinstance(dec.decorator, cst.Attribute):
                        decorators.append(dec.decorator.attr.value)

                # Determinar categoria
                category = self._categorize(name, decorators)

                return MethodInfo(
                    name=name, category=category, node=node, decorators=decorators
                )

            def _categorize(self, name: str, decorators: list[str]) -> MethodCategory:
                """Categoriza um método."""
                # Verificar decorators
                if any(
                    d in decorators
                    for d in ["property", "cached_property", "computed_field"]
                ):
                    return MethodCategory.PROPERTY
                if "staticmethod" in decorators:
                    return MethodCategory.STATIC
                if "classmethod" in decorators:
                    return MethodCategory.CLASS

                # Verificar nome
                if name.startswith("__") and name.endswith("__"):
                    return MethodCategory.MAGIC
                elif name.startswith("__"):
                    return MethodCategory.PRIVATE
                elif name.startswith("_"):
                    return MethodCategory.PROTECTED
                else:
                    return MethodCategory.PUBLIC

            def _sort_methods(self, methods: list[MethodInfo]) -> list[MethodInfo]:
                """Ordena métodos por categoria."""
                category_order = [
                    MethodCategory.MAGIC,
                    MethodCategory.PROPERTY,
                    MethodCategory.STATIC,
                    MethodCategory.CLASS,
                    MethodCategory.PUBLIC,
                    MethodCategory.PROTECTED,
                    MethodCategory.PRIVATE,
                ]

                return sorted(
                    methods, key=lambda m: (category_order.index(m.category), m.name)
                )

        return tree.visit(ClassReconstructor()), changes


class MRORedundancyChecker(RefactorRule):
    """Detecta e corrige redeclarações via MRO."""

    def apply(
        self, tree: cst.Module, file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        changes = []

        class MRORemover(cst.CSTTransformer):
            def leave_ClassDef(self, original_node, updated_node):
                # Verificar classes aninhadas que herdam de algo do pai
                if not isinstance(updated_node.body, cst.IndentedBlock):
                    return updated_node

                new_body = []
                for stmt in updated_node.body.body:
                    if isinstance(stmt, cst.ClassDef) and stmt.bases:
                        # Verificar se herda de classe externa
                        for base in stmt.bases:
                            if isinstance(base.value, cst.Attribute):
                                # Ex: class Platform(FlextConstants.Platform)
                                changes.append(
                                    f"Fixed MRO redeclaration: {stmt.name.value}"
                                )
                                stmt = stmt.with_changes(bases=[])
                                break
                    new_body.append(stmt)

                return updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_body)
                )

        return tree.visit(MRORemover()), changes


class FlextRefactorEngine(FlextService[dict[str, Any]]):
    """Engine de refatoração que orquestra regras declarativas."""

    def __init__(self, config_path: Path | None = None):
        super().__init__()
        self.config_path = config_path or self._default_config_path()
        self.config: dict[str, Any] = {}
        self.rules: list[RefactorRule] = []
        self.rule_filters: list[str] = []

    def _default_config_path(self) -> Path:
        """Retorna caminho padrão do config dentro de flext_infra."""
        return Path(__file__).parent / "config.yml"

    def set_rule_filters(self, filters: list[str]) -> None:
        """Define filtros para regras (apenas regras que correspondem serão executadas)."""
        self.rule_filters = [f.lower() for f in filters]

    def load_config(self) -> r[None]:
        """Carrega configuração do YAML."""
        try:
            self.config = yaml.safe_load(self.config_path.read_text())
            output.info(f"Loaded config from {self.config_path}")
            return r[None].ok(None)
        except Exception as e:
            return r[None].fail(f"Failed to load config: {e}")

    def load_rules(self) -> r[None]:
        """Carrega regras de arquivos YAML."""
        try:
            rules_dir = self.config_path.parent / "rules"

            for rule_file in sorted(rules_dir.glob("*.yml")):
                output.info(f"Loading rules from {rule_file.name}")
                rule_config = yaml.safe_load(rule_file.read_text())
                rules = rule_config.get("rules", [])

                for rule_def in rules:
                    if not rule_def.get("enabled", True):
                        continue

                    rule_id = rule_def.get("id", "unknown")

                    # Instanciar regra apropriada
                    rule: RefactorRule | None = None
                    if any(x in rule_id for x in ["legacy", "alias", "deprecated"]):
                        rule = LegacyRemovalRule(rule_def)
                    elif any(x in rule_id for x in ["import", "modernize"]):
                        rule = ImportModernizerRule(rule_def)
                    elif any(x in rule_id for x in ["class", "reorder", "method"]):
                        rule = ClassReconstructorRule(rule_def)
                    elif "mro" in rule_id:
                        rule = MRORedundancyChecker(rule_def)

                    if rule:
                        # Aplicar filtros se existirem
                        if self.rule_filters:
                            if any(rule.matches_filter(f) for f in self.rule_filters):
                                self.rules.append(rule)
                        else:
                            self.rules.append(rule)

            output.info(f"Loaded {len(self.rules)} rules")
            if self.rule_filters:
                output.info(f"Active filters: {', '.join(self.rule_filters)}")
            return r[None].ok(None)
        except Exception as e:
            return r[None].fail(f"Failed to load rules: {e}")

    def list_rules(self) -> list[dict[str, Any]]:
        """Lista todas as regras disponíveis."""
        rules_info = []
        for rule in self.rules:
            rules_info.append({
                "id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
                "severity": rule.severity,
            })
        return rules_info

    def refactor_file(self, file_path: Path, dry_run: bool = False) -> RefactorResult:
        """Refatora um único arquivo."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = cst.parse_module(source)

            all_changes = []

            # Aplicar todas as regras
            for rule in self.rules:
                if rule.enabled:
                    tree, changes = rule.apply(tree, file_path)
                    all_changes.extend(changes)

            result_code = tree.code if hasattr(tree, "code") else str(tree)
            modified = result_code != source

            if not dry_run and modified:
                file_path.write_text(result_code, encoding="utf-8")

            return RefactorResult(
                file_path=file_path,
                success=True,
                modified=modified,
                changes=all_changes,
            )
        except Exception as e:
            return RefactorResult(
                file_path=file_path,
                success=False,
                modified=False,
                error=str(e),
                changes=[],
            )

    def refactor_files(
        self, file_paths: list[Path], dry_run: bool = False
    ) -> list[RefactorResult]:
        """Refatora múltiplos arquivos."""
        results = []

        for file_path in file_paths:
            result = self.refactor_file(file_path, dry_run=dry_run)
            results.append(result)

            # Log do resultado
            if result.success:
                if result.modified:
                    output.info(
                        f"{'[DRY-RUN] ' if dry_run else ''}Modified: {file_path.name}"
                    )
                    for change in result.changes:
                        output.info(f"  - {change}")
                else:
                    output.info(f"Unchanged: {file_path.name}")
            else:
                output.error(f"Failed: {file_path.name} - {result.error}")

        return results

    def refactor_project(
        self, project_path: Path, dry_run: bool = False, pattern: str = "*.py"
    ) -> list[RefactorResult]:
        """Refatora projeto inteiro."""
        src_dir = project_path / "src"
        if not src_dir.exists():
            output.error(f"No src/ directory in {project_path}")
            return []

        # Coletar arquivos
        files = []
        for py_file in src_dir.rglob(pattern):
            # Ignorar arquivos especiais
            if py_file.name in {"__init__.py", "conftest.py"}:
                continue
            files.append(py_file)

        output.info(f"Found {len(files)} files to process")

        # Processar arquivos
        return self.refactor_files(files, dry_run=dry_run)

    def execute(self) -> r[dict[str, Any]]:
        """Executa refatoração (interface FlextService)."""
        result = self.load_config()
        if not result.is_success:
            return r[dict[str, Any]].fail(f"Config error: {result.error}")

        result = self.load_rules()
        if not result.is_success:
            return r[dict[str, Any]].fail(f"Rules error: {result.error}")

        return r[dict[str, Any]].ok({
            "status": "ready",
            "rules_loaded": len(self.rules),
        })


def print_rules_table(rules: list[dict[str, Any]]) -> None:
    """Imprime tabela de regras formatada."""
    output.header("Available Rules")

    if not rules:
        output.info("No rules loaded.")
        return

    # Calcular larguras
    id_width = max(len(r["id"]) for r in rules) + 2
    name_width = max(len(r["name"]) for r in rules) + 2

    # Cabeçalho
    header = f"{'ID':<{id_width}} {'Name':<{name_width}} {'Severity':<10} {'Status'}"
    output.info(header)
    output.info("-" * len(header))

    # Linhas
    for rule in rules:
        status = "✓" if rule["enabled"] else "✗"
        line = f"{rule['id']:<{id_width}} {rule['name']:<{name_width}} {rule['severity']:<10} {status}"
        output.info(line)
        if rule["description"]:
            output.info(f"  └─ {rule['description']}")


def print_summary(results: list[RefactorResult], dry_run: bool) -> None:
    """Imprime resumo dos resultados."""
    modified = sum(1 for r in results if r.modified)
    failed = sum(1 for r in results if not r.success)
    unchanged = sum(1 for r in results if r.success and not r.modified)

    output.header("Summary")
    output.info(f"Total files: {len(results)}")
    output.info(f"Modified: {modified}")
    output.info(f"Unchanged: {unchanged}")
    output.info(f"Failed: {failed}")

    if dry_run:
        output.info("\n[DRY-RUN] No changes applied")
    elif failed == 0:
        output.info("\n✓ All changes applied successfully")
    else:
        output.info(f"\n⚠ {failed} files failed")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Flext Refactor Engine - Declarative code transformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available rules
  python -m flext_infra refactor --list-rules
  
  # Refactor entire project (dry-run)
  python -m flext_infra refactor --project ../flext-quality --dry-run
  
  # Refactor specific file
  python -m flext_infra refactor --file src/module.py
  
  # Run only specific rules
  python -m flext_infra refactor --project ../flext-quality --rules legacy,import
  
  # Refactor all test files
  python -m flext_infra refactor --project ../flext-quality --pattern "test_*.py"
        """,
    )

    # Modo de operação
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--project",
        "-p",
        type=Path,
        help="Path do projeto a refatorar (processa src/*.py)",
    )
    mode_group.add_argument(
        "--file", "-f", type=Path, help="Path do arquivo específico a refatorar"
    )
    mode_group.add_argument(
        "--files", nargs="+", type=Path, help="Paths dos arquivos a refatorar"
    )
    mode_group.add_argument(
        "--list-rules",
        "-l",
        action="store_true",
        help="Listar regras disponíveis e sair",
    )

    # Opções de filtro
    parser.add_argument(
        "--rules",
        "-r",
        type=str,
        help="Regras específicas a executar (comma-separated, ex: legacy,import,mro)",
    )
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="Padrão de arquivos a processar (default: *.py)",
    )

    # Opções de execução
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Mostrar o que seria feito sem aplicar",
    )
    parser.add_argument(
        "--config", "-c", type=Path, help="Path do arquivo de configuração YAML"
    )

    args = parser.parse_args()

    # Inicializar engine
    engine = FlextRefactorEngine(config_path=args.config)

    # Carregar config e regras
    result = engine.load_config()
    if not result.is_success:
        output.error(f"Config error: {result.error}")
        sys.exit(1)

    result = engine.load_rules()
    if not result.is_success:
        output.error(f"Rules error: {result.error}")
        sys.exit(1)

    # Modo: listar regras
    if args.list_rules:
        rules = engine.list_rules()
        print_rules_table(rules)
        sys.exit(0)

    # Aplicar filtros de regras
    if args.rules:
        rule_filters = [f.strip() for f in args.rules.split(",")]
        engine.set_rule_filters(rule_filters)

        # Recarregar regras com filtros
        engine.rules = []  # Limpar regras
        result = engine.load_rules()  # Recarregar com filtros
        if not result.is_success:
            output.error(f"Rules error: {result.error}")
            sys.exit(1)

    # Executar refatoração
    output.header(f"Refactoring")
    output.info(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    output.info(f"Rules: {len(engine.rules)} active")
    if args.rules:
        output.info(f"Filter: {args.rules}")

    results: list[RefactorResult] = []

    if args.project:
        output.info(f"Project: {args.project}")
        output.info(f"Pattern: {args.pattern}")
        results = engine.refactor_project(
            args.project, dry_run=args.dry_run, pattern=args.pattern
        )
    elif args.file:
        output.info(f"File: {args.file}")
        if not args.file.exists():
            output.error(f"File not found: {args.file}")
            sys.exit(1)
        result_single = engine.refactor_file(args.file, dry_run=args.dry_run)
        results = [result_single]
    elif args.files:
        output.info(f"Files: {len(args.files)}")
        existing_files = [f for f in args.files if f.exists()]
        missing = [f for f in args.files if not f.exists()]
        for f in missing:
            output.error(f"File not found: {f}")
        results = engine.refactor_files(existing_files, dry_run=args.dry_run)

    # Imprimir resumo
    print_summary(results, args.dry_run)

    # Exit code
    failed = sum(1 for r in results if not r.success)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
