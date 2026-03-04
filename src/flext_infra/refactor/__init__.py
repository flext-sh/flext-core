"""Refactor engine baseado em libcst e regras declarativas.

Uso:
    python -m flext_infra refactor --project ../projeto --dry-run
    python -m flext_infra refactor --project ../projeto
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, override

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


class RefactorRule:
    """Base para regras de refatoração."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rule_id = config.get("id", "unknown")
        self.enabled = config.get("enabled", True)

    def apply(self, tree: cst.Module) -> cst.Module:
        """Aplica a regra na AST."""
        return tree


class LegacyRemovalRule(RefactorRule):
    """Remove código legado: aliases, deprecated."""

    def apply(self, tree: cst.Module) -> cst.Module:
        # Remover aliases simples (OldName = NewName)
        tree = self._remove_aliases(tree)

        # Remover classes deprecated
        tree = self._remove_deprecated(tree)

        return tree

    def _remove_aliases(self, tree: cst.Module) -> cst.Module:
        """Remove aliases de compatibilidade no nível do módulo."""

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
                        output.info(f"  Removing alias: {target} = {value}")
                        return cst.RemovalSentinel.REMOVE
                return updated_node

        return tree.visit(AliasRemover())

    def _remove_deprecated(self, tree: cst.Module) -> cst.Module:
        """Remove classes marcadas como deprecated."""

        class DeprecatedRemover(cst.CSTTransformer):
            def leave_ClassDef(self, original_node, updated_node):
                # Verificar se nome contém Deprecated
                if "deprecated" in original_node.name.value.lower():
                    output.info(
                        f"  Removing deprecated class: {original_node.name.value}"
                    )
                    return cst.RemovalSentinel.REMOVE

                # Verificar se tem warnings.warn com DeprecationWarning no __init__
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
                                                    output.info(
                                                        f"  Removing deprecated class: {original_node.name.value}"
                                                    )
                                                    return cst.RemovalSentinel.REMOVE

                return updated_node

        return tree.visit(DeprecatedRemover())


class ImportModernizerRule(RefactorRule):
    """Moderniza imports para usar runtime aliases explícitos."""

    def apply(self, tree: cst.Module) -> cst.Module:
        forbidden = self.config.get("forbidden_imports", [])

        if not forbidden:
            return tree

        # Mapear imports a substituir
        imports_to_remove = []
        symbols_to_replace: dict[str, str] = {}
        needed_aliases: set[str] = set()

        for rule in forbidden:
            module = rule.get("module", "")
            symbols = rule.get("symbols", [])
            mapping = rule.get("symbol_mapping", {})

            imports_to_remove.append(module)

            for symbol, alias_path in mapping.items():
                symbols_to_replace[symbol] = alias_path
                # Extrair alias principal (c, m, r, t, u)
                alias = alias_path.split(".")[0]
                needed_aliases.add(alias)

        # Remover imports antigos e adicionar novo import de aliases
        class ImportModernizer(cst.CSTTransformer):
            def __init__(self, imports_to_remove, symbols_to_replace, needed_aliases):
                self.imports_to_remove = imports_to_remove
                self.symbols_to_replace = symbols_to_replace
                self.needed_aliases = sorted(needed_aliases)
                self.modified_imports = False

            def leave_ImportFrom(self, original_node, updated_node):
                module_name = self._get_module_name(original_node.module)

                for mod in self.imports_to_remove:
                    if mod in module_name:
                        self.modified_imports = True
                        output.info(f"  Removing import: from {module_name} import ...")
                        return cst.RemovalSentinel.REMOVE

                return updated_node

            def leave_Name(self, original_node, updated_node):
                if original_node.value in self.symbols_to_replace:
                    alias_path = self.symbols_to_replace[original_node.value]
                    parts = alias_path.split(".")

                    # Construir atributo: c.System.PLATFORM
                    result = cst.Name(parts[0])
                    for part in parts[1:]:
                        result = cst.Attribute(value=result, attr=cst.Name(part))

                    output.info(f"  Replacing: {original_node.value} -> {alias_path}")
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
                if self.modified_imports and self.needed_aliases:
                    # Adicionar import específico dos aliases
                    # from flext_core import c, m, r, t, u
                    new_import = cst.SimpleStatementLine(
                        body=[
                            cst.ImportFrom(
                                module=cst.Name("flext_core"),
                                names=cst.ImportStar(),  # Simplificado
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

                    new_body = body[:insert_idx] + [new_import] + body[insert_idx:]
                    return updated_node.with_changes(body=new_body)
                return updated_node

        modernizer = ImportModernizer(
            imports_to_remove, symbols_to_replace, needed_aliases
        )
        return tree.visit(modernizer)


class ClassReconstructorRule(RefactorRule):
    """Reconstrói classes: ordena métodos, organiza estrutura."""

    def apply(self, tree: cst.Module) -> cst.Module:
        order_config = self.config.get("order", [])

        if not order_config:
            return tree

        # Aplicar reconstrução
        class ClassReconstructor(cst.CSTTransformer):
            def __init__(self, order_config):
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

                # Ordenar métodos
                sorted_methods = self._sort_methods(methods)

                # Reconstruir corpo
                new_body = other_members + [m.node for m in sorted_methods]

                if methods:
                    output.info(
                        f"  Reordered {len(methods)} methods in class {original_node.name.value}"
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
                if (
                    "property" in decorators
                    or "cached_property" in decorators
                    or "computed_field" in decorators
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

        return tree.visit(ClassReconstructor(order_config))


class MRORedundancyChecker(RefactorRule):
    """Detecta e corrige redeclarações via MRO."""

    def apply(self, tree: cst.Module) -> cst.Module:

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
                                # Isso é redeclaração MRO - remover herança
                                output.info(
                                    f"  Fixing MRO redeclaration: {stmt.name.value}"
                                )
                                stmt = stmt.with_changes(bases=[])
                                break
                    new_body.append(stmt)

                return updated_node.with_changes(
                    body=updated_node.body.with_changes(body=new_body)
                )

        return tree.visit(MRORemover())


class FacadeGenerator:
    """Gera facades automaticamente a partir de código existente."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def extract_constants(self, project_path: Path) -> dict[str, Any]:
        """Extrai constantes de arquivos existentes."""
        constants = {}

        for py_file in (project_path / "src").rglob("*.py"):
            if py_file.name == "constants.py":
                try:
                    tree = cst.parse_module(py_file.read_text())

                    for stmt in tree.body:
                        if isinstance(stmt, (cst.Assign, cst.AnnAssign)):
                            # Extrair nome e valor
                            if isinstance(stmt, cst.Assign) and len(stmt.targets) == 1:
                                if isinstance(stmt.targets[0].target, cst.Name):
                                    name = stmt.targets[0].target.value
                                    # Tentar extrair valor como string
                                    try:
                                        value = tree.code_for_node(stmt.value)
                                        constants[name] = value
                                    except:
                                        pass
                            elif isinstance(stmt, cst.AnnAssign):
                                if isinstance(stmt.target, cst.Name):
                                    name = stmt.target.value
                                    try:
                                        value = (
                                            tree.code_for_node(stmt.value)
                                            if stmt.value
                                            else None
                                        )
                                        constants[name] = value
                                    except:
                                        pass
                except Exception as e:
                    output.error(f"Error parsing {py_file}: {e}")

        return constants

    def generate_facades(self, project_path: Path) -> r[None]:
        """Gera facades para o projeto."""
        output.info("Generating facades...")

        # Extrair constants
        constants = self.extract_constants(project_path)
        output.info(f"Found {len(constants)} constants")

        # TODO: Gerar arquivo de facade usando templates
        # Isso seria implementado com Jinja2 templates

        return r[None].ok(None)


class FlextRefactorEngine(FlextService[dict[str, Any]]):
    """Engine de refatoração que orquestra regras declarativas."""

    def __init__(self, config_path: Path | None = None):
        super().__init__()
        self.config_path = config_path or self._default_config_path()
        self.config: dict[str, Any] = {}
        self.rules: list[RefactorRule] = []

    def _default_config_path(self) -> Path:
        """Retorna caminho padrão do config dentro de flext_infra."""
        return Path(__file__).parent / "config.yml"

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
                    if any(x in rule_id for x in ["legacy", "alias", "deprecated"]):
                        self.rules.append(LegacyRemovalRule(rule_def))
                    elif any(x in rule_id for x in ["import", "modernize"]):
                        self.rules.append(ImportModernizerRule(rule_def))
                    elif any(x in rule_id for x in ["class", "reorder", "method"]):
                        self.rules.append(ClassReconstructorRule(rule_def))
                    elif "mro" in rule_id:
                        self.rules.append(MRORedundancyChecker(rule_def))

            output.info(f"Loaded {len(self.rules)} rules")
            return r[None].ok(None)
        except Exception as e:
            return r[None].fail(f"Failed to load rules: {e}")

    def refactor_file(self, file_path: Path, dry_run: bool = False) -> r[str]:
        """Refatora um único arquivo."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = cst.parse_module(source)

            # Aplicar todas as regras
            for rule in self.rules:
                if rule.enabled:
                    tree = rule.apply(tree)

            result = tree.code if hasattr(tree, "code") else str(tree)

            if not dry_run and result != source:
                file_path.write_text(result, encoding="utf-8")

            return r[str].ok(result)
        except Exception as e:
            return r[str].fail(f"Failed to refactor {file_path}: {e}")

    def refactor_project(
        self, project_path: Path, dry_run: bool = False
    ) -> r[dict[str, Any]]:
        """Refatora projeto inteiro."""
        results = {"modified": [], "failed": [], "unchanged": [], "files_processed": 0}

        src_dir = project_path / "src"
        if not src_dir.exists():
            return r[dict[str, Any]].fail(f"No src/ directory in {project_path}")

        for py_file in src_dir.rglob("*.py"):
            # Ignorar arquivos especiais
            if py_file.name in {"__init__.py", "conftest.py"}:
                continue

            results["files_processed"] += 1
            result = self.refactor_file(py_file, dry_run=dry_run)

            if result.is_success:
                original = py_file.read_text(encoding="utf-8")
                if result.value != original:
                    results["modified"].append(str(py_file.relative_to(project_path)))
                    if dry_run:
                        output.info(f"[DRY-RUN] Would modify: {py_file.name}")
                else:
                    results["unchanged"].append(str(py_file.relative_to(project_path)))
            else:
                results["failed"].append({
                    "file": str(py_file.relative_to(project_path)),
                    "error": result.error,
                })
                output.error(f"Failed: {py_file.name} - {result.error}")

        return r[dict[str, Any]].ok(results)

    @override
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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Flext Refactor Engine - Declarative code transformation"
    )
    parser.add_argument(
        "--project", "-p", required=True, type=Path, help="Path do projeto a refatorar"
    )
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

    if not args.project.exists():
        output.error(f"Project not found: {args.project}")
        sys.exit(1)

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

    output.header(f"Refactoring: {args.project.name}")
    output.info(f"Loaded {len(engine.rules)} rules")
    output.info(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")

    # Executar refatoração
    result = engine.refactor_project(args.project, dry_run=args.dry_run)

    if not result.is_success:
        output.error(f"Refactor failed: {result.error}")
        sys.exit(1)

    results = result.value

    # Resumo
    output.header("Summary")
    output.info(f"Files processed: {results['files_processed']}")
    output.info(f"Modified: {len(results['modified'])}")
    output.info(f"Unchanged: {len(results['unchanged'])}")
    output.info(f"Failed: {len(results['failed'])}")

    if args.dry_run:
        output.info("\n[DRY-RUN] No changes applied")
    else:
        output.info("\nChanges applied successfully")

    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
