"""Refactor engine baseado em libcst e regras declarativas.

Uso:
    python -m flext_infra refactor --project ../projeto --dry-run
    python -m flext_infra refactor --project ../projeto
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import libcst as cst
import yaml
from libcst.codemod import CodemodContext

from flext_core import FlextService, r, t
from flext_infra.constants import c
from flext_infra.output import output


class RefactorRule:
    """Base para regras de refatoração carregadas de YAML."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rule_id = config.get("id", "unknown")
        self.enabled = config.get("enabled", True)

    def apply(self, tree: cst.Module) -> cst.Module:
        """Aplica a regra na AST. Deve ser sobrescrito."""
        return tree


class LegacyRemovalRule(RefactorRule):
    """Remove código legado: aliases, wrappers, deprecated."""

    def apply(self, tree: cst.Module) -> cst.Module:
        patterns = self.config.get("patterns", {})

        # Remover aliases
        if patterns.get("aliases"):
            tree = self._remove_aliases(tree)

        # Remover wrappers
        if patterns.get("wrappers"):
            tree = self._remove_wrappers(tree)

        # Remover deprecated
        if patterns.get("deprecated"):
            tree = self._remove_deprecated(tree)

        return tree

    def _remove_aliases(self, tree: cst.Module) -> cst.Module:
        """Remove OldName = NewName no nível do módulo."""

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
                    # Remover se for alias (nomes diferentes)
                    if target != value:
                        return cst.RemovalSentinel.REMOVE
                return updated_node

        return tree.visit(AliasRemover())

    def _remove_wrappers(self, tree: cst.Module) -> cst.Module:
        """Remove funções que só chamam outra função."""

        class WrapperRemover(cst.CSTTransformer):
            def leave_FunctionDef(self, original_node, updated_node):
                # Verificar se corpo tem apenas return func()
                body = original_node.body.body
                if (
                    len(body) == 1
                    and isinstance(body[0], cst.SimpleStatementLine)
                    and len(body[0].body) == 1
                    and isinstance(body[0].body[0], cst.Return)
                    and isinstance(body[0].body[0].value, cst.Call)
                ):
                    # É um wrapper - remover
                    return cst.RemovalSentinel.REMOVE
                return updated_node

        return tree.visit(WrapperRemover())

    def _remove_deprecated(self, tree: cst.Module) -> cst.Module:
        """Remove classes deprecated."""

        class DeprecatedRemover(cst.CSTTransformer):
            def leave_ClassDef(self, original_node, updated_node):
                # Verificar se nome contém Deprecated
                if "deprecated" in original_node.name.value.lower():
                    return cst.RemovalSentinel.REMOVE

                # Verificar se tem warnings.warn no corpo
                for stmt in original_node.body.body:
                    if isinstance(stmt, cst.SimpleStatementLine):
                        for line in stmt.body:
                            if isinstance(line, cst.Expr):
                                if isinstance(line.value, cst.Call):
                                    func = line.value.func
                                    if isinstance(func, cst.Attribute):
                                        if func.attr.value == "warn":
                                            return cst.RemovalSentinel.REMOVE

                return updated_node

        return tree.visit(DeprecatedRemover())


class ImportModernizerRule(RefactorRule):
    """Moderniza imports para usar runtime aliases."""

    def apply(self, tree: cst.Module) -> cst.Module:
        forbidden = self.config.get("forbidden_imports", [])

        if not forbidden:
            return tree

        # Coletar imports a substituir
        imports_to_remove = []
        symbols_to_replace: dict[str, str] = {}

        for rule in forbidden:
            module = rule.get("module", "")
            replacement = rule.get("replacement", "")
            symbols = rule.get("symbols", [])
            mapping = rule.get("symbol_mapping", {})

            imports_to_remove.append({
                "module": module,
                "replacement": replacement,
                "symbols": symbols,
            })

            for symbol, alias_path in mapping.items():
                symbols_to_replace[symbol] = alias_path

        # Remover imports antigos e substituir usos
        class ImportModernizer(cst.CSTTransformer):
            def __init__(self, imports_to_remove, symbols_to_replace):
                self.imports_to_remove = imports_to_remove
                self.symbols_to_replace = symbols_to_replace
                self.needs_runtime_import = False

            def leave_ImportFrom(self, original_node, updated_node):
                module_name = self._get_module_name(original_node.module)

                for rule in self.imports_to_remove:
                    if rule["module"] in module_name:
                        self.needs_runtime_import = True
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
                    return result
                return updated_node

            def _get_module_name(self, module):
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
                if self.needs_runtime_import:
                    # Adicionar import dos aliases no topo
                    new_import = cst.SimpleStatementLine(
                        body=[
                            cst.ImportFrom(
                                module=cst.Name("flext_core"), names=cst.ImportStar()
                            )
                        ]
                    )
                    new_body = [new_import] + list(updated_node.body)
                    return updated_node.with_changes(body=new_body)
                return updated_node

        modernizer = ImportModernizer(imports_to_remove, symbols_to_replace)
        return tree.visit(modernizer)


class FlextRefactorEngine(FlextService[dict[str, Any]]):
    """Engine de refatoração que orquestra regras declarativas."""

    def __init__(self, config_path: Path | None = None):
        super().__init__()
        self.config_path = config_path or self._default_config_path()
        self.config: dict[str, Any] = {}
        self.rules: list[RefactorRule] = []

    def _default_config_path(self) -> Path:
        """Retorna caminho padrão do config."""
        return Path(__file__).parent / "config.yml"

    def load_config(self) -> r[None]:
        """Carrega configuração do YAML."""
        try:
            self.config = yaml.safe_load(self.config_path.read_text())
            return r[None].ok(None)
        except Exception as e:
            return r[None].fail(f"Failed to load config: {e}")

    def load_rules(self) -> r[None]:
        """Carrega regras de arquivos YAML."""
        try:
            rules_dir = self.config_path.parent / "rules"

            for rule_file in rules_dir.glob("*.yml"):
                rule_config = yaml.safe_load(rule_file.read_text())
                rules = rule_config.get("rules", [])

                for rule_def in rules:
                    if not rule_def.get("enabled", True):
                        continue

                    rule_id = rule_def.get("id", "unknown")

                    # Instanciar regra apropriada
                    if (
                        "legacy" in rule_id
                        or "alias" in rule_id
                        or "wrapper" in rule_id
                        or "deprecated" in rule_id
                    ):
                        self.rules.append(LegacyRemovalRule(rule_def))
                    elif "import" in rule_id or "modernize" in rule_id:
                        self.rules.append(ImportModernizerRule(rule_def))

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
        results = {"modified": [], "failed": [], "unchanged": []}

        src_dir = project_path / "src"
        if not src_dir.exists():
            return r[dict[str, Any]].fail(f"No src/ directory in {project_path}")

        for py_file in src_dir.rglob("*.py"):
            # Ignorar arquivos especiais
            if py_file.name in {"__init__.py", "conftest.py"}:
                continue

            result = self.refactor_file(py_file, dry_run=dry_run)

            if result.is_success:
                original = py_file.read_text(encoding="utf-8")
                if result.value != original:
                    results["modified"].append(str(py_file.relative_to(project_path)))
                    output.info(
                        f"{'[DRY-RUN] ' if dry_run else ''}Modified: {py_file.name}"
                    )
                else:
                    results["unchanged"].append(str(py_file.relative_to(project_path)))
            else:
                results["failed"].append({
                    "file": str(py_file.relative_to(project_path)),
                    "error": result.error,
                })
                output.error(f"Failed: {py_file.name} - {result.error}")

        return r[dict[str, Any]].ok(results)

    def execute(self) -> r[dict[str, Any]]:
        """Executa refatoração (interface FlextService)."""
        # Carregar config e regras
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

    output.info(f"Loaded {len(engine.rules)} rules")

    # Executar refatoração
    result = engine.refactor_project(args.project, dry_run=args.dry_run)

    if not result.is_success:
        output.error(f"Refactor failed: {result.error}")
        sys.exit(1)

    results = result.value

    # Resumo
    output.header("Refactor Summary")
    output.info(f"Modified: {len(results['modified'])} files")
    output.info(f"Unchanged: {len(results['unchanged'])} files")
    output.info(f"Failed: {len(results['failed'])} files")

    if args.dry_run:
        output.info("\n[DRY-RUN] No changes applied")

    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
