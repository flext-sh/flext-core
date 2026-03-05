"""Rule that removes legacy compatibility code patterns."""

from __future__ import annotations

from pathlib import Path
from typing import override

import libcst as cst

from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.alias_remover import (
    FlextInfraRefactorAliasRemover,
)
from flext_infra.refactor.transformers.deprecated_remover import (
    FlextInfraRefactorDeprecatedRemover,
)
from flext_infra.refactor.transformers.import_bypass_remover import (
    FlextInfraRefactorImportBypassRemover,
)


class FlextInfraRefactorLegacyRemovalRule(FlextInfraRefactorRule):
    """Remove aliases, deprecated classes, wrappers and import bypass blocks."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        """Apply configured legacy-removal transforms to module tree."""
        changes: list[str] = []
        fix_action = str(self.config.get("fix_action", "")).strip().lower()

        if "alias" in self.rule_id or fix_action == "remove":
            tree, alias_changes = self._remove_aliases(tree)
            changes.extend(alias_changes)

        if "deprecated" in self.rule_id or fix_action == "remove_and_update_refs":
            tree, deprecated_changes = self._remove_deprecated(tree)
            changes.extend(deprecated_changes)

        if "wrapper" in self.rule_id or fix_action == "inline_and_remove":
            tree, wrapper_changes = self._remove_wrappers(tree)
            changes.extend(wrapper_changes)

        if "bypass" in self.rule_id or fix_action == "keep_try_only":
            tree, bypass_changes = self._remove_import_bypasses(tree)
            changes.extend(bypass_changes)

        return tree, changes

    def _remove_wrappers(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        changes: list[str] = []

        new_body: list[cst.BaseStatement] = []
        for stmt in tree.body:
            if not isinstance(stmt, cst.FunctionDef):
                new_body.append(stmt)
                continue

            target_name = self._get_passthrough_target(stmt)
            if target_name is None:
                new_body.append(stmt)
                continue

            alias_assign = cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(stmt.name.value))],
                        value=cst.Name(target_name),
                    )
                ]
            )
            new_body.append(alias_assign)
            changes.append(f"Inlined wrapper: {stmt.name.value} -> {target_name}")

        return tree.with_changes(body=new_body), changes

    def _get_passthrough_target(self, func: cst.FunctionDef) -> str | None:
        if not isinstance(func.body, cst.IndentedBlock):
            return None
        if len(func.body.body) != 1:
            return None
        stmt = func.body.body[0]
        if not isinstance(stmt, cst.SimpleStatementLine):
            return None
        if len(stmt.body) != 1:
            return None

        small_stmt = stmt.body[0]
        if not isinstance(small_stmt, cst.Return):
            return None
        if not isinstance(small_stmt.value, cst.Call):
            return None
        if not isinstance(small_stmt.value.func, cst.Name):
            return None

        call_args = small_stmt.value.args
        param_names = [param.name.value for param in func.params.params]
        if len(call_args) != len(param_names):
            return None

        for idx, arg in enumerate(call_args):
            if arg.keyword is not None:
                return None
            if arg.star not in {"", None}:
                return None
            if not isinstance(arg.value, cst.Name):
                return None
            if arg.value.value != param_names[idx]:
                return None

        return small_stmt.value.func.value

    def _remove_import_bypasses(
        self,
        tree: cst.Module,
    ) -> tuple[cst.Module, list[str]]:
        transformer = FlextInfraRefactorImportBypassRemover()
        new_tree = tree.visit(transformer)
        return new_tree, transformer.changes

    def _remove_aliases(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        allow_aliases = set(self.config.get("allow_aliases", []))
        allow_target_suffixes = tuple(self.config.get("allow_target_suffixes", []))
        transformer = FlextInfraRefactorAliasRemover(
            allow_aliases=allow_aliases,
            allow_target_suffixes=allow_target_suffixes,
        )
        new_tree = tree.visit(transformer)
        return new_tree, transformer.changes

    def _remove_deprecated(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        transformer = FlextInfraRefactorDeprecatedRemover()
        new_tree = tree.visit(transformer)
        return new_tree, transformer.changes


__all__ = ["FlextInfraRefactorLegacyRemovalRule"]
