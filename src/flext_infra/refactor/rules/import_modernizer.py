"""Rule that modernizes imports into runtime alias references."""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import libcst as cst
from libcst.metadata import MetadataWrapper

from flext_infra import c
from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.import_modernizer import (
    FlextInfraRefactorImportModernizer,
)
from flext_infra.refactor.transformers.lazy_import_fixer import (
    FlextInfraRefactorLazyImportFixer,
)


class FlextInfraRefactorImportModernizerRule(FlextInfraRefactorRule):
    """Modernize forbidden imports and map symbols to runtime aliases."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        """Apply import modernizer or lazy-import hoisting based on fix action."""
        fix_action = str(self.config.get("fix_action", "")).strip().lower()
        if "lazy-import" in self.rule_id or fix_action == "hoist_to_module_top":
            return self._fix_lazy_imports(tree)

        runtime_aliases = set(c.Infra.Refactor.RUNTIME_ALIAS_NAMES)
        blocked_aliases = self._collect_blocked_aliases(tree, runtime_aliases)
        blocked_aliases.update(
            self._collect_function_shadowed_aliases(tree, runtime_aliases)
        )

        forbidden = self.config.get("forbidden_imports")
        if forbidden is None:
            forbidden = [self.config]
        if not forbidden:
            return tree, []

        imports_to_remove: list[str] = []
        symbols_to_replace: dict[str, str] = {}
        for rule_config in cast("list[dict[str, object]]", forbidden):
            module_name = rule_config.get("module", "")
            mapping = rule_config.get("symbol_mapping", {})
            imports_to_remove.append(str(module_name))
            if isinstance(mapping, dict):
                for k, v in cast("dict[str, object]", mapping).items():
                    symbols_to_replace[k] = str(v)

        transformer = FlextInfraRefactorImportModernizer(
            imports_to_remove=imports_to_remove,
            symbols_to_replace=symbols_to_replace,
            runtime_aliases=runtime_aliases,
            blocked_aliases=blocked_aliases,
        )
        wrapper = MetadataWrapper(tree)
        return wrapper.visit(transformer), transformer.changes

    def _bound_name_from_import_alias(self, alias: cst.ImportAlias) -> str:
        if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
            return alias.asname.name.value
        if isinstance(alias.name, cst.Name):
            return alias.name.value
        return alias.name.attr.value

    def _collect_blocked_aliases(
        self,
        tree: cst.Module,
        runtime_aliases: set[str],
    ) -> set[str]:
        blocked_aliases: set[str] = set()

        for stmt in tree.body:
            if isinstance(stmt, cst.ImportFrom):
                continue

            if not isinstance(stmt, cst.SimpleStatementLine):
                if (
                    isinstance(stmt, cst.FunctionDef)
                    and stmt.name.value in runtime_aliases
                ):
                    blocked_aliases.add(stmt.name.value)
                if (
                    isinstance(stmt, cst.ClassDef)
                    and stmt.name.value in runtime_aliases
                ):
                    blocked_aliases.add(stmt.name.value)
                continue

            for small_stmt in stmt.body:
                if isinstance(small_stmt, cst.ImportFrom):
                    module_name = self._module_name_from_expr(small_stmt.module)
                    if module_name == "flext_core":
                        continue
                    if isinstance(small_stmt.names, cst.ImportStar):
                        continue
                    for alias in tuple(small_stmt.names):
                        bound_name = self._bound_name_from_import_alias(alias)
                        if bound_name in runtime_aliases:
                            blocked_aliases.add(bound_name)
                    continue

                if isinstance(small_stmt, cst.Import):
                    for alias in small_stmt.names:
                        bound_name = self._bound_name_from_import_alias(alias)
                        if bound_name in runtime_aliases:
                            blocked_aliases.add(bound_name)
                    continue

                if isinstance(small_stmt, cst.Assign):
                    for assign_target in small_stmt.targets:
                        target = assign_target.target
                        if (
                            isinstance(target, cst.Name)
                            and target.value in runtime_aliases
                        ):
                            blocked_aliases.add(target.value)
                    continue

                if isinstance(small_stmt, cst.AnnAssign):
                    target = small_stmt.target
                    if isinstance(target, cst.Name) and target.value in runtime_aliases:
                        blocked_aliases.add(target.value)

        return blocked_aliases

    def _collect_function_shadowed_aliases(
        self,
        tree: cst.Module,
        runtime_aliases: set[str],
    ) -> set[str]:
        shadowed_aliases: set[str] = set()

        class FunctionShadowCollector(cst.CSTVisitor):
            def __init__(self) -> None:
                self._function_depth = 0

            @override
            def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
                del original_node
                self._function_depth -= 1

            @override
            def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
                if self._function_depth == 0:
                    return
                target = node.target
                if isinstance(target, cst.Name) and target.value in runtime_aliases:
                    shadowed_aliases.add(target.value)

            @override
            def visit_Assign(self, node: cst.Assign) -> None:
                if self._function_depth == 0:
                    return
                for assign_target in node.targets:
                    target = assign_target.target
                    if isinstance(target, cst.Name) and target.value in runtime_aliases:
                        shadowed_aliases.add(target.value)

            @override
            def visit_For(self, node: cst.For) -> None:
                if self._function_depth == 0:
                    return
                target = node.target
                if isinstance(target, cst.Name) and target.value in runtime_aliases:
                    shadowed_aliases.add(target.value)

            @override
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                self._function_depth += 1
                param_names = [
                    param.name.value
                    for param in (
                        list(node.params.params)
                        + list(node.params.posonly_params)
                        + list(node.params.kwonly_params)
                    )
                ]
                if isinstance(node.params.star_arg, cst.Param):
                    param_names.append(node.params.star_arg.name.value)
                if isinstance(node.params.star_kwarg, cst.Param):
                    param_names.append(node.params.star_kwarg.name.value)

                for param_name in param_names:
                    if param_name in runtime_aliases:
                        shadowed_aliases.add(param_name)

        tree.visit(FunctionShadowCollector())
        return shadowed_aliases

    def _fix_lazy_imports(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        transformer = FlextInfraRefactorLazyImportFixer()
        new_tree = tree.visit(transformer)
        return new_tree, transformer.changes

    def _module_name_from_expr(self, module: cst.BaseExpression | None) -> str:
        if module is None:
            return ""
        if isinstance(module, cst.Name):
            return module.value
        if isinstance(module, cst.Attribute):
            parts: list[str] = []
            current: cst.BaseExpression | cst.BaseAssignTargetExpression = module
            while isinstance(current, cst.Attribute):
                parts.append(current.attr.value)
                current = current.value
            if isinstance(current, cst.Name):
                parts.append(current.value)
            return ".".join(reversed(parts))
        return ""


__all__ = ["FlextInfraRefactorImportModernizerRule"]
