"""Pattern correction rules for high-volume violations."""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import libcst as cst

from flext_infra.refactor.rule import FlextInfraRefactorRule


class _DictToMappingTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.changes: list[str] = []
        self._has_mapping_import = False

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module = node.module
        if not isinstance(module, cst.Attribute) and not isinstance(module, cst.Name):
            return
        module_name = cst.Module(body=[]).code_for_node(module)
        if module_name != "collections.abc":
            return
        names = node.names
        if isinstance(names, cst.ImportStar):
            self._has_mapping_import = True
            return
        for alias in names:
            if isinstance(alias.name, cst.Name):
                if alias.name.value == "Mapping":
                    self._has_mapping_import = True

    @override
    def leave_Annotation(
        self,
        original_node: cst.Annotation,
        updated_node: cst.Annotation,
    ) -> cst.Annotation:
        del original_node
        annotation = updated_node.annotation
        if not isinstance(annotation, cst.Subscript):
            return updated_node
        if not isinstance(annotation.value, cst.Name):
            return updated_node
        if annotation.value.value != "dict":
            return updated_node

        replacement = annotation.with_changes(value=cst.Name("Mapping"))
        self.changes.append("Converted annotation dict[...] to Mapping[...]")
        return updated_node.with_changes(annotation=replacement)

    @override
    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        del original_node
        if not self.changes or self._has_mapping_import:
            return updated_node

        import_stmt = cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Attribute(
                        value=cst.Name("collections"),
                        attr=cst.Name("abc"),
                    ),
                    names=[cst.ImportAlias(name=cst.Name("Mapping"))],
                )
            ]
        )
        body = list(updated_node.body)
        insert_at = 0
        if body and isinstance(body[0], cst.SimpleStatementLine):
            if body[0].body and isinstance(body[0].body[0], cst.Expr):
                expr = body[0].body[0].value
                if isinstance(expr, cst.SimpleString):
                    insert_at = 1
                    if len(body) > 1 and isinstance(body[1], cst.EmptyLine):
                        insert_at = 2
        body.insert(insert_at, import_stmt)
        return updated_node.with_changes(body=body)


class _RedundantCastRemover(cst.CSTTransformer):
    def __init__(self, removable_types: set[str]) -> None:
        self.removable_types = removable_types
        self.changes: list[str] = []

    @override
    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call,
    ) -> cst.BaseExpression:
        del original_node
        func = updated_node.func
        if not isinstance(func, cst.Name) or func.value != "cast":
            return updated_node
        if len(updated_node.args) != 2:
            return updated_node
        type_arg, value_arg = updated_node.args
        if type_arg.keyword is not None or value_arg.keyword is not None:
            return updated_node
        if not isinstance(type_arg.value, cst.SimpleString):
            return updated_node

        target = type_arg.value.evaluated_value
        if not isinstance(target, str):
            return updated_node
        if target not in self.removable_types:
            return updated_node

        self.changes.append(f"Removed redundant cast for {target}")
        return value_arg.value


class FlextInfraRefactorPatternCorrectionsRule(FlextInfraRefactorRule):
    """Apply corrective refactors for high-volume pattern violations."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        fix_action = str(self.config.get("fix_action", "")).strip().lower()
        if fix_action == "convert_dict_to_mapping_annotations":
            transformer = _DictToMappingTransformer()
            updated = tree.visit(transformer)
            return updated, transformer.changes

        if fix_action == "remove_redundant_casts":
            raw_types = self.config.get("redundant_type_targets", [])
            removable_types = {
                str(item)
                for item in cast("list[object]", raw_types)
                if isinstance(item, str)
            }
            transformer = _RedundantCastRemover(removable_types=removable_types)
            updated = tree.visit(transformer)
            return updated, transformer.changes

        return tree, []


__all__ = ["FlextInfraRefactorPatternCorrectionsRule"]
