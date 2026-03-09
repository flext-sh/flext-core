"""Rule ensuring future annotations import in module header."""

from __future__ import annotations

from pathlib import Path
from typing import override

import libcst as cst

from flext_infra.refactor.rule import FlextInfraRefactorRule


class FlextInfraRefactorEnsureFutureAnnotationsRule(FlextInfraRefactorRule):
    """Ensure `from __future__ import annotations` exists and is properly placed."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        """Ensure future annotations import exists after docstring/header."""
        changes: list[str] = []
        body = list(tree.body)
        insert_idx = 0
        has_docstring = False
        if (
            body
            and isinstance(body[0], cst.SimpleStatementLine)
            and (len(body[0].body) == 1)
            and isinstance(body[0].body[0], cst.Expr)
            and isinstance(body[0].body[0].value, cst.SimpleString)
        ):
            has_docstring = True
            insert_idx = 1
        existing_annotations_stmt: cst.SimpleStatementLine | None = None
        non_annotation_future_stmts: list[cst.BaseStatement] = []
        body_without_future: list[cst.BaseStatement] = []
        for stmt in body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                body_without_future.append(stmt)
                continue
            if (
                len(stmt.body) == 1
                and isinstance(stmt.body[0], cst.ImportFrom)
                and isinstance(stmt.body[0].module, cst.Name)
                and (stmt.body[0].module.value == "__future__")
            ):
                import_from = stmt.body[0]
                aliases: tuple[cst.ImportAlias, ...] = ()
                if not isinstance(import_from.names, cst.ImportStar):
                    aliases = tuple(import_from.names)
                contains_annotations = any(
                    isinstance(alias.name, cst.Name)
                    and alias.name.value == "annotations"
                    for alias in aliases
                )
                if contains_annotations:
                    existing_annotations_stmt = stmt
                else:
                    non_annotation_future_stmts.append(stmt)
                continue
            body_without_future.append(stmt)
        needs_leading_blank_line = has_docstring
        if existing_annotations_stmt is not None:
            annotations_stmt = existing_annotations_stmt
        else:
            annotations_stmt = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name("__future__"),
                        names=[cst.ImportAlias(name=cst.Name("annotations"))],
                    ),
                ],
            )
            changes.append("Ensured: from __future__ import annotations")
        if needs_leading_blank_line:
            annotations_stmt = annotations_stmt.with_changes(
                leading_lines=[cst.EmptyLine()],
            )
        future_block = [annotations_stmt, *non_annotation_future_stmts]
        new_body = (
            body_without_future[:insert_idx]
            + future_block
            + body_without_future[insert_idx:]
        )
        if (
            new_body != body
            and "Ensured: from __future__ import annotations" not in changes
        ):
            changes.append("Moved: from __future__ import annotations")
        return (tree.with_changes(body=new_body), changes)


__all__ = ["FlextInfraRefactorEnsureFutureAnnotationsRule"]
