"""LibCST transformations for migrate-to-mro."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from flext_infra import c

from .scanner import MROFileScan


@dataclass(frozen=True)
class MROFileMigration:
    file: str
    module: str
    moved_symbols: tuple[str, ...]
    created_classes: tuple[str, ...]


@dataclass(frozen=True)
class _MoveItem:
    line: int
    symbol: str
    class_name: str
    statement: cst.BaseStatement


class _ModuleStatementMover(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, *, move_items: list[_MoveItem]) -> None:
        self._move_lines = {item.line for item in move_items}
        self._by_class: dict[str, list[cst.BaseStatement]] = {}
        for item in move_items:
            self._by_class.setdefault(item.class_name, []).append(item.statement)
        self.touched_classes: set[str] = set()

    def leave_SimpleStatementLine(
        self,
        original_node: Any,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.BaseStatement | cst.RemovalSentinel:
        pos = self.get_metadata(PositionProvider, original_node, None)
        if pos is not None and pos.start.line in self._move_lines:
            return cst.RemoveFromParent()
        return updated_node

    def leave_ClassDef(
        self,
        original_node: Any,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        class_name = original_node.name.value
        additions = self._by_class.get(class_name)
        if additions is None:
            return updated_node
        self.touched_classes.add(class_name)
        return updated_node.with_changes(
            body=updated_node.body.with_changes(
                body=[*updated_node.body.body, *additions]
            )
        )


class FlextInfraRefactorMROTransformer:
    @staticmethod
    def migrate_file(
        *,
        scan_result: MROFileScan,
    ) -> tuple[str, MROFileMigration, dict[str, str]]:
        file_path = Path(scan_result.file)
        source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        tree = ast.parse(source)

        by_line = {candidate.line: candidate for candidate in scan_result.candidates}
        lines = source.splitlines(keepends=True)
        move_items: list[_MoveItem] = []
        symbol_alias_map: dict[str, str] = {}
        for stmt in tree.body:
            line = getattr(stmt, "lineno", None)
            if not isinstance(line, int):
                continue
            candidate = by_line.get(line)
            if candidate is None:
                continue
            end_line = getattr(stmt, "end_lineno", line)
            segment = "".join(lines[line - 1 : end_line])
            parsed = cst.parse_statement(segment)
            move_items.append(
                _MoveItem(
                    line=line,
                    symbol=candidate.symbol,
                    class_name=candidate.class_name,
                    statement=parsed,
                )
            )
            symbol_alias_map[candidate.symbol] = (
                "c" if candidate.kind == "constant" else "t"
            )

        if len(move_items) == 0:
            return (
                source,
                MROFileMigration(
                    file=scan_result.file,
                    module=scan_result.module,
                    moved_symbols=(),
                    created_classes=(),
                ),
                {},
            )

        module = cst.parse_module(source)
        mover = _ModuleStatementMover(move_items=move_items)
        updated = MetadataWrapper(module).visit(mover)

        missing_classes = sorted({
            item.class_name
            for item in move_items
            if item.class_name not in mover.touched_classes
        })
        if len(missing_classes) > 0:
            updated = FlextInfraRefactorMROTransformer._append_missing_classes(
                module=updated,
                class_names=missing_classes,
                move_items=move_items,
            )

        migration = MROFileMigration(
            file=scan_result.file,
            module=scan_result.module,
            moved_symbols=tuple(sorted(symbol_alias_map)),
            created_classes=tuple(missing_classes),
        )
        return updated.code, migration, symbol_alias_map

    @staticmethod
    def _append_missing_classes(
        *,
        module: cst.Module,
        class_names: list[str],
        move_items: list[_MoveItem],
    ) -> cst.Module:
        by_class: dict[str, list[cst.BaseStatement]] = {}
        for item in move_items:
            by_class.setdefault(item.class_name, []).append(item.statement)

        body = list(module.body)
        for class_name in class_names:
            class_body = by_class.get(class_name, [])
            if len(class_body) == 0:
                continue
            template = cst.parse_statement(f"class {class_name}:\n    pass\n")
            if not isinstance(template, cst.ClassDef):
                continue
            body.append(
                template.with_changes(body=template.body.with_changes(body=class_body))
            )
        return module.with_changes(body=body)


__all__ = ["FlextInfraRefactorMROTransformer", "MROFileMigration"]
