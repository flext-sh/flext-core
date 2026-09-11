"""Add explicit ``encoding="utf-8"`` to text-mode ``open()`` calls.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, ClassVar, Literal, override

from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorOpenEncoding(FlextInfraRopeTransformer):
    """Inject ``encoding="utf-8"`` into text-mode ``open()`` calls that omit it."""

    _description = "add encoding to text-mode open() calls"
    _MODE_ARG_INDEX_BUILTIN: ClassVar[int] = 1
    _MODE_ARG_INDEX_PATH: ClassVar[int] = 0

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Add encoding kwarg when missing and mode is known text."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, list(self.changes)
        targets = self._find_open_calls(tree)
        if not targets:
            return source, list(self.changes)
        lines = source.splitlines(keepends=True)
        for lineno, col_offset in sorted(targets, reverse=True):
            line_index = lineno - 1
            if line_index >= len(lines):
                continue
            line = lines[line_index]
            close = line.rfind(")", col_offset)
            if close == -1:
                continue
            lines[line_index] = f'{line[:close]}, encoding="utf-8"){line[close + 1 :]}'
            self._record_change('Added encoding="utf-8" to open() call')
        return "".join(lines), list(self.changes)

    @classmethod
    def _find_open_calls(cls, tree: ast.Module) -> list[tuple[int, int]]:
        """Return (lineno, col_offset) for text-mode open calls missing encoding."""
        targets: list[tuple[int, int]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_kind = cls._open_call_kind(node.func)
            if call_kind is None:
                continue
            if any(kw.arg == "encoding" for kw in node.keywords):
                continue
            if not cls._is_text_mode(cls._mode_expr(node, call_kind)):
                continue
            targets.append((node.lineno, node.col_offset))
        return targets

    @classmethod
    def _open_call_kind(cls, func: ast.expr) -> Literal["builtin", "path"] | None:
        """Return the supported open-call kind for ``func``."""
        if isinstance(func, ast.Name) and func.id == "open":
            return "builtin"
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "open"
            and cls._is_path_constructor_call(func.value)
        ):
            return "path"
        return None

    @staticmethod
    def _is_path_constructor_call(node: ast.expr) -> bool:
        """Return whether ``node`` is a direct pathlib ``Path(...)`` constructor."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Name) and func.id == "Path") or (
            isinstance(func, ast.Attribute) and func.attr == "Path"
        )

    @classmethod
    def _mode_expr(
        cls, node: ast.Call, call_kind: Literal["builtin", "path"]
    ) -> ast.expr | None:
        """Return the explicit mode expression, or None for default text mode."""
        for keyword in node.keywords:
            if keyword.arg == "mode":
                return keyword.value
        mode_index = (
            cls._MODE_ARG_INDEX_BUILTIN
            if call_kind == "builtin"
            else cls._MODE_ARG_INDEX_PATH
        )
        return node.args[mode_index] if len(node.args) > mode_index else None

    @staticmethod
    def _is_text_mode(mode: ast.expr | None) -> bool:
        """Return whether ``mode`` is statically known to be a text mode."""
        if mode is None:
            return True
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            return "b" not in mode.value
        return False


__all__: list[str] = ["FlextInfraRefactorOpenEncoding"]
