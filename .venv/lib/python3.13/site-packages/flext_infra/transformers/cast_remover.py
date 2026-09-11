"""Remove ``typing.cast()`` calls outside flext-core result internals.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, override

from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorCastRemover(FlextInfraRopeTransformer):
    """Rewrite ``cast(T, expr)`` and ``typing.cast(T, expr)`` to ``expr``.

    ``typing.cast`` has no runtime effect; removing it is runtime-equivalent.
    The underlying typing issue must still be resolved by the developer, but
    this transformer eliminates the forbidden ``cast()`` surface mechanically.
    """

    _description = "remove typing.cast calls"

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Return source with all ``cast()`` calls replaced by their argument."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, list(self.changes)
        rewriter = _CastCallRewriter()
        new_tree = rewriter.visit(tree)
        if not rewriter.removed_count:
            return source, list(self.changes)
        ast.fix_missing_locations(new_tree)
        updated = ast.unparse(new_tree)
        self._record_change(f"Removed {rewriter.removed_count} cast() call(s)")
        return updated, list(self.changes)


class _CastCallRewriter(ast.NodeTransformer):
    """AST visitor that replaces ``cast(T, value)`` with ``value``."""

    def __init__(self) -> None:
        """Initialize removal counter."""
        super().__init__()
        self.removed_count = 0

    @override
    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Replace a cast() call with its second positional argument."""
        if self._is_cast_call(node):
            self.removed_count += 1
            # cast(T, value, ...) -> value
            if node.args:
                return node.args[1] if len(node.args) > 1 else node.args[0]
        return self.generic_visit(node)

    @staticmethod
    def _is_cast_call(node: ast.Call) -> bool:
        """Return True when the call is ``cast(...)`` or ``typing.cast(...)``."""
        func = node.func
        return (isinstance(func, ast.Name) and func.id == "cast") or (
            isinstance(func, ast.Attribute)
            and func.attr == "cast"
            and isinstance(func.value, ast.Name)
            and func.value.id == "typing"
        )


__all__: list[str] = ["FlextInfraRefactorCastRemover"]
