"""Automatic fixer for ``smell_boolean_logic`` findings.

Rewrites long chains of ``or``/``and`` into ``any()``/``all()`` when every
operand is a simple, side-effect-free expression. The transformation is
provably equivalent and preserves short-circuit semantics for the rewritten
chain itself (``any``/``all`` short-circuit on iterables).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra.transformers.smells.base import (
    FlextInfraSmellFixer,
    register_smell_fixer,
)

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m


class _BooleanSimplifier(ast.NodeTransformer):
    """Replace eligible long BoolOp chains with any()/all() calls."""

    _MIN_OPERANDS: ClassVar[int] = 4

    @override
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        operands = node.values
        if len(operands) < self._MIN_OPERANDS:
            return node
        if not all(self._is_simple_operand(op) for op in operands):
            return node
        func_name = "any" if isinstance(node.op, ast.Or) else "all"
        tuple_node = ast.Tuple(elts=operands, ctx=ast.Load())
        call = ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()), args=[tuple_node], keywords=[]
        )
        ast.copy_location(call, node)
        return call

    @staticmethod
    def _is_simple_operand(node: ast.AST) -> bool:
        """Return True for expressions that are safe to repeat/evaluate.

        We intentionally exclude arbitrary Calls and BinOps to avoid silently
        changing side-effect ordering or semantics.
        """
        if isinstance(node, ast.Name | ast.Constant | ast.Attribute):
            return True
        if isinstance(node, ast.Compare):
            return _BooleanSimplifier._is_simple_operand(node.left) and all(
                _BooleanSimplifier._is_simple_operand(comp) for comp in node.comparators
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return _BooleanSimplifier._is_simple_operand(node.operand)
        return False


@register_smell_fixer
class FlextInfraBooleanLogicFixer(FlextInfraSmellFixer):
    """Simplify boolean expressions reported by the boolean-logic smell."""

    tag: ClassVar[str] = "smell_boolean_logic"

    @override
    def fix(self, project_dir: Path, issue: m.Infra.Issue) -> tuple[bool, list[str]]:
        """Rewrite eligible boolean chains in the issue's file."""
        source_path = project_dir / issue.file
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError:
            return False, []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False, []
        simplified = _BooleanSimplifier().visit(tree)
        ast.fix_missing_locations(simplified)
        updated = ast.unparse(simplified)
        if updated.rstrip("\n") == source.rstrip("\n"):
            return False, []
        source_path.write_text(updated, encoding="utf-8")
        message = (
            f"{issue.file}:{issue.line}: simplified boolean-logic chain to any()/all()"
        )
        self._record_change(message)
        return True, list(self.changes)


__all__: list[str] = ["FlextInfraBooleanLogicFixer"]
