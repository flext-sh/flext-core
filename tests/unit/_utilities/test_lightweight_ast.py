"""Lightweight AST helper behavior contract.

Covers SKIP semantics on missing/unreadable source (`r.ok(None)`),
successful parse (`r.ok(ast.Module)`), and SyntaxError surfaced via
`r.fail(...)` so callers decide whether to SKIP or HIT.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests import u


class TestsFlextCoreUtilitiesLightweightAst:
    """Behavior contract for FlextUtilitiesLightweightAst.lightweight_ast_parse."""

    def test_missing_path_returns_ok_none(self, tmp_path: Path) -> None:
        absent = tmp_path / "does_not_exist.py"
        result = u.lightweight_ast_parse(absent)
        assert result.success is True
        assert result.value is None

    def test_valid_source_returns_module(self, tmp_path: Path) -> None:
        sample = tmp_path / "sample.py"
        sample.write_text("x = 1\n", encoding="utf-8")
        result = u.lightweight_ast_parse(sample)
        assert result.success is True
        assert isinstance(result.value, ast.Module)

    def test_syntax_error_returns_fail(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.py"
        broken.write_text("def x(:\n", encoding="utf-8")
        result = u.lightweight_ast_parse(broken)
        assert result.success is False
        assert result.error
