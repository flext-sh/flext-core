"""Tests for u.lightweight_ast_parse — source-optional AST helper.

Lane A-CH Phase 0 Task 0.2. Validates SKIP-on-missing-source semantics:
- missing path → r.ok(None)
- valid Python → r.ok(ast.Module)
- broken Python → r.fail(...)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests import u


class TestsFlextCoreUtilitiesLightweightAst:
    """Behavior contract for lightweight_ast_parse skip semantics."""

    def test_skip_when_source_missing_returns_ok_none(self, tmp_path: Path) -> None:
        result = u.lightweight_ast_parse(tmp_path / "missing.py")
        assert result.success is True
        assert result.value is None

    def test_parse_when_source_present_returns_module(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "ok.py"
        src.write_text("VALUE = 1\n", encoding="utf-8")
        result = u.lightweight_ast_parse(src)
        assert result.success is True
        assert isinstance(result.value, ast.Module)

    def test_parse_when_source_has_syntax_error_returns_fail(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "broken.py"
        src.write_text("def broken(:\n    pass\n", encoding="utf-8")
        result = u.lightweight_ast_parse(src)
        assert result.success is False
        assert result.error is not None
        assert "syntax error" in str(result.error).lower()
