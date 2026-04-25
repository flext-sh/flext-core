"""Lightweight AST helper — read + parse a Python source path safely.

Returns ``r.ok(ast.Module)`` on success, ``r.ok(None)`` when the path is
absent or unreadable (skip-on-missing-source contract per AGENTS.md
§3.8 Skepticism by Default), and ``r.fail(...)`` on ``SyntaxError`` so
callers decide whether to SKIP or HIT.

Used by both the beartype A-PT hooks (``check_*`` predicates) and the
upcoming ``MINIMAL_AST`` source dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path

from flext_core.result import FlextResult as r


class FlextUtilitiesLightweightAst:
    """Skip-on-missing-source AST loader for static-analysis predicates."""

    @classmethod
    def lightweight_ast_parse(cls, path: Path) -> r[ast.Module | None]:
        """Read ``path`` and return its parsed ``ast.Module``.

        ``r.ok(None)`` — missing or unreadable source (SKIP semantics).
        ``r.ok(ast.Module)`` — successfully parsed.
        ``r.fail(error)`` — source read but ``SyntaxError`` raised.
        """
        if not path.exists():
            return r[ast.Module | None].ok(None)
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            return r[ast.Module | None].ok(None)
        try:
            return r[ast.Module | None].ok(ast.parse(source))
        except SyntaxError as exc:
            return r[ast.Module | None].fail(
                f"SyntaxError parsing {path}: {exc}",
                exception=exc,
            )
