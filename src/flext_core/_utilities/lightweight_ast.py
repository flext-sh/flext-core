"""Lightweight ``ast.parse`` with source-optional skip semantics.

Used by ``MINIMAL_AST`` enforcement source kinds. When the source file is
not on disk (zipped wheel, frozen module, namespace package without
``__file__``) the parse skips and returns ``r.ok(None)``. Callers translate
``None`` into ``ViolationOutcome.SKIP``. Real ``SyntaxError`` is reported via
``r.fail(...)`` so detectors don't silently miss broken sources.

Per AGENTS.md §3.3 fallible operations return ``r[T]``. Per §3.2 no ``Any``.
Exposed via ``u.lightweight_ast_parse(path)``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path

from flext_core._constants.serialization import FlextConstantsSerialization
from flext_core.result import FlextResult


class FlextUtilitiesLightweightAst:
    """Source-optional ``ast.parse`` helper used by minimal-AST detectors."""

    @staticmethod
    def lightweight_ast_parse(source: Path) -> FlextResult[ast.Module | None]:
        """Parse ``source`` if readable; otherwise emit a SKIP-shaped result.

        Returns:
            ``r.ok(ast.Module)`` when the source is parseable.
            ``r.ok(None)`` when the source is missing or unreadable
                (caller emits ``ViolationOutcome.SKIP``).
            ``r.fail(...)`` when the source is present but contains a
                ``SyntaxError`` — caller decides whether this is a HIT or SKIP
                for the specific rule.

        """
        if not source.exists():
            return FlextResult[ast.Module | None].ok(None)
        try:
            text = source.read_text(
                encoding=FlextConstantsSerialization.DEFAULT_ENCODING
            )
        except OSError:
            return FlextResult[ast.Module | None].ok(None)
        try:
            return FlextResult[ast.Module | None].ok(
                ast.parse(text, filename=str(source))
            )
        except SyntaxError as exc:
            return FlextResult[ast.Module | None].fail(
                f"syntax error in {source}: {exc.msg}",
                exception=exc,
            )


__all__: list[str] = ["FlextUtilitiesLightweightAst"]
