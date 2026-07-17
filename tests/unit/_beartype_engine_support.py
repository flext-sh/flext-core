"""Shared beartype engine test helpers."""

from __future__ import annotations

import sys
import typing

from tests import m
from tests import p
from tests import t
from tests import u

if typing.TYPE_CHECKING:
    from pathlib import Path

type AnyAlias = str | typing.Any
type CleanAlias = str | int
type NestedAnyAlias = t.MappingKV[str, typing.Any]


class TestsFlextBeartypeEngine:
    """Shared beartype engine test support."""

    FORBIDDEN: frozenset[str] = frozenset({"dict", "list", "set"})

    @staticmethod
    def _run_python(script: str, cwd: Path) -> p.Cli.CommandOutput:
        """Run a Python snippet in a subprocess and capture text output."""
        result = u.Cli.run_raw([sys.executable, "-c", script], cwd=cwd)
        if result.success:
            return result.value
        return m.Cli.CommandOutput(
            stdout="",
            stderr=result.error or "python snippet execution failed",
            exit_code=1,
        )
