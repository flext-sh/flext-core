"""Post-apply formatting helpers for refactor census fixes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class FlextInfraRefactorCensusApplyFormattingMixin:
    """Mixin for normalizing files touched by census apply operations."""

    @staticmethod
    def _ruff_fix_touched_files(paths: Iterable[Path]) -> None:
        """Normalize trailing newlines + import sort on touched files."""
        existing = sorted({str(path) for path in paths if path.is_file()})
        if not existing:
            return
        check_result = u.Cli.run_raw(
            ["ruff", "check", "--fix", "--select", "I,W", *existing],
            timeout=c.Infra.TIMEOUT_SHORT,
        )
        if check_result.failure:
            msg = (
                "ruff check --fix failed after refactor apply: "
                f"{check_result.error or 'unknown error'}; files={existing!r}"
            )
            raise RuntimeError(msg)
        format_result = u.Cli.run_raw(
            ["ruff", "format", *existing], timeout=c.Infra.TIMEOUT_SHORT
        )
        if format_result.failure:
            msg = (
                "ruff format failed after refactor apply: "
                f"{format_result.error or 'unknown error'}; files={existing!r}"
            )
            raise RuntimeError(msg)


__all__: list[str] = ["FlextInfraRefactorCensusApplyFormattingMixin"]
