"""Extract error information from verb log files.

Provides utilities for reading the tail of orchestration log files
and extracting error-like lines for display in orchestrator output.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra.constants import c

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra.typings import t


class FlextInfraUtilitiesLogParser:
    """Extract error information from verb log files."""

    # Inline patterns moved to c.Infra.LogParser

    @staticmethod
    def append_process_failure(log_path: Path, failure: str) -> str:
        """Persist one process-lifecycle failure beside the child output."""
        diagnostic = f"ERROR: workspace process lifecycle failed: {failure}"
        try:
            with log_path.open("a", encoding=c.Cli.ENCODING_DEFAULT) as stream:
                stream.write(f"\n{diagnostic}\n")
        except OSError as exc:
            return f"{diagnostic}; diagnostic persistence failed: {exc}"
        return diagnostic

    @staticmethod
    def extract_errors(
        log_path: Path, *, max_lines: int = 5
    ) -> t.Pair[int, t.StrSequence]:
        """Read log tail and extract error lines.

        Args:
            log_path: Path to the log file.
            max_lines: Maximum number of error lines to return.

        Returns:
            Tuple of (total_error_count, first_n_error_lines).

        """
        if not log_path.exists():
            return (0, [])
        text = log_path.read_text(encoding=c.Cli.ENCODING_DEFAULT, errors="replace")
        tail = text.splitlines()[-c.Infra.LOG_TAIL_LINES :]
        error_lines: t.MutableSequenceOf[str] = []
        for line in tail:
            stripped = line.strip()
            if not stripped:
                continue
            if any(pattern.search(stripped) for pattern in c.Infra.LOG_NOISE_PATTERNS):
                continue
            if any(pattern.search(stripped) for pattern in c.Infra.LOG_ERROR_PATTERNS):
                error_lines.append(stripped)
        total = len(error_lines)
        return (total, error_lines[:max_lines])

    @staticmethod
    def extract_make_child_exit_code(log_path: Path) -> int | None:
        """Return the recipe exit code GNU make reported, when present.

        GNU make always exits 2 on a failed recipe, so the child's real exit
        code survives only in make's own error line. Returns None when the log
        is unreadable or carries no such line.
        """
        if not log_path.is_file():
            return None
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        matches = tuple(c.Infra.LOG_MAKE_CHILD_EXIT_PATTERN.finditer(content))
        if not matches:
            return None
        return int(matches[-1].group("code"))


__all__: list[str] = ["FlextInfraUtilitiesLogParser"]
