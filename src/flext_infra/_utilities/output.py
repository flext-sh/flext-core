"""Terminal output utility with ANSI color detection and structured formatting.

Provides color-aware terminal output for infrastructure commands. All output
is written to sys.stderr to preserve stdout for machine-readable content.

Color detection priority:
    NO_COLOR env → FORCE_COLOR env → CI detection → isatty() → TERM env

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from typing import Final, TextIO

from flext_infra import c
from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal


class FlextInfraUtilitiesOutput:
    """Structured terminal output for infrastructure commands.

    All methods write to ``sys.stderr`` so that stdout remains clean for
    machine-readable output (JSON, CSV, exit codes).

    Uses c.Infra.Style for colors and symbols.
    """

    def __init__(
        self,
        *,
        use_color: bool | None = None,
        use_unicode: bool | None = None,
        stream: TextIO | None = None,
    ) -> None:
        """Initialize structured terminal output."""
        self.use_color = (
            FlextInfraUtilitiesTerminal.terminal_should_use_color()
            if use_color is None
            else use_color
        )
        self.use_unicode = (
            FlextInfraUtilitiesTerminal.terminal_should_use_unicode()
            if use_unicode is None
            else use_unicode
        )
        self.stream = sys.stderr if stream is None else stream
        # Cache style lookups
        style = c.Infra.Style
        self._reset = style.RESET if self.use_color else ""
        self._red = style.RED if self.use_color else ""
        self._green = style.GREEN if self.use_color else ""
        self._yellow = style.YELLOW if self.use_color else ""
        self._blue = style.BLUE if self.use_color else ""
        self._bold = style.BOLD if self.use_color else ""
        self._sym_ok = style.OK if self.use_unicode else "[OK]"
        self._sym_fail = style.FAIL if self.use_unicode else "[FAIL]"
        self._sym_warn = style.WARN if self.use_unicode else "[WARN]"
        self._sym_skip = style.SKIP if self.use_unicode else "[SKIP]"

    def debug(self, message: str) -> None:
        """Write a debug message.

        Args:
            message: Debug text.

        """
        self._write(f"{self._green}DEBUG{self._reset}: {message}")

    def error(self, message: str, detail: str | None = None) -> None:
        """Write an error message.

        Args:
            message: Primary error message.
            detail: Optional detail text shown on the next line.

        """
        self._write(f"{self._red}ERROR{self._reset}: {message}")
        if detail:
            self._write(f"  {detail}")

    def header(self, title: str) -> None:
        """Write a bold section header.

        Args:
            title: Section title text.

        """
        sep = "═" if self.use_unicode else "="
        line = sep * 60
        self._write("")
        self._write(f"{self._bold}{line}{self._reset}")
        self._write(f"{self._bold}  {title}{self._reset}")
        self._write(f"{self._bold}{line}{self._reset}")

    def info(self, message: str) -> None:
        """Write an informational message.

        Args:
            message: Information text.

        """
        self._write(f"{self._blue}INFO{self._reset}: {message}")

    def progress(self, index: int, total: int, project: str, verb: str) -> None:
        """Write a progress indicator line.

        Example::

            [01/33] flext-core check ...

        Args:
            index: Current 1-based index.
            total: Total number of items.
            project: Project identifier.
            verb: Operation name.

        """
        width = len(str(total))
        counter = f"[{index:0{width}d}/{total:0{width}d}]"
        self._write(f"{self._bold}{counter}{self._reset} {project} {verb} ...")

    def status(self, verb: str, project: str, result: bool, elapsed: float) -> None:
        """Write a formatted status line for a project operation.

        Example::

            ✓ check  flext-core  1.23s
            ✗ check  flext-api   0.45s

        Args:
            verb: Operation name (e.g. ``check``, ``test``, ``lint``).
            project: Project identifier.
            result: ``True`` for success, ``False`` for failure.
            elapsed: Duration in seconds.

        """
        if result:
            sym = f"{self._green}{self._sym_ok}{self._reset}"
        else:
            sym = f"{self._red}{self._sym_fail}{self._reset}"
        elapsed_str = f"{elapsed:.2f}s"
        self._write(f"  {sym} {verb:<8} {project:<24} {elapsed_str}")

    def summary(
        self,
        verb: str,
        total: int,
        success: int,
        failed: int,
        skipped: int,
        elapsed: float,
    ) -> None:
        """Write an operation summary with counts.

        Example::

            ── check summary ──
            Total: 33  Success: 30  Failed: 2  Skipped: 1  (12.34s)

        Args:
            verb: Operation name.
            total: Total project count.
            success: Successful count.
            failed: Failed count.
            skipped: Skipped count.
            elapsed: Total duration in seconds.

        """
        sep = "──" if self.use_unicode else "--"
        header = f"{self._bold}{sep} {verb} summary {sep}{self._reset}"
        self._write("")
        self._write(header)
        parts: list[str] = [f"Total: {total}"]
        success_str = f"{self._green}Success: {success}{self._reset}"
        parts.append(success_str)
        if failed > 0:
            failed_str = f"{self._red}Failed: {failed}{self._reset}"
        else:
            failed_str = f"Failed: {failed}"
        parts.append(failed_str)
        if skipped > 0:
            skipped_str = f"{self._yellow}Skipped: {skipped}{self._reset}"
        else:
            skipped_str = f"Skipped: {skipped}"
        parts.append(skipped_str)
        elapsed_str = f"({elapsed:.2f}s)"
        line = "  ".join(parts) + f"  {elapsed_str}"
        self._write(line)

    def warning(self, message: str) -> None:
        """Write a warning message.

        Args:
            message: Warning text.

        """
        self._write(f"{self._yellow}WARN{self._reset}: {message}")

    def _write(self, message: str) -> None:
        """Write a line to the output stream with newline."""
        self.stream.write(message + "\n")
        self.stream.flush()


output: Final[FlextInfraUtilitiesOutput] = FlextInfraUtilitiesOutput()
"Module-level singleton for direct use: ``from flext_infra import output``"
__all__ = [
    "FlextInfraUtilitiesOutput",
    "output",
]
