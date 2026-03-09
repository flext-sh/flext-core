"""Terminal output utility with ANSI color detection and structured formatting.

Static facade delegates to a module-level ``_OutputBackend`` singleton.
All output is written to sys.stderr to preserve stdout for machine-readable
content.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from typing import Final, TextIO

from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
from flext_infra.constants import FlextInfraConstants as c


class _OutputBackend:
    """Private output backend with instance state for color/unicode/stream."""

    __slots__ = (
        "blue",
        "bold",
        "green",
        "red",
        "reset",
        "stream",
        "sym_fail",
        "sym_ok",
        "sym_skip",
        "sym_warn",
        "use_color",
        "use_unicode",
        "yellow",
    )

    def __init__(
        self,
        *,
        use_color: bool | None = None,
        use_unicode: bool | None = None,
        stream: TextIO | None = None,
    ) -> None:
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
        style = c.Infra.Style
        self.reset = style.RESET if self.use_color else ""
        self.red = style.RED if self.use_color else ""
        self.green = style.GREEN if self.use_color else ""
        self.yellow = style.YELLOW if self.use_color else ""
        self.blue = style.BLUE if self.use_color else ""
        self.bold = style.BOLD if self.use_color else ""
        self.sym_ok = style.OK if self.use_unicode else "[OK]"
        self.sym_fail = style.FAIL if self.use_unicode else "[FAIL]"
        self.sym_warn = style.WARN if self.use_unicode else "[WARN]"
        self.sym_skip = style.SKIP if self.use_unicode else "[SKIP]"

    def write(self, message: str) -> None:
        """Write a line to the output stream with newline."""
        self.stream.write(message + "\n")
        self.stream.flush()


_backend: Final[_OutputBackend] = _OutputBackend()


class FlextInfraUtilitiesOutput:
    """Static output facade — delegates to ``_backend`` singleton.

    All methods are ``@staticmethod`` — exposed via ``u.Infra.info()`` etc.
    """

    @staticmethod
    def info(message: str) -> None:
        """Write an informational message."""
        _backend.write(f"{_backend.blue}INFO{_backend.reset}: {message}")

    @staticmethod
    def error(message: str, detail: str | None = None) -> None:
        """Write an error message with optional detail."""
        _backend.write(f"{_backend.red}ERROR{_backend.reset}: {message}")
        if detail:
            _backend.write(f"  {detail}")

    @staticmethod
    def warning(message: str) -> None:
        """Write a warning message."""
        _backend.write(f"{_backend.yellow}WARN{_backend.reset}: {message}")

    @staticmethod
    def debug(message: str) -> None:
        """Write a debug message."""
        _backend.write(f"{_backend.green}DEBUG{_backend.reset}: {message}")

    @staticmethod
    def header(title: str) -> None:
        """Write a bold section header."""
        sep = "═" if _backend.use_unicode else "="
        line = sep * 60
        _backend.write("")
        _backend.write(f"{_backend.bold}{line}{_backend.reset}")
        _backend.write(f"{_backend.bold}  {title}{_backend.reset}")
        _backend.write(f"{_backend.bold}{line}{_backend.reset}")

    @staticmethod
    def progress(index: int, total: int, project: str, verb: str) -> None:
        """Write a progress indicator line."""
        width = len(str(total))
        counter = f"[{index:0{width}d}/{total:0{width}d}]"
        _backend.write(
            f"{_backend.bold}{counter}{_backend.reset} {project} {verb} ...",
        )

    @staticmethod
    def status(verb: str, project: str, result: bool, elapsed: float) -> None:
        """Write a formatted status line for a project operation."""
        if result:
            sym = f"{_backend.green}{_backend.sym_ok}{_backend.reset}"
        else:
            sym = f"{_backend.red}{_backend.sym_fail}{_backend.reset}"
        _backend.write(f"  {sym} {verb:<8} {project:<24} {elapsed:.2f}s")

    @staticmethod
    def summary(
        verb: str,
        total: int,
        success: int,
        failed: int,
        skipped: int,
        elapsed: float,
    ) -> None:
        """Write an operation summary with counts."""
        sep = "──" if _backend.use_unicode else "--"
        hdr = f"{_backend.bold}{sep} {verb} summary {sep}{_backend.reset}"
        _backend.write("")
        _backend.write(hdr)
        parts: list[str] = [f"Total: {total}"]
        parts.append(f"{_backend.green}Success: {success}{_backend.reset}")
        if failed > 0:
            parts.append(f"{_backend.red}Failed: {failed}{_backend.reset}")
        else:
            parts.append(f"Failed: {failed}")
        if skipped > 0:
            parts.append(f"{_backend.yellow}Skipped: {skipped}{_backend.reset}")
        else:
            parts.append(f"Skipped: {skipped}")
        _backend.write("  ".join(parts) + f"  ({elapsed:.2f}s)")

    @staticmethod
    def gate_result(
        gate: str,
        count: int,
        passed: bool,
        elapsed: float,
    ) -> None:
        """Write per-gate result during check execution."""
        if passed:
            sym = f"{_backend.green}{_backend.sym_ok}{_backend.reset}"
        else:
            sym = f"{_backend.red}{_backend.sym_fail}{_backend.reset}"
        count_str = (
            f"{count:>5} errors"
            if count > 0
            else f"{_backend.green}    0{_backend.reset} errors"
        )
        _backend.write(f"    {sym} {gate:<10} {count_str}  ({elapsed:.2f}s)")


output: Final[FlextInfraUtilitiesOutput] = FlextInfraUtilitiesOutput()
"Module-level singleton for direct use: ``from flext_infra import output``"
__all__ = [
    "FlextInfraUtilitiesOutput",
    "output",
]
