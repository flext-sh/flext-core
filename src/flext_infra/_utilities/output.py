"""Terminal output utility with ANSI color detection and structured formatting.

Static facade delegates to a module-level ``OutputBackend`` singleton.
All output is written to sys.stderr to preserve stdout for machine-readable
content.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TextIO, TypeAlias

from flext_infra import t
from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
from flext_infra.constants import FlextInfraConstants as c

MetricValue: TypeAlias = t.Infra.MetricValue


class OutputBackend:
    """Private output backend with instance state for color/unicode/stream.

    All formatting methods live here as instance methods. Tests create
    instances directly with custom config; production uses the module-level
    ``_backend`` singleton through the ``FlextInfraUtilitiesOutput`` facade.
    """

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

    def info(self, message: str) -> None:
        """Write an informational message."""
        self.write(f"{self.blue}INFO{self.reset}: {message}")

    def error(self, message: str, detail: str | None = None) -> None:
        """Write an error message with optional detail."""
        self.write(f"{self.red}ERROR{self.reset}: {message}")
        if detail:
            self.write(f"  {detail}")

    def warning(self, message: str) -> None:
        """Write a warning message."""
        self.write(f"{self.yellow}WARN{self.reset}: {message}")

    def debug(self, message: str) -> None:
        """Write a debug message."""
        self.write(f"{self.green}DEBUG{self.reset}: {message}")

    def header(self, title: str) -> None:
        """Write a bold section header."""
        sep = "═" if self.use_unicode else "="
        line = sep * 60
        self.write("")
        self.write(f"{self.bold}{line}{self.reset}")
        self.write(f"{self.bold}  {title}{self.reset}")
        self.write(f"{self.bold}{line}{self.reset}")

    def progress(self, index: int, total: int, project: str, verb: str) -> None:
        """Write a progress indicator line."""
        width = len(str(total))
        counter = f"[{index:0{width}d}/{total:0{width}d}]"
        self.write(f"{self.bold}{counter}{self.reset} {project} {verb} ...")

    def status(self, verb: str, project: str, result: bool, elapsed: float) -> None:
        """Write a formatted status line for a project operation."""
        sym = (
            f"{self.green}{self.sym_ok}{self.reset}"
            if result
            else f"{self.red}{self.sym_fail}{self.reset}"
        )
        self.write(f"  {sym} {verb:<8} {project:<24} {elapsed:.2f}s")

    def summary(
        self,
        verb: str,
        total: int,
        success: int,
        failed: int,
        skipped: int,
        elapsed: float,
    ) -> None:
        """Write an operation summary with counts."""
        sep = "──" if self.use_unicode else "--"
        hdr = f"{self.bold}{sep} {verb} summary {sep}{self.reset}"
        self.write("")
        self.write(hdr)
        parts: list[str] = [f"Total: {total}"]
        parts.append(f"{self.green}Success: {success}{self.reset}")
        if failed > 0:
            parts.append(f"{self.red}Failed: {failed}{self.reset}")
        else:
            parts.append(f"Failed: {failed}")
        if skipped > 0:
            parts.append(f"{self.yellow}Skipped: {skipped}{self.reset}")
        else:
            parts.append(f"Skipped: {skipped}")
        self.write("  ".join(parts) + f"  ({elapsed:.2f}s)")

    def gate_result(
        self,
        gate: str,
        count: int,
        passed: bool,
        elapsed: float,
    ) -> None:
        """Write per-gate result during check execution."""
        sym = (
            f"{self.green}{self.sym_ok}{self.reset}"
            if passed
            else f"{self.red}{self.sym_fail}{self.reset}"
        )
        count_str = (
            f"{count:>5} errors"
            if count > 0
            else f"{self.green}    0{self.reset} errors"
        )
        self.write(f"    {sym} {gate:<10} {count_str}  ({elapsed:.2f}s)")

    def metrics(
        self,
        *instances: t.Infra.MetricRecord,
        **kwargs: MetricValue,
    ) -> None:
        """Write key-value metrics for machine-readable output."""
        for item in instances:
            iterable = item.items() if isinstance(item, Mapping) else item
            for key, value in iterable:
                if isinstance(value, str | int | float | bool | Path) or value is None:
                    sys.stdout.write(f"{key}={value}\n")

        for key, value in kwargs.items():
            if isinstance(value, str | int | float | bool | Path) or value is None:
                sys.stdout.write(f"{key}={value}\n")

        sys.stdout.flush()


_backend: Final[OutputBackend] = OutputBackend()


class FlextInfraUtilitiesOutput:
    """Static output facade — delegates to ``_backend`` singleton.

    All methods are ``@staticmethod`` — exposed via ``u.Infra.info()`` etc.
    """

    @staticmethod
    def write(message: str) -> None:
        """Write raw message."""
        _backend.write(message)

    @staticmethod
    def info(message: str) -> None:
        """Write an informational message."""
        _backend.info(message)

    @staticmethod
    def error(message: str, detail: str | None = None) -> None:
        """Write an error message with optional detail."""
        _backend.error(message, detail)

    @staticmethod
    def warning(message: str) -> None:
        """Write a warning message."""
        _backend.warning(message)

    @staticmethod
    def debug(message: str) -> None:
        """Write a debug message."""
        _backend.debug(message)

    @staticmethod
    def header(title: str) -> None:
        """Write a bold section header."""
        _backend.header(title)

    @staticmethod
    def progress(index: int, total: int, project: str, verb: str) -> None:
        """Write a progress indicator line."""
        _backend.progress(index, total, project, verb)

    @staticmethod
    def status(verb: str, project: str, result: bool, elapsed: float) -> None:
        """Write a formatted status line for a project operation."""
        _backend.status(verb, project, result, elapsed)

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
        _backend.summary(verb, total, success, failed, skipped, elapsed)

    @staticmethod
    def gate_result(gate: str, count: int, passed: bool, elapsed: float) -> None:
        """Write per-gate result during check execution."""
        _backend.gate_result(gate, count, passed, elapsed)

    @staticmethod
    def metrics(
        *instances: t.Infra.MetricRecord,
        **kwargs: MetricValue,
    ) -> None:
        """Write key-value metrics for machine readable stdout."""
        _backend.metrics(*instances, **kwargs)


output: Final[FlextInfraUtilitiesOutput] = FlextInfraUtilitiesOutput()
"Module-level singleton for direct use: ``from flext_infra import output``"
__all__ = [
    "FlextInfraUtilitiesOutput",
    "OutputBackend",
    "output",
]
