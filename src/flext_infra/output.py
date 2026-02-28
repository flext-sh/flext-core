"""Terminal output utility with ANSI color detection and structured formatting.

Provides color-aware terminal output for infrastructure commands. All output
is written to sys.stderr to preserve stdout for machine-readable content.

Color detection priority:
    NO_COLOR env → FORCE_COLOR env → CI detection → isatty() → TERM env

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import sys
from typing import Final, TextIO


def _should_use_color(stream: TextIO | None = None) -> bool:
    """Detect whether ANSI colors should be used on the given stream.

    Priority chain:
        1. NO_COLOR env set (any value) → disable
        2. FORCE_COLOR env set (any value) → enable
        3. CI / GITHUB_ACTIONS / GITLAB_CI env set → disable
        4. stream.isatty() → check TERM
        5. TERM == "dumb" or empty → disable
        6. Otherwise → enable

    Args:
        stream: Output stream to check for TTY. Defaults to sys.stderr.

    Returns:
        True if ANSI escape codes should be emitted.

    """
    target = stream if stream is not None else sys.stderr

    # 1. NO_COLOR takes absolute precedence (https://no-color.org/)
    if os.environ.get("NO_COLOR") is not None:
        return False

    # 2. FORCE_COLOR overrides all detection
    if os.environ.get("FORCE_COLOR") is not None:
        return True

    # 3. CI environments generally don't support colors well
    ci_vars = ("CI", "GITHUB_ACTIONS", "GITLAB_CI")
    if any(os.environ.get(var) is not None for var in ci_vars):
        return False

    # 4. Check if stream is a TTY
    if hasattr(target, "isatty") and target.isatty():
        # 5. TERM=dumb or empty → no colors
        term = os.environ.get("TERM", "")
        return term not in {"dumb", ""}

    return False


def _should_use_unicode() -> bool:
    """Detect whether Unicode symbols are safe to use.

    Checks LANG and LC_ALL for UTF-8 indicators. Falls back to ASCII
    when the locale suggests a non-Unicode terminal (e.g., Docker Alpine,
    bare SSH sessions).

    Returns:
        True if UTF-8 symbols can be used safely.

    """
    for var in ("LC_ALL", "LANG"):
        value = os.environ.get(var, "")
        if value and "utf" in value.lower():
            return True
    return False


_USE_COLOR: Final[bool] = _should_use_color()
_USE_UNICODE: Final[bool] = _should_use_unicode()

RESET: Final[str] = "\033[0m" if _USE_COLOR else ""
RED: Final[str] = "\033[31m" if _USE_COLOR else ""
GREEN: Final[str] = "\033[32m" if _USE_COLOR else ""
YELLOW: Final[str] = "\033[33m" if _USE_COLOR else ""
BLUE: Final[str] = "\033[34m" if _USE_COLOR else ""
BOLD: Final[str] = "\033[1m" if _USE_COLOR else ""


SYM_OK: Final[str] = "✓" if _USE_UNICODE else "[OK]"
SYM_FAIL: Final[str] = "✗" if _USE_UNICODE else "[FAIL]"
SYM_WARN: Final[str] = "⚠" if _USE_UNICODE else "[WARN]"
SYM_SKIP: Final[str] = "–" if _USE_UNICODE else "[SKIP]"
SYM_ARROW: Final[str] = "→" if _USE_UNICODE else "->"
SYM_BULLET: Final[str] = "•" if _USE_UNICODE else "*"


class InfraOutput:
    """Structured terminal output for infrastructure commands.

    All methods write to ``sys.stderr`` so that stdout remains clean for
    machine-readable output (JSON, CSV, exit codes).

    The class reads color/unicode settings at construction time but defers
    to the module-level detection functions so behaviour can be overridden
    in tests or downstream code.

    Args:
        use_color: Override automatic color detection. ``None`` uses the
            module-level ``_USE_COLOR`` value.
        use_unicode: Override automatic unicode detection. ``None`` uses
            the module-level ``_USE_UNICODE`` value.
        stream: Output stream. Defaults to ``sys.stderr``.

    """

    def __init__(
        self,
        *,
        use_color: bool | None = None,
        use_unicode: bool | None = None,
        stream: TextIO | None = None,
    ) -> None:
        """Initialize structured terminal output.

        Args:
            use_color: Override automatic color detection.
            use_unicode: Override automatic unicode detection.
            stream: Output stream. Defaults to sys.stderr.

        """
        self._color: bool = use_color if use_color is not None else _USE_COLOR
        self._unicode: bool = use_unicode if use_unicode is not None else _USE_UNICODE
        self._stream: TextIO = stream if stream is not None else sys.stderr

        self._reset = "\033[0m" if self._color else ""
        self._red = "\033[31m" if self._color else ""
        self._green = "\033[32m" if self._color else ""
        self._yellow = "\033[33m" if self._color else ""
        self._blue = "\033[34m" if self._color else ""
        self._bold = "\033[1m" if self._color else ""

        self._sym_ok = "✓" if self._unicode else "[OK]"
        self._sym_fail = "✗" if self._unicode else "[FAIL]"
        self._sym_warn = "⚠" if self._unicode else "[WARN]"
        self._sym_skip = "–" if self._unicode else "[SKIP]"

    def _write(self, message: str) -> None:
        """Write a line to the output stream with newline."""
        self._stream.write(message + "\n")
        self._stream.flush()

    def status(
        self,
        verb: str,
        project: str,
        result: bool,
        elapsed: float,
    ) -> None:
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
        sep = "──" if self._unicode else "--"
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

    def error(self, message: str, detail: str | None = None) -> None:
        """Write an error message in red.

        Args:
            message: Primary error message.
            detail: Optional detail text shown on the next line.

        """
        self._write(f"{self._red}ERROR{self._reset}: {message}")
        if detail:
            self._write(f"  {detail}")

    def warning(self, message: str) -> None:
        """Write a warning message in yellow.

        Args:
            message: Warning text.

        """
        self._write(f"{self._yellow}WARN{self._reset}: {message}")

    def info(self, message: str) -> None:
        """Write an informational message in blue.

        Args:
            message: Information text.

        """
        self._write(f"{self._blue}INFO{self._reset}: {message}")

    def header(self, title: str) -> None:
        """Write a bold section header.

        Args:
            title: Section title text.

        """
        sep = "═" if self._unicode else "="
        line = sep * 60
        self._write("")
        self._write(f"{self._bold}{line}{self._reset}")
        self._write(f"{self._bold}  {title}{self._reset}")
        self._write(f"{self._bold}{line}{self._reset}")

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


output: Final[InfraOutput] = InfraOutput()
"""Module-level singleton for direct use: ``from flext_infra import output``."""

__all__ = [
    "BLUE",
    "BOLD",
    "GREEN",
    "RED",
    "RESET",
    "SYM_ARROW",
    "SYM_BULLET",
    "SYM_FAIL",
    "SYM_OK",
    "SYM_SKIP",
    "SYM_WARN",
    "YELLOW",
    "InfraOutput",
    "output",
]
