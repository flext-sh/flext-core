"""Terminal detection helpers for infrastructure output.

Centralizes terminal capability detection (color, unicode).
All constants (colors, symbols) are in constants.py via c.Infra.Style.*

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import sys
from typing import TextIO


class FlextInfraUtilitiesTerminal:
    """Terminal capability detection helpers.

    Centralizes terminal capability detection (color, unicode).
    Style constants are in c.Infra.Style - import from there.

    Usage via namespace::

        from flext_infra import c, u

        if u.Infra.terminal_should_use_color():
            print(f"{c.Infra.Style.RED}Error{c.Infra.Style.RESET}")
    """

    @staticmethod
    def terminal_should_use_color(stream: TextIO | None = None) -> bool:
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
        if os.environ.get("NO_COLOR") is not None:
            return False
        if os.environ.get("FORCE_COLOR") is not None:
            return True
        ci_vars = ("CI", "GITHUB_ACTIONS", "GITLAB_CI")
        if any(os.environ.get(var) is not None for var in ci_vars):
            return False
        if hasattr(target, "isatty") and target.isatty():
            term = os.environ.get("TERM", "")
            return term not in {"dumb", ""}
        return False

    @staticmethod
    def terminal_should_use_unicode() -> bool:
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


__all__ = ["FlextInfraUtilitiesTerminal"]
