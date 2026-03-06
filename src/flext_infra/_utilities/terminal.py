"""Terminal detection helpers for infrastructure output.

Centralizes terminal capability detection (color, unicode) previously
defined as module-level functions in ``flext_infra.output``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import sys
from typing import TextIO


class FlextInfraUtilitiesTerminal:
    """Terminal capability detection helpers.

    Usage via namespace::

        from flext_infra import u

        if u.Infra.Terminal.should_use_color():
            ...
    """

    @staticmethod
    def should_use_color(stream: TextIO | None = None) -> bool:
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

    @staticmethod
    def should_use_unicode() -> bool:
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
