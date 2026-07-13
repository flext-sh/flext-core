"""Unified console output helpers absorbing stdout/stderr into ``u.*``.

The single canonical sink for user-facing process output. Nothing in the
FLEXT fleet writes to ``sys.stdout``/``sys.stderr`` or calls ``print`` at a
call site: they route through ``u.out`` (stdout) and ``u.err`` (stderr),
keeping presentation coherent with structured logging (``u.fetch_logger``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys


class FlextUtilitiesConsole:
    """Canonical stdout/stderr sinks exposed flat through ``u``.

    ``out`` is the user-facing result channel (stdout); ``err`` is the
    diagnostic/error channel (stderr). Both are the only sanctioned way to
    emit process output — raw ``print``/``sys.stdout``/``sys.stderr`` at call
    sites are governance violations absorbed here.
    """

    @staticmethod
    def out(message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        """Write ``message`` to stdout (the user-facing result channel)."""
        sys.stdout.write(f"{message}{end}")
        if flush:
            sys.stdout.flush()

    @staticmethod
    def err(message: str = "", *, end: str = "\n", flush: bool = False) -> None:
        """Write ``message`` to stderr (the diagnostic/error channel)."""
        sys.stderr.write(f"{message}{end}")
        if flush:
            sys.stderr.flush()
