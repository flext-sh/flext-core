"""Fail-loud native stream handler for the logging handler graph.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import sys
from typing import override

from flext_core import t


class FlextUtilitiesLoggingHandlerStream:
    """Own the native stream handler that surfaces its sink error."""

    class _StreamHandler(logging.StreamHandler[t.TextStream]):
        """Retain native rendering/locking while surfacing the native sink error."""

        include_callsite: bool = True

        @override
        def handleError(self, record: t.LoggingNativeRecord) -> None:
            error = sys.exception()
            if error is not None:
                raise error
            msg = "Logging handler failed without an active exception"
            raise RuntimeError(msg)


__all__: list[str] = ["FlextUtilitiesLoggingHandlerStream"]
