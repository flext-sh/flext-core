"""Extracted facade class for logging.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._decorators._logging import FlextDecoratorsLoggingPayloads


class FlextDecoratorsLogging(
    FlextDecoratorsLoggingPayloads,
):
    """Facade for logging."""


__all__: list[str] = ["FlextDecoratorsLogging"]
