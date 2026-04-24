"""FlextConstantsLogging - log level + async + log message constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsLogging:
    """SSOT for logging level enumeration and log message templates."""

    @unique
    class LogLevel(StrEnum):
        """Standard log levels."""

        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    ASYNC_ENABLED: Final[bool] = True
    ASYNC_BLOCK_ON_FULL: Final[bool] = False

    MAX_FILE_SIZE: Final[int] = 10485760
    BACKUP_COUNT: Final[int] = 5

    _TEMPLATE_REGISTERED: Final[str] = "Registered {subject}"
    LOG_REGISTERED_AUTO_DISCOVERY_HANDLER: Final[str] = _TEMPLATE_REGISTERED.format(
        subject="auto-discovery handler"
    )
    LOG_REGISTERED_EVENT_SUBSCRIBER: Final[str] = _TEMPLATE_REGISTERED.format(
        subject="event subscriber"
    )
    LOG_REGISTERED_HANDLER: Final[str] = _TEMPLATE_REGISTERED.format(subject="handler")
    LOG_HANDLER_EXECUTION_FAILED: Final[str] = "Handler execution failed"
    LOG_HANDLER_PIPELINE_FAILURE: Final[str] = "Critical handler pipeline failure"
    LOG_TRACKED_OPERATION_EXPECTED_EXCEPTION: Final[str] = (
        "Tracked operation raised expected exception"
    )
    LOG_SERVICE_REGISTRATION_FAILED: Final[str] = "Service registration failed"
