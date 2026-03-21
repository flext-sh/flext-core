"""FlextConstantsDomain - logging and domain constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from types import MappingProxyType
from typing import Final


class FlextConstantsDomain:
    """Constants for logging and domain models."""

    DEFAULT_LEVEL: Final[str] = "INFO"
    DEFAULT_LEVEL_DEVELOPMENT: Final[str] = "DEBUG"
    DEFAULT_LEVEL_PRODUCTION: Final[str] = "WARNING"
    DEFAULT_LEVEL_TESTING: Final[str] = "INFO"
    VALID_LEVELS: Final[tuple[str, ...]] = (
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    )
    JSON_OUTPUT_DEFAULT: Final[bool] = False
    DEFAULT_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    STRUCTURED_OUTPUT: Final[bool] = True
    INCLUDE_SOURCE: Final[bool] = True
    VERBOSITY: Final[str] = "compact"
    MAX_FILE_SIZE: Final[int] = 10485760
    BACKUP_COUNT: Final[int] = 5
    CONSOLE_ENABLED: Final[bool] = True
    CONSOLE_COLOR_ENABLED: Final[bool] = True
    TRACK_PERFORMANCE: Final[bool] = False
    TRACK_TIMING: Final[bool] = False
    INCLUDE_CONTEXT: Final[bool] = True
    INCLUDE_CORRELATION_ID: Final[bool] = True
    MAX_CONTEXT_KEYS: Final[int] = 50
    MASK_SENSITIVE_DATA: Final[bool] = True
    ASYNC_ENABLED: Final[bool] = True
    ASYNC_QUEUE_SIZE: Final[int] = 10000
    ASYNC_WORKERS: Final[int] = 1
    ASYNC_BLOCK_ON_FULL: Final[bool] = False
    LEVEL_HIERARCHY: Final[MappingProxyType[str, int]] = MappingProxyType({
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40,
        "critical": 50,
    })
    "Numeric log levels for comparison (lower = more verbose)."

    @unique
    class ContextOperation(StrEnum):
        """Context operation types enumeration."""

        BIND = "bind"
        UNBIND = "unbind"
        CLEAR = "clear"
        GET = "get"

    @unique
    class Status(StrEnum):
        """Status values for domain entities."""

        ACTIVE = "active"
        INACTIVE = "inactive"
        ARCHIVED = "archived"

    @unique
    class Currency(StrEnum):
        """Currency enumeration for monetary operations."""

        USD = "USD"
        EUR = "EUR"
        GBP = "GBP"
        BRL = "BRL"

    @unique
    class OrderStatus(StrEnum):
        """Order status enumeration for order lifecycle."""

        PENDING = "pending"
        CONFIRMED = "confirmed"
        SHIPPED = "shipped"
        DELIVERED = "delivered"
        CANCELLED = "cancelled"
