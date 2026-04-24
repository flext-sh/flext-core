"""FlextConstantsInfrastructure - runtime infrastructure constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final

from flext_core._constants.timeout import FlextConstantsTimeout


class FlextConstantsInfrastructure:
    """Constants for context, container, dispatcher, resilience, and persistence."""

    DEFAULT_MAX_FACTORIES: Final[int] = 500
    MAX_FACTORIES: Final[int] = 5000
    DEFAULT_LOGGER_MODULE: Final[str] = "flext_core"

    @unique
    class ContextScope(StrEnum):
        """Context scope identifiers for FlextContext operations."""

        GLOBAL = "global"
        REQUEST = "request"
        USER = "user"
        SESSION = "session"
        TRANSACTION = "transaction"
        APPLICATION = "application"
        OPERATION = "operation"

    @unique
    class ContextKey(StrEnum):
        """Standard context dictionary key names."""

        OPERATION_ID = "operation_id"
        USER_ID = "user_id"
        CORRELATION_ID = "correlation_id"
        PARENT_CORRELATION_ID = "parent_correlation_id"
        SERVICE_NAME = "service_name"
        OPERATION_NAME = "operation_name"
        REQUEST_ID = "request_id"
        SERVICE_VERSION = "service_version"
        OPERATION_START_TIME = "operation_start_time"
        OPERATION_METADATA = "operation_metadata"
        REQUEST_TIMESTAMP = "request_timestamp"
        SERVICE_MODULE = "service_module"

    @unique
    class MetadataKey(StrEnum):
        """Metadata dictionary key names for operation timing."""

        START_TIME = "start_time"
        END_TIME = "end_time"
        DURATION_SECONDS = "duration_seconds"

    @unique
    class ServiceName(StrEnum):
        """Standard service registration names used in FlextContainer."""

        LOGGER = "logger"
        COMMAND_BUS = "command_bus"

    @unique
    class HandlerMode(StrEnum):
        """Dispatcher handler processing modes."""

        COMMAND = "command"
        QUERY = "query"

    DEFAULT_HANDLER_MODE: Final[str] = HandlerMode.COMMAND

    @unique
    class BackoffStrategy(StrEnum):
        """Retry backoff strategy identifiers."""

        EXPONENTIAL = "exponential"
        LINEAR = "linear"

    DEFAULT_BACKOFF_STRATEGY: Final[str] = BackoffStrategy.EXPONENTIAL
    MAX_RETRY_ATTEMPTS: Final[int] = 3
    DEFAULT_RETRY_DELAY_SECONDS: Final[int] = 1
    DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = (
        FlextConstantsTimeout.DEFAULT_RECOVERY_TIMEOUT_SECONDS
    )

    DATABASE_URL: Final[str] = "sqlite:///:memory:"
    DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 10
    MAX_CONNECTION_POOL_SIZE: Final[int] = 100
    MIN_POOL_SIZE: Final[int] = 1
