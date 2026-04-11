"""FlextConstantsInfrastructure - context and infrastructure constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final

from flext_core import FlextConstantsBase


class FlextConstantsInfrastructure:
    """Constants for context, container, dispatcher, and pagination."""

    DEBUG_CONTEXT_KEYS: Final[frozenset[str]] = frozenset({
        "schema",
        "params",
    })
    "Keys whose values are bound at DEBUG level in operation context."

    ERROR_CONTEXT_KEYS: Final[frozenset[str]] = frozenset({
        "stack_trace",
        "exception",
        "traceback",
        "error_details",
    })
    "Keys whose values are bound at ERROR level in operation context."

    CORRELATION_ID_PREFIX: Final[str] = "flext-"
    CORRELATION_ID_LENGTH: Final[int] = 12
    DEFAULT_CONTEXT_TIMEOUT: Final[int] = FlextConstantsBase.DEFAULT_TIMEOUT_SECONDS
    MAX_CONTEXT_DEPTH: Final[int] = 10
    MAX_CONTEXT_SIZE: Final[int] = 1000
    MILLISECONDS_PER_SECOND: Final[int] = 1000
    SENTINEL_MISSING: Final[str] = "__sentinel_missing__"
    """Sentinel value to distinguish 'not provided' from None in context operations."""

    INFRA_TIMEOUT_SECONDS: Final[int] = FlextConstantsBase.DEFAULT_TIMEOUT_SECONDS
    # MIN_TIMEOUT_SECONDS inherited from FlextConstantsBase via FlextConstants MRO
    INFRA_MAX_TIMEOUT_SECONDS: Final[int] = 300
    MAX_CACHE_SIZE: Final[int] = 100
    DEFAULT_MAX_SERVICES: Final[int] = 1000
    "Default maximum number of services allowed in container."
    DEFAULT_MAX_FACTORIES: Final[int] = 500
    "Default maximum number of factories allowed in container."
    MAX_FACTORIES: Final[int] = 5000
    "Maximum number of factories allowed in container."

    THREAD_NAME_PREFIX: Final[str] = "flext-dispatcher"
    "Thread name prefix for dispatcher thread pool executor."
    DEFAULT_AUTO_CONTEXT: Final[bool] = True
    DEFAULT_ENABLE_LOGGING: Final[bool] = True
    DEFAULT_ENABLE_METRICS: Final[bool] = True
    # DEFAULT_TIMEOUT_SECONDS inherited from FlextConstantsBase via FlextConstants MRO
    MIN_REGISTRATION_ID_LENGTH: Final[int] = 1
    DEFAULT_DISPATCHER_PATH: Final[str] = "flext_core.dispatcher:FlextDispatcher"
    "Default dispatcher implementation path."
    DEFAULT_SERVICE_NAME: Final[str] = "default_service"
    "Default service name for service models."
    DEFAULT_RESOURCE_TYPE: Final[str] = "default_resource"
    "Default resource type for service models."
    MIN_REQUEST_ID_LENGTH: Final[int] = 1
    SINGLE_HANDLER_ARG_COUNT: Final[int] = 1
    TWO_HANDLER_ARG_COUNT: Final[int] = 2
    DEFAULT_LOGGER_MODULE: Final[str] = "flext_core"
    "Default module name for logger creation."

    DEFAULT_PAGE_NUMBER: Final[int] = 1
    # DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, MIN_PAGE_SIZE inherited from FlextConstantsBase via FlextConstants MRO
    MIN_PAGE_NUMBER: Final[int] = 1
    MAX_PAGE_NUMBER: Final[int] = 10000
    DEFAULT_PAGE_SIZE_EXAMPLE: Final[int] = 20
    "Default page size for examples and utilities (different from DEFAULT_PAGE_SIZE)."
    MAX_PAGE_SIZE_EXAMPLE: Final[int] = 1000
    "Maximum page size for examples and utilities."

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
    class ExportFormat(StrEnum):
        """Supported context export formats."""

        JSON = "json"
        DICT = "dict"

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

    @unique
    class ContextHeader(StrEnum):
        """HTTP header names used for distributed tracing."""

        CORRELATION_ID = "X-Correlation-Id"
        PARENT_CORRELATION_ID = "X-Parent-Correlation-Id"
        SERVICE_NAME = "X-Service-Name"
        USER_ID = "X-User-Id"

    @unique
    class MetadataKey(StrEnum):
        """Metadata dictionary key names for operation timing."""

        START_TIME = "start_time"
        END_TIME = "end_time"
        DURATION_SECONDS = "duration_seconds"

    @unique
    class MetadataField(StrEnum):
        """Metadata field names used in context operations."""

        USER_ID = "user_id"
        CORRELATION_ID = "correlation_id"
        REQUEST_ID = "request_id"
        SESSION_ID = "session_id"
        TENANT_ID = "tenant_id"

    @unique
    class ContainerKind(StrEnum):
        """Container registration kind identifiers."""

        SERVICE = "service"
        "Service: singleton instance registered directly."
        FACTORY = "factory"
        "Factory: callable that creates instances on demand."
        RESOURCE = "resource"
        "Resource: lifecycle-managed service (start/stop)."

    @unique
    class ServiceName(StrEnum):
        """Standard service registration names used in FlextContainer."""

        LOGGER = "logger"
        "Logger service registration name."
        COMMAND_BUS = "command_bus"
        "Command bus service registration name."

    @unique
    class RegistrationStatus(StrEnum):
        """Handler registration lifecycle statuses."""

        ACTIVE = "active"
        INACTIVE = "inactive"
        ERROR = "error"

    @unique
    class HandlerMode(StrEnum):
        """Dispatcher handler processing modes."""

        COMMAND = "command"
        QUERY = "query"

    DEFAULT_HANDLER_MODE: Final[str] = HandlerMode.COMMAND
