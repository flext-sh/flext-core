"""FlextConstantsInfrastructure - context and infrastructure constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final

from flext_core._constants.base import FlextConstantsBase


class FlextConstantsInfrastructure:
    """Constants for context, container, dispatcher, and pagination."""

    SCOPE_GLOBAL: Final[str] = "global"
    SCOPE_REQUEST: Final[str] = "request"
    SCOPE_USER: Final[str] = "user"
    SCOPE_SESSION: Final[str] = "session"
    SCOPE_TRANSACTION: Final[str] = "transaction"
    SCOPE_APPLICATION: Final[str] = "application"
    SCOPE_OPERATION: Final[str] = "operation"

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
    EXPORT_FORMAT_JSON: Final[str] = "json"
    EXPORT_FORMAT_DICT: Final[str] = "dict"

    @unique
    class MetadataField(StrEnum):
        """Metadata field names used in context operations."""

        USER_ID = "user_id"
        CORRELATION_ID = "correlation_id"
        REQUEST_ID = "request_id"
        SESSION_ID = "session_id"
        TENANT_ID = "tenant_id"

    OPERATION_SET: Final[str] = "set"
    OPERATION_GET: Final[str] = "get"
    OPERATION_REMOVE: Final[str] = "remove"
    OPERATION_CLEAR: Final[str] = "clear"
    KEY_OPERATION_ID: Final[str] = "operation_id"
    KEY_USER_ID: Final[str] = "user_id"
    KEY_CORRELATION_ID: Final[str] = "correlation_id"
    KEY_PARENT_CORRELATION_ID: Final[str] = "parent_correlation_id"
    KEY_SERVICE_NAME: Final[str] = "service_name"
    KEY_OPERATION_NAME: Final[str] = "operation_name"
    KEY_REQUEST_ID: Final[str] = "request_id"
    KEY_SERVICE_VERSION: Final[str] = "service_version"
    KEY_OPERATION_START_TIME: Final[str] = "operation_start_time"
    KEY_OPERATION_METADATA: Final[str] = "operation_metadata"
    KEY_REQUEST_TIMESTAMP: Final[str] = "request_timestamp"
    HEADER_CORRELATION_ID: Final[str] = "X-Correlation-Id"
    HEADER_PARENT_CORRELATION_ID: Final[str] = "X-Parent-Correlation-Id"
    HEADER_SERVICE_NAME: Final[str] = "X-Service-Name"
    HEADER_USER_ID: Final[str] = "X-User-Id"
    METADATA_KEY_START_TIME: Final[str] = "start_time"
    METADATA_KEY_END_TIME: Final[str] = "end_time"
    METADATA_KEY_DURATION_SECONDS: Final[str] = "duration_seconds"
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
    HANDLER_MODE_COMMAND: Final[str] = "command"
    HANDLER_MODE_QUERY: Final[str] = "query"
    VALID_HANDLER_MODES: Final[tuple[str, ...]] = (
        HANDLER_MODE_COMMAND,
        HANDLER_MODE_QUERY,
    )
    DEFAULT_HANDLER_MODE: Final[str] = HANDLER_MODE_COMMAND
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
    ERROR_INVALID_HANDLER_MODE: Final[str] = "handler_mode must be 'command' or 'query'"
    ERROR_HANDLER_REQUIRED: Final[str] = "handler cannot be None"
    ERROR_MESSAGE_REQUIRED: Final[str] = "message cannot be None"
    ERROR_POSITIVE_TIMEOUT: Final[str] = "timeout must be positive"
    ERROR_INVALID_REGISTRATION_ID: Final[str] = (
        "registration_id must be non-empty string"
    )
    ERROR_INVALID_REQUEST_ID: Final[str] = "request_id must be non-empty string"
    REGISTRATION_STATUS_ACTIVE: Final[str] = "active"
    "Registration status: active (matches Cqrs.RegistrationStatus.ACTIVE.value)."
    REGISTRATION_STATUS_INACTIVE: Final[str] = "inactive"
    "Registration status: inactive (matches Cqrs.RegistrationStatus.INACTIVE.value)."
    REGISTRATION_STATUS_ERROR: Final[str] = "error"
    "Registration status: error (not part of RegistrationStatus StrEnum)."
    VALID_REGISTRATION_STATUSES: Final[tuple[str, ...]] = (
        REGISTRATION_STATUS_ACTIVE,
        REGISTRATION_STATUS_INACTIVE,
        REGISTRATION_STATUS_ERROR,
    )

    CONTAINER_KIND_SERVICE: Final[str] = "service"
    "Container registration kind: service (singleton instance)."
    CONTAINER_KIND_FACTORY: Final[str] = "factory"
    "Container registration kind: factory (callable)."
    CONTAINER_KIND_RESOURCE: Final[str] = "resource"
    "Container registration kind: resource (lifecycle-managed)."
    SERVICE_NAME_LOGGER: Final[str] = "logger"
    "Standard service name for logger registration."
    SERVICE_NAME_COMMAND_BUS: Final[str] = "command_bus"
    "Standard service name for command bus registration."
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
