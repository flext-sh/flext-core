"""FlextConstantsMixins - mixins, processing, discovery, and test constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsMixins:
    """Constants for mixins and processing support."""

    class Mixins:
        """Constants for mixin operations."""

        FIELD_ID: Final[str] = "unique_id"
        FIELD_NAME: Final[str] = "name"
        FIELD_TYPE: Final[str] = "type"
        FIELD_STATUS: Final[str] = "status"
        FIELD_DATA: Final[str] = "data"
        FIELD_CONFIG: Final[str] = "config"
        FIELD_STATE: Final[str] = "state"
        FIELD_CREATED_AT: Final[str] = "created_at"
        FIELD_UPDATED_AT: Final[str] = "updated_at"
        FIELD_VALIDATED: Final[str] = "validated"
        FIELD_CLASS: Final[str] = "class"
        FIELD_MODULE: Final[str] = "module"
        FIELD_REGISTERED: Final[str] = "registered"
        FIELD_EVENT_NAME: Final[str] = "event_name"
        FIELD_AGGREGATE_ID: Final[str] = "aggregate_id"
        FIELD_OCCURRED_AT: Final[str] = "occurred_at"
        STATE_ACTIVE: Final[str] = "active"
        "State: active."
        STATE_INACTIVE: Final[str] = "inactive"
        "State: inactive."
        STATE_SENT: Final[str] = "sent"
        STATE_IDLE: Final[str] = "idle"
        STATE_HEALTHY: Final[str] = "healthy"
        STATE_DEGRADED: Final[str] = "degraded"
        STATE_UNHEALTHY: Final[str] = "unhealthy"
        STATUS_PASS: Final[str] = "PASS"  # noqa: S105 — health-check status label, not a credential
        STATUS_FAIL: Final[str] = "FAIL"
        STATUS_NO_TARGET: Final[str] = "NO_TARGET"
        STATUS_SKIP: Final[str] = "SKIP"
        STATUS_UNKNOWN: Final[str] = "UNKNOWN"
        IDENTIFIER_UNKNOWN: Final[str] = "unknown"
        IDENTIFIER_DEFAULT: Final[str] = "default"
        IDENTIFIER_ANONYMOUS: Final[str] = "anonymous"
        IDENTIFIER_GUEST: Final[str] = "guest"
        IDENTIFIER_SYSTEM: Final[str] = "system"
        METHOD_HANDLE: Final[str] = "handle"
        METHOD_PROCESS: Final[str] = "process"
        METHOD_EXECUTE: Final[str] = "execute"
        METHOD_PROCESS_COMMAND: Final[str] = "process_command"
        OPERATION_OVERRIDE: Final[str] = "override"
        "Override operation mode."
        OPERATION_COLLECTION: Final[str] = "collection"
        "Collection operation mode."
        AUTH_BEARER: Final[str] = "bearer"
        AUTH_API_KEY: Final[str] = "api_key"
        AUTH_JWT: Final[str] = "jwt"
        HANDLER_COMMAND: Final[str] = "command"
        HANDLER_QUERY: Final[str] = "query"
        METHOD_VALIDATE: Final[str] = "validate"
        DEFAULT_JSON_INDENT: Final[int] = 2
        DEFAULT_SORT_KEYS: Final[bool] = False
        DEFAULT_ENSURE_ASCII: Final[bool] = False

        @unique
        class BoolTrueValue(StrEnum):
            """String representations of boolean true values."""

            TRUE = "true"
            ONE = "1"
            YES = "yes"
            ON = "on"
            ENABLED = "enabled"

        @unique
        class BoolFalseValue(StrEnum):
            """String representations of boolean false values."""

            FALSE = "false"
            ZERO = "0"
            NO = "no"
            OFF = "off"
            DISABLED = "disabled"

        STRING_TRUE: Final[str] = "true"
        STRING_FALSE: Final[str] = "false"
        DEFAULT_USE_UTC: Final[bool] = True
        DEFAULT_AUTO_UPDATE: Final[bool] = True
        MAX_OPERATION_NAME_LENGTH: Final[int] = 100
        MAX_STATE_VALUE_LENGTH: Final[int] = 50
        MAX_FIELD_NAME_LENGTH: Final[int] = 50
        MIN_FIELD_NAME_LENGTH: Final[int] = 1
        ERROR_EMPTY_OPERATION: Final[str] = "Operation name cannot be empty"
        ERROR_EMPTY_STATE: Final[str] = "State value cannot be empty"
        ERROR_EMPTY_FIELD_NAME: Final[str] = "Field name cannot be empty"
        ERROR_INVALID_ENCODING: Final[str] = "Invalid character encoding"
        ERROR_MISSING_TIMESTAMP_FIELDS: Final[str] = "Required timestamp fields missing"
        ERROR_INVALID_LOG_LEVEL: Final[str] = "Invalid log level"

    class Processing:
        """Processing pipeline constants."""

        DEFAULT_MAX_WORKERS: Final[int] = 4
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        PATTERN_TUPLE_MIN_LENGTH: Final[int] = 2
        "Minimum length for tuple patterns in parsing operations."
        PATTERN_TUPLE_MAX_LENGTH: Final[int] = 3
        "Maximum length for tuple patterns in parsing operations."

    class Discovery:
        """Constants for auto-discovery of handlers/factories.

        Used by `@h.handler()` and `@d.factory()` decorators to mark methods
        for automatic discovery and registration by FlextService and FlextContainer.

        Attributes:
            HANDLER_ATTR: Attribute name for storing handler decorator configuration.
            FACTORY_ATTR: Attribute name for storing factory decorator configuration.
            DEFAULT_PRIORITY: Default handler priority (higher = processed first).
            DEFAULT_TIMEOUT: Default handler timeout (None = no timeout).

        """

        HANDLER_ATTR: Final[str] = "_flext_handler_config_"
        "Attribute name for storing handler decorator configuration on methods."
        FACTORY_ATTR: Final[str] = "_flext_factory_config_"
        "Attribute name for storing factory decorator configuration on functions."
        DEFAULT_PRIORITY: Final[int] = 0
        "Default priority for handlers (0 = normal priority)."
        DEFAULT_TIMEOUT: Final[float | None] = None
        "Default timeout for handlers (None = no timeout)."

    class Test:
        """Test constants for unit and integration testing.

        Provides standardized test data values used across the FLEXT ecosystem.
        These constants ensure consistent test behavior and avoid magic strings.
        """

        DEFAULT_PASSWORD: Final[str] = "test_password"  # noqa: S105 — test-only fixture constant, never used in production
        "Default password for test user authentication."
        NONEXISTENT_USERNAME: Final[str] = "nonexistent"
        "Username that should not exist in test scenarios."
