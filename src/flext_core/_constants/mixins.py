"""FlextConstantsMixins - mixins, processing, discovery, and test constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final

from flext_core._constants.base import FlextConstantsBase


class FlextConstantsMixins:
    """Constants for mixins and processing support."""

    FIELD_ID: Final[str] = "unique_id"
    FIELD_NAME: Final[str] = "name"
    FIELD_TYPE: Final[str] = "type"
    FIELD_STATUS: Final[str] = "status"
    FIELD_DATA: Final[str] = "data"
    FIELD_CONFIG: Final[str] = "settings"
    FIELD_METADATA: Final[str] = "metadata"
    FIELD_ATTRIBUTES: Final[str] = "attributes"
    FIELD_DESCRIPTION: Final[str] = "description"
    FIELD_CONTEXT: Final[str] = "context"
    FIELD_HANDLER_MODE: Final[str] = "handler_mode"
    FIELD_AUTO_LOG: Final[str] = "auto_log"
    FIELD_AUTO_CORRELATION: Final[str] = "auto_correlation"
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

    IDENTIFIER_UNKNOWN: Final[str] = "unknown"
    IDENTIFIER_DEFAULT: Final[str] = "default"
    IDENTIFIER_ANONYMOUS: Final[str] = "anonymous"
    IDENTIFIER_GUEST: Final[str] = "guest"
    IDENTIFIER_SYSTEM: Final[str] = "system"

    DEFAULT_JSON_INDENT: Final[int] = 2
    DEFAULT_SORT_KEYS: Final[bool] = False
    DEFAULT_ENSURE_ASCII: Final[bool] = False
    DEFAULT_USE_UTC: Final[bool] = True
    DEFAULT_AUTO_UPDATE: Final[bool] = True
    # MAX_OPERATION_NAME_LENGTH inherited from FlextConstantsBase via FlextConstants MRO
    MAX_STATE_VALUE_LENGTH: Final[int] = 50
    MAX_FIELD_NAME_LENGTH: Final[int] = 50
    MIN_FIELD_NAME_LENGTH: Final[int] = 1

    DEFAULT_MAX_WORKERS: Final[int] = FlextConstantsBase.DEFAULT_WORKERS
    # DEFAULT_BATCH_SIZE inherited from FlextConstantsBase via FlextConstants MRO
    PATTERN_TUPLE_MIN_LENGTH: Final[int] = 2
    "Minimum length for tuple patterns in parsing operations."
    PATTERN_TUPLE_MAX_LENGTH: Final[int] = 3
    "Maximum length for tuple patterns in parsing operations."

    HANDLER_ATTR: Final[str] = "_flext_handler_config_"
    "Attribute name for storing handler decorator configuration on methods."
    FACTORY_ATTR: Final[str] = "_flext_factory_config_"
    "Attribute name for storing factory decorator configuration on functions."
    DEFAULT_PRIORITY: Final[int] = 0
    "Default priority for handlers (0 = normal priority)."
    DEFAULT_HANDLER_TIMEOUT: Final[float | None] = None
    "Default timeout for handlers (None = no timeout)."

    DEFAULT_TEST_CREDENTIAL: Final[str] = "test_password"
    "Default credential for test user authentication."
    NONEXISTENT_USERNAME: Final[str] = "nonexistent"
    "Username that should not exist in test scenarios."

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

    @unique
    class RegistrationScope(StrEnum):
        """Plugin registration scopes for registry operations."""

        INSTANCE = "instance"
        CLASS = "class"

    @unique
    class CheckStatus(StrEnum):
        """Check/validation result status codes."""

        PASS = "PASS"
        FAIL = "FAIL"
        NO_TARGET = "NO_TARGET"
        SKIP = "SKIP"
        UNKNOWN = "UNKNOWN"

    @unique
    class MethodName(StrEnum):
        """Standard method names used in handler and mixin resolution."""

        HANDLE = "handle"
        PROCESS = "process"
        EXECUTE = "execute"
        PROCESS_COMMAND = "process_command"
        VALIDATE = "validate"

    @unique
    class OperationMode(StrEnum):
        """Operation processing modes."""

        OVERRIDE = "override"
        COLLECTION = "collection"
