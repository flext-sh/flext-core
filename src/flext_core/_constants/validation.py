"""FlextConstantsValidation - validation, errors, and exception constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsValidation:
    """Constants for validation and error handling."""

    MIN_NAME_LENGTH: Final[int] = 2
    # MAX_NAME_LENGTH inherited from FlextConstantsBase via FlextConstants MRO
    MAX_EMAIL_LENGTH: Final[int] = 254
    EMAIL_PARTS_COUNT: Final[int] = 2
    LEVEL_PREFIX_PARTS_COUNT: Final[int] = 4
    MIN_PHONE_DIGITS: Final[int] = 10
    MAX_PHONE_DIGITS: Final[int] = 20
    MIN_USERNAME_LENGTH: Final[int] = 3
    MAX_AGE: Final[int] = 150
    MIN_AGE: Final[int] = 0
    # PREVIEW_LENGTH inherited from FlextConstantsBase via FlextConstants MRO
    VALIDATION_TIMEOUT_MS: Final[float] = 100.0
    MAX_UNCOMMITTED_EVENTS: Final[int] = 100
    DISCOUNT_THRESHOLD: Final[int] = 100
    DISCOUNT_RATE: Final[float] = 0.05
    FILTER_THRESHOLD: Final[int] = 5
    RETRY_COUNT_MAX: Final[int] = 3
    MAX_WORKERS_LIMIT: Final[int] = 100
    MAX_RETRY_STATUS_CODES: Final[int] = 100
    MAX_CUSTOM_VALIDATORS: Final[int] = 50

    @unique
    class ErrorCode(StrEnum):
        """Structured error code identifiers for exception classification."""

        VALIDATION_ERROR = "VALIDATION_ERROR"
        TYPE_ERROR = "TYPE_ERROR"
        ATTRIBUTE_ERROR = "ATTRIBUTE_ERROR"
        CONFIG_ERROR = "CONFIG_ERROR"
        GENERIC_ERROR = "GENERIC_ERROR"
        COMMAND_PROCESSING_FAILED = "COMMAND_PROCESSING_FAILED"
        UNKNOWN_ERROR = "UNKNOWN_ERROR"
        SERIALIZATION_ERROR = "SERIALIZATION_ERROR"
        MAP_ERROR = "MAP_ERROR"
        BIND_ERROR = "BIND_ERROR"
        CHAIN_ERROR = "CHAIN_ERROR"
        UNWRAP_ERROR = "UNWRAP_ERROR"
        OPERATION_ERROR = "OPERATION_ERROR"
        SERVICE_ERROR = "SERVICE_ERROR"
        BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"
        BUSINESS_RULE_ERROR = "BUSINESS_RULE_ERROR"
        NOT_FOUND_ERROR = "NOT_FOUND_ERROR"
        NOT_FOUND = "NOT_FOUND"
        RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
        ALREADY_EXISTS = "ALREADY_EXISTS"
        COMMAND_BUS_ERROR = "COMMAND_BUS_ERROR"
        COMMAND_HANDLER_NOT_FOUND = "COMMAND_HANDLER_NOT_FOUND"
        DOMAIN_EVENT_ERROR = "DOMAIN_EVENT_ERROR"
        TIMEOUT_ERROR = "TIMEOUT_ERROR"
        PROCESSING_ERROR = "PROCESSING_ERROR"
        CONNECTION_ERROR = "CONNECTION_ERROR"
        CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
        EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
        PERMISSION_ERROR = "PERMISSION_ERROR"
        AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
        AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
        EXCEPTION_ERROR = "EXCEPTION_ERROR"
        CRITICAL_ERROR = "CRITICAL_ERROR"

    @unique
    class ErrorType(StrEnum):
        """Structured error categories for validation and runtime failures."""

        VALIDATION = "validation"
        CONFIGURATION = "configuration"
        OPERATION = "operation"
        CONNECTION = "connection"
        TIMEOUT = "timeout"
        AUTHORIZATION = "authorization"
        AUTHENTICATION = "authentication"
        NOT_FOUND = "not_found"
        ATTRIBUTE_ACCESS = "attribute_access"
        CONFLICT = "conflict"
        RATE_LIMIT = "rate_limit"
        CIRCUIT_BREAKER = "circuit_breaker"
        TYPE_ERROR = "type_error"
        VALUE_ERROR = "value_error"
        RUNTIME_ERROR = "runtime_error"
        SYSTEM_ERROR = "system_error"

    @unique
    class FailureLevel(StrEnum):
        """Exception failure levels."""

        STRICT = "strict"
        WARN = "warn"
        PERMISSIVE = "permissive"

    @unique
    class ParserCase(StrEnum):
        """Supported parser normalization cases."""

        LOWER = "lower"
        UPPER = "upper"
        TITLE = "title"

    @unique
    class ParserBooleanToken(StrEnum):
        """Canonical string tokens accepted for boolean coercion."""

        TRUE = "true"
        ONE = "1"
        YES = "yes"
        ON = "on"
        FALSE = "false"
        ZERO = "0"
        NO = "no"
        OFF = "off"

    FAILURE_LEVEL_DEFAULT: Final[FailureLevel] = FailureLevel.PERMISSIVE
    PARSER_BOOLEAN_TRUTHY: frozenset[str] = frozenset(
        {
            ParserBooleanToken.TRUE.value,
            ParserBooleanToken.ONE.value,
            ParserBooleanToken.YES.value,
            ParserBooleanToken.ON.value,
        },
    )
    PARSER_BOOLEAN_FALSY: frozenset[str] = frozenset(
        {
            ParserBooleanToken.FALSE.value,
            ParserBooleanToken.ZERO.value,
            ParserBooleanToken.NO.value,
            ParserBooleanToken.OFF.value,
        },
    )

    STRING_METHOD_MAP: frozenset[str] = frozenset({
        "str",
        "dict",
        "list",
        "tuple",
        "sequence",
        "mapping",
        "list_or_tuple",
        "sequence_not_str",
        "sequence_not_str_bytes",
        "sized",
        "callable",
        "bytes",
        "int",
        "float",
        "bool",
        "none",
        "string_non_empty",
        "dict_non_empty",
        "list_non_empty",
    })
