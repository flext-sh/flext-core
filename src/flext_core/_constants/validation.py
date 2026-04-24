"""FlextConstantsValidation - validation and error classification constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsValidation:
    """SSOT for validation thresholds, error codes, and parser tokens."""

    LEVEL_PREFIX_PARTS_COUNT: Final[int] = 4

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
    PARSER_BOOLEAN_TRUTHY: Final[frozenset[str]] = frozenset({
        ParserBooleanToken.TRUE.value,
        ParserBooleanToken.ONE.value,
        ParserBooleanToken.YES.value,
        ParserBooleanToken.ON.value,
    })
    PARSER_BOOLEAN_FALSY: Final[frozenset[str]] = frozenset({
        ParserBooleanToken.FALSE.value,
        ParserBooleanToken.ZERO.value,
        ParserBooleanToken.NO.value,
        ParserBooleanToken.OFF.value,
    })

    STRING_METHOD_MAP: Final[frozenset[str]] = frozenset({
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
