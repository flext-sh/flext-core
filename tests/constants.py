"""Constants for flext-core tests.

Provides TestsFlextConstants, extending FlextTestsConstants with flext-core-specific
constants. All generic test constants come from flext_tests.

Architecture:
- FlextTestsConstants (flext_tests) = Generic constants for all FLEXT projects
- TestsFlextConstants (tests/) = flext-core-specific constants extending FlextTestsConstants

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from flext_core.constants import c as core_c
from flext_core.models import m as core_m
from flext_tests.constants import FlextTestsConstants


class TestsFlextConstants(FlextTestsConstants):
    """Constants for flext-core tests - extends FlextTestsConstants.

    Architecture: Extends FlextTestsConstants with flext-core-specific constants.
    All generic constants from FlextTestsConstants are available through inheritance.

    Rules:
    - NEVER duplicate constants from FlextTestsConstants
    - Only flext-core-specific constants allowed (not generic for other projects)
    - All generic constants come from FlextTestsConstants
    """

    class Strings:
        """Flext-core-specific test strings organized by complexity."""

        EMPTY: Final[str] = ""
        SINGLE_CHAR: Final[str] = "a"
        BASIC_WORD: Final[str] = "hello"
        BASIC_LIST: Final[str] = "a,b,c"
        NUMERIC_LIST: Final[str] = "1,2,3"
        WITH_SPACES: Final[str] = "a, b, c"
        EXCESSIVE_SPACES: Final[str] = "  a  ,  b  ,  c  "
        LEADING_SPACES: Final[str] = "  hello"
        TRAILING_SPACES: Final[str] = "hello  "
        LEADING_TRAILING: Final[str] = ",a,b,c,"
        WITH_EMPTY: Final[str] = "a,,c"
        ONLY_DELIMITERS: Final[str] = ",,,"
        UNICODE_CHARS: Final[str] = "héllo,wörld"
        VALID_EMAIL: Final[str] = "test@example.com"
        INVALID_EMAIL: Final[str] = "invalid-email"
        USER_ID_VALID: Final[str] = "123"
        USER_ID_INVALID: Final[str] = "invalid"
        USER_ID_EMPTY: Final[str] = ""

    class Delimiters:
        """Flext-core-specific delimiter characters for string parsing."""

        COMMA: Final[str] = ","
        SEMICOLON: Final[str] = ";"
        PIPE: Final[str] = "|"
        COLON: Final[str] = ":"
        TAB: Final[str] = "\t"
        NEWLINE: Final[str] = "\n"

    class EscapeChars:
        """Flext-core-specific escape characters for string parsing."""

        BACKSLASH: Final[str] = "\\"
        HASH: Final[str] = "#"
        AT: Final[str] = "@"
        QUOTE: Final[str] = '"'
        SINGLE_QUOTE: Final[str] = "'"

    class Replacements:
        """Flext-core-specific replacement strings for string processing."""

        SPACE: Final[str] = " "
        UNDERSCORE: Final[str] = "_"
        DASH: Final[str] = "-"
        EQUALS: Final[str] = "="
        COMMA: Final[str] = ","
        EMPTY: Final[str] = ""

    class Patterns:
        """Flext-core-specific regex patterns for string processing."""

        WHITESPACE: Final[str] = r"\s+"
        DASH: Final[str] = r"-+"
        EQUALS_SPACE: Final[str] = r"\s+="
        COMMA_SPACE: Final[str] = r",\s+"
        EMAIL: Final[str] = r"^[^@]+@[^@]+\.[^@]+$"
        ALPHA_ONLY: Final[str] = r"^[a-zA-Z]+$"
        NUMERIC_ONLY: Final[str] = r"^\d+$"

    class TestErrors:
        """Flext-core-specific error message patterns for validation."""

        DELIMITER_EMPTY: Final[str] = "Delimiter must be exactly one character"
        DELIMITER_MULTI: Final[str] = "Delimiter must be exactly one character"
        DELIMITER_WHITESPACE: Final[str] = "Delimiter cannot be a whitespace"
        SPLIT_EMPTY: Final[str] = "Split character cannot be empty"
        ESCAPE_EMPTY: Final[str] = "Escape character cannot be empty"
        SPLIT_ESCAPE_SAME: Final[str] = (
            "Split character and escape character cannot be the same"
        )
        FAILED_PARSE: Final[str] = "Failed to parse"
        FAILED_SPLIT: Final[str] = "Failed to split"
        FAILED_NORMALIZE: Final[str] = "Failed to normalize"
        FAILED_PIPELINE: Final[str] = "validation error"
        INVALID_REGEX: Final[str] = "Invalid regex pattern"
        USER_NOT_FOUND: Final[str] = "User not found"
        INVALID_EMAIL: Final[str] = "Invalid email address"
        VALUE_TOO_LOW: Final[str] = "Value must be positive"
        VALUE_TOO_HIGH: Final[str] = "Value must be <= 100"
        PROCESSING_ERROR: Final[str] = "Processing error occurred"

    class TestValidation:
        """Flext-core-specific validation constants for various checks."""

        MIN_LENGTH_DEFAULT: Final[int] = 3
        MAX_LENGTH_DEFAULT: Final[int] = 100
        MIN_VALUE: Final[int] = 0
        MAX_VALUE: Final[int] = 100

    class Services:
        """Flext-core-specific service-related constants."""

        DEFAULT_USER_NAME_PREFIX: Final[str] = "User "
        DEFAULT_EMAIL_DOMAIN: Final[str] = "@example.com"
        DEFAULT_ERROR_MESSAGE: Final[str] = "Test error"

    class Railway:
        """Flext-core-specific railway pattern operation constants."""

        class Operation(StrEnum):
            """Railway operation types for testing."""

            GET_EMAIL = "get_email"
            SEND_EMAIL = "send_email"
            GET_STATUS = "get_status"
            DOUBLE = "double"
            SQUARE = "square"
            NEGATE = "negate"

        type OperationLiteral = Literal[
            "get_email",
            "send_email",
            "get_status",
            "double",
            "square",
            "negate",
        ]

        OP_GET_EMAIL: Final[str] = Operation.GET_EMAIL
        OP_SEND_EMAIL: Final[str] = Operation.SEND_EMAIL
        OP_GET_STATUS: Final[str] = Operation.GET_STATUS
        OP_DOUBLE: Final[str] = Operation.DOUBLE
        OP_SQUARE: Final[str] = Operation.SQUARE
        OP_NEGATE: Final[str] = Operation.NEGATE

    class HTTP:
        """Flext-core-specific HTTP-related constants for testing."""

        class Method(StrEnum):
            """HTTP methods for testing."""

            GET = "GET"
            POST = "POST"

        type StatusLiteral = Literal[200, 404, 400]
        type MethodLiteral = Literal["GET", "POST"]

        STATUS_OK: Final[int] = 200
        STATUS_NOT_FOUND: Final[int] = 404
        STATUS_BAD_REQUEST: Final[int] = 400
        CONTENT_TYPE_JSON: Final[str] = "application/json"
        METHOD_GET: Final[str] = Method.GET
        METHOD_POST: Final[str] = Method.POST

    class Mapper:
        """Flext-core-specific Mapper utilities test constants."""

        OLD_KEY: Final[str] = "old_key"
        NEW_KEY: Final[str] = "new_key"
        FOO: Final[str] = "foo"
        BAR: Final[str] = "bar"
        UNMAPPED: Final[str] = "unmapped"
        VALUE1: Final[str] = "value1"
        VALUE2: Final[str] = "value2"
        FLAGS_READ: Final[str] = "read"
        FLAGS_WRITE: Final[str] = "write"
        FLAGS_DELETE: Final[str] = "delete"
        CAN_READ: Final[str] = "can_read"
        CAN_WRITE: Final[str] = "can_write"
        CAN_DELETE: Final[str] = "can_delete"
        HELLO: Final[str] = "hello"
        WORLD: Final[str] = "world"
        HELLO_UPPER: Final[str] = "HELLO"
        WORLD_UPPER: Final[str] = "WORLD"
        A: Final[str] = "a"
        B: Final[str] = "b"
        C: Final[str] = "c"
        NUM_1: Final[int] = 1
        NUM_2: Final[int] = 2
        NUM_3: Final[int] = 3
        X: Final[str] = "x"
        Y: Final[str] = "y"

    class TestDomain:
        """Flext-core-specific domain utilities test constants."""

        ENTITY_NAME_ALICE: Final[str] = "Alice"
        ENTITY_NAME_BOB: Final[str] = "Bob"
        ENTITY_VALUE_10: Final[int] = 10
        ENTITY_VALUE_20: Final[int] = 20
        VALUE_DATA_TEST: Final[str] = "test"
        VALUE_COUNT_5: Final[int] = 5
        VALUE_COUNT_10: Final[int] = 10
        CUSTOM_ID_1: Final[str] = "id1"
        CUSTOM_ID_2: Final[str] = "id2"
        COMPLEX_ITEMS: Final[tuple[str, ...]] = ("a", "b")

    class Result:
        """Flext-core-specific FlextResult test constants."""

        TEST_VALUE: Final[str] = "test_value"
        TEST_INT: Final[int] = 42
        TEST_INT_DOUBLE: Final[int] = 84
        TEST_ERROR: Final[str] = "test_error"
        TEST_ERROR_CODE: Final[str] = "TEST_ERROR"
        UNKNOWN_ERROR: Final[str] = "Unknown error occurred"
        DEFAULT_VALUE: Final[str] = "default"
        MISSING_VALUE: Final[str] = "Missing value"
        INVALID_INDEX: Final[str] = "only supports indices 0 (data) and 1 (error)"
        CANNOT_ACCEPT_NONE: Final[str] = "cannot accept None"
        TEST_DATA: Final[core_m.ConfigMap] = core_m.ConfigMap({
            "key": "value",
            "value": 5,
        })
        TEST_DICT: Final[core_m.ConfigMap] = core_m.ConfigMap({"key": "value"})
        TEST_LIST: Final[tuple[int, ...]] = (1, 2, 3)
        MAX_EXECUTION_TIME: Final[float] = 1.0
        ITERATION_COUNT: Final[int] = 1000
        TEST_BATCH_SIZE: Final[int] = 10

    class Network(core_c.Network):
        """Network-related defaults for tests - real inheritance."""

    class Validation(core_c.Validation):
        """Input validation limits for tests - real inheritance."""

    class Errors(core_c.Errors):
        """Error codes for tests - real inheritance."""

    class Exceptions(core_c.Exceptions):
        """Exception handling configuration for tests."""

        FailureLevel = core_c.Exceptions.FailureLevel

    class Messages(core_c.Messages):
        """User-facing message templates for tests - real inheritance."""

    class Defaults(core_c.Defaults):
        """Default values for tests - real inheritance."""

    class Utilities(core_c.Utilities):
        """Utility constants for tests - real inheritance."""

    class Settings(core_c.Settings):
        """Configuration defaults for tests."""

        LogLevel = core_c.Settings.LogLevel
        Environment = core_c.Settings.Environment

    class ModelConfig(core_c.ModelConfig):
        """Pydantic model configuration defaults for tests - real inheritance."""

    class Platform(core_c.Platform):
        """Platform-specific constants for tests - real inheritance."""

    class Performance(core_c.Performance):
        """Performance thresholds for tests - real inheritance."""

        class BatchProcessing(core_c.Performance.BatchProcessing):
            """Batch processing constants for tests - real inheritance."""

    class Reliability(core_c.Reliability):
        """Reliability thresholds for tests - real inheritance."""

        CircuitBreakerState = core_c.Reliability.CircuitBreakerState

    class Security(core_c.Security):
        """Security constants for tests - real inheritance."""

    class Logging(core_c.Logging):
        """Logging configuration for tests - real inheritance."""

        ContextOperation = core_c.Logging.ContextOperation

    class Literals(core_c.Literals):
        """Literal type aliases for tests - real inheritance."""

    class Domain(core_c.Domain):
        """Domain-specific constants for tests."""

        Status = core_c.Domain.Status
        Currency = core_c.Domain.Currency
        OrderStatus = core_c.Domain.OrderStatus

    class Cqrs(core_c.Cqrs):
        """CQRS pattern constants for tests."""

        Status = core_c.Cqrs.Status
        HandlerType = core_c.Cqrs.HandlerType
        CommonStatus = core_c.Cqrs.CommonStatus
        MetricType = core_c.Cqrs.MetricType
        ProcessingMode = core_c.Cqrs.ProcessingMode
        ValidationLevel = core_c.Cqrs.ValidationLevel
        ProcessingPhase = core_c.Cqrs.ProcessingPhase
        BindType = core_c.Cqrs.BindType
        MergeStrategy = core_c.Cqrs.MergeStrategy
        HealthStatus = core_c.Cqrs.HealthStatus
        SpecialStatus = core_c.Cqrs.SpecialStatus
        TokenType = core_c.Cqrs.TokenType
        OperationStatus = core_c.Cqrs.OperationStatus
        SerializationFormat = core_c.Cqrs.SerializationFormat
        Compression = core_c.Cqrs.Compression
        Aggregation = core_c.Cqrs.Aggregation
        Action = core_c.Cqrs.Action
        PersistenceLevel = core_c.Cqrs.PersistenceLevel
        TargetFormat = core_c.Cqrs.TargetFormat
        WarningLevel = core_c.Cqrs.WarningLevel
        OutputFormat = core_c.Cqrs.OutputFormat
        Mode = core_c.Cqrs.Mode
        RegistrationStatus = core_c.Cqrs.RegistrationStatus

    class Context(core_c.Context):
        """Context management constants for tests - real inheritance."""

    class Container(core_c.Container):
        """Dependency injection container constants for tests - real inheritance."""

    class Dispatcher(core_c.Dispatcher):
        """Message dispatcher constants for tests - real inheritance."""

    class Pagination(core_c.Pagination):
        """Pagination configuration for tests - real inheritance."""

    class Mixins(core_c.Mixins):
        """Constants for mixin operations for tests - real inheritance."""

    class Processing(core_c.Processing):
        """Processing pipeline constants for tests - real inheritance."""

    class Fixtures:
        """Test fixture dataclasses for flext-core tests."""

        @dataclass(frozen=True, slots=True)
        class Identifiers:
            """Test identifiers and IDs."""

            user_id: str = "test_user_123"
            session_id: str = "test_session_123"
            service_name: str = "test_service"
            operation_id: str = "test_operation"
            request_id: str = "test-request-456"
            correlation_id: str = "test-corr-123"

        @dataclass(frozen=True, slots=True)
        class Names:
            """Test module and component names."""

            module_name: str = "test_module"
            handler_name: str = "test_handler"
            chain_name: str = "test_chain"
            command_type: str = "test_command"
            query_type: str = "test_query"
            logger_name: str = "test_logger"
            app_name: str = "test-app"
            validation_app: str = "validation-test"
            source_service: str = "test_service"

        @dataclass(frozen=True, slots=True)
        class ErrorData:
            """Test error codes and messages."""

            error_code: str = "TEST_ERROR_001"
            validation_error: str = "test_error"
            operation_error: str = "Op failed"
            config_error: str = "Config failed"
            timeout_error: str = "Operation timeout"

        @dataclass(frozen=True, slots=True)
        class Data:
            """Test field names and data values."""

            field_name: str = "test_field"
            config_key: str = "test_key"
            username: str = "test_user"
            email: str = "test@example.com"
            password: str = "test_pass"
            string_value: str = "test_value"
            input_data: str = "test_input"
            request_data: str = "test_request"
            result_data: str = "test_result"
            message: str = "test_message"

        @dataclass(frozen=True, slots=True)
        class PatternData:
            """Test patterns and formats."""

            slug_input: str = "Test_String"
            slug_expected: str = "test_string"
            uuid_format: str = "550e8400-e29b-41d4-a716-446655440000"

        @dataclass(frozen=True, slots=True)
        class NumericValues:
            """Test port and numeric values."""

            port: int = 8080
            timeout: int = 30
            retry_count: int = 3
            batch_size: int = 100


# Short alias per FLEXT convention
c = TestsFlextConstants

__all__ = [
    "TestsFlextConstants",
    "c",
]
