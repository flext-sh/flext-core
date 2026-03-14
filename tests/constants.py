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

from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from flext_core import m
from flext_infra import FlextInfraConstants
from flext_tests import FlextTestsConstants


class TestsFlextConstants(FlextTestsConstants, FlextInfraConstants):
    """Constants for flext-core tests - extends FlextTestsConstants.

    Architecture: Extends FlextTestsConstants with flext-core-specific constants.
    All generic constants from FlextTestsConstants are available through inheritance.

    Rules:
    - NEVER duplicate constants from FlextTestsConstants
    - Only flext-core-specific constants allowed (not generic for other projects)
    - All generic constants come from FlextTestsConstants
    """

    class Tests(FlextTestsConstants.Tests):
        """flext-core-specific test namespaces."""

        class ServiceTestType(StrEnum):
            """Service test type enum for test scenarios."""

            GET_USER = "get_user"
            VALIDATE = "validate"
            FAIL = "fail"

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

            WHITESPACE: Final[str] = "\\s+"
            DASH: Final[str] = "-+"
            EQUALS_SPACE: Final[str] = "\\s+="
            COMMA_SPACE: Final[str] = ",\\s+"
            EMAIL: Final[str] = "^[^@]+@[^@]+\\.[^@]+$"
            ALPHA_ONLY: Final[str] = "^[a-zA-Z]+$"
            NUMERIC_ONLY: Final[str] = "^\\d+$"

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
            """Flext-core-specific r test constants."""

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
            TEST_DATA: Final[m.ConfigMap] = m.ConfigMap({"key": "value", "value": 5})
            TEST_DICT: Final[m.ConfigMap] = m.ConfigMap({"key": "value"})
            TEST_LIST: Final[tuple[int, ...]] = (1, 2, 3)
            MAX_EXECUTION_TIME: Final[float] = 1.0
            ITERATION_COUNT: Final[int] = 1000
            TEST_BATCH_SIZE: Final[int] = 10

        class Exceptions(FlextTestsConstants.Exceptions):
            """Exception handling configuration for tests."""

            FailureLevel = FlextTestsConstants.Exceptions.FailureLevel

        class Settings(FlextTestsConstants.Settings):
            """Configuration defaults for tests."""

            LogLevel = FlextTestsConstants.Settings.LogLevel
            Environment = FlextTestsConstants.Settings.Environment

        class Logging(FlextTestsConstants.Logging):
            """Logging configuration for tests - real inheritance."""

            ContextOperation = FlextTestsConstants.Logging.ContextOperation

        class Domain(FlextTestsConstants.Domain):
            """Domain-specific constants for tests."""

            Status = FlextTestsConstants.Domain.Status
            Currency = FlextTestsConstants.Domain.Currency
            OrderStatus = FlextTestsConstants.Domain.OrderStatus

        class Cqrs(FlextTestsConstants.Cqrs):
            """CQRS pattern constants for tests."""

            Status = FlextTestsConstants.Cqrs.Status
            HandlerType = FlextTestsConstants.Cqrs.HandlerType
            CommonStatus = FlextTestsConstants.Cqrs.CommonStatus
            MetricType = FlextTestsConstants.Cqrs.MetricType
            ProcessingMode = FlextTestsConstants.Cqrs.ProcessingMode
            ProcessingPhase = FlextTestsConstants.Cqrs.ProcessingPhase
            BindType = FlextTestsConstants.Cqrs.BindType
            MergeStrategy = FlextTestsConstants.Cqrs.MergeStrategy
            HealthStatus = FlextTestsConstants.Cqrs.HealthStatus
            SpecialStatus = FlextTestsConstants.Cqrs.SpecialStatus
            TokenType = FlextTestsConstants.Cqrs.TokenType
            OperationStatus = FlextTestsConstants.Cqrs.OperationStatus
            SerializationFormat = FlextTestsConstants.Cqrs.SerializationFormat
            Compression = FlextTestsConstants.Cqrs.Compression
            Aggregation = FlextTestsConstants.Cqrs.Aggregation
            Action = FlextTestsConstants.Cqrs.Action
            PersistenceLevel = FlextTestsConstants.Cqrs.PersistenceLevel
            TargetFormat = FlextTestsConstants.Cqrs.TargetFormat
            WarningLevel = FlextTestsConstants.Cqrs.WarningLevel
            OutputFormat = FlextTestsConstants.Cqrs.OutputFormat
            Mode = FlextTestsConstants.Cqrs.Mode
            RegistrationStatus = FlextTestsConstants.Cqrs.RegistrationStatus

        class StatusEnum(StrEnum):
            """Reusable test status enum for test fixtures.

            Standard three-state status enum used across multiple test modules.
            """

            ACTIVE = "active"
            PENDING = "pending"
            INACTIVE = "inactive"

        class PriorityEnum(StrEnum):
            """Reusable test priority enum for test fixtures.

            Standard three-level priority enum used across multiple test modules.
            """

            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"

        class Fixtures:
            """Test fixture dataclasses for flext-core tests."""

            class Identifiers(BaseModel):
                """Test identifiers and IDs."""

                model_config = ConfigDict(frozen=True)

                user_id: Annotated[str, Field(default="test_user_123", description="Default test user identifier",)] = "test_user_123"
                session_id: Annotated[str, Field(default="test_session_123", description="Default test session identifier",)] = "test_session_123"
                service_name: Annotated[str, Field(default="test_service", description="Default test service name")] = "test_service"
                operation_id: Annotated[str, Field(default="test_operation", description="Default test operation identifier",)] = "test_operation"
                request_id: Annotated[str, Field(default="test-request-456", description="Default test request identifier",)] = "test-request-456"
                correlation_id: Annotated[str, Field(default="test-corr-123", description="Default test correlation identifier",)] = "test-corr-123"

            class Names(BaseModel):
                """Test module and component names."""

                model_config = ConfigDict(frozen=True)

                module_name: Annotated[str, Field(default="test_module", description="Default test module name")] = "test_module"
                handler_name: Annotated[str, Field(default="test_handler", description="Default test handler name")] = "test_handler"
                chain_name: Annotated[str, Field(default="test_chain", description="Default test chain name")] = "test_chain"
                command_type: Annotated[str, Field(default="test_command", description="Default test command type")] = "test_command"
                query_type: Annotated[str, Field(default="test_query", description="Default test query type")] = "test_query"
                logger_name: Annotated[str, Field(default="test_logger", description="Default test logger name")] = "test_logger"
                app_name: Annotated[str, Field(default="test-app", description="Default test application name")] = "test-app"
                validation_app: Annotated[str, Field(default="validation-test", description="Default validation test application name",)] = "validation-test"
                source_service: Annotated[str, Field(default="test_service", description="Default source service name",)] = "test_service"

            class ErrorData(BaseModel):
                """Test error codes and messages."""

                model_config = ConfigDict(frozen=True)

                error_code: Annotated[str, Field(default="TEST_ERROR_001", description="Default test error code")] = "TEST_ERROR_001"
                validation_error: Annotated[str, Field(default="test_error", description="Default validation error message",)] = "test_error"
                operation_error: Annotated[str, Field(default="Op failed", description="Default operation error message",)] = "Op failed"
                config_error: Annotated[str, Field(default="Config failed", description="Default configuration error message",)] = "Config failed"
                timeout_error: Annotated[str, Field(default="Operation timeout", description="Default timeout error message",)] = "Operation timeout"

            class Data(BaseModel):
                """Test field names and data values."""

                model_config = ConfigDict(frozen=True)

                field_name: Annotated[str, Field(default="test_field", description="Default test field name")] = "test_field"
                config_key: Annotated[str, Field(default="test_key", description="Default test config key")] = "test_key"
                username: Annotated[str, Field(default="test_user", description="Default test username")] = "test_user"
                email: Annotated[str, Field(default="test@example.com", description="Default test email")] = "test@example.com"
                password: Annotated[str, Field(default="test_pass", description="Default test password")] = "test_pass"
                string_value: Annotated[str, Field(default="test_value", description="Default test string value")] = "test_value"
                input_data: Annotated[str, Field(default="test_input", description="Default test input data")] = "test_input"
                request_data: Annotated[str, Field(default="test_request", description="Default test request data")] = "test_request"
                result_data: Annotated[str, Field(default="test_result", description="Default test result data")] = "test_result"
                message: Annotated[str, Field(default="test_message", description="Default test message")] = "test_message"

            class PatternData(BaseModel):
                """Test patterns and formats."""

                model_config = ConfigDict(frozen=True)

                slug_input: Annotated[str, Field(default="Test_String", description="Input value for slug conversion tests",)] = "Test_String"
                slug_expected: Annotated[str, Field(default="test_string", description="Expected slug conversion output",)] = "test_string"
                uuid_format: Annotated[str, Field(default="550e8400-e29b-41d4-a716-446655440000", description="Sample UUID format for tests",)] = "550e8400-e29b-41d4-a716-446655440000"

            class NumericValues(BaseModel):
                """Test port and numeric values."""

                model_config = ConfigDict(frozen=True)

                port: Annotated[int, Field(default=8080, description="Default test port")] = 8080
                timeout: Annotated[int, Field(default=30, description="Default timeout in seconds")] = 30
                retry_count: Annotated[int, Field(default=3, description="Default retry count")] = 3
                batch_size: Annotated[int, Field(default=100, description="Default test batch size")] = 100

    Strings = Tests.Strings
    Delimiters = Tests.Delimiters
    EscapeChars = Tests.EscapeChars
    Replacements = Tests.Replacements
    Patterns = Tests.Patterns
    TestErrors = Tests.TestErrors
    TestValidation = Tests.TestValidation
    Services = Tests.Services
    Railway = Tests.Railway
    HTTP = Tests.HTTP
    Mapper = Tests.Mapper
    TestDomain = Tests.TestDomain
    Result = Tests.Result


c = TestsFlextConstants

__all__ = ["TestsFlextConstants", "c"]
