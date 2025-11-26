"""Test constants organized by domain namespace.

Provides centralized constants for all test domains, organized in nested namespaces
for maximum reuse and consistency across test modules. Uses advanced Python patterns
for better organization and type safety.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class TestConstants:
    """Base namespace for all test constants using Final annotations for immutability."""

    class Delimiters:
        """Delimiter characters for string parsing."""

        COMMA: Final[str] = ","
        SEMICOLON: Final[str] = ";"
        PIPE: Final[str] = "|"
        COLON: Final[str] = ":"
        TAB: Final[str] = "\t"
        NEWLINE: Final[str] = "\n"

    class EscapeChars:
        """Escape characters for string parsing."""

        BACKSLASH: Final[str] = "\\"
        HASH: Final[str] = "#"
        AT: Final[str] = "@"
        QUOTE: Final[str] = '"'
        SINGLE_QUOTE: Final[str] = "'"

    class Strings:
        """Common test strings organized by complexity."""

        # Basic strings
        EMPTY: Final[str] = ""
        SINGLE_CHAR: Final[str] = "a"
        BASIC_WORD: Final[str] = "hello"
        BASIC_LIST: Final[str] = "a,b,c"
        NUMERIC_LIST: Final[str] = "1,2,3"

        # With spaces
        WITH_SPACES: Final[str] = "a, b, c"
        EXCESSIVE_SPACES: Final[str] = "  a  ,  b  ,  c  "
        LEADING_SPACES: Final[str] = "  hello"
        TRAILING_SPACES: Final[str] = "hello  "

        # Edge cases
        LEADING_TRAILING: Final[str] = ",a,b,c,"
        WITH_EMPTY: Final[str] = "a,,c"
        ONLY_DELIMITERS: Final[str] = ",,,"
        UNICODE_CHARS: Final[str] = "héllo,wörld"

        # Service-related
        VALID_EMAIL: Final[str] = "test@example.com"
        INVALID_EMAIL: Final[str] = "invalid-email"
        USER_ID_VALID: Final[str] = "123"
        USER_ID_INVALID: Final[str] = "invalid"
        USER_ID_EMPTY: Final[str] = ""

    class Patterns:
        """Regex patterns for string processing."""

        WHITESPACE: Final[str] = r"\s+"
        DASH: Final[str] = r"-+"
        EQUALS_SPACE: Final[str] = r"\s+="
        COMMA_SPACE: Final[str] = r",\s+"
        EMAIL: Final[str] = r"^[^@]+@[^@]+\.[^@]+$"
        ALPHA_ONLY: Final[str] = r"^[a-zA-Z]+$"
        NUMERIC_ONLY: Final[str] = r"^\d+$"

    class Replacements:
        """Replacement strings for string processing."""

        SPACE: Final[str] = " "
        UNDERSCORE: Final[str] = "_"
        DASH: Final[str] = "-"
        EQUALS: Final[str] = "="
        COMMA: Final[str] = ","
        EMPTY: Final[str] = ""

    class Errors:
        """Error message patterns for validation."""

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
        FAILED_PIPELINE: Final[str] = "Failed to apply"
        INVALID_REGEX: Final[str] = "Invalid regex pattern"
        USER_NOT_FOUND: Final[str] = "User not found"
        INVALID_EMAIL: Final[str] = "Invalid email address"
        VALUE_TOO_LOW: Final[str] = "Value must be positive"
        VALUE_TOO_HIGH: Final[str] = "Value must be <= 100"
        PROCESSING_ERROR: Final[str] = "Processing error occurred"

    class Validation:
        """Validation constants for various checks."""

        MIN_LENGTH_DEFAULT: Final[int] = 3
        MAX_LENGTH_DEFAULT: Final[int] = 100
        MIN_VALUE: Final[int] = 0
        MAX_VALUE: Final[int] = 100

    class Services:
        """Service-related constants."""

        DEFAULT_USER_NAME_PREFIX: Final[str] = "User "
        DEFAULT_EMAIL_DOMAIN: Final[str] = "@example.com"
        DEFAULT_ERROR_MESSAGE: Final[str] = "Test error"

    class Railway:
        """Railway pattern operation constants."""

        OP_GET_EMAIL: Final[str] = "get_email"
        OP_SEND_EMAIL: Final[str] = "send_email"
        OP_GET_STATUS: Final[str] = "get_status"
        OP_DOUBLE: Final[str] = "double"
        OP_SQUARE: Final[str] = "square"
        OP_NEGATE: Final[str] = "negate"

    class HTTP:
        """HTTP-related constants for testing."""

        STATUS_OK: Final[int] = 200
        STATUS_NOT_FOUND: Final[int] = 404
        STATUS_BAD_REQUEST: Final[int] = 400
        CONTENT_TYPE_JSON: Final[str] = "application/json"
        METHOD_GET: Final[str] = "GET"
        METHOD_POST: Final[str] = "POST"

    class DataMapper:
        """DataMapper utilities test constants."""

        # Mapping test data
        OLD_KEY: Final[str] = "old_key"
        NEW_KEY: Final[str] = "new_key"
        FOO: Final[str] = "foo"
        BAR: Final[str] = "bar"
        UNMAPPED: Final[str] = "unmapped"
        VALUE1: Final[str] = "value1"
        VALUE2: Final[str] = "value2"

        # Flags test data
        FLAGS_READ: Final[str] = "read"
        FLAGS_WRITE: Final[str] = "write"
        FLAGS_DELETE: Final[str] = "delete"
        CAN_READ: Final[str] = "can_read"
        CAN_WRITE: Final[str] = "can_write"
        CAN_DELETE: Final[str] = "can_delete"

        # Transform test data
        HELLO: Final[str] = "hello"
        WORLD: Final[str] = "world"
        HELLO_UPPER: Final[str] = "HELLO"
        WORLD_UPPER: Final[str] = "WORLD"

        # Filter test data
        A: Final[str] = "a"
        B: Final[str] = "b"
        C: Final[str] = "c"
        NUM_1: Final[int] = 1
        NUM_2: Final[int] = 2
        NUM_3: Final[int] = 3

        # Invert test data
        X: Final[str] = "x"
        Y: Final[str] = "y"

    class Domain:
        """Domain utilities test constants."""

        # Entity test data
        ENTITY_NAME_ALICE: Final[str] = "Alice"
        ENTITY_NAME_BOB: Final[str] = "Bob"
        ENTITY_VALUE_10: Final[int] = 10
        ENTITY_VALUE_20: Final[int] = 20

        # Value object test data
        VALUE_DATA_TEST: Final[str] = "test"
        VALUE_COUNT_5: Final[int] = 5
        VALUE_COUNT_10: Final[int] = 10

        # Custom entity ID
        CUSTOM_ID_1: Final[str] = "id1"
        CUSTOM_ID_2: Final[str] = "id2"

        # Complex value test data
        COMPLEX_ITEMS: Final[list[str]] = ["a", "b"]

    class Result:
        """FlextResult test constants."""

        # Test values
        TEST_VALUE: Final[str] = "test_value"
        TEST_INT: Final[int] = 42
        TEST_INT_DOUBLE: Final[int] = 84
        TEST_ERROR: Final[str] = "test_error"
        TEST_ERROR_CODE: Final[str] = "TEST_ERROR"
        UNKNOWN_ERROR: Final[str] = "Unknown error occurred"
        DEFAULT_VALUE: Final[str] = "default"

        # Error messages
        MISSING_VALUE: Final[str] = "Missing value"
        INVALID_INDEX: Final[str] = "only supports indices 0 (data) and 1 (error)"
        CANNOT_ACCEPT_NONE: Final[str] = "cannot accept None"

        # Test data
        TEST_DATA: Final[dict[str, object]] = {"key": "value", "value": 5}
        TEST_DICT: Final[dict[str, object]] = {"key": "value"}
        TEST_LIST: Final[list[int]] = [1, 2, 3]

        # Performance thresholds
        MAX_EXECUTION_TIME: Final[float] = 1.0
        ITERATION_COUNT: Final[int] = 1000
        TEST_BATCH_SIZE: Final[int] = 10

    class Payload:
        """Payload model test constants."""

        # Test IDs
        USER_ID_1: Final[str] = "1"
        USER_ID_2: Final[str] = "2"
        USER_ID_123: Final[str] = "123"
        PRODUCT_ID_456: Final[str] = "456"

        # Test names
        USER_NAME_TEST: Final[str] = "Test"
        PRODUCT_TITLE_TEST: Final[str] = "Test"
        PRODUCT_TITLE_WRONG: Final[str] = "Wrong"

        # Error patterns
        TYPE_MISMATCH_PATTERN: Final[str] = (
            r"Payload\[PayloadUser\] received data of type PayloadProduct"
        )

        # Class name patterns
        PAYLOAD_IN_NAME: Final[str] = "Payload"
        USER_IN_NAME: Final[str] = "User"
