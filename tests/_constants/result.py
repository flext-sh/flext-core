"""Constants mixin for result.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final, Literal

from flext_core import FlextModels as m


class TestsFlextCoreConstantsResult:
    class Railway:
        """Flext-core-specific railway pattern operation constants."""

        @unique
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
        TEST_DATA: Final[m.ConfigMap] = m.ConfigMap(
            root={
                "key": "value",
                "value": 5,
            }
        )
        TEST_DICT: Final[m.ConfigMap] = m.ConfigMap(root={"key": "value"})
        TEST_LIST: Final[tuple[int, ...]] = (1, 2, 3)
        MAX_EXECUTION_TIME: Final[float] = 1.0
        ITERATION_COUNT: Final[int] = 1000
        TEST_BATCH_SIZE: Final[int] = 10
