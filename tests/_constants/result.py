"""Constants mixin for result.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final, Literal

from flext_core import FlextModels as m


class TestsFlextConstantsResult:
    """Flat result and railway constants for flext-core tests."""

    @unique
    class RailwayOperation(StrEnum):
        """Railway operation types for testing."""

        GET_EMAIL = "get_email"
        SEND_EMAIL = "send_email"
        GET_STATUS = "get_status"
        DOUBLE = "double"
        SQUARE = "square"
        NEGATE = "negate"

    type RailwayOperationLiteral = Literal[
        "get_email",
        "send_email",
        "get_status",
        "double",
        "square",
        "negate",
    ]
    OP_GET_EMAIL: Final[str] = RailwayOperation.GET_EMAIL
    OP_SEND_EMAIL: Final[str] = RailwayOperation.SEND_EMAIL
    OP_GET_STATUS: Final[str] = RailwayOperation.GET_STATUS
    OP_DOUBLE: Final[str] = RailwayOperation.DOUBLE
    OP_SQUARE: Final[str] = RailwayOperation.SQUARE
    OP_NEGATE: Final[str] = RailwayOperation.NEGATE

    # Mapper test constants
    MAPPER_OLD_KEY: Final[str] = "old_key"
    MAPPER_NEW_KEY: Final[str] = "new_key"
    MAPPER_FOO: Final[str] = "foo"
    MAPPER_BAR: Final[str] = "bar"
    MAPPER_UNMAPPED: Final[str] = "unmapped"
    MAPPER_VALUE1: Final[str] = "value1"
    MAPPER_VALUE2: Final[str] = "value2"
    MAPPER_FLAGS_READ: Final[str] = "read"
    MAPPER_FLAGS_WRITE: Final[str] = "write"
    MAPPER_FLAGS_DELETE: Final[str] = "delete"
    MAPPER_CAN_READ: Final[str] = "can_read"
    MAPPER_CAN_WRITE: Final[str] = "can_write"
    MAPPER_CAN_DELETE: Final[str] = "can_delete"
    MAPPER_HELLO: Final[str] = "hello"
    MAPPER_WORLD: Final[str] = "world"
    MAPPER_HELLO_UPPER: Final[str] = "HELLO"
    MAPPER_WORLD_UPPER: Final[str] = "WORLD"
    MAPPER_A: Final[str] = "a"
    MAPPER_B: Final[str] = "b"
    MAPPER_C: Final[str] = "c"
    MAPPER_NUM_1: Final[int] = 1
    MAPPER_NUM_2: Final[int] = 2
    MAPPER_NUM_3: Final[int] = 3
    MAPPER_X: Final[str] = "x"
    MAPPER_Y: Final[str] = "y"

    # Result test constants
    RESULT_TEST_VALUE: Final[str] = "test_value"
    RESULT_TEST_INT: Final[int] = 42
    RESULT_TEST_INT_DOUBLE: Final[int] = 84
    RESULT_TEST_ERROR: Final[str] = "test_error"
    RESULT_TEST_ERROR_CODE: Final[str] = "TEST_ERROR"
    RESULT_UNKNOWN_ERROR: Final[str] = "Unknown error occurred"
    RESULT_DEFAULT_VALUE: Final[str] = "default"
    RESULT_MISSING_VALUE: Final[str] = "Missing value"
    RESULT_INVALID_INDEX: Final[str] = "only supports indices 0 (data) and 1 (error)"
    RESULT_CANNOT_ACCEPT_NONE: Final[str] = "cannot accept None"
    RESULT_TEST_DATA: Final[m.ConfigMap] = m.ConfigMap(
        root={
            "key": "value",
            "value": 5,
        }
    )
    RESULT_TEST_DICT: Final[m.ConfigMap] = m.ConfigMap(root={"key": "value"})
    RESULT_TEST_LIST: Final[tuple[int, ...]] = (1, 2, 3)
    RESULT_MAX_EXECUTION_TIME: Final[float] = 1.0
    RESULT_ITERATION_COUNT: Final[int] = 1000
    RESULT_TEST_BATCH_SIZE: Final[int] = 10
