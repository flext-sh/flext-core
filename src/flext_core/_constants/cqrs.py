"""FlextConstantsCqrs - CQRS constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsCqrs:
    """Constants for CQRS patterns and workflows."""

    DEFAULT_COMMAND_TYPE: Final[str] = "generic_command"
    DEFAULT_TIMESTAMP: Final[str] = ""
    DEFAULT_RETRIES: Final[int] = 0
    MIN_RETRIES: Final[int] = 0
    MAX_RETRIES: Final[int] = 5
    DEFAULT_MAX_COMMAND_RETRIES: Final[int] = 0
    # DEFAULT_PAGE_SIZE and MAX_PAGE_SIZE inherited from FlextConstantsBase via FlextConstants MRO
    DEFAULT_MAX_VALIDATION_ERRORS: Final[int] = 10
    DEFAULT_MINIMUM_THROUGHPUT: Final[int] = 10
    DEFAULT_PARALLEL_EXECUTION: Final[bool] = False
    DEFAULT_STOP_ON_ERROR: Final[bool] = True
    CQRS_OPERATION_FAILED: Final[str] = "CQRS_OPERATION_FAILED"
    COMMAND_VALIDATION_FAILED: Final[str] = "COMMAND_VALIDATION_FAILED"
    QUERY_VALIDATION_FAILED: Final[str] = "QUERY_VALIDATION_FAILED"
    HANDLER_CONFIG_INVALID: Final[str] = "HANDLER_CONFIG_INVALID"

    @unique
    class HandlerType(StrEnum):
        """CQRS handler types enumeration."""

        COMMAND = "command"
        QUERY = "query"
        EVENT = "event"
        OPERATION = "operation"
        SAGA = "saga"

    @unique
    class MetricType(StrEnum):
        """Service metric types enumeration."""

        COUNTER = "counter"
        GAUGE = "gauge"
        HISTOGRAM = "histogram"
        SUMMARY = "summary"

    DEFAULT_HANDLER_TYPE: HandlerType = HandlerType.COMMAND

    @unique
    class ProcessingMode(StrEnum):
        """CQRS processing modes enumeration."""

        BATCH = "batch"
        STREAM = "stream"
        PARALLEL = "parallel"
        SEQUENTIAL = "sequential"

    @unique
    class MergeStrategy(StrEnum):
        """CQRS merge strategies enumeration."""

        REPLACE = "replace"
        UPDATE = "update"
        DEEP = "deep"
        MERGE_DEEP = "merge_deep"
        OVERRIDE = "override"
        APPEND = "append"
        FILTER_NONE = "filter_none"
        FILTER_EMPTY = "filter_empty"
        FILTER_BOTH = "filter_both"

    @unique
    class Aggregation(StrEnum):
        """CQRS aggregation functions enumeration."""

        SUM = "sum"
        AVG = "avg"
        MIN = "min"
        MAX = "max"
        COUNT = "count"

    @unique
    class Action(StrEnum):
        """CQRS action types enumeration.

        DRY Pattern:
            StrEnum is the single source of truth. Use Action.GET.value
            or Action.GET directly - no base strings needed.
        """

        GET = "get"
        CREATE = "create"
        UPDATE = "update"
        DELETE = "delete"
        LIST = "list"

    @unique
    class WarningLevel(StrEnum):
        """CQRS warning levels enumeration.

        DRY Pattern:
            StrEnum is the single source of truth. Use WarningLevel.NONE.value
            or WarningLevel.NONE directly - no base strings needed.
        """

        NONE = "none"
        WARN = "warn"
        ERROR = "error"

    @unique
    class Mode(StrEnum):
        """CQRS operation modes enumeration.

        DRY Pattern:
            StrEnum is the single source of truth. Use Mode.VALIDATION.value
            or Mode.VALIDATION directly - no base strings needed.
        """

        VALIDATION = "validation"
        SERIALIZATION = "serialization"
