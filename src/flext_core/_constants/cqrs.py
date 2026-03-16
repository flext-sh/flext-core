"""FlextConstantsCqrs - CQRS constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class FlextConstantsCqrs:
    """Constants for CQRS patterns and workflows."""

    class Cqrs:
        """CQRS pattern constants."""

        class Status(StrEnum):
            """CQRS status enumeration."""

            STOPPED = "stopped"

        class HandlerType(StrEnum):
            """CQRS handler types enumeration."""

            COMMAND = "command"
            QUERY = "query"
            EVENT = "event"
            OPERATION = "operation"
            SAGA = "saga"

        class CommonStatus(StrEnum):
            """CQRS common status enumeration."""

            ACTIVE = "active"
            INACTIVE = "inactive"
            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            CANCELLED = "cancelled"
            COMPENSATING = "compensating"
            ARCHIVED = "archived"

        class MetricType(StrEnum):
            """Service metric types enumeration."""

            COUNTER = "counter"
            GAUGE = "gauge"
            HISTOGRAM = "histogram"
            SUMMARY = "summary"

        class ServiceMetricCategory(StrEnum):
            """Service metric categories enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use ServiceMetricCategory.PERFORMANCE.value
                or ServiceMetricCategory.PERFORMANCE directly - no base strings needed.
            """

            PERFORMANCE = "performance"
            ERRORS = "errors"
            THROUGHPUT = "throughput"

        DEFAULT_METRIC_CATEGORIES: Final[tuple[str, ...]] = (
            ServiceMetricCategory.PERFORMANCE,
            ServiceMetricCategory.ERRORS,
            ServiceMetricCategory.THROUGHPUT,
        )
        "Default metric categories for service metrics requests."
        DEFAULT_HANDLER_TYPE: HandlerType = HandlerType.COMMAND

        class ProcessingMode(StrEnum):
            """CQRS processing modes enumeration."""

            BATCH = "batch"
            STREAM = "stream"
            PARALLEL = "parallel"
            SEQUENTIAL = "sequential"

        class ProcessingPhase(StrEnum):
            """CQRS processing phases enumeration."""

            PREPARE = "prepare"
            EXECUTE = "execute"
            VALIDATE = "validate"
            COMPLETE = "complete"

        class BindType(StrEnum):
            """CQRS binding types enumeration."""

            TEMPORARY = "temporary"
            PERMANENT = "permanent"

        class MergeStrategy(StrEnum):
            """CQRS merge strategies enumeration."""

            REPLACE = "replace"
            UPDATE = "update"
            MERGE_DEEP = "merge_deep"

        class HealthStatus(StrEnum):
            """CQRS health status enumeration."""

            HEALTHY = "healthy"
            DEGRADED = "degraded"
            UNHEALTHY = "unhealthy"

        class SpecialStatus(StrEnum):
            """Special status values not in CommonStatus."""

            SENT = "sent"
            IDLE = "idle"
            PROCESSING = "processing"

        class TokenType(StrEnum):
            """CQRS token types enumeration."""

            BEARER = "bearer"
            API_KEY = "api_key"
            JWT = "jwt"

        class OperationStatus(StrEnum):
            """CQRS operation status enumeration."""

            SUCCESS = "success"
            FAILURE = "failure"
            PARTIAL = "partial"

        class SerializationFormat(StrEnum):
            """CQRS serialization formats enumeration."""

            JSON = "json"
            YAML = "yaml"
            TOML = "toml"
            MSGPACK = "msgpack"

        class Compression(StrEnum):
            """CQRS compression formats enumeration."""

            NONE = "none"
            GZIP = "gzip"
            BZIP2 = "bzip2"
            LZ4 = "lz4"

        class Aggregation(StrEnum):
            """CQRS aggregation functions enumeration."""

            SUM = "sum"
            AVG = "avg"
            MIN = "min"
            MAX = "max"
            COUNT = "count"

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

        class PersistenceLevel(StrEnum):
            """CQRS persistence levels enumeration."""

            MEMORY = "memory"
            DISK = "disk"
            DISTRIBUTED = "distributed"

        class TargetFormat(StrEnum):
            """CQRS target formats enumeration."""

            FULL = "full"
            COMPACT = "compact"
            MINIMAL = "minimal"

        class WarningLevel(StrEnum):
            """CQRS warning levels enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use WarningLevel.NONE.value
                or WarningLevel.NONE directly - no base strings needed.
            """

            NONE = "none"
            WARN = "warn"
            ERROR = "error"

        class OutputFormat(StrEnum):
            """CQRS output formats enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use OutputFormat.DICT.value
                or OutputFormat.DICT directly - no base strings needed.
            """

            DICT = "dict"
            JSON = "json"

        class Mode(StrEnum):
            """CQRS operation modes enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use Mode.VALIDATION.value
                or Mode.VALIDATION directly - no base strings needed.
            """

            VALIDATION = "validation"
            SERIALIZATION = "serialization"

        class RegistrationStatus(StrEnum):
            """CQRS registration status enumeration.

            DRY Pattern:
                Values match Cqrs.CommonStatus. These StrEnum values are the
                single source of truth.
            """

            ACTIVE = "active"
            INACTIVE = "inactive"

        DEFAULT_COMMAND_TYPE: Final[str] = "generic_command"
        DEFAULT_TIMESTAMP: Final[str] = ""
        DEFAULT_TIMEOUT: Final[int] = 30000
        MIN_TIMEOUT: Final[int] = 1000
        MAX_TIMEOUT: Final[int] = 300000
        DEFAULT_COMMAND_TIMEOUT: Final[int] = 0
        DEFAULT_RETRIES: Final[int] = 0
        MIN_RETRIES: Final[int] = 0
        MAX_RETRIES: Final[int] = 5
        DEFAULT_MAX_COMMAND_RETRIES: Final[int] = 0
        DEFAULT_PAGE_SIZE: Final[int] = 10
        MAX_PAGE_SIZE: Final[int] = 1000
        DEFAULT_MAX_VALIDATION_ERRORS: Final[int] = 10
        DEFAULT_MINIMUM_THROUGHPUT: Final[int] = 10
        DEFAULT_PARALLEL_EXECUTION: Final[bool] = False
        DEFAULT_STOP_ON_ERROR: Final[bool] = True
        CQRS_OPERATION_FAILED: Final[str] = "CQRS_OPERATION_FAILED"
        COMMAND_VALIDATION_FAILED: Final[str] = "COMMAND_VALIDATION_FAILED"
        QUERY_VALIDATION_FAILED: Final[str] = "QUERY_VALIDATION_FAILED"
        HANDLER_CONFIG_INVALID: Final[str] = "HANDLER_CONFIG_INVALID"
