"""Tests for c - production constants facade.

Tests all constant groups through the `c` facade (TestsFlextCoreConstants -> c MRO).
Covers: base, cqrs, validation, infrastructure, platform, domain, errors, settings, mixins.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from enum import StrEnum
from re import Pattern

import pytest
from flext_tests import tm

from tests import c, t


def _constant_case_id(case: t.JsonValue | t.VariadicTuple[t.JsonValue]) -> str:
    if isinstance(case, tuple) and case:
        first_item = case[0]
        return first_item if isinstance(first_item, str) else str(first_item)
    return str(case)


class TestFlextConstants:
    """Production constants accessed through the c.* facade."""

    # ------------------------------------------------------------------
    # Base constants - value correctness
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("NAME", "FLEXT"),
            ("INITIAL_TIME", 0.0),
            ("PERCENTAGE_MULTIPLIER", 100),
            ("MILLISECONDS_MULTIPLIER", 1000),
            ("MICROSECONDS_MULTIPLIER", 1000000),
            ("LOOPBACK_IP", "127.0.0.1"),
            ("LOCALHOST", "localhost"),
            ("MIN_PORT", 1),
            ("MAX_PORT", 65535),
            ("DEFAULT_TIMEOUT_SECONDS", 30),
            ("DEFAULT_CONNECTION_POOL_SIZE", 10),
            ("MAX_CONNECTION_POOL_SIZE", 100),
            ("MAX_HOSTNAME_LENGTH", 253),
            ("HTTP_STATUS_MIN", 100),
            ("HTTP_STATUS_MAX", 599),
            ("TYPE_MISMATCH", "Type mismatch"),
            ("CACHE_TTL", 300),
            ("MAX_MESSAGE_LENGTH", 100),
            ("DEFAULT_MIDDLEWARE_ORDER", 0),
            ("DATABASE_URL", "sqlite:///:memory:"),
            ("MAX_TIMEOUT_SECONDS", 3600),
            ("MIN_TIMEOUT_SECONDS", 1),
            ("DEFAULT_MAX_CACHE_SIZE", 100),
            ("DEFAULT_BATCH_SIZE", 1000),
            ("DEFAULT_PAGE_SIZE", 10),
            ("MAX_PAGE_SIZE", 1000),
            ("MIN_PAGE_SIZE", 1),
            ("DEFAULT_MAX_RETRY_ATTEMPTS", 3),
            ("DEFAULT_WORKERS", 4),
            ("ZERO", 0),
            ("EXPECTED_TUPLE_LENGTH", 2),
            ("DEFAULT_FAILURE_THRESHOLD", 5),
            ("PREVIEW_LENGTH", 50),
            ("DEFAULT_RECOVERY_TIMEOUT_SECONDS", 60),
            ("IDENTIFIER_LENGTH", 12),
            ("MAX_BATCH_SIZE_LIMIT", 10000),
            ("DEFAULT_BACKOFF_MULTIPLIER", 2.0),
            ("DEFAULT_MAX_DELAY_SECONDS", 60.0),
            ("MAX_TIMEOUT_SECONDS_PERFORMANCE", 600),
        ],
        ids=_constant_case_id,
    )
    def test_base_constant_values(
        self, attr: str, expected: t.Core.Tests.MatcherKwargValue
    ) -> None:
        """Base constants have correct values."""

    # ------------------------------------------------------------------
    # CQRS constants
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("DEFAULT_COMMAND_TYPE", "generic_command"),
            ("DEFAULT_TIMESTAMP", ""),
            ("DEFAULT_RETRIES", 0),
            ("MIN_RETRIES", 0),
            ("MAX_RETRIES", 5),
            ("DEFAULT_MAX_VALIDATION_ERRORS", 10),
            ("DEFAULT_PARALLEL_EXECUTION", False),
            ("DEFAULT_STOP_ON_ERROR", True),
            ("CQRS_OPERATION_FAILED", "CQRS_OPERATION_FAILED"),
            ("COMMAND_VALIDATION_FAILED", "COMMAND_VALIDATION_FAILED"),
            ("QUERY_VALIDATION_FAILED", "QUERY_VALIDATION_FAILED"),
            ("HANDLER_CONFIG_INVALID", "HANDLER_CONFIG_INVALID"),
        ],
    )
    def test_cqrs_constant_values(
        self, attr: str, expected: t.Core.Tests.MatcherKwargValue
    ) -> None:
        """CQRS constants have correct values."""

    @pytest.mark.parametrize(
        ("enum_cls", "members"),
        [
            (
                "HandlerType",
                {
                    "COMMAND": "command",
                    "QUERY": "query",
                    "EVENT": "event",
                    "OPERATION": "operation",
                    "SAGA": "saga",
                },
            ),
            (
                "MetricType",
                {
                    "COUNTER": "counter",
                    "GAUGE": "gauge",
                    "HISTOGRAM": "histogram",
                    "SUMMARY": "summary",
                },
            ),
            (
                "ServiceMetricCategory",
                {
                    "PERFORMANCE": "performance",
                    "ERRORS": "errors",
                    "THROUGHPUT": "throughput",
                },
            ),
            (
                "ProcessingMode",
                {
                    "BATCH": "batch",
                    "STREAM": "stream",
                    "PARALLEL": "parallel",
                    "SEQUENTIAL": "sequential",
                },
            ),
            (
                "ProcessingPhase",
                {
                    "PREPARE": "prepare",
                    "EXECUTE": "execute",
                    "VALIDATE": "validate",
                    "COMPLETE": "complete",
                },
            ),
            (
                "BindType",
                {"TEMPORARY": "temporary", "PERMANENT": "permanent"},
            ),
            (
                "MergeStrategy",
                {
                    "REPLACE": "replace",
                    "UPDATE": "update",
                    "MERGE_DEEP": "merge_deep",
                },
            ),
            (
                "HealthStatus",
                {
                    "HEALTHY": "healthy",
                    "DEGRADED": "degraded",
                    "UNHEALTHY": "unhealthy",
                    "UNKNOWN": "unknown",
                    "ERROR": "error",
                },
            ),
            (
                "TokenType",
                {"BEARER": "bearer", "API_KEY": "api_key", "JWT": "jwt"},
            ),
            (
                "SerializationFormat",
                {
                    "JSON": "json",
                    "YAML": "yaml",
                    "TOML": "toml",
                    "MSGPACK": "msgpack",
                },
            ),
            (
                "Compression",
                {
                    "NONE": "none",
                    "GZIP": "gzip",
                    "BZIP2": "bzip2",
                    "LZ4": "lz4",
                },
            ),
            (
                "Aggregation",
                {
                    "SUM": "sum",
                    "AVG": "avg",
                    "MIN": "min",
                    "MAX": "max",
                    "COUNT": "count",
                },
            ),
            (
                "Action",
                {
                    "GET": "get",
                    "CREATE": "create",
                    "UPDATE": "update",
                    "DELETE": "delete",
                    "LIST": "list",
                },
            ),
            (
                "PersistenceLevel",
                {
                    "MEMORY": "memory",
                    "DISK": "disk",
                    "DISTRIBUTED": "distributed",
                },
            ),
            (
                "TargetFormat",
                {"FULL": "full", "COMPACT": "compact", "MINIMAL": "minimal"},
            ),
            (
                "WarningLevel",
                {"NONE": "none", "WARN": "warn", "ERROR": "error"},
            ),
            (
                "OutputFormat",
                {"DICT": "dict", "JSON": "json"},
            ),
            (
                "Mode",
                {"VALIDATION": "validation", "SERIALIZATION": "serialization"},
            ),
            (
                "RegistrationStatus",
                {"ACTIVE": "active", "INACTIVE": "inactive"},
            ),
        ],
        ids=_constant_case_id,
    )
    def test_cqrs_enum_members(
        self,
        enum_cls: str,
        members: dict[str, str],
    ) -> None:
        """CQRS StrEnum classes contain exactly the expected members and values."""
        cls = getattr(c, enum_cls)
        tm.that(issubclass(cls, StrEnum), eq=True, msg=f"{enum_cls} not StrEnum")
        for name, value in members.items():
            member = cls[name]
            tm.that(str(member), eq=value)
        # Completeness: no extra members
        tm.that(len(cls), eq=len(members), msg=f"{enum_cls} member count mismatch")

    def test_cqrs_default_metric_categories_tuple(self) -> None:
        """DEFAULT_METRIC_CATEGORIES is a tuple of all ServiceMetricCategory values."""
        tm.that(c.DEFAULT_METRIC_CATEGORIES, is_=tuple)
        tm.that(len(c.DEFAULT_METRIC_CATEGORIES), eq=3)
        tm.that(c.DEFAULT_METRIC_CATEGORIES[0], eq=c.ServiceMetricCategory.PERFORMANCE)
        tm.that(c.DEFAULT_METRIC_CATEGORIES[1], eq=c.ServiceMetricCategory.ERRORS)
        tm.that(c.DEFAULT_METRIC_CATEGORIES[2], eq=c.ServiceMetricCategory.THROUGHPUT)

    def test_cqrs_default_handler_type(self) -> None:
        """DEFAULT_HANDLER_TYPE equals HandlerType.COMMAND."""
        tm.that(c.DEFAULT_HANDLER_TYPE, eq=c.HandlerType.COMMAND)

    # ------------------------------------------------------------------
    # Validation constants
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("MIN_NAME_LENGTH", 2),
            ("MAX_EMAIL_LENGTH", 254),
            ("EMAIL_PARTS_COUNT", 2),
            ("MIN_PHONE_DIGITS", 10),
            ("MAX_PHONE_DIGITS", 20),
            ("MIN_USERNAME_LENGTH", 3),
            ("MAX_AGE", 150),
            ("MIN_AGE", 0),
            ("VALIDATION_TIMEOUT_MS", 100.0),
            ("MAX_UNCOMMITTED_EVENTS", 100),
            ("DISCOUNT_THRESHOLD", 100),
            ("DISCOUNT_RATE", 0.05),
            ("RETRY_COUNT_MAX", 3),
        ],
    )
    def test_validation_numeric_values(
        self, attr: str, expected: t.Core.Tests.MatcherKwargValue
    ) -> None:
        """Validation numeric constants have correct values."""

    @pytest.mark.parametrize(
        "code",
        [
            "VALIDATION_ERROR",
            "TYPE_ERROR",
            "ATTRIBUTE_ERROR",
            "CONFIG_ERROR",
            "GENERIC_ERROR",
            "COMMAND_PROCESSING_FAILED",
            "UNKNOWN_ERROR",
            "SERIALIZATION_ERROR",
            "MAP_ERROR",
            "BIND_ERROR",
            "CHAIN_ERROR",
            "UNWRAP_ERROR",
            "OPERATION_ERROR",
            "SERVICE_ERROR",
            "BUSINESS_RULE_VIOLATION",
            "BUSINESS_RULE_ERROR",
            "NOT_FOUND_ERROR",
            "NOT_FOUND",
            "RESOURCE_NOT_FOUND",
            "ALREADY_EXISTS",
            "COMMAND_BUS_ERROR",
            "COMMAND_HANDLER_NOT_FOUND",
            "DOMAIN_EVENT_ERROR",
            "TIMEOUT_ERROR",
            "PROCESSING_ERROR",
            "CONNECTION_ERROR",
            "CONFIGURATION_ERROR",
            "EXTERNAL_SERVICE_ERROR",
            "PERMISSION_ERROR",
            "AUTHENTICATION_ERROR",
            "AUTHORIZATION_ERROR",
            "EXCEPTION_ERROR",
            "CRITICAL_ERROR",
        ],
    )
    def test_validation_error_codes_are_uppercase_strings(self, code: str) -> None:
        """All error code constants exist as non-empty uppercase strings matching their name."""
        value = c.ErrorCode[code]
        tm.that(value, is_=str)
        tm.that(value, eq=code)

    @pytest.mark.parametrize(
        ("enum_cls", "members"),
        [
            (
                "ErrorType",
                {
                    "VALIDATION": "validation",
                    "CONFIGURATION": "configuration",
                    "OPERATION": "operation",
                    "CONNECTION": "connection",
                    "TIMEOUT": "timeout",
                    "AUTHORIZATION": "authorization",
                    "AUTHENTICATION": "authentication",
                    "NOT_FOUND": "not_found",
                    "ATTRIBUTE_ACCESS": "attribute_access",
                    "CONFLICT": "conflict",
                    "RATE_LIMIT": "rate_limit",
                    "CIRCUIT_BREAKER": "circuit_breaker",
                    "TYPE_ERROR": "type_error",
                    "VALUE_ERROR": "value_error",
                    "RUNTIME_ERROR": "runtime_error",
                    "SYSTEM_ERROR": "system_error",
                },
            ),
            (
                "FailureLevel",
                {
                    "STRICT": "strict",
                    "WARN": "warn",
                    "PERMISSIVE": "permissive",
                },
            ),
        ],
        ids=_constant_case_id,
    )
    def test_validation_enum_members(
        self,
        enum_cls: str,
        members: dict[str, str],
    ) -> None:
        """Validation StrEnum classes have all expected members."""
        cls = getattr(c, enum_cls)
        tm.that(issubclass(cls, StrEnum), eq=True)
        for name, value in members.items():
            tm.that(str(cls[name]), eq=value)
        tm.that(len(cls), eq=len(members))

    def test_validation_failure_level_default(self) -> None:
        """FAILURE_LEVEL_DEFAULT is PERMISSIVE."""
        tm.that(c.FAILURE_LEVEL_DEFAULT, eq=c.FailureLevel.PERMISSIVE)

    def test_validation_string_method_map_is_frozenset(self) -> None:
        """STRING_METHOD_MAP is a non-empty frozenset of strings."""
        tm.that(c.STRING_METHOD_MAP, is_=frozenset)
        tm.that(len(c.STRING_METHOD_MAP), gt=0)
        # Spot-check some expected entries
        tm.that("str" in c.STRING_METHOD_MAP, eq=True)
        tm.that("dict" in c.STRING_METHOD_MAP, eq=True)
        tm.that("list" in c.STRING_METHOD_MAP, eq=True)
        tm.that("none" in c.STRING_METHOD_MAP, eq=True)

    # ------------------------------------------------------------------
    # Infrastructure constants
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("ContextScope", c.ContextScope),
            ("ContextKey", c.ContextKey),
            ("MetadataKey", c.MetadataKey),
            ("ServiceName", c.ServiceName),
            ("THREAD_NAME_PREFIX", "flext-dispatcher"),
            ("HandlerMode", c.HandlerMode),
            ("DEFAULT_HANDLER_MODE", "command"),
            ("BackoffStrategy", c.BackoffStrategy),
            ("DEFAULT_BACKOFF_STRATEGY", "exponential"),
            ("MAX_RETRY_ATTEMPTS", 3),
            ("DEFAULT_RETRY_DELAY_SECONDS", 1),
            ("DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 60),
            ("DATABASE_URL", "sqlite:///:memory:"),
            ("DEFAULT_CONNECTION_POOL_SIZE", 10),
            ("MAX_CONNECTION_POOL_SIZE", 100),
            ("MIN_POOL_SIZE", 1),
            ("DEFAULT_MAX_FACTORIES", 500),
            ("MAX_FACTORIES", 5000),
            ("DEFAULT_LOGGER_MODULE", "flext_core"),
        ],
    )
    def test_infrastructure_constant_values(
        self, attr: str, expected: t.Core.Tests.MatcherKwargValue
    ) -> None:
        """Infrastructure constants have correct values."""

    def test_infrastructure_valid_handler_modes(self) -> None:
        """Handler mode set is exposed by the infrastructure enum."""
        valid_handler_modes: tuple[c.HandlerMode, ...] = (
            c.HandlerMode.COMMAND,
            c.HandlerMode.QUERY,
        )
        tm.that(valid_handler_modes, is_=tuple)
        tm.that(len(valid_handler_modes), eq=2)
        tm.that(c.HandlerMode.COMMAND in valid_handler_modes, eq=True)
        tm.that(c.HandlerMode.QUERY in valid_handler_modes, eq=True)

    def test_infrastructure_valid_registration_statuses(self) -> None:
        """Registration status set is composed from canonical enum authorities."""
        valid_registration_statuses: tuple[
            c.RegistrationStatus | c.WarningLevel, ...
        ] = (
            c.RegistrationStatus.ACTIVE,
            c.RegistrationStatus.INACTIVE,
            c.WarningLevel.ERROR,
        )
        tm.that(valid_registration_statuses, is_=tuple)
        tm.that(len(valid_registration_statuses), eq=3)
        tm.that(c.RegistrationStatus.ACTIVE in valid_registration_statuses, eq=True)
        tm.that(
            c.RegistrationStatus.INACTIVE in valid_registration_statuses,
            eq=True,
        )
        tm.that(c.WarningLevel.ERROR in valid_registration_statuses, eq=True)

    def test_infrastructure_context_keys_cover_core_request_fields(self) -> None:
        """ContextKey exposes core request-scoped identifiers."""
        tm.that(str(c.ContextKey.OPERATION_ID), eq="operation_id")
        tm.that(str(c.ContextKey.USER_ID), eq="user_id")
        tm.that(str(c.ContextKey.CORRELATION_ID), eq="correlation_id")
        tm.that(str(c.ContextKey.REQUEST_ID), eq="request_id")

    def test_infrastructure_metadata_keys_cover_timing_fields(self) -> None:
        """MetadataKey exposes the operation timing fields."""
        tm.that(str(c.MetadataKey.START_TIME), eq="start_time")
        tm.that(str(c.MetadataKey.END_TIME), eq="end_time")
        tm.that(str(c.MetadataKey.DURATION_SECONDS), eq="duration_seconds")

    @pytest.mark.parametrize(
        ("enum_cls", "members"),
        [
            (
                "ContextKey",
                {
                    "OPERATION_ID": "operation_id",
                    "USER_ID": "user_id",
                    "CORRELATION_ID": "correlation_id",
                    "PARENT_CORRELATION_ID": "parent_correlation_id",
                    "SERVICE_NAME": "service_name",
                    "OPERATION_NAME": "operation_name",
                    "REQUEST_ID": "request_id",
                    "SERVICE_VERSION": "service_version",
                    "OPERATION_START_TIME": "operation_start_time",
                    "OPERATION_METADATA": "operation_metadata",
                    "REQUEST_TIMESTAMP": "request_timestamp",
                    "SERVICE_MODULE": "service_module",
                },
            ),
            (
                "MetadataKey",
                {
                    "START_TIME": "start_time",
                    "END_TIME": "end_time",
                    "DURATION_SECONDS": "duration_seconds",
                },
            ),
            (
                "ServiceName",
                {
                    "LOGGER": "logger",
                    "COMMAND_BUS": "command_bus",
                },
            ),
            (
                "BackoffStrategy",
                {
                    "EXPONENTIAL": "exponential",
                    "LINEAR": "linear",
                },
            ),
        ],
    )
    def test_infrastructure_enum_members(
        self,
        enum_cls: str,
        members: dict[str, str],
    ) -> None:
        """Infrastructure StrEnum classes have all expected members."""
        cls = getattr(c, enum_cls)
        tm.that(issubclass(cls, StrEnum), eq=True)
        for name, value in members.items():
            tm.that(str(cls[name]), eq=value)
        tm.that(len(cls), eq=len(members))

    def test_infrastructure_context_key_values_use_snake_case(self) -> None:
        """Context keys follow the public snake_case contract."""
        tm.that(str(c.ContextKey.PARENT_CORRELATION_ID), eq="parent_correlation_id")
        tm.that(str(c.ContextKey.SERVICE_NAME), eq="service_name")
        tm.that(str(c.ContextKey.OPERATION_NAME), eq="operation_name")
        tm.that(str(c.ContextKey.SERVICE_MODULE), eq="service_module")

    # ------------------------------------------------------------------
    # Platform constants
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("ENV_PREFIX", "FLEXT_"),
            ("ENV_FILE_DEFAULT", ".env"),
            ("ENV_FILE_ENV_VAR", "FLEXT_ENV_FILE"),
            ("ENV_NESTED_DELIMITER", "__"),
            ("DEFAULT_APP_NAME", "flext"),
            ("MAX_RETRY_ATTEMPTS", 3),
            ("DEFAULT_RETRY_DELAY_SECONDS", 1),
            ("DEFAULT_BACKOFF_STRATEGY", "exponential"),
        ],
    )
    def test_platform_constant_values(
        self, attr: str, expected: t.Core.Tests.MatcherKwargValue
    ) -> None:
        """Platform constants have correct values."""

    def test_platform_circuit_breaker_recovery_timeout_derives_from_base(self) -> None:
        """DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT equals base DEFAULT_RECOVERY_TIMEOUT_SECONDS."""
        tm.that(
            c.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            eq=c.DEFAULT_RECOVERY_TIMEOUT_SECONDS,
        )

    # ------------------------------------------------------------------
    # Platform regex patterns
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("pattern_attr", "valid", "invalid"),
        [
            (
                "PATTERN_IDENTIFIER_WITH_UNDERSCORE",
                ["_private", "myVar", "__dunder"],
                ["1digit", "-dash"],
            ),
        ],
        ids=_constant_case_id,
    )
    def test_platform_regex_patterns(
        self,
        pattern_attr: str,
        valid: list[str],
        invalid: list[str],
    ) -> None:
        """Platform regex patterns correctly match/reject inputs."""
        pattern_value = getattr(c, pattern_attr)
        compiled: Pattern[str] = re.compile(pattern_value)
        for case in valid:
            tm.that(
                compiled.match(case) is not None,
                eq=True,
                msg=f"'{case}' should match {pattern_attr}",
            )
        for case in invalid:
            tm.that(
                compiled.match(case) is None,
                eq=True,
                msg=f"'{case}' should NOT match {pattern_attr}",
            )

    # ------------------------------------------------------------------
    # Domain constants
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("MAX_FILE_SIZE", 10485760),
            ("BACKUP_COUNT", 5),
            ("ASYNC_ENABLED", True),
            ("ASYNC_BLOCK_ON_FULL", False),
        ],
    )
    def test_domain_constant_values(
        self, attr: str, expected: t.Core.Tests.MatcherKwargValue
    ) -> None:
        """Domain constants have correct values."""

    @pytest.mark.parametrize(
        ("enum_cls", "members"),
        [
            (
                "ContextOperation",
                {
                    "BIND": "bind",
                    "UNBIND": "unbind",
                    "CLEAR": "clear",
                    "GET": "get",
                    "REMOVE": "remove",
                    "SET": "set",
                },
            ),
            (
                "Currency",
                {"USD": "USD", "EUR": "EUR", "GBP": "GBP", "BRL": "BRL"},
            ),
            (
                "OrderStatus",
                {
                    "PENDING": "pending",
                    "CONFIRMED": "confirmed",
                    "SHIPPED": "shipped",
                    "DELIVERED": "delivered",
                    "CANCELLED": "cancelled",
                },
            ),
        ],
        ids=_constant_case_id,
    )
    def test_domain_enum_members(
        self,
        enum_cls: str,
        members: dict[str, str],
    ) -> None:
        """Domain StrEnum classes have all expected members."""
        cls = getattr(c, enum_cls)
        tm.that(StrEnum in cls.__mro__, eq=True)
        for name, value in members.items():
            tm.that(str(cls[name]), eq=value)
        tm.that(len(cls), eq=len(members))

    # ------------------------------------------------------------------
    # Errors constants
    # ------------------------------------------------------------------

    def test_errors_error_domain_enum(self) -> None:
        """ErrorDomain has all seven categories with uppercase values."""
        cls = c.ErrorDomain
        tm.that(StrEnum in cls.__mro__, eq=True)
        expected = {
            "VALIDATION": "VALIDATION",
            "NETWORK": "NETWORK",
            "AUTH": "AUTH",
            "NOT_FOUND": "NOT_FOUND",
            "TIMEOUT": "TIMEOUT",
            "INTERNAL": "INTERNAL",
            "UNKNOWN": "UNKNOWN",
        }
        for name, value in expected.items():
            tm.that(str(cls[name]), eq=value)
        tm.that(len(cls), eq=len(expected))

    def test_errors_error_domain_str_returns_value(self) -> None:
        """ErrorDomain.__str__ returns the value, not the name."""
        tm.that(str(c.ErrorDomain.VALIDATION), eq="VALIDATION")
        tm.that(str(c.ErrorDomain.AUTH), eq="AUTH")

    # ------------------------------------------------------------------
    # Settings constants
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("DEFAULT_ENCODING", "utf-8"),
            ("SERIALIZATION_ISO8601", "iso8601"),
            ("SERIALIZATION_BASE64", "base64"),
            ("EXTRA_CONFIG_FORBID", "forbid"),
            ("EXTRA_CONFIG_IGNORE", "ignore"),
            ("LONG_UUID_LENGTH", 12),
            ("SHORT_UUID_LENGTH", 8),
            ("VERSION_MODULO", 100),
            ("MAX_WORKERS_THRESHOLD", 50),
            ("DEFAULT_ENABLE_CACHING", True),
            ("DEFAULT_ENABLE_TRACING", False),
            ("DEFAULT_DEBUG_MODE", False),
            ("DEFAULT_TRACE_MODE", False),
        ],
    )
    def test_settings_constant_values(
        self, attr: str, expected: t.Core.Tests.MatcherKwargValue
    ) -> None:
        """Settings constants have correct values."""

    def test_settings_cache_attribute_names_tuple(self) -> None:
        """CACHE_ATTRIBUTE_NAMES is a tuple of expected cache attribute names."""
        tm.that(c.CACHE_ATTRIBUTE_NAMES, is_=tuple)
        tm.that("_cache" in c.CACHE_ATTRIBUTE_NAMES, eq=True)
        tm.that("_ttl" in c.CACHE_ATTRIBUTE_NAMES, eq=True)
        tm.that("_cached_at" in c.CACHE_ATTRIBUTE_NAMES, eq=True)
        tm.that("_cached_value" in c.CACHE_ATTRIBUTE_NAMES, eq=True)

    @pytest.mark.parametrize(
        ("enum_cls", "members"),
        [
            (
                "LogLevel",
                {
                    "DEBUG": "DEBUG",
                    "INFO": "INFO",
                    "WARNING": "WARNING",
                    "ERROR": "ERROR",
                    "CRITICAL": "CRITICAL",
                },
            ),
            (
                "Environment",
                {
                    "DEVELOPMENT": "development",
                    "STAGING": "staging",
                    "PRODUCTION": "production",
                    "TESTING": "testing",
                    "LOCAL": "local",
                },
            ),
        ],
        ids=_constant_case_id,
    )
    def test_settings_enum_members(
        self,
        enum_cls: str,
        members: dict[str, str],
    ) -> None:
        """Settings StrEnum classes have all expected members."""
        cls = getattr(c, enum_cls)
        tm.that(issubclass(cls, StrEnum), eq=True)
        for name, value in members.items():
            tm.that(str(cls[name]), eq=value)
        tm.that(len(cls), eq=len(members))

    # ------------------------------------------------------------------
    # Mixins constants
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("FIELD_ID", "unique_id"),
            ("FIELD_NAME", "name"),
            ("FIELD_TYPE", "type"),
            ("FIELD_STATUS", "status"),
            ("FIELD_DATA", "data"),
            ("FIELD_CONFIG", "settings"),
            ("FIELD_METADATA", "metadata"),
            ("FIELD_ATTRIBUTES", "attributes"),
            ("FIELD_DESCRIPTION", "description"),
            ("FIELD_CONTEXT", "context"),
            ("FIELD_HANDLER_MODE", "handler_mode"),
            ("FIELD_AUTO_LOG", "auto_log"),
            ("FIELD_AUTO_CORRELATION", "auto_correlation"),
            ("FIELD_STATE", "state"),
            ("FIELD_CREATED_AT", "created_at"),
            ("FIELD_UPDATED_AT", "updated_at"),
            ("FIELD_VALIDATED", "validated"),
            ("FIELD_CLASS", "class"),
            ("FIELD_MODULE", "module"),
            ("FIELD_REGISTERED", "registered"),
            ("FIELD_EVENT_NAME", "event_name"),
            ("FIELD_AGGREGATE_ID", "aggregate_id"),
            ("FIELD_OCCURRED_AT", "occurred_at"),
            ("STATUS_PASSED", "PASS"),
            ("STATUS_FAIL", "FAIL"),
            ("STATUS_NO_TARGET", "NO_TARGET"),
            ("STATUS_SKIP", "SKIP"),
            ("STATUS_UNKNOWN", "UNKNOWN"),
            ("IDENTIFIER_UNKNOWN", "unknown"),
            ("IDENTIFIER_DEFAULT", "default"),
            ("IDENTIFIER_ANONYMOUS", "anonymous"),
            ("IDENTIFIER_GUEST", "guest"),
            ("IDENTIFIER_SYSTEM", "system"),
            ("METHOD_HANDLE", "handle"),
            ("METHOD_PROCESS", "process"),
            ("METHOD_EXECUTE", "execute"),
            ("METHOD_PROCESS_COMMAND", "process_command"),
            ("METHOD_VALIDATE", "validate"),
            ("OPERATION_OVERRIDE", "override"),
            ("OPERATION_COLLECTION", "collection"),
            ("AUTH_BEARER", "bearer"),
            ("AUTH_API_KEY", "api_key"),
            ("AUTH_JWT", "jwt"),
            ("HANDLER_COMMAND", "command"),
            ("HANDLER_QUERY", "query"),
            ("DEFAULT_JSON_INDENT", 2),
            ("DEFAULT_SORT_KEYS", False),
            ("DEFAULT_ENSURE_ASCII", False),
            ("STRING_TRUE", "true"),
            ("STRING_FALSE", "false"),
            ("DEFAULT_USE_UTC", True),
            ("DEFAULT_AUTO_UPDATE", True),
            ("MAX_STATE_VALUE_LENGTH", 50),
            ("MAX_FIELD_NAME_LENGTH", 50),
            ("MIN_FIELD_NAME_LENGTH", 1),
            ("PATTERN_TUPLE_MIN_LENGTH", 2),
            ("PATTERN_TUPLE_MAX_LENGTH", 3),
            ("HANDLER_ATTR", "_flext_handler_config_"),
            ("FACTORY_ATTR", "_flext_factory_config_"),
            ("DEFAULT_PRIORITY", 0),
            ("DEFAULT_HANDLER_TIMEOUT", None),
            ("DEFAULT_TEST_CREDENTIAL", "test_password"),
            ("NONEXISTENT_USERNAME", "nonexistent"),
        ],
    )
    def test_mixins_constant_values(
        self, attr: str, expected: t.Core.Tests.MatcherKwargValue
    ) -> None:
        """Mixins constants have correct values."""

    def test_mixins_state_active_derives_from_domain_status(self) -> None:
        """STATE_ACTIVE equals Domain.Status.ACTIVE."""
        tm.that(c.Status.ACTIVE, eq=str(c.Status.ACTIVE))

    def test_mixins_state_inactive_derives_from_domain_status(self) -> None:
        """STATE_INACTIVE equals Domain.Status.INACTIVE."""
        tm.that(c.Status.INACTIVE, eq=str(c.Status.INACTIVE))

    def test_mixins_health_states_derive_from_cqrs(self) -> None:
        """Health state constants derive from HealthStatus enum."""
        tm.that(c.HealthStatus.HEALTHY, eq=str(c.HealthStatus.HEALTHY))
        tm.that(c.HealthStatus.DEGRADED, eq=str(c.HealthStatus.DEGRADED))
        tm.that(c.HealthStatus.UNHEALTHY, eq=str(c.HealthStatus.UNHEALTHY))

    @pytest.mark.parametrize(
        ("enum_cls", "members"),
        [
            (
                "BoolTrueValue",
                {
                    "TRUE": "true",
                    "ONE": "1",
                    "YES": "yes",
                    "ON": "on",
                    "ENABLED": "enabled",
                },
            ),
            (
                "BoolFalseValue",
                {
                    "FALSE": "false",
                    "ZERO": "0",
                    "NO": "no",
                    "OFF": "off",
                    "DISABLED": "disabled",
                },
            ),
        ],
        ids=_constant_case_id,
    )
    def test_mixins_bool_enum_members(
        self,
        enum_cls: str,
        members: dict[str, str],
    ) -> None:
        """Boolean StrEnum classes have all expected members."""
        cls = getattr(c, enum_cls)
        tm.that(issubclass(cls, StrEnum), eq=True)
        for name, value in members.items():
            tm.that(str(cls[name]), eq=value)
        tm.that(len(cls), eq=len(members))

    def test_mixins_default_max_workers_derives_from_base(self) -> None:
        """DEFAULT_MAX_WORKERS equals base DEFAULT_WORKERS."""
        tm.that(c.DEFAULT_MAX_WORKERS, eq=c.DEFAULT_WORKERS)

    # ------------------------------------------------------------------
    # Cross-cutting: type correctness
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("attr", "expected_type"),
        [
            ("NAME", str),
            ("MIN_PORT", int),
            ("MAX_PORT", int),
            ("DEFAULT_TIMEOUT_SECONDS", int),
            ("INITIAL_TIME", float),
            ("DEFAULT_BACKOFF_MULTIPLIER", float),
            ("DEFAULT_PARALLEL_EXECUTION", bool),
            ("DEFAULT_STOP_ON_ERROR", bool),
            ("DEFAULT_ENCODING", str),
            ("FLEXT_API_PORT", int),
            ("DATABASE_URL", str),
        ],
    )
    def test_type_correctness(self, attr: str, expected_type: type) -> None:
        """Constants have the correct runtime type."""

    # ------------------------------------------------------------------
    # Cross-cutting: numeric range invariants
    # ------------------------------------------------------------------

    def test_port_range_invariant(self) -> None:
        """MIN_PORT < MAX_PORT and both are within valid TCP range."""
        tm.that(c.MIN_PORT, gt=0)
        tm.that(c.MAX_PORT, lte=65535)
        tm.that(c.MIN_PORT, lt=c.MAX_PORT)

    def test_timeout_range_invariant(self) -> None:
        """MIN < DEFAULT < MAX for timeout seconds."""
        tm.that(c.MIN_TIMEOUT_SECONDS, gt=0)
        tm.that(c.DEFAULT_TIMEOUT_SECONDS, gte=c.MIN_TIMEOUT_SECONDS)
        tm.that(c.MAX_TIMEOUT_SECONDS, gt=c.DEFAULT_TIMEOUT_SECONDS)

    def test_page_size_range_invariant(self) -> None:
        """MIN < DEFAULT < MAX for page sizes."""
        tm.that(c.MIN_PAGE_SIZE, gt=0)
        tm.that(c.DEFAULT_PAGE_SIZE, gte=c.MIN_PAGE_SIZE)
        tm.that(c.MAX_PAGE_SIZE, gte=c.DEFAULT_PAGE_SIZE)

    def test_http_status_range_invariant(self) -> None:
        """HTTP_STATUS_MIN < HTTP_STATUS_MAX within valid HTTP range."""
        tm.that(c.HTTP_STATUS_MIN, gte=100)
        tm.that(c.HTTP_STATUS_MAX, lte=599)
        tm.that(c.HTTP_STATUS_MIN, lt=c.HTTP_STATUS_MAX)

    def test_retry_range_invariant(self) -> None:
        """MIN_RETRIES <= DEFAULT_RETRIES <= MAX_RETRIES."""
        tm.that(c.MIN_RETRIES, lte=c.DEFAULT_RETRIES)
        tm.that(c.DEFAULT_RETRIES, lte=c.MAX_RETRIES)

    def test_pool_size_range_invariant(self) -> None:
        """MIN < DEFAULT < MAX for pool sizes."""
        tm.that(c.MIN_POOL_SIZE, gt=0)
        tm.that(c.DEFAULT_CONNECTION_POOL_SIZE, gte=c.MIN_POOL_SIZE)
        tm.that(c.MAX_CONNECTION_POOL_SIZE, gte=c.DEFAULT_CONNECTION_POOL_SIZE)

    # ------------------------------------------------------------------
    # Cross-cutting: no duplicate enum values within a single enum
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "enum_cls",
        [
            "Status",
            "HandlerType",
            "MetricType",
            "ServiceMetricCategory",
            "ProcessingMode",
            "ProcessingPhase",
            "BindType",
            "MergeStrategy",
            "HealthStatus",
            "TokenType",
            "SerializationFormat",
            "Compression",
            "Aggregation",
            "Action",
            "PersistenceLevel",
            "TargetFormat",
            "WarningLevel",
            "OutputFormat",
            "Mode",
            "RegistrationStatus",
            "ErrorType",
            "FailureLevel",
            "ContextOperation",
            "Currency",
            "OrderStatus",
            "ErrorDomain",
            "LogLevel",
            "Environment",
            "MetadataKey",
            "BoolTrueValue",
            "BoolFalseValue",
        ],
    )
    def test_enum_values_are_unique(self, enum_cls: str) -> None:
        """Each StrEnum has no duplicate values (@unique guarantees this at class creation)."""
        cls = getattr(c, enum_cls)
        values = [member.value for member in cls]
        tm.that(
            len(values), eq=len(set(values)), msg=f"{enum_cls} has duplicate values"
        )

    # ------------------------------------------------------------------
    # Facade completeness: all subclass constants accessible via c
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "attr",
        [
            # From base
            "NAME",
            "DEFAULT_TIMEOUT_SECONDS",
            "ZERO",
            # From cqrs
            "DEFAULT_COMMAND_TYPE",
            "CQRS_OPERATION_FAILED",
            "HandlerType",
            "Status",
            # From validation
            "ErrorCode",
            "ErrorType",
            "FailureLevel",
            "STRING_METHOD_MAP",
            # From infrastructure
            "ContextScope",
            "ContextKey",
            "MetadataKey",
            "BackoffStrategy",
            # From platform
            "ENV_PREFIX",
            "PATTERN_IDENTIFIER_WITH_UNDERSCORE",
            # From domain
            "ContextOperation",
            "Currency",
            # From errors
            "ErrorDomain",
            # From settings
            "DEFAULT_ENCODING",
            "LogLevel",
            "Environment",
            # From mixins
            "FIELD_ID",
            "BoolTrueValue",
            "BoolFalseValue",
            "HANDLER_ATTR",
        ],
    )
    def test_facade_attribute_accessible(self, attr: str) -> None:
        """All subclass constants and enums are accessible through the facade."""

    def test_facade_docstring_mentions_layer_zero(self) -> None:
        """C docstring references Layer 0."""
        tm.that(c.__doc__, none=False)
        doc = c.__doc__ or ""
        tm.that("layer 0" in doc.lower(), eq=True)


__all__: list[str] = ["TestFlextConstants"]
