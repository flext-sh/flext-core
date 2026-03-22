"""Tests for e - Exception Type Definitions and Implementations.

Module: flext_core.exceptions
Scope: e - all exception types and factory methods

Tests e functionality including:
- BaseError initialization and configuration
- Exception type hierarchy (ValidationError, ConfigurationError, etc.)
- Exception with metadata, correlation IDs, error codes
- Exception factory methods (create_error, create)
- Exception serialization and string representation
- Exception chaining and context propagation

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from typing import cast, override

import pytest
from flext_tests import t, tm, u

from flext_core import FlextConstants, FlextRuntime, e
from tests import m


class Teste:
    """Comprehensive test suite for e using u."""

    def test_exception_class_hierarchy(self) -> None:
        """Test exception class inheritance hierarchy."""
        tm.that(
            all(
                issubclass(cls, e.BaseError)
                for cls in [
                    e.ValidationError,
                    e.NotFoundError,
                    e.AuthenticationError,
                    e.TimeoutError,
                ]
            ),
            eq=True,
        )
        tm.that(issubclass(e.BaseError, Exception), eq=True)

    def test_timestamp_generation(self) -> None:
        """Test that timestamp is automatically generated."""
        before = time.time()
        error = e.BaseError("Test error")
        after = time.time()
        tm.that(before <= error.timestamp <= after, eq=True)

    def test_metadata_merge_with_kwargs(self) -> None:
        """Test that metadata and kwargs are properly merged."""
        metadata = {"existing": "value"}
        error = e.BaseError("Test error", metadata=metadata, new_field="new_value")
        tm.that(error.metadata.attributes["existing"], eq="value")
        tm.that(error.metadata.attributes["new_field"], eq="new_value")

    def test_exception_repr(self) -> None:
        """Test repr() for all exception types."""
        exceptions_to_test: list[e.BaseError] = [
            e.BaseError("base"),
            e.ValidationError("validation"),
            e.NotFoundError("not_found"),
            e.ConflictError("conflict"),
            e.AuthenticationError("auth"),
            e.AuthorizationError("authz"),
            e.TimeoutError("timeout"),
            e.ConnectionError("connection"),
            e.RateLimitError("rate_limit"),
            e.CircuitBreakerError("circuit"),
            e.ConfigurationError("config"),
            e.OperationError("operation"),
            e.TypeError("type"),
        ]
        for exc in exceptions_to_test:
            repr_str = repr(exc)
            tm.that(repr_str is not None and len(repr_str) > 0, eq=True)

    def test_exception_serialization(self) -> None:
        """Test exception serialization to dict."""
        u.Tests.ExceptionHelpers.create_metadata_object({
            "field": "email",
            "value": "invalid",
        })
        exc = e.ValidationError(
            "Validation failed",
            error_code="INVALID_INPUT",
            field="email",
            value="invalid",
        )
        if hasattr(exc, "to_dict"):
            result = exc.to_dict()
            tm.that(isinstance(result, dict), eq=True)
            tm.that("error_code" in result or "message" in result, eq=True)

    def test_base_error_str_without_code(self) -> None:
        """Test __str__ without error code - tests line 118.

        Note: BaseError has default error_code=UNKNOWN_ERROR, so to test
        line 118 (no code path), we need to set error_code to empty string
        after creation or use a different approach.
        """
        error = e.BaseError("Test message")
        error.error_code = ""
        tm.that(str(error), eq="Test message")

    def test_base_error_str_with_code(self) -> None:
        """Test __str__ with error code - tests line 117."""
        error = e.BaseError("Test message", error_code="TEST_ERROR")
        tm.that(str(error), eq="[TEST_ERROR] Test message")

    def test_normalize_metadata_fallback(self) -> None:
        """Test _normalize_metadata fallback path - tests line 219."""
        result = e.BaseError._normalize_metadata(12345, {})
        tm.that(hasattr(result, "attributes"), eq=True)
        tm.that(len(result.attributes), gt=0)

    def test_normalize_metadata_with_merged_kwargs(self) -> None:
        """Test _normalize_metadata with merged_kwargs - tests lines 211-212."""
        metadata = {"key1": "value1"}
        merged_kwargs = {"key2": "value2"}
        merged_kwargs_cast = cast("dict[str, t.MetadataAttributeValue]", merged_kwargs)
        result = e.BaseError._normalize_metadata(metadata, merged_kwargs_cast)
        tm.that(hasattr(result, "attributes"), eq=True)
        tm.that(result.attributes["key1"], eq="value1")
        tm.that(result.attributes["key2"], eq="value2")

    def test_validation_error_with_context(self) -> None:
        """Test ValidationError with context - tests lines 243-244."""
        context_raw: dict[str, t.NormalizedValue] = {"key1": "value1", "key2": 123}
        context: dict[str, t.MetadataValue] = {
            k: FlextRuntime.normalize_to_metadata(v) for k, v in context_raw.items()
        }
        error = e.ValidationError(
            "Validation failed",
            field="test_field",
            context=context,
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")
        tm.that(error.metadata.attributes, has="key2")

    @pytest.mark.parametrize(
        ("error_class", "msg", "context_key"),
        [
            (e.ConfigurationError, "Config failed", "key1"),
            (e.ConnectionError, "Connection failed", "key1"),
            (e.AuthenticationError, "Auth failed", "key1"),
            (e.AuthorizationError, "Authz failed", "key1"),
        ],
        ids=["config_context", "connection_context", "auth_context", "authz_context"],
    )
    def test_exception_with_context(
        self,
        error_class: type[e.BaseError],
        msg: str,
        context_key: str,
    ) -> None:
        """Test exception classes with context metadata."""
        context = {context_key: "value1"}
        error = error_class(msg, context=context)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has=context_key)

    def test_not_found_error_with_context(self) -> None:
        """Test NotFoundError with context - tests lines 518-547."""
        context_raw: dict[str, t.NormalizedValue] = {
            "key1": "value1",
            "correlation_id": "test-id",
            "metadata": "test",
            "auto_log": True,
            "auto_correlation": True,
        }
        context: dict[str, t.MetadataValue] = {
            k: FlextRuntime.normalize_to_metadata(v) for k, v in context_raw.items()
        }
        error = e.NotFoundError(
            "Not found",
            resource_type="User",
            resource_id="123",
            context=context,
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")
        tm.that(error.metadata.attributes, lacks="correlation_id")

    def test_conflict_error_with_context(self) -> None:
        """Test ConflictError with context - tests line 624."""
        context = {"key1": "value1"}
        error = e.ConflictError(
            "Conflict",
            resource_type="User",
            resource_id="123",
            context=context,
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_conflict_error_build_context(self) -> None:
        """Test ConflictError with context - tests line 624."""
        context = {"key1": "value1"}
        error = e.ConflictError(
            "Conflict",
            resource_type="User",
            resource_id="123",
            context=context,
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_conflict_error_build_context_with_extra_kwargs(self) -> None:
        """Test ConflictError with extra_kwargs - tests line 625."""
        error = e.ConflictError(
            "Conflict",
            custom="value",
            resource_type="User",
            resource_id="123",
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="custom")

    def test_rate_limit_error_with_context(self) -> None:
        """Test RateLimitError with context - tests line 624."""
        context = {"key1": "value1"}
        error = e.RateLimitError(
            "Rate limit",
            limit=100,
            window_seconds=60,
            context=context,
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_circuit_breaker_error_with_context(self) -> None:
        """Test CircuitBreakerError with context - tests line 659."""
        context = {"key1": "value1"}
        error = e.CircuitBreakerError(
            "Circuit open",
            service="test_service",
            context=context,
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_type_error_with_context(self) -> None:
        """Test TypeError with context - tests line 701."""
        context = {"key1": "value1"}
        error = e.TypeError(
            "Type error",
            expected_type=str,
            actual_type=int,
            context=context,
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_operation_error_with_context(self) -> None:
        """Test OperationError with context - tests lines 757-761."""
        context = {"key1": "value1"}
        error = e.OperationError(
            "Operation failed",
            operation="test_op",
            context=context,
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_operation_error_with_context_and_reason(self) -> None:
        """Test OperationError with context and reason - tests lines 870-872, 875, 878."""
        context = {"key1": "value1"}
        error = e.OperationError(
            "Operation failed",
            operation="test_op",
            reason="Test reason",
            context=context,
            custom_key="custom_value",
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")
        tm.that(error.metadata.attributes, has="reason")
        tm.that(error.metadata.attributes, has="custom_key")

    def test_type_error_normalize_type(self) -> None:
        """Test TypeError._normalize_type."""
        type_map: dict[str, type] = {"str": str, "int": int}
        extra_kwargs_raw = {"expected_type": "str"}
        extra_kwargs: dict[str, t.Container] = dict(extra_kwargs_raw)
        result = e.TypeError._normalize_type(
            None,
            type_map,
            extra_kwargs,
            "expected_type",
        )
        tm.that(result is str, eq=True)
        tm.that(extra_kwargs, lacks="expected_type")
        extra_kwargs_type_raw = {"expected_type": "int"}
        extra_kwargs_type: dict[str, t.Container] = dict(extra_kwargs_type_raw)
        result = e.TypeError._normalize_type(
            None,
            type_map,
            extra_kwargs_type,
            "expected_type",
        )
        tm.that(result is int, eq=True)
        tm.that(extra_kwargs_type, lacks="expected_type")
        result = e.TypeError._normalize_type("str", type_map, {}, "expected_type")
        tm.that(result is str, eq=True)
        result = e.TypeError._normalize_type(str, type_map, {}, "expected_type")
        tm.that(result is str, eq=True)
        result = e.TypeError._normalize_type(None, type_map, {}, "expected_type")
        tm.that(result is None, eq=True)
        result = e.TypeError._normalize_type("float", type_map, {}, "expected_type")
        tm.that(result is None, eq=True)

    def test_type_error_build_type_context(self) -> None:
        """Test TypeError._build_type_context."""
        context = {"key1": "value1"}
        type_context = e.TypeError._build_type_context(
            str,
            int,
            context,
            {"custom": "value"},
        )
        tm.that(type_context, has="key1")
        tm.that(type_context, has="expected_type")
        tm.that(type_context["expected_type"], eq="str")
        tm.that(type_context, has="actual_type")
        tm.that(type_context["actual_type"], eq="int")
        tm.that(type_context, has="custom")

    def test_type_error_build_type_context_none_types(self) -> None:
        """Test TypeError._build_type_context with None types."""
        type_context = e.TypeError._build_type_context(None, None, None, {})
        tm.that(type_context, lacks="expected_type")
        tm.that(type_context, lacks="actual_type")

    def test_type_error_build_type_context_with_type_objects(self) -> None:
        """Test TypeError._build_type_context with type objects."""
        type_context = e.TypeError._build_type_context(str, int, None, {})
        tm.that(type_context["expected_type"], eq="str")
        tm.that(type_context["actual_type"], eq="int")

    def test_type_error_build_type_context_with_string_types(self) -> None:
        """Test TypeError._build_type_context with string types."""
        type_context = e.TypeError._build_type_context("str", "int", None, {})
        tm.that(type_context["expected_type"], eq="str")
        tm.that(type_context["actual_type"], eq="int")

    def test_type_error_build_type_context_with_context(self) -> None:
        """Test TypeError._build_type_context with context."""
        context = {"key1": "value1"}
        type_context = e.TypeError._build_type_context(None, None, context, {})
        tm.that(type_context, has="key1")
        tm.that(type_context, lacks="expected_type")
        tm.that(type_context, lacks="actual_type")

    def test_type_error_build_type_context_expected_type_string(self) -> None:
        """Test TypeError._build_type_context with string expected_type."""
        type_context = e.TypeError._build_type_context("custom_type", None, None, {})
        tm.that(type_context["expected_type"], eq="custom_type")

    def test_type_error_build_type_context_actual_type_string(self) -> None:
        """Test TypeError._build_type_context with string actual_type."""
        type_context = e.TypeError._build_type_context(None, "custom_type", None, {})
        tm.that(type_context["actual_type"], eq="custom_type")

    def test_type_error_build_type_context_with_extra_kwargs(self) -> None:
        """Test TypeError._build_type_context with extra_kwargs."""
        type_context = e.TypeError._build_type_context(
            None,
            None,
            None,
            {"custom": "value"},
        )
        tm.that(type_context, lacks="expected_type")
        tm.that(type_context, lacks="actual_type")
        tm.that(type_context["custom"], eq="value")

    def test_create_error_with_invalid_type(self) -> None:
        """Test create_error with invalid error type - tests lines 1031-1032."""
        with pytest.raises(ValueError, match="Unknown error type"):
            e.create("invalid_type", "message")

    def test_create_with_invalid_type(self) -> None:
        """Test create with invalid error type - tests line 1346."""
        error = e.create("message", invalid_key="value")
        tm.that(isinstance(error, e.BaseError), eq=True)
        tm.that(error.message, eq="message")

    def test_create_error_factory_methods(self) -> None:
        """Test create_error factory methods - tests lines 1014-1033."""
        error = e.create("ValidationError", "Test message")
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.message, eq="Test message")
        error = e.create("ConfigurationError", "Config error")
        tm.that(isinstance(error, e.ConfigurationError), eq=True)
        error = e.create("ConnectionError", "Conn error")
        tm.that(isinstance(error, e.ConnectionError), eq=True)
        error = e.create("TimeoutError", "Timeout error")
        tm.that(isinstance(error, e.TimeoutError), eq=True)
        error = e.create("AuthenticationError", "Auth error")
        tm.that(isinstance(error, e.AuthenticationError), eq=True)
        error = e.create("AuthorizationError", "Authz error")
        tm.that(isinstance(error, e.AuthorizationError), eq=True)
        error = e.create("NotFoundError", "Not found error")
        tm.that(isinstance(error, e.NotFoundError), eq=True)
        error = e.create("ConflictError", "Conflict error")
        tm.that(isinstance(error, e.ConflictError), eq=True)
        error = e.create("RateLimitError", "Rate limit error")
        tm.that(isinstance(error, e.RateLimitError), eq=True)
        error = e.create("CircuitBreakerError", "Circuit error")
        tm.that(isinstance(error, e.CircuitBreakerError), eq=True)
        error = e.create("TypeError", "Type error")
        tm.that(isinstance(error, e.TypeError), eq=True)
        error = e.create("OperationError", "Operation error")
        tm.that(isinstance(error, e.OperationError), eq=True)
        error = e.create("AttributeError", "Attribute error")
        tm.that(isinstance(error, e.AttributeAccessError), eq=True)

    def test_create_factory_methods(self) -> None:
        """Test create factory methods - tests lines 1368-1379."""
        error = e.create("Test message", field="test_field")
        tm.that(isinstance(error, e.ValidationError), eq=True)
        error = e.create("Config error", config_key="test_key")
        tm.that(isinstance(error, e.ConfigurationError), eq=True)
        error = e.create("Conn error", host=FlextConstants.LOCALHOST)
        tm.that(isinstance(error, e.ConnectionError), eq=True)
        error = e.create("Timeout error", timeout_seconds=30.0)
        tm.that(isinstance(error, e.TimeoutError), eq=True)
        error = e.create("Auth error", user_id="user1", auth_method="password")
        tm.that(isinstance(error, e.AuthenticationError), eq=True)
        error = e.create("Authz error", user_id="user1", permission="read")
        tm.that(isinstance(error, e.AuthorizationError), eq=True)
        error = e.create("Not found error", resource_type="User", resource_id="123")
        tm.that(isinstance(error, e.NotFoundError), eq=True)
        error = e.create("Operation error", operation="test_op")
        tm.that(isinstance(error, e.OperationError), eq=True)
        error = e.create("Attribute error", attribute_name="test_attr")
        tm.that(isinstance(error, e.AttributeAccessError), eq=True)

    def test_create_with_metadata_dict(self) -> None:
        """Test create with metadata as dict - tests lines 1372-1375."""
        error = e.create(
            "Test message",
            field="test_field",
            metadata={"key1": "value1"},
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_create_with_metadata_metadata_object(self) -> None:
        """Test create with metadata as Metadata t.NormalizedValue - tests lines 1369-1371."""
        metadata_obj = m.Metadata(attributes={"key1": "value1"})
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast(
                "t.MetadataAttributeValue", cast("t.NormalizedValue", metadata_obj)
            ),
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)

    def test_create_with_dict_like_metadata_basic(self) -> None:
        """Test create with dict-like metadata - tests lines 1376-1379."""

        class DictLike(Mapping[str, t.NormalizedValue]):
            @override
            def __getitem__(self, key: str) -> str:
                if key == "key1":
                    return "value1"
                raise KeyError(key)

            @override
            def __iter__(self) -> Iterator[str]:
                return iter(["key1"])

            @override
            def __len__(self) -> int:
                return 1

        dict_like = DictLike()
        dict_like_converted: dict[str, t.NormalizedValue] = {
            k: FlextRuntime.normalize_to_metadata(
                str(v)
                if not isinstance(v, (str, int, float, bool, type(None), list, dict))
                else cast("t.NormalizedValue", v),
            )
            for k, v in dict_like.items()
        }
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", dict_like_converted),
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_create_with_correlation_id(self) -> None:
        """Test create with correlation_id - tests lines 1365-1366."""
        error = e.create(
            "Test message",
            field="test_field",
            correlation_id="test-correlation-id",
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.correlation_id, eq="test-correlation-id")

    def test_create_error_by_type_with_error_code(self) -> None:
        """Test _create_error_by_type with error_code - tests lines 1322-1323."""
        error = e._create_error_by_type(
            "validation",
            "Test message",
            error_code="CUSTOM_ERROR",
            context=None,
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.error_code, eq="CUSTOM_ERROR")

    def test_create_error_by_type_with_context(self) -> None:
        """Test _create_error_by_type with context - tests lines 1320-1321."""
        context = {"key1": "value1"}
        error = e._create_error_by_type(
            "validation",
            "Test message",
            error_code=None,
            context=context,
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_create_error_by_type_base_error(self) -> None:
        """Test _create_error_by_type returns BaseError for None type - tests lines 1346-1350."""
        error = e._create_error_by_type(
            None,
            "Test message",
            error_code=None,
            context=None,
        )
        tm.that(isinstance(error, e.BaseError), eq=True)
        tm.that(error.message, eq="Test message")

    def test_create_error_by_type_invalid_type(self) -> None:
        """Test _create_error_by_type with invalid type - tests line 1346."""
        error = e._create_error_by_type(
            "invalid",
            "Test message",
            error_code=None,
            context=None,
        )
        tm.that(isinstance(error, e.BaseError), eq=True)
        tm.that(error.message, eq="Test message")

    def test_attribute_access_error_with_extra_kwargs(self) -> None:
        """Test AttributeAccessError with extra_kwargs - tests line 918."""
        error = e.AttributeAccessError(
            "Attribute error",
            attribute_name="test_attr",
            custom_key="custom_value",
        )
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="custom_key")

    def test_prepare_kwargs(self) -> None:
        """Test prepare_exception_kwargs - tests lines 945-970."""
        specific_params_raw = {"field": "test_field"}
        specific_params: dict[str, t.MetadataValue] = {
            k: cast("t.MetadataAttributeValue", v)
            for k, v in specific_params_raw.items()
        }
        kwargs_raw = {
            "correlation_id": "test-id",
            "metadata": m.Metadata(attributes={"key": "value"}),
            "auto_log": True,
            "auto_correlation": True,
            "config": "test_config",
            "field": "override_field",
            "custom": "value",
        }
        kwargs: dict[str, t.MetadataValue] = {
            k: cast("t.MetadataAttributeValue", v) for k, v in kwargs_raw.items()
        }
        result = e.prepare_exception_kwargs(kwargs, specific_params)
        corr_id, metadata, auto_log, auto_corr, _config, extra = result
        tm.that(corr_id, eq="test-id")
        tm.that(metadata, none=False)
        tm.that(auto_log, eq=True)
        tm.that(auto_corr, eq=True)
        tm.that(extra, has="field")
        tm.that(extra["field"], eq="test_field")
        tm.that(extra, has="custom")
        tm.that(extra, lacks="correlation_id")
        tm.that(extra, lacks="metadata")
        tm.that(extra, lacks="auto_log")
        tm.that(extra, lacks="auto_correlation")
        tm.that(extra, lacks="config")

    def test_prepare_kwargs_with_empty_specific_params(self) -> None:
        """Test prepare_exception_kwargs with empty specific_params - tests line 945."""
        kwargs_raw = {"field": "test_field"}
        kwargs: dict[str, t.MetadataValue] = {
            k: cast("t.MetadataAttributeValue", v) for k, v in kwargs_raw.items()
        }
        result = e.prepare_exception_kwargs(kwargs, {})
        _corr_id, _metadata, _auto_log, _auto_corr, _config, extra = result
        tm.that(extra, has="field")
        tm.that(extra["field"], eq="test_field")

    def test_prepare_kwargs_setdefault_behavior(self) -> None:
        """Test prepare_exception_kwargs setdefault behavior - tests line 948."""
        specific_params_raw = {"field": "test_field"}
        specific_params: dict[str, t.MetadataValue] = {
            k: cast("t.MetadataAttributeValue", v)
            for k, v in specific_params_raw.items()
        }
        kwargs: dict[str, t.MetadataValue] = {}
        result = e.prepare_exception_kwargs(kwargs, specific_params)
        _corr_id, _metadata, _auto_log, _auto_corr, _config, extra = result
        tm.that(extra, has="field")
        tm.that(extra["field"], eq="test_field")

    def test_prepare_kwargs_with_specific_params_none(self) -> None:
        """Test prepare_exception_kwargs with None in specific_params - tests lines 947-948."""
        specific_params_raw: dict[str, None] = {"field": None}
        specific_params: dict[str, t.MetadataValue | None] = {
            k: cast("t.MetadataAttributeValue", v)
            for k, v in specific_params_raw.items()
        }
        kwargs_raw = {"field": "test_field"}
        kwargs: dict[str, t.MetadataValue] = {
            k: cast("t.MetadataAttributeValue", v) for k, v in kwargs_raw.items()
        }
        result = e.prepare_exception_kwargs(kwargs, specific_params)
        _corr_id, _metadata, _auto_log, _auto_corr, _config, extra = result
        tm.that(extra["field"], eq="test_field")

    def test_prepare_exception_kwargs_with_non_string_correlation_id(self) -> None:
        """Test prepare_exception_kwargs with non-string correlation_id - tests lines 961-966."""
        kwargs = {"correlation_id": 123}
        kwargs_cast = cast("dict[str, t.MetadataAttributeValue]", kwargs)
        result = e.prepare_exception_kwargs(kwargs_cast, None)
        corr_id, _metadata, _auto_log, _auto_corr, _config, _extra = result
        tm.that(corr_id is None, eq=True)

    def test_prepare_exception_kwargs_return_tuple(self) -> None:
        """Test prepare_exception_kwargs returns correct tuple - tests lines 967-970."""
        kwargs = {"field": "test"}
        kwargs_cast = cast("dict[str, t.MetadataAttributeValue]", kwargs)
        result = e.prepare_exception_kwargs(kwargs_cast, None)
        tm.that(len(result), eq=6)
        corr_id, metadata, auto_log, auto_corr, _metadata_val, extra = result
        tm.that(corr_id is None or isinstance(corr_id, str), eq=True)
        tm.that(metadata is None or isinstance(metadata, (m.Metadata, dict)), eq=True)
        tm.that(isinstance(auto_log, bool), eq=True)
        tm.that(isinstance(auto_corr, bool), eq=True)
        tm.that(isinstance(extra, dict), eq=True)

    def test_create_error_with_context(self) -> None:
        """Test create_error with context parameter - tests lines 1086-1087."""
        error = e.create("ValidationError", "Test message")
        tm.that(isinstance(error, e.ValidationError), eq=True)

    def test_create_with_dict_like_metadata_normalization(self) -> None:
        """Test create normalizes dict-like metadata values - tests lines 1376-1379."""

        class DictLike(Mapping[str, t.NormalizedValue]):
            _obj: t.NormalizedValue

            @override
            def __init__(self) -> None:
                self._obj = str(id(self))

            @override
            def __getitem__(self, key: str) -> t.NormalizedValue:
                if key == "key1":
                    return "value1"
                if key == "key2":
                    return self._obj
                raise KeyError(key)

            @override
            def __iter__(self) -> Iterator[str]:
                return iter(["key1", "key2"])

            @override
            def __len__(self) -> int:
                return 2

        dict_like = DictLike()
        dict_like_converted: dict[str, t.NormalizedValue] = {
            k: FlextRuntime.normalize_to_metadata(
                str(v)
                if not isinstance(v, (str, int, float, bool, type(None), list, dict))
                else cast("t.NormalizedValue", v),
            )
            for k, v in dict_like.items()
        }
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", dict_like_converted),
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")
        tm.that(error.metadata.attributes, has="key2")

    def test_create_with_dict_like_metadata_items_iteration(self) -> None:
        """Test create iterates dict-like metadata items - tests lines 1378-1379."""

        class DictLike(Mapping[str, t.NormalizedValue]):
            @override
            def __getitem__(self, key: str) -> t.NormalizedValue:
                mapping: dict[str, t.NormalizedValue] = {"key1": "value1", "key2": 123}
                if key in mapping:
                    return mapping[key]
                raise KeyError(key)

            @override
            def __iter__(self) -> Iterator[str]:
                return iter(["key1", "key2"])

            @override
            def __len__(self) -> int:
                return 2

        dict_like = DictLike()
        dict_like_converted: dict[str, t.NormalizedValue] = {
            k: FlextRuntime.normalize_to_metadata(
                str(v)
                if not isinstance(v, (str, int, float, bool, type(None), list, dict))
                else cast("t.NormalizedValue", v),
            )
            for k, v in dict_like.items()
        }
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", dict_like_converted),
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")
        tm.that(error.metadata.attributes, has="key2")

    def test_create_with_dict_like_metadata_normalize_values(self) -> None:
        """Test create normalizes dict-like metadata values - tests line 1379."""

        class DictLike(Mapping[str, t.NormalizedValue]):
            _obj: t.NormalizedValue

            @override
            def __init__(self) -> None:
                self._obj = str(id(self))

            @override
            def __getitem__(self, key: str) -> t.NormalizedValue:
                if key == "key1":
                    return self._obj
                raise KeyError(key)

            @override
            def __iter__(self) -> Iterator[str]:
                return iter(["key1"])

            @override
            def __len__(self) -> int:
                return 1

        dict_like = DictLike()
        dict_like_converted: dict[str, t.NormalizedValue] = {
            k: FlextRuntime.normalize_to_metadata(
                str(v)
                if not isinstance(v, (str, int, float, bool, type(None), list, dict))
                else cast("t.NormalizedValue", v),
            )
            for k, v in dict_like.items()
        }
        error = e.create(
            "Test message",
            field="test_field",
            metadata=cast("t.MetadataAttributeValue", dict_like_converted),
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="key1")

    def test_create_error_instance(self) -> None:
        """Test create_error instance method (__call__) - tests lines 1453-1456."""
        error_factory = e()
        error = error_factory(
            "Test message",
            error_code="TEST_ERROR",
            field="test_field",
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.error_code, eq="TEST_ERROR")
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="field")

    def test_create_error_instance_normalizes_kwargs(self) -> None:
        """Test create_error instance method normalizes kwargs - tests lines 1453-1456."""
        error_factory = e()
        error = error_factory.create(
            "Test message",
            error_code="TEST_ERROR",
            field="test_field",
            value=123,
            custom_obj=cast("t.MetadataAttributeValue", str(t.NormalizedValue())),
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="field")
        tm.that(error.metadata.attributes, has="value")
        tm.that(error.metadata.attributes, has="custom_obj")

    def test_create_error_instance_normalizes_all_kwargs(self) -> None:
        """Test create_error instance method normalizes all kwargs - tests lines 1454-1455."""
        error_factory = e()
        error = error_factory.create(
            "Test message",
            field="test_field",
            value1=123,
            value2="string",
            value3=True,
            value4="none",
        )
        tm.that(isinstance(error, e.ValidationError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="field")
        tm.that(error.metadata.attributes, has="value1")
        tm.that(error.metadata.attributes, has="value2")
        tm.that(error.metadata.attributes, has="value3")
        tm.that(error.metadata.attributes, has="value4")

    def test_create_error_instance_normalize_loop(self) -> None:
        """Test create_error instance method normalization loop - tests lines 1454-1455."""
        error_factory = e()
        error = error_factory.create(
            "Test message",
            obj=str(t.NormalizedValue()),
            lst=[1, 2, 3],
            dct={"key": "value"},
        )
        tm.that(isinstance(error, e.BaseError), eq=True)
        tm.that(error.metadata, none=False)
        tm.that(error.metadata.attributes, has="obj")
        tm.that(error.metadata.attributes, has="lst")
        tm.that(error.metadata.attributes, has="dct")

    def test_determine_error_type(self) -> None:
        """Test _determine_error_type - tests lines 1105-1308."""
        error_type = e._determine_error_type({"field": "test"})
        tm.that(error_type, eq="validation")
        error_type = e._determine_error_type({"config_key": "test"})
        tm.that(error_type, eq="configuration")
        error_type = e._determine_error_type({"host": FlextConstants.LOCALHOST})
        tm.that(error_type, eq="connection")
        error_type = e._determine_error_type({"timeout_seconds": 30.0})
        tm.that(error_type, eq="timeout")
        error_type = e._determine_error_type({
            "user_id": "user1",
            "auth_method": "password",
        })
        tm.that(error_type, eq="authentication")
        error_type = e._determine_error_type({"user_id": "user1", "permission": "read"})
        tm.that(error_type, eq="authorization")
        error_type = e._determine_error_type({"resource_id": "123"})
        tm.that(error_type, eq="not_found")
        error_type = e._determine_error_type({"attribute_name": "test"})
        tm.that(error_type, eq="attribute_access")
        error_type = e._determine_error_type({"unknown": "value"})
        tm.that(error_type is None, eq=True)

    def test_determine_error_type_with_conflict(self) -> None:
        """Test _determine_error_type with conflict pattern - tests line 585."""
        error_type = e._determine_error_type({
            "resource_type": "User",
            "resource_id": "123",
        })
        tm.that(error_type, eq="not_found")

    def test_extract_common_kwargs(self) -> None:
        """Test extract_common_kwargs - tests lines 999-1008."""
        kwargs = {
            "correlation_id": "test-id",
            "metadata": m.Metadata(attributes={"key": "value"}),
            "field": "test_field",
        }
        kwargs_cast = cast("dict[str, t.MetadataAttributeValue]", kwargs)
        corr_id, metadata = e.extract_common_kwargs(kwargs_cast)
        tm.that(corr_id, eq="test-id")
        tm.that(isinstance(metadata, m.Metadata), eq=True)
        kwargs_dict_raw = {
            "correlation_id": "test-id",
            "metadata": {"key": "value"},
            "field": "test_field",
        }
        kwargs_dict: dict[str, t.MetadataValue] = {
            k: cast("t.MetadataAttributeValue", v)
            if not isinstance(v, dict)
            else cast(
                "t.MetadataAttributeValue",
                {str(k2): cast("t.MetadataAttributeValue", v2) for k2, v2 in v.items()},
            )
            for k, v in kwargs_dict_raw.items()
        }
        corr_id_dict, metadata_dict = e.extract_common_kwargs(kwargs_dict)
        tm.that(corr_id_dict, eq="test-id")
        tm.that(isinstance(metadata_dict, dict), eq=True)

        class DictLike(Mapping[str, t.NormalizedValue]):
            @override
            def __getitem__(self, key: str) -> str:
                if key == "key":
                    return "value"
                raise KeyError(key)

            @override
            def __iter__(self) -> Iterator[str]:
                return iter(["key"])

            @override
            def __len__(self) -> int:
                return 1

        dict_like_obj = DictLike()
        kwargs_dict_like_raw = {
            "correlation_id": "test-id",
            "metadata": dict_like_obj,
            "field": "test_field",
        }
        kwargs_dict_like: dict[str, t.MetadataValue] = {}
        for k, v in kwargs_dict_like_raw.items():
            if k == "metadata" and isinstance(v, DictLike):
                dict_like_dict: dict[str, t.MetadataValue] = {
                    k2: str(v2)
                    if not isinstance(v2, (str, int, float, bool, type(None)))
                    else cast("t.MetadataAttributeValue", v2)
                    for k2, v2 in v.items()
                }
                kwargs_dict_like[k] = cast("t.MetadataAttributeValue", dict_like_dict)
            else:
                kwargs_dict_like[k] = cast("t.MetadataAttributeValue", v)
        corr_id_dl, metadata_dl = e.extract_common_kwargs(kwargs_dict_like)
        tm.that(corr_id_dl, eq="test-id")
        tm.that(isinstance(metadata_dl, dict), eq=True)
        tm.that(metadata_dl, has="key")

    def test_extract_common_kwargs_metadata_protocol_like(self) -> None:

        class _MetadataLike:
            def __init__(self) -> None:
                self.attributes = {"source": "protocol"}

        kwargs = {
            "correlation_id": "test-id",
            "metadata": cast("t.MetadataAttributeValue", _MetadataLike()),
        }
        kwargs_cast = cast("Mapping[str, t.MetadataValue]", kwargs)
        corr_id, metadata = e.extract_common_kwargs(kwargs_cast)
        tm.that(corr_id, eq="test-id")
        tm.that(metadata, none=False)
        attrs = getattr(metadata, "attributes", {})
        tm.that(attrs.get("source"), eq="protocol")


__all__ = ["Teste"]
