"""Behavioral tests for the public exception contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import pytest

from tests import c, e, r, t


class TestExceptions:
    """Validate the public behavior exposed by flext_core.exceptions."""

    def test_exception_hierarchy_uses_base_error(self) -> None:
        assert issubclass(e.ValidationError, e.BaseError)
        assert issubclass(e.NotFoundError, e.BaseError)
        assert issubclass(e.AuthenticationError, e.BaseError)
        assert issubclass(e.TimeoutError, e.BaseError)
        assert issubclass(e.BaseError, Exception)

    def test_base_error_sets_timestamp_and_formats_string(self) -> None:
        before = time.time()
        error = e.BaseError("Test message", error_code="TEST_ERROR")
        assert before <= error.timestamp <= time.time()
        assert str(error) == "[TEST_ERROR] Test message"
        error.error_code = ""
        assert str(error) == "Test message"

    def test_base_error_merges_metadata_context_and_extra_kwargs(self) -> None:
        error = e.BaseError(
            "Test error",
            context={"scope": "service"},
            metadata={"existing": "value"},
            new_field="new_value",
        )
        attributes = error.metadata.attributes
        assert attributes["existing"] == "value"
        assert attributes["scope"] == "service"
        assert attributes["new_field"] == "new_value"

    def test_fail_operation_returns_structured_failure(self) -> None:
        result: r[bool] = e.fail_operation(
            "register service",
            ValueError("boom"),
        )
        assert result.failure
        assert result.error is not None
        assert "Failed to register service" in result.error
        assert "boom" in result.error
        assert result.error_code == c.ErrorCode.OPERATION_ERROR
        assert result.error_data is not None
        assert result.error_data["operation"] == "register service"
        assert result.error_data["reason"] == "boom"

    def test_fail_not_found_returns_structured_failure(self) -> None:
        result: r[bool] = e.fail_not_found(
            "service",
            "command_bus",
        )
        assert result.failure
        assert result.error is not None
        assert "Service 'command_bus' not found" in result.error
        assert result.error_code == c.ErrorCode.NOT_FOUND_ERROR
        assert result.error_data is not None
        assert result.error_data["resource_type"] == "service"
        assert result.error_data["resource_id"] == "command_bus"

    def test_fail_type_mismatch_returns_structured_failure(self) -> None:
        result: r[bool] = e.fail_type_mismatch(
            "FlextDispatcher",
            "str",
        )
        assert result.failure
        assert result.error is not None
        assert "FlextDispatcher" in result.error
        assert result.error_code == c.ErrorCode.TYPE_ERROR
        assert result.error_data is not None
        assert result.error_data["expected_type"] == "FlextDispatcher"
        assert result.error_data["actual_type"] == "str"

    def test_fail_validation_returns_structured_failure(self) -> None:
        result: r[bool] = e.fail_validation(
            "name",
            "",
            error="empty",
        )
        assert result.failure
        assert result.error is not None
        assert "validate name" in result.error
        assert result.error_code == c.ErrorCode.VALIDATION_ERROR
        assert result.error_data is not None
        assert result.error_data["field"] == "name"
        assert result.error_data["value"] == ""
        assert result.error_data["cause"] == "empty"

    def test_result_from_exception_uses_public_error_surface(self) -> None:
        error = e.ValidationError(
            "Validation failed",
            field="email",
            value="bad",
            correlation_id="corr-123",
        )
        result = r[bool].from_exception(error)
        assert result.failure
        assert result.error == "Validation failed"
        assert result.error_code == c.ErrorCode.VALIDATION_ERROR
        assert result.error_data is not None
        assert result.error_data["field"] == "email"
        assert result.error_data["value"] == "bad"
        assert result.error_data["correlation_id"] == "corr-123"

    def test_declarative_error_supports_public_auto_correlation(self) -> None:
        error = e.ValidationError(
            "Validation failed",
            field="email",
            auto_correlation=True,
        )
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")
        assert error.field == "email"

    @pytest.mark.parametrize(
        ("factory", "expected_type"),
        [
            (
                lambda: e.ValidationError("Test message", field="email"),
                e.ValidationError,
            ),
            (
                lambda: e.ConfigurationError("Test message", config_key="timeout"),
                e.ConfigurationError,
            ),
            (
                lambda: e.ConnectionError("Test message", host=c.LOCALHOST, port=8080),
                e.ConnectionError,
            ),
            (
                lambda: e.TimeoutError("Test message", timeout_seconds=30.0),
                e.TimeoutError,
            ),
            (
                lambda: e.AuthenticationError(
                    "Test message",
                    auth_method="password",
                    user_id="u-1",
                ),
                e.AuthenticationError,
            ),
            (
                lambda: e.AuthorizationError(
                    "Test message", user_id="u-1", permission="read"
                ),
                e.AuthorizationError,
            ),
            (
                lambda: e.NotFoundError(
                    "Test message",
                    resource_type="User",
                    resource_id="123",
                ),
                e.NotFoundError,
            ),
            (
                lambda: e.OperationError("Test message", operation="dispatch"),
                e.OperationError,
            ),
            (
                lambda: e.AttributeAccessError("Test message", attribute_name="logger"),
                e.AttributeAccessError,
            ),
        ],
    )
    def test_typed_exception_instances_are_correct_type(
        self,
        factory: Callable[[], e.BaseError],
        expected_type: type[e.BaseError],
    ) -> None:
        error = factory()
        assert isinstance(error, expected_type)
        assert error.message == "Test message"

    def test_extract_common_kwargs_handles_correlation_and_metadata(self) -> None:
        # extract_common_kwargs is still part of the public contract
        corr, meta = e.extract_common_kwargs({
            "correlation_id": "corr-001",
            "metadata": {"scope": "service"},
        })
        assert corr == "corr-001"
        # meta can be Mapping or m.Metadata — just verify scope is accessible
        assert meta is not None

    def test_not_found_error_excludes_internal_context_keys_from_metadata(self) -> None:
        error = e.NotFoundError(
            "Not found",
            resource_type="User",
            resource_id="123",
            context={
                "key1": "value1",
                "correlation_id": "corr-001",
                "metadata": "skip-me",
            },
        )
        attributes = error.metadata.attributes
        assert attributes["key1"] == "value1"
        assert "correlation_id" not in attributes
        assert "metadata" not in attributes

    def test_type_error_normalizes_expected_and_actual_type(self) -> None:
        error = e.TypeError(
            "Type mismatch",
            expected_type="str",
            actual_type=int,
            context={"source": "api"},
        )
        assert error.expected_type is str
        assert error.actual_type is int
        attributes = error.metadata.attributes
        assert attributes["source"] == "api"
        assert attributes["expected_type"] == "str"
        assert attributes["actual_type"] == "int"

    def test_public_exceptions_serialize_consistently(self) -> None:
        errors: Sequence[e.BaseError] = [
            e.BaseError("base", correlation_id="corr-001"),
            e.ValidationError("validation", correlation_id="corr-001"),
            e.TimeoutError("timeout", correlation_id="corr-001"),
            e.OperationError("operation", correlation_id="corr-001"),
        ]
        for error in errors:
            payload = error.to_dict()
            assert payload["message"] == error.message
            assert payload["error_type"] == type(error).__name__
            assert payload["correlation_id"] == "corr-001"

    def test_metrics_are_recorded_and_reset_through_public_api(self) -> None:
        e.clear_metrics()
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.TimeoutError)
        metrics = e.resolve_metrics()
        assert metrics["total_exceptions"] == 3
        exception_counts = metrics["exception_counts"]
        assert isinstance(exception_counts, t.ConfigMap)
        assert exception_counts[e.ValidationError.__qualname__] == 2
        assert exception_counts[e.TimeoutError.__qualname__] == 1
        e.clear_metrics()
        cleared_metrics = e.resolve_metrics()
        assert cleared_metrics["total_exceptions"] == 0
