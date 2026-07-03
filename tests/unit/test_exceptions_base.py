"""Base public exception behavior tests."""

from __future__ import annotations

import time

from flext_tests import e

from tests.constants import c
from tests.models import m
from tests.protocols import p


class TestsFlextExceptionsBase:
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
        result: p.Result[bool] = e.fail_operation(
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
        result: p.Result[bool] = e.fail_not_found(
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
        result: p.Result[bool] = e.fail_type_mismatch(
            "Dispatcher",
            "str",
        )
        assert result.failure
        assert result.error is not None
        assert "Dispatcher" in result.error
        assert result.error_code == c.ErrorCode.TYPE_ERROR
        assert result.error_data is not None
        assert result.error_data["expected_type"] == "Dispatcher"
        assert result.error_data["actual_type"] == "str"

    def test_fail_type_mismatch_accepts_service_lookup_params(self) -> None:
        result: p.Result[bool] = e.fail_type_mismatch(
            m.ServiceLookupParams(
                service_name="connection",
                expected_type="ldap3.Connection",
                actual_type="str",
            ),
        )

        assert result.failure
        assert result.error is not None
        assert "ldap3.Connection" in result.error
        assert result.error_data is not None
        assert result.error_data["service_name"] == "connection"
        assert result.error_data["expected_type"] == "ldap3.Connection"
        assert result.error_data["actual_type"] == "str"

    def test_fail_validation_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_validation(
            m.ValidationErrorParams(field="name", value=""),
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

    def test_fail_validation_accepts_validation_error_params(self) -> None:
        result: p.Result[bool] = e.fail_validation(
            m.ValidationErrorParams(field="email", value="bad"),
            error="invalid",
        )

        assert result.failure
        assert result.error is not None
        assert "validate email" in result.error
        assert result.error_data is not None
        assert result.error_data["field"] == "email"
        assert result.error_data["value"] == "bad"
        assert result.error_data["cause"] == "invalid"

    def test_declarative_error_supports_public_auto_correlation(self) -> None:
        error = e.ValidationError(
            "Validation failed",
            field="email",
            auto_correlation=True,
        )
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")
        assert error.field == "email"
