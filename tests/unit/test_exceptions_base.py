"""Base public exception behavior tests."""

from __future__ import annotations

import time

import pytest
from flext_tests import e

from tests import c
from tests import m

from tests import p


class TestsFlextCoreExceptionsBase:
    @pytest.mark.parametrize(
        "subclass",
        [
            e.ValidationError,
            e.NotFoundError,
            e.AuthenticationError,
            e.TimeoutError,
            e.ConflictError,
            e.ConfigurationError,
        ],
    )
    def test_typed_exceptions_are_base_error_subclasses(
        self, subclass: type[e.BaseError]
    ) -> None:
        assert issubclass(subclass, e.BaseError)
        assert issubclass(subclass, Exception)

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

    def test_typed_exception_raises_and_is_caught_as_base_error(self) -> None:
        error = e.ValidationError("bad", field="email")
        with pytest.raises(e.BaseError) as excinfo:
            raise error
        raised = excinfo.value
        assert isinstance(raised, e.ValidationError)
        assert raised.field == "email"
        assert "bad" in str(raised)

    def test_fail_operation_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_operation(
            "register service", ValueError("boom")
        )
        assert result.failure
        assert result.error is not None
        assert "Failed to register service" in result.error
        assert "boom" in result.error
        assert result.error_code == c.ErrorCode.OPERATION_ERROR
        assert result.error_data is not None
        assert result.error_data["operation"] == "register service"
        assert result.error_data["reason"] == "boom"

    def test_failure_result_short_circuits_map_and_recovers(self) -> None:
        result: p.Result[bool] = e.fail_operation(
            "register service", ValueError("boom")
        )
        mapped = result.map(lambda _value: False)
        assert mapped.failure
        assert mapped.error == result.error
        assert result.unwrap_or(True) is True

    def test_fail_not_found_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_not_found("service", "command_bus")
        assert result.failure
        assert result.error is not None
        assert "Service 'command_bus' not found" in result.error
        assert result.error_code == c.ErrorCode.NOT_FOUND_ERROR
        assert result.error_data is not None
        assert result.error_data["resource_type"] == "service"
        assert result.error_data["resource_id"] == "command_bus"

    def test_fail_type_mismatch_returns_structured_failure(self) -> None:
        result: p.Result[bool] = e.fail_type_mismatch("Dispatcher", "str")
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
            )
        )

        assert result.failure
        assert result.error is not None
        assert "ldap3.Connection" in result.error
        assert result.error_data is not None
        assert result.error_data["service_name"] == "connection"
        assert result.error_data["expected_type"] == "ldap3.Connection"
        assert result.error_data["actual_type"] == "str"

    @pytest.mark.parametrize(
        ("field", "value", "cause"),
        [("name", "", "empty"), ("email", "bad", "invalid")],
    )
    def test_fail_validation_returns_structured_failure(
        self, field: str, value: str, cause: str
    ) -> None:
        result: p.Result[bool] = e.fail_validation(
            m.ValidationErrorParams(field=field, value=value), error=cause
        )
        assert result.failure
        assert result.error is not None
        assert f"validate {field}" in result.error
        assert result.error_code == c.ErrorCode.VALIDATION_ERROR
        assert result.error_data is not None
        assert result.error_data["field"] == field
        assert result.error_data["value"] == value
        assert result.error_data["cause"] == cause

    def test_declarative_error_supports_public_auto_correlation(self) -> None:
        error = e.ValidationError(
            "Validation failed", field="email", auto_correlation=True
        )
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")
        assert error.field == "email"
