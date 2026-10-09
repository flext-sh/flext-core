"""Base public exception behavior tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from flext_tests import e

from tests import c, m

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreExceptionsBase:
    """Tests for ``FlextCoreExceptionsBase``."""

    @pytest.mark.parametrize(
        "subclass",
        [
            e.ValidationError,
            e.NotFoundError,
            e.AuthenticationError,
            e.FlextTimeoutError,
            e.ConflictError,
            e.ConfigurationError,
        ],
    )
    @staticmethod
    def test_typed_exceptions_are_base_error_subclasses(
        subclass: type[e.BaseError],
    ) -> None:
        """Test typed exceptions are base error subclasses."""
        assert issubclass(subclass, e.BaseError)
        assert issubclass(subclass, Exception)

    @staticmethod
    def test_base_error_sets_timestamp_and_formats_string() -> None:
        """Test base error sets timestamp and formats string."""
        before = time.time()
        error = e.BaseError(
            "Test message",
            options=m.ExceptionInitOptions(error_code="TEST_ERROR"),
        )
        assert before <= error.timestamp <= time.time()
        assert str(error) == "[TEST_ERROR] Test message"
        error.error_code = ""
        assert str(error) == "Test message"

    @staticmethod
    def test_base_error_merges_metadata_context_and_extra_kwargs() -> None:
        """Test base error merges metadata context and extra kwargs."""
        error = e.BaseError(
            "Test error",
            options=m.ExceptionInitOptions(
                context={"scope": "service"},
                metadata={"existing": "value"},
            ),
            new_field="new_value",
        )
        attributes = error.metadata.attributes
        assert attributes["existing"] == "value"
        assert attributes["scope"] == "service"
        assert attributes["new_field"] == "new_value"

    @staticmethod
    def test_typed_exception_raises_and_is_caught_as_base_error() -> None:
        """Test typed exception raises and is caught as base error."""
        error = e.ValidationError("bad", field="email")
        with pytest.raises(e.BaseError) as excinfo:
            raise error
        raised = excinfo.value
        assert isinstance(raised, e.ValidationError)
        assert raised.field == "email"
        assert "bad" in str(raised)

    @staticmethod
    def test_fail_operation_returns_structured_failure() -> None:
        """Test fail operation returns structured failure."""
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

    @staticmethod
    def test_fail_operation_message_carries_the_exception_notes() -> None:
        """Notes attached to the cause (PEP 678) reach the failure message."""
        cause = ValueError("Aborted with 1 warnings in strict mode!")
        cause.add_note("WARNING: Doc file 'a.md' contains a link to 'b.md'")
        result: p.Result[bool] = e.fail_operation("build docs", cause)
        assert result.failure
        assert result.error is not None
        assert "Aborted with 1 warnings in strict mode!" in result.error
        assert "WARNING: Doc file 'a.md' contains a link to 'b.md'" in result.error

    @staticmethod
    def test_failure_result_short_circuits_map_and_rejects_unwrap() -> None:
        """Test failure result short circuits map and rejects unwrap."""
        result: p.Result[bool] = e.fail_operation(
            "register service",
            ValueError("boom"),
        )
        mapped = result.map(lambda _value: False)
        assert mapped.failure
        assert mapped.error == result.error
        with pytest.raises(RuntimeError) as raised:
            mapped.unwrap()
        assert str(raised.value) == c.ERR_RESULT_CANNOT_UNWRAP.format(
            error=result.error,
        )

    @staticmethod
    def test_fail_not_found_returns_structured_failure() -> None:
        """Test fail not found returns structured failure."""
        result: p.Result[bool] = e.fail_not_found("service", "command_bus")
        assert result.failure
        assert result.error is not None
        assert "Service 'command_bus' not found" in result.error
        assert result.error_code == c.ErrorCode.NOT_FOUND_ERROR
        assert result.error_data is not None
        assert result.error_data["resource_type"] == "service"
        assert result.error_data["resource_id"] == "command_bus"

    @staticmethod
    def test_fail_type_mismatch_returns_structured_failure() -> None:
        """Test fail type mismatch returns structured failure."""
        result: p.Result[bool] = e.fail_type_mismatch("Dispatcher", "str")
        assert result.failure
        assert result.error is not None
        assert "Dispatcher" in result.error
        assert result.error_code == c.ErrorCode.TYPE_ERROR
        assert result.error_data is not None
        assert result.error_data["expected_type"] == "Dispatcher"
        assert result.error_data["actual_type"] == "str"

    @staticmethod
    def test_fail_type_mismatch_accepts_service_lookup_params() -> None:
        """Test fail type mismatch accepts service lookup params."""
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

    @pytest.mark.parametrize(
        ("field", "value", "cause"),
        [("name", "", "empty"), ("email", "bad", "invalid")],
    )
    @staticmethod
    def test_fail_validation_returns_structured_failure(
        field: str,
        value: str,
        cause: str,
    ) -> None:
        """Test fail validation returns structured failure."""
        result: p.Result[bool] = e.fail_validation(
            m.ValidationErrorParams(field=field, value=value),
            error=cause,
        )
        assert result.failure
        assert result.error is not None
        assert f"validate {field}" in result.error
        assert result.error_code == c.ErrorCode.VALIDATION_ERROR
        assert result.error_data is not None
        assert result.error_data["field"] == field
        assert result.error_data["value"] == value
        assert result.error_data["cause"] == cause

    @staticmethod
    def test_declarative_error_supports_public_auto_correlation() -> None:
        """Test declarative error supports public auto correlation."""
        error = e.ValidationError(
            "Validation failed",
            field="email",
            options=m.ExceptionInitOptions(auto_correlation=True),
        )
        assert error.correlation_id is not None
        assert error.correlation_id.startswith("exc_")
        assert error.field == "email"
