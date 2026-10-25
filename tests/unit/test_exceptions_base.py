"""Base public exception behavior tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import cast

import pytest

from tests import c, e, m, p, tm


class TestsFlextCoreExceptionsBase:
    """Tests for ``FlextCoreExceptionsBase``."""

    @staticmethod
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
    def test_typed_exceptions_are_base_error_subclasses(
        subclass: type[e.BaseError],
    ) -> None:
        """Test typed exceptions are base error subclasses."""
        tm.that(subclass, where=e.BaseError.__subclasscheck__)
        tm.that(subclass, where=Exception.__subclasscheck__)

    @staticmethod
    def test_base_error_sets_timestamp_and_formats_string() -> None:
        """Test base error sets timestamp and formats string."""
        before = time.time()
        error = e.BaseError(
            "Test message",
            options=m.ExceptionInitOptions(error_code="TEST_ERROR"),
        )
        tm.that(before <= error.timestamp <= time.time(), truth=True)
        tm.that(str(error), eq="[TEST_ERROR] Test message")
        error.error_code = ""
        tm.that(str(error), eq="Test message")

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
        tm.that(attributes["existing"], eq="value")
        tm.that(attributes["scope"], eq="service")
        tm.that(attributes["new_field"], eq="new_value")

    @staticmethod
    def test_typed_exception_raises_and_is_caught_as_base_error() -> None:
        """Test typed exception raises and is caught as base error."""
        error = e.ValidationError("bad", field="email")
        with pytest.raises(e.BaseError) as excinfo:
            raise error
        raised = excinfo.value
        tm.ok(raised is error)
        tm.that(error.field, eq="email")
        tm.that("bad" in str(raised), truth=True)

    @staticmethod
    def test_fail_operation_message_carries_the_exception_notes() -> None:
        """Notes attached to the cause (PEP 678) reach the failure message."""
        cause = ValueError("Aborted with 1 warnings in strict mode!")
        cause.add_note("WARNING: Doc file 'a.md' contains a link to 'b.md'")
        result: p.Result[bool] = e.fail_operation("build docs", cause)
        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that(
            "Aborted with 1 warnings in strict mode!" in str(result.error), truth=True
        )
        tm.that(
            "WARNING: Doc file 'a.md' contains a link to 'b.md'" in str(result.error),
            truth=True,
        )

    @staticmethod
    def test_fail_type_mismatch_returns_structured_failure() -> None:
        """Test fail type mismatch returns structured failure."""
        result: p.Result[bool] = e.fail_type_mismatch("Dispatcher", "str")
        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that("Dispatcher" in str(result.error), truth=True)
        tm.that(result.error_code, eq=c.ErrorCode.TYPE_ERROR)
        tm.that(result.error_data, none=False)
        tm.not_none(result.error_data)
        tm.that(
            cast("dict[str, object]", result.error_data)["expected_type"],
            eq="Dispatcher",
        )
        tm.that(cast("dict[str, object]", result.error_data)["actual_type"], eq="str")

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

        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that("ldap3.Connection" in str(result.error), truth=True)
        tm.that(result.error_data, none=False)
        tm.not_none(result.error_data)
        tm.that(
            cast("dict[str, object]", result.error_data)[c.ContextKey.SERVICE_NAME],
            eq="connection",
        )
        tm.that(
            cast("dict[str, object]", result.error_data)["expected_type"],
            eq="ldap3.Connection",
        )
        tm.that(cast("dict[str, object]", result.error_data)["actual_type"], eq="str")

    @staticmethod
    @pytest.mark.parametrize(
        ("field", "value", "cause"),
        [("name", "", "empty"), ("email", "bad", "invalid")],
    )
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
        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that(f"validate {field}" in str(result.error), truth=True)
        tm.that(result.error_code, eq=c.ErrorCode.VALIDATION_ERROR)
        tm.that(result.error_data, none=False)
        tm.not_none(result.error_data)
        tm.that(cast("dict[str, object]", result.error_data)["field"], eq=field)
        tm.that(cast("dict[str, object]", result.error_data)["value"], eq=value)
        tm.that(cast("dict[str, object]", result.error_data)["cause"], eq=cause)

    @staticmethod
    def test_declarative_error_supports_public_auto_correlation() -> None:
        """Test declarative error supports public auto correlation."""
        error = e.ValidationError(
            "Validation failed",
            field="email",
            options=m.ExceptionInitOptions(auto_correlation=True),
        )
        correlation_id = tm.not_none(error.correlation_id)
        tm.that(correlation_id.startswith("exc_"), truth=True)
        tm.that(error.field, eq="email")
