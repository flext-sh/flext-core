"""Structured exception public contract tests.

Behavioral only: every assertion exercises the observable public surface of the
``FlextExceptions`` family and the ``r[T]`` failure contract exposed through the
``flext_tests`` facades. No private attributes, no monkeypatching, no spying on
internal collaborators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator

import pytest

from tests import c, e, p, t, tm
from tests.unit._exceptions_failure_support import FAILURES, FailureFactory
from tests.unit._exceptions_structured_support import STRUCTURED_ERRORS, ErrorFactory


class TestsFlextCoreExceptionsStructuredContracts:
    """Public-contract behavior of structured errors and failure results."""

    @pytest.mark.parametrize(
        ("_name", "factory", "expected_domain", "expected_code", "expected_payload"),
        STRUCTURED_ERRORS,
    )
    @staticmethod
    def test_structured_errors_expose_public_contract(
        _name: str,
        factory: ErrorFactory,
        expected_domain: str,
        expected_code: str,
        expected_payload: t.MappingKV[str, str | int | None],
    ) -> None:
        # Arrange / Act
        """Test structured errors expose public contract."""
        error = factory()

        # Assert: structured protocol surface
        tm.that(error, is_=p.StructuredError)
        tm.that(error.error_domain, eq=expected_domain)
        tm.that(error.error_code, eq=expected_code)
        tm.that(error.error_message, eq=error.message)
        tm.that(error.matches_error_domain(expected_domain), truth=True)
        tm.that(error.matches_error_domain("definitely-not-a-domain"), truth=False)

        # Assert: caller-visible payload flows through public metadata
        for key, value in expected_payload.items():
            tm.that(error.metadata.attributes[key], eq=value)

    @pytest.mark.parametrize(
        ("_name", "factory", "expected_domain", "expected_code", "_expected_payload"),
        STRUCTURED_ERRORS,
    )
    @staticmethod
    def test_structured_errors_are_raisable_and_preserve_contract(
        _name: str,
        factory: ErrorFactory,
        expected_domain: str,
        expected_code: str,
        _expected_payload: t.MappingKV[str, str | int | None],
    ) -> None:
        # Act / Assert: the error is a real exception that callers can catch,
        # and its structured contract survives raise/except unchanged.
        """Test structured errors are raisable and preserve contract."""
        with pytest.raises(e.BaseError) as caught:
            raise factory()

        error = caught.value
        tm.that(error.error_domain, eq=expected_domain)
        tm.that(error.error_code, eq=expected_code)
        tm.that(error.matches_error_domain(expected_domain), truth=True)
        tm.that(error.message in str(error), truth=True)
        tm.that(error.error_message, eq=error.message)

    @staticmethod
    def test_base_error_defaults_to_unknown_domain_protocol_surface() -> None:
        # Arrange / Act
        """Test base error defaults to unknown domain protocol surface."""
        error = e.BaseError("Base failure")

        # Assert
        tm.that(error, is_=p.StructuredError)
        tm.that(error.error_domain, eq=c.ErrorDomain.UNKNOWN.value)
        tm.that(error.error_code, eq=c.ErrorCode.UNKNOWN_ERROR)
        tm.that(error.error_message, eq="Base failure")
        tm.that(error.matches_error_domain(c.ErrorDomain.UNKNOWN.value), truth=True)

    @staticmethod
    def test_specific_errors_are_catchable_as_base_error() -> None:
        # Assert: the family shares a common catchable base contract.
        """Test specific errors are catchable as base error."""
        not_found = e.NotFoundError(
            "User missing",
            resource_type="User",
            resource_id="123",
        )
        with pytest.raises(e.BaseError) as caught:
            raise not_found

        error = caught.value
        tm.that(isinstance(error, e.NotFoundError), truth=True)
        tm.that(error.error_code, eq=c.ErrorCode.NOT_FOUND_ERROR)
        tm.that(error.error_domain, eq=c.ErrorDomain.NOT_FOUND.value)

    @staticmethod
    def test_metadata_attributes_preserve_correlation_id() -> None:
        # Arrange / Act
        """Test metadata attributes preserve correlation id."""
        error = e.OperationError(
            "Insert failed",
            operation="insert_user",
            reason="constraint",
            correlation_id="corr-123",
            metadata={"scope": "users"},
            attempt=2,
        )

        # Assert: caller-supplied context is observable on the public surface
        tm.that(error.message, eq="Insert failed")
        tm.that(error.error_code, eq=c.ErrorCode.OPERATION_ERROR)
        tm.that(error.error_domain, eq=c.ErrorDomain.INTERNAL.value)
        tm.that(error.correlation_id, eq="corr-123")
        tm.that(error.metadata.attributes["scope"], eq="users")
        tm.that(error.metadata.attributes["operation"], eq="insert_user")
        tm.that(error.metadata.attributes["reason"], eq="constraint")
        tm.that(error.metadata.attributes["attempt"], eq=2)

    @pytest.mark.parametrize(
        ("_name", "factory", "expected_fragment", "expected_code", "expected_data"),
        FAILURES,
    )
    @staticmethod
    def test_failure_factories_return_public_result_contract(
        _name: str,
        factory: FailureFactory,
        expected_fragment: str,
        expected_code: str,
        expected_data: t.MappingKV[str, str | int | None],
    ) -> None:
        # Arrange / Act
        """Test failure factories return public result contract."""
        result = factory()
        result_data = result.error_data or {}

        # Assert: r[T] failure contract
        tm.that(result.failure, truth=True)
        tm.that(result.success, truth=False)
        tm.that(result.error, none=False)
        tm.that(expected_fragment in str(result.error), truth=True)
        tm.that(result.error_code, eq=expected_code)
        tm.that(result.error_data, none=False)
        for key, value in expected_data.items():
            tm.that(result_data[key], eq=value)

    @pytest.mark.parametrize(
        ("_name", "factory", "_expected_fragment", "expected_code", "_expected_data"),
        FAILURES,
    )
    @staticmethod
    def test_failure_results_honor_combinator_contract(
        _name: str,
        factory: FailureFactory,
        _expected_fragment: str,
        expected_code: str,
        _expected_data: t.MappingKV[str, str | int | None],
    ) -> None:
        # Arrange
        """Test failure results honor combinator contract."""
        result = factory()

        # Act / Assert: map short-circuits, preserving the failure and its code.
        mapped = result.map(operator.not_)
        tm.that(mapped.failure, truth=True)
        tm.that(mapped.error_code, eq=expected_code)

        # Strict extraction preserves the failure instead of inventing a value.
        if result.exception is not None:
            with pytest.raises(type(result.exception)) as raised:
                result.unwrap()
            tm.that(raised.value is result.exception, truth=True)
        else:
            with pytest.raises(RuntimeError) as logical:
                result.unwrap()
            tm.that(
                str(logical.value),
                eq=c.ERR_RESULT_CANNOT_UNWRAP.format(error=result.error),
            )

        # recover converts a failure into a caller-defined success value.
        recovered = result.recover(lambda _error: False)
        tm.ok(recovered, eq=False)

        # tap_error observes the error without altering the failure channel.
        observed: list[bool] = []
        tapped = result.tap_error(lambda _err: observed.append(True))
        tm.that(tapped.failure, truth=True)
        tm.that(len(observed), eq=1)
