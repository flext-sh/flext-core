"""Structured exception public contract tests.

Behavioral only: every assertion exercises the observable public surface of the
``FlextExceptions`` family and the ``r[T]`` failure contract exposed through the
``flext_tests`` facades. No private attributes, no monkeypatching, no spying on
internal collaborators.
"""

from __future__ import annotations

import pytest

from flext_tests import e
from tests.constants import c
from tests.protocols import p
from tests.unit._exceptions_failure_support import FAILURES, FailureFactory
from tests.unit._exceptions_structured_support import STRUCTURED_ERRORS, ErrorFactory
import operator


class TestsFlextCoreExceptionsStructuredContracts:
    """Public-contract behavior of structured errors and failure results."""

    @pytest.mark.parametrize(
        ("name", "factory", "expected_domain", "expected_code", "expected_payload"),
        STRUCTURED_ERRORS,
    )
    def test_structured_errors_expose_public_contract(
        self,
        name: str,
        factory: ErrorFactory,
        expected_domain: str,
        expected_code: str,
        expected_payload: dict[str, str | int | None],
    ) -> None:
        # Arrange / Act
        error = factory()

        # Assert: structured protocol surface
        assert isinstance(error, p.StructuredError)
        assert error.error_domain == expected_domain
        assert error.error_code == expected_code
        assert error.error_message == error.message
        assert error.matches_error_domain(expected_domain)
        assert not error.matches_error_domain("definitely-not-a-domain")

        # Assert: caller-visible payload flows through public metadata
        for key, value in expected_payload.items():
            assert error.metadata.attributes[key] == value

    @pytest.mark.parametrize(
        ("name", "factory", "expected_domain", "expected_code", "expected_payload"),
        STRUCTURED_ERRORS,
    )
    def test_structured_errors_are_raisable_and_preserve_contract(
        self,
        name: str,
        factory: ErrorFactory,
        expected_domain: str,
        expected_code: str,
        expected_payload: dict[str, str | int | None],
    ) -> None:
        # Act / Assert: the error is a real exception that callers can catch,
        # and its structured contract survives raise/except unchanged.
        with pytest.raises(e.BaseError) as caught:
            raise factory()

        error = caught.value
        assert error.error_domain == expected_domain
        assert error.error_code == expected_code
        assert error.matches_error_domain(expected_domain)
        assert error.message in str(error)
        assert error.error_message == error.message

    def test_base_error_defaults_to_unknown_domain_protocol_surface(self) -> None:
        # Arrange / Act
        error = e.BaseError("Base failure")

        # Assert
        assert isinstance(error, p.StructuredError)
        assert error.error_domain == c.ErrorDomain.UNKNOWN.value
        assert error.error_code == c.ErrorCode.UNKNOWN_ERROR
        assert error.error_message == "Base failure"
        assert error.matches_error_domain(c.ErrorDomain.UNKNOWN.value)

    def test_specific_errors_are_catchable_as_base_error(self) -> None:
        # Assert: the family shares a common catchable base contract.
        not_found = e.NotFoundError(
            "User missing", resource_type="User", resource_id="123"
        )
        with pytest.raises(e.BaseError) as caught:
            raise not_found

        error = caught.value
        assert isinstance(error, e.NotFoundError)
        assert error.error_code == c.ErrorCode.NOT_FOUND_ERROR
        assert error.error_domain == c.ErrorDomain.NOT_FOUND.value

    def test_metadata_attributes_preserve_correlation_id(self) -> None:
        # Arrange / Act
        error = e.OperationError(
            "Insert failed",
            operation="insert_user",
            reason="constraint",
            correlation_id="corr-123",
            metadata={"scope": "users"},
            attempt=2,
        )

        # Assert: caller-supplied context is observable on the public surface
        assert error.message == "Insert failed"
        assert error.error_code == c.ErrorCode.OPERATION_ERROR
        assert error.error_domain == c.ErrorDomain.INTERNAL.value
        assert error.correlation_id == "corr-123"
        assert error.metadata.attributes["scope"] == "users"
        assert error.metadata.attributes["operation"] == "insert_user"
        assert error.metadata.attributes["reason"] == "constraint"
        assert error.metadata.attributes["attempt"] == 2

    @pytest.mark.parametrize(
        ("name", "factory", "expected_fragment", "expected_code", "expected_data"),
        FAILURES,
    )
    def test_failure_factories_return_public_result_contract(
        self,
        name: str,
        factory: FailureFactory,
        expected_fragment: str,
        expected_code: str,
        expected_data: dict[str, str | int | None],
    ) -> None:
        # Arrange / Act
        result = factory()
        result_data = result.error_data or {}

        # Assert: r[T] failure contract
        assert result.failure
        assert not result.success
        assert result.error is not None
        assert expected_fragment in result.error
        assert result.error_code == expected_code
        assert result.error_data is not None
        for key, value in expected_data.items():
            assert result_data[key] == value

    @pytest.mark.parametrize(
        ("name", "factory", "expected_fragment", "expected_code", "expected_data"),
        FAILURES,
    )
    def test_failure_results_honor_combinator_contract(
        self,
        name: str,
        factory: FailureFactory,
        expected_fragment: str,
        expected_code: str,
        expected_data: dict[str, str | int | None],
    ) -> None:
        # Arrange
        result = factory()

        # Act / Assert: map short-circuits, preserving the failure and its code.
        mapped = result.map(operator.not_)
        assert mapped.failure
        assert mapped.error_code == expected_code

        # unwrap_or yields the caller's default on failure.
        assert result.unwrap_or(True) is True

        # recover converts a failure into a caller-defined success value.
        recovered = result.recover(lambda _error: False)
        assert recovered.success
        assert recovered.unwrap_or(True) is False

        # tap_error observes the error without altering the failure channel.
        observed: list[bool] = []
        tapped = result.tap_error(lambda _err: observed.append(True))
        assert tapped.failure
        assert len(observed) == 1
