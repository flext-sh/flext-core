"""Structured exception public contract tests."""

from __future__ import annotations

import pytest

from tests import c, e, p
from tests.unit._exceptions_failure_support import FAILURES, FailureFactory
from tests.unit._exceptions_structured_support import STRUCTURED_ERRORS, ErrorFactory


class TestsFlextCoverageExceptionContracts:
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
        error = factory()

        assert isinstance(error, p.StructuredError)
        assert error.error_domain == expected_domain
        assert error.error_code == expected_code
        assert error.error_message == error.message
        assert error.matches_error_domain(expected_domain)

        for key, value in expected_payload.items():
            assert error.metadata.attributes[key] == value

    def test_base_error_defaults_to_unknown_domain_protocol_surface(self) -> None:
        error = e.BaseError("Base failure")

        assert isinstance(error, p.StructuredError)
        assert error.error_domain == c.ErrorDomain.UNKNOWN.value
        assert error.error_code == c.ErrorCode.UNKNOWN_ERROR
        assert error.error_message == "Base failure"
        assert error.matches_error_domain(c.ErrorDomain.UNKNOWN.value)

    def test_metadata_attributes_preserve_correlation_id(self) -> None:
        error = e.OperationError(
            "Insert failed",
            operation="insert_user",
            reason="constraint",
            correlation_id="corr-123",
            metadata={"scope": "users"},
            attempt=2,
        )

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
        result = factory()
        result_data = result.error_data or {}

        assert result.failure
        assert result.error is not None
        assert expected_fragment in result.error
        assert result.error_code == expected_code
        assert result.error_data is not None
        for key, value in expected_data.items():
            assert result_data[key] == value
