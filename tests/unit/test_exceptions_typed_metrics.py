"""Typed exception and metrics behavior tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from flext_tests import e

from tests.constants import c
from tests.models import m


class TestsFlextExceptionsTypedMetrics:
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

    def test_base_error_handles_correlation_and_metadata(self) -> None:
        err = e.BaseError(
            "boom",
            correlation_id="corr-001",
            metadata={"scope": "service"},
        )
        assert err.correlation_id == "corr-001"
        assert err.metadata.attributes.get("scope") == "service"

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

    def test_metrics_are_recorded_and_reset_through_public_api(self) -> None:
        e.clear_metrics()
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.TimeoutError)
        metrics = e.resolve_metrics_snapshot()
        assert isinstance(metrics, m.ExceptionMetricsSnapshot)
        assert metrics.total_exceptions == 3
        assert metrics.exception_counts[e.ValidationError.__qualname__] == 2
        assert metrics.exception_counts[e.TimeoutError.__qualname__] == 1
        e.clear_metrics()
        cleared_metrics = e.resolve_metrics_snapshot()
        assert cleared_metrics.total_exceptions == 0
