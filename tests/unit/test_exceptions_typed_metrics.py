"""Typed exception and metrics behavior tests.

Behavioral contract tests over the public ``FlextExceptions`` (``e``) surface:
typed exception construction, structured fields, error-code / routing-domain
derivation, exception raising, correlation metadata, and the public metrics
snapshot API. No private attributes, internal collaborators, or patched
internals are exercised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import e

from tests import c
from tests import m

if TYPE_CHECKING:
    from collections.abc import Callable


class TestsFlextCoreExceptionsTypedMetrics:
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
                    "Test message", auth_method="password", user_id="u-1"
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
                    "Test message", resource_type="User", resource_id="123"
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
    def test_typed_exception_is_expected_type_and_raisable(
        self, factory: Callable[[], e.BaseError], expected_type: type[e.BaseError]
    ) -> None:
        # Arrange / Act
        error = factory()

        # Assert: public identity + message contract, and it behaves as an exception
        assert isinstance(error, expected_type)
        assert isinstance(error, Exception)
        assert error.message == "Test message"
        with pytest.raises(expected_type) as caught:
            raise error
        assert caught.value is error
        assert caught.value.message == "Test message"

    @pytest.mark.parametrize(
        ("factory", "expected_domain"),
        [
            (lambda: e.ValidationError("m", field="email"), "VALIDATION"),
            (lambda: e.ConfigurationError("m", config_key="timeout"), "INTERNAL"),
            (lambda: e.ConnectionError("m", host=c.LOCALHOST, port=8080), "NETWORK"),
            (lambda: e.TimeoutError("m", timeout_seconds=30.0), "TIMEOUT"),
            (lambda: e.AuthenticationError("m", auth_method="password"), "AUTH"),
            (
                lambda: e.NotFoundError("m", resource_type="User", resource_id="1"),
                "NOT_FOUND",
            ),
        ],
    )
    def test_error_code_maps_to_routing_domain(
        self, factory: Callable[[], e.BaseError], expected_domain: str
    ) -> None:
        # Arrange / Act
        error = factory()

        # Assert: routing domain derived from the structured error code
        assert error.error_domain == expected_domain
        assert error.matches_error_domain(expected_domain) is True
        assert error.matches_error_domain("nonexistent-domain") is False

    def test_string_representation_prefixes_error_code(self) -> None:
        # Arrange
        error = e.ValidationError("Bad email", field="email")

        # Act / Assert: caller-visible str() carries the structured code
        rendered = str(error)
        assert error.error_code in rendered
        assert error.message in rendered
        assert rendered == f"[{error.error_code}] {error.message}"

    @pytest.mark.parametrize(
        ("factory", "attribute", "expected_value"),
        [
            (lambda: e.ValidationError("m", field="email"), "field", "email"),
            (
                lambda: e.ConfigurationError("m", config_key="timeout"),
                "config_key",
                "timeout",
            ),
            (lambda: e.ConnectionError("m", host=c.LOCALHOST), "host", c.LOCALHOST),
            (lambda: e.ConnectionError("m", port=8080), "port", 8080),
            (
                lambda: e.TimeoutError("m", timeout_seconds=30.0),
                "timeout_seconds",
                30.0,
            ),
            (
                lambda: e.NotFoundError("m", resource_type="User"),
                "resource_type",
                "User",
            ),
            (
                lambda: e.OperationError("m", operation="dispatch"),
                "operation",
                "dispatch",
            ),
            (
                lambda: e.AttributeAccessError("m", attribute_name="logger"),
                "attribute_name",
                "logger",
            ),
        ],
    )
    def test_typed_exception_exposes_structured_fields(
        self, factory: Callable[[], e.BaseError], attribute: str, expected_value: object
    ) -> None:
        # Arrange / Act
        error = factory()

        # Assert: structured field is readable through the public attribute
        assert getattr(error, attribute) == expected_value

    def test_base_error_exposes_correlation_and_metadata(self) -> None:
        # Arrange / Act
        err = e.BaseError(
            "boom", correlation_id="corr-001", metadata={"scope": "service"}
        )

        # Assert: correlation id + metadata attributes reachable publicly
        assert err.correlation_id == "corr-001"
        assert err.error_message == "boom"
        assert err.metadata.attributes.get("scope") == "service"

    def test_not_found_error_excludes_internal_context_keys_from_metadata(self) -> None:
        # Arrange / Act
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

        # Assert: caller-provided context survives; reserved keys are excluded
        attributes = error.metadata.attributes
        assert attributes["key1"] == "value1"
        assert "correlation_id" not in attributes
        assert "metadata" not in attributes

    def test_type_error_normalizes_expected_and_actual_type(self) -> None:
        # Arrange / Act: mixed str + type inputs
        error = e.TypeError(
            "Type mismatch",
            expected_type="str",
            actual_type=int,
            context={"source": "api"},
        )

        # Assert: both resolve to concrete types on the public attributes
        assert error.expected_type is str
        assert error.actual_type is int
        attributes = error.metadata.attributes
        assert attributes["source"] == "api"
        assert attributes["expected_type"] == "str"
        assert attributes["actual_type"] == "int"

    def test_metrics_snapshot_counts_recorded_exceptions(self) -> None:
        # Arrange
        e.clear_metrics()

        # Act
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.TimeoutError)
        metrics = e.resolve_metrics_snapshot()

        # Assert: public snapshot contract
        assert isinstance(metrics, m.ExceptionMetricsSnapshot)
        assert metrics.total_exceptions == 3
        assert metrics.unique_exception_types == 2
        assert metrics.has_exceptions is True
        assert metrics.exception_counts[e.ValidationError.__qualname__] == 2
        assert metrics.exception_counts[e.TimeoutError.__qualname__] == 1

        # Assert: flat config export mirrors the snapshot
        config_map = metrics.to_config_map()
        assert config_map["total_exceptions"] == 3
        assert config_map["unique_exception_types"] == 2

        # Cleanup so the shared counter does not leak into other tests
        e.clear_metrics()

    def test_clear_metrics_resets_snapshot_and_is_idempotent(self) -> None:
        # Arrange
        e.clear_metrics()
        e.record_exception(e.ValidationError)

        # Act
        e.clear_metrics()
        first = e.resolve_metrics_snapshot()
        e.clear_metrics()
        second = e.resolve_metrics_snapshot()

        # Assert: cleared state is empty and clearing again is a no-op
        assert first.total_exceptions == 0
        assert first.unique_exception_types == 0
        assert first.has_exceptions is False
        assert second.total_exceptions == 0
