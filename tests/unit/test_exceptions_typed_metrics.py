"""Typed exception and metrics behavior tests.

Behavioral contract tests over the public ``FlextExceptions`` (``e``) surface:
typed exception construction, structured fields, error-code / routing-domain
derivation, exception raising, correlation metadata, and the public metrics
snapshot API. No private attributes, internal collaborators, or patched
internals are exercised.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import c, e, m, t, tm

if TYPE_CHECKING:
    from collections.abc import Callable


class TestsFlextCoreExceptionsTypedMetrics:
    """Tests for ``FlextCoreExceptionsTypedMetrics``."""

    @staticmethod
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
                lambda: e.FlextConnectionError(
                    "Test message",
                    host=c.LOCALHOST,
                    port=8080,
                ),
                e.FlextConnectionError,
            ),
            (
                lambda: e.FlextTimeoutError("Test message", timeout_seconds=30.0),
                e.FlextTimeoutError,
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
                    "Test message",
                    user_id="u-1",
                    permission="read",
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
    def test_typed_exception_is_expected_type_and_raisable(
        factory: Callable[[], e.BaseError],
        expected_type: type[e.BaseError],
    ) -> None:
        # Arrange / Act
        """Test typed exception is expected type and raisable."""
        error = factory()

        # Assert: public identity + message contract, and it behaves as an exception
        tm.that(isinstance(error, expected_type), truth=True)
        tm.that(error, is_=Exception)
        tm.that(error.message, eq="Test message")
        with pytest.raises(expected_type) as caught:
            raise error
        tm.that(caught.value is error, truth=True)
        tm.that(caught.value.message, eq="Test message")

    @staticmethod
    @pytest.mark.parametrize(
        ("factory", "expected_domain"),
        [
            (lambda: e.ValidationError("m", field="email"), "VALIDATION"),
            (lambda: e.ConfigurationError("m", config_key="timeout"), "INTERNAL"),
            (
                lambda: e.FlextConnectionError("m", host=c.LOCALHOST, port=8080),
                "NETWORK",
            ),
            (lambda: e.FlextTimeoutError("m", timeout_seconds=30.0), "TIMEOUT"),
            (lambda: e.AuthenticationError("m", auth_method="password"), "AUTH"),
            (
                lambda: e.NotFoundError("m", resource_type="User", resource_id="1"),
                "NOT_FOUND",
            ),
        ],
    )
    def test_error_code_maps_to_routing_domain(
        factory: Callable[[], e.BaseError],
        expected_domain: str,
    ) -> None:
        # Arrange / Act
        """Test error code maps to routing domain."""
        error = factory()

        # Assert: routing domain derived from the structured error code
        tm.that(error.error_domain, eq=expected_domain)
        tm.that(error.matches_error_domain(expected_domain) is True, truth=True)
        tm.that(error.matches_error_domain("nonexistent-domain") is False, truth=True)

    @staticmethod
    def test_string_representation_prefixes_error_code() -> None:
        # Arrange
        """Test string representation prefixes error code."""
        error = e.ValidationError("Bad email", field="email")

        # Act / Assert: caller-visible str() carries the structured code
        rendered = str(error)
        tm.that(error.error_code in rendered, truth=True)
        tm.that(error.message in rendered, truth=True)
        tm.that(rendered, eq=f"[{error.error_code}] {error.message}")

    @staticmethod
    @pytest.mark.parametrize(
        ("factory", "attribute", "expected_value"),
        [
            (lambda: e.ValidationError("m", field="email"), "field", "email"),
            (
                lambda: e.ConfigurationError("m", config_key="timeout"),
                "config_key",
                "timeout",
            ),
            (
                lambda: e.FlextConnectionError("m", host=c.LOCALHOST),
                "host",
                c.LOCALHOST,
            ),
            (lambda: e.FlextConnectionError("m", port=8080), "port", 8080),
            (
                lambda: e.FlextTimeoutError("m", timeout_seconds=30.0),
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
                c.ContextScope.OPERATION,
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
        factory: Callable[[], e.BaseError],
        attribute: str,
        expected_value: t.Scalar,
    ) -> None:
        # Arrange / Act
        """Test typed exception exposes structured fields."""
        error = factory()

        # Assert: structured field is readable through the public attribute
        tm.that(getattr(error, attribute), eq=expected_value)

    @staticmethod
    def test_base_error_exposes_correlation_and_metadata() -> None:
        # Arrange / Act
        """Test base error exposes correlation and metadata."""
        err = e.BaseError(
            "boom",
            options=m.ExceptionInitOptions(
                correlation_id="corr-001",
                metadata={"scope": "service"},
            ),
        )

        # Assert: correlation id + metadata attributes reachable publicly
        tm.that(err.correlation_id, eq="corr-001")
        tm.that(err.error_message, eq="boom")
        tm.that(err.metadata.attributes.get("scope"), eq="service")

    @staticmethod
    def test_not_found_error_excludes_internal_context_keys_from_metadata() -> None:
        # Arrange / Act
        """Test not found error excludes internal context keys from metadata."""
        error = e.NotFoundError(
            "Not found",
            resource_type="User",
            resource_id="123",
            options=m.ExceptionInitOptions(
                context={
                    "key1": "value1",
                    c.ContextKey.CORRELATION_ID: "corr-001",
                    "metadata": "skip-me",
                },
            ),
        )

        # Assert: caller-provided context survives; reserved keys are excluded
        attributes = error.metadata.attributes
        tm.that(attributes["key1"], eq="value1")
        tm.that(c.ContextKey.CORRELATION_ID not in attributes, truth=True)
        tm.that("metadata" not in attributes, truth=True)

    @staticmethod
    def test_type_error_normalizes_expected_and_actual_type() -> None:
        # Arrange / Act: mixed str + type inputs
        """Test type error normalizes expected and actual type."""
        error = e.FlextTypeError(
            "Type mismatch",
            expected_type="str",
            actual_type=int,
            options=m.ExceptionInitOptions(context={"source": "api"}),
        )

        # Assert: both resolve to concrete types on the public attributes
        tm.that(error.expected_type is str, truth=True)
        tm.that(error.actual_type is int, truth=True)
        attributes = error.metadata.attributes
        tm.that(attributes["source"], eq="api")
        tm.that(attributes["expected_type"], eq="str")
        tm.that(attributes["actual_type"], eq="int")

    @staticmethod
    def test_metrics_snapshot_counts_recorded_exceptions() -> None:
        # Arrange
        """Test metrics snapshot counts recorded exceptions."""
        e.clear_metrics()

        # Act
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.FlextTimeoutError)
        metrics = e.resolve_metrics_snapshot()

        # Assert: public snapshot contract
        tm.that(metrics, is_=m.ExceptionMetricsSnapshot)
        tm.that(metrics.total_exceptions, eq=3)
        tm.that(metrics.unique_exception_types, eq=2)
        tm.that(metrics.has_exceptions is True, truth=True)
        tm.that(metrics.exception_counts[e.ValidationError.__qualname__], eq=2)
        tm.that(metrics.exception_counts[e.FlextTimeoutError.__qualname__], eq=1)

        # Assert: flat config export mirrors the snapshot
        config_map = metrics.to_config_map()
        tm.that(config_map["total_exceptions"], eq=3)
        tm.that(config_map["unique_exception_types"], eq=2)

        # Cleanup so the shared counter does not leak into other tests
        e.clear_metrics()

    @staticmethod
    def test_clear_metrics_resets_snapshot_and_is_idempotent() -> None:
        # Arrange
        """Test clear metrics resets snapshot and is idempotent."""
        e.clear_metrics()
        e.record_exception(e.ValidationError)

        # Act
        e.clear_metrics()
        first = e.resolve_metrics_snapshot()
        e.clear_metrics()
        second = e.resolve_metrics_snapshot()

        # Assert: cleared state is empty and clearing again is a no-op
        tm.that(first.total_exceptions, eq=0)
        tm.that(first.unique_exception_types, eq=0)
        tm.that(first.has_exceptions is False, truth=True)
        tm.that(second.total_exceptions, eq=0)
