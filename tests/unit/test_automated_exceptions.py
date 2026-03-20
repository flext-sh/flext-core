"""Tests for exceptions module real API via flext_tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from flext_tests import tm
from hypothesis import given, settings, strategies as st

from flext_core import FlextExceptions

EXCEPTION_CLASSES: tuple[type[FlextExceptions.BaseError], ...] = (
    FlextExceptions.BaseError,
    FlextExceptions.ValidationError,
    FlextExceptions.ConfigurationError,
    FlextExceptions.ConnectionError,
    FlextExceptions.TimeoutError,
    FlextExceptions.AuthenticationError,
    FlextExceptions.AuthorizationError,
    FlextExceptions.NotFoundError,
    FlextExceptions.ConflictError,
    FlextExceptions.RateLimitError,
    FlextExceptions.CircuitBreakerError,
    FlextExceptions.TypeError,
    FlextExceptions.OperationError,
)


class TestAutomatedExceptions:
    """Real functionality tests for FlextExceptions."""

    def test_base_error_init_to_dict_and_string(self) -> None:
        cfg = m.Tests.Config()
        tm.that(hasattr(cfg, "model_dump"), eq=True)
        ctx = {"source": "test", "debug": True}
        err = FlextExceptions.BaseError(
            "boom",
            error_code="E-BASE",
            context=ctx,
            correlation_id="corr-001",
            auto_log=True,
        )
        payload = err.to_dict()
        tm.that(payload, keys=["message", "error_code", "correlation_id", "timestamp"])
        tm.that(payload["message"], eq="boom")
        tm.that(payload["error_code"], eq="E-BASE")
        tm.that(payload["correlation_id"], eq="corr-001")
        tm.that(str(err), has="[E-BASE] boom")
        tm.that(err.auto_log, eq=True)

    @pytest.mark.parametrize("exc_cls", EXCEPTION_CLASSES)
    def test_exception_hierarchy(self, exc_cls: type[Exception]) -> None:
        tm.that(issubclass(exc_cls, FlextExceptions.BaseError), eq=True)
        instance = exc_cls("x")
        tm.that(str(instance), has="x")

    @pytest.mark.parametrize(
        ("factory", "expected_field", "expected_value"),
        [
            (
                lambda: FlextExceptions.ValidationError("bad", field="email"),
                "field",
                "email",
            ),
            (
                lambda: FlextExceptions.ConfigurationError("bad", config_key="db_url"),
                "config_key",
                "db_url",
            ),
            (
                lambda: FlextExceptions.TimeoutError("late", operation="dispatch"),
                "operation",
                "dispatch",
            ),
            (
                lambda: FlextExceptions.NotFoundError("missing", resource_id="42"),
                "resource_id",
                "42",
            ),
        ],
    )
    def test_specialized_exception_fields(
        self,
        factory: Callable[[], FlextExceptions.BaseError],
        expected_field: str,
        expected_value: str,
    ) -> None:
        err = factory()
        payload = err.to_dict()
        tm.that(payload[expected_field], eq=expected_value)
        tm.that(payload["message"], is_=str, len=(1, 200))

    def test_correlation_id_propagation(self) -> None:
        target = FlextExceptions.OperationError(
            "failed op",
            operation="sync",
            correlation_id="cid-123",
        )
        tm.that(target.correlation_id, eq="cid-123")
        tm.that(target.to_dict()["correlation_id"], eq="cid-123")

    def test_auto_correlation_generation(self) -> None:
        err = FlextExceptions.BaseError("auto", auto_correlation=True)
        tm.that(err.correlation_id, is_=str)
        tm.that(err.correlation_id, starts="exc_")

    def test_auto_log_flag_is_respected(self) -> None:
        err = FlextExceptions.BaseError("flag", auto_log=False)
        tm.that(err.auto_log, eq=False)

    @given(message=st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_base_error_to_dict_always_contains_message(self, message: str) -> None:
        payload = FlextExceptions.BaseError(message).to_dict()
        tm.that(payload, keys=["message"])
        tm.that(payload["message"], eq=message)

    @given(
        message=st.text(min_size=1, max_size=80),
        cid=st.text(min_size=1, max_size=24),
    )
    @settings(max_examples=50)
    def test_all_exception_types_with_arbitrary_inputs(
        self,
        message: str,
        cid: str,
    ) -> None:
        errors = [
            FlextExceptions.BaseError(message, correlation_id=cid),
            FlextExceptions.ValidationError(message, correlation_id=cid),
            FlextExceptions.ConfigurationError(message, correlation_id=cid),
            FlextExceptions.ConnectionError(message, correlation_id=cid),
            FlextExceptions.TimeoutError(message, correlation_id=cid),
            FlextExceptions.AuthenticationError(message, correlation_id=cid),
            FlextExceptions.AuthorizationError(message, correlation_id=cid),
            FlextExceptions.NotFoundError(message, correlation_id=cid),
            FlextExceptions.ConflictError(message, correlation_id=cid),
            FlextExceptions.RateLimitError(message, correlation_id=cid),
            FlextExceptions.CircuitBreakerError(message, correlation_id=cid),
            FlextExceptions.TypeError(message, correlation_id=cid),
            FlextExceptions.OperationError(message, correlation_id=cid),
        ]
        for err in errors:
            payload = err.to_dict()
            tm.that(payload["message"], eq=message)
            tm.that(payload["correlation_id"], eq=cid)

    @pytest.mark.performance
    def test_exception_creation_to_dict_benchmark(
        self,
        benchmark: Callable[..., object],
    ) -> None:
        def create_and_dump() -> dict[str, str | float | None]:
            err = FlextExceptions.OperationError(
                "bench",
                operation="publish",
                reason="transient",
                correlation_id="bench-1",
            )
            return {
                "message": err.message,
                "error_code": err.error_code,
                "correlation_id": err.correlation_id,
                "timestamp": err.timestamp,
            }

        _ = benchmark(create_and_dump)
        payload = create_and_dump()
        tm.that(payload, keys=["message", "error_code", "correlation_id"])
