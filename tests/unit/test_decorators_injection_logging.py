"""Behavioral tests for injection and logging decorators.

Covers the public contract of ``d.inject``, ``d.log_operation``,
``d.with_correlation`` and ``d.deprecated`` through observable behavior only:
return values, propagated exceptions, emitted log output, correlation state and
warnings. No private attributes, internal collaborators or implementation
structures are asserted.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextContainer, FlextContext
from tests import c, d, m, tm, u


class TestsFlextCoreDecoratorsInjectionLogging:
    """Tests for ``FlextCoreDecoratorsInjectionLogging``."""

    @staticmethod
    @pytest.fixture
    def container() -> FlextContainer:
        """Provide ``container``.

        Returns:
            The resulting ``FlextContainer``.

        """
        return FlextContainer.shared()

    # --- d.inject --------------------------------------------------------
    @staticmethod
    def test_inject_resolves_registered_service_from_container(
        container: FlextContainer,
    ) -> None:
        """Test inject resolves registered service from container."""
        container.bind("inject_greeter", "HELLO")

        @d.inject(greeter="inject_greeter")
        def greet(name: str, *, greeter: str = "DEFAULT") -> str:
            return f"{greeter}:{name}"

        tm.that(greet("bob"), eq="HELLO:bob")

    @staticmethod
    def test_inject_explicit_kwarg_overrides_container_value(
        container: FlextContainer,
    ) -> None:
        """Test inject explicit kwarg overrides container value."""
        container.bind("inject_override", "CONTAINER")

        @d.inject(greeter="inject_override")
        def greet(*, greeter: str = "DEFAULT") -> str:
            return greeter

        tm.that(greet(greeter="EXPLICIT"), eq="EXPLICIT")

    @staticmethod
    def test_inject_missing_key_preserves_default_argument() -> None:
        """Test inject missing key preserves default argument."""

        @d.inject(dependency="unregistered_key_xyz")
        def use(*, dependency: str = "fallback") -> str:
            return dependency

        tm.that(use(), eq="fallback")

    @staticmethod
    def test_inject_delivers_pydantic_model_instance(
        container: FlextContainer,
    ) -> None:
        """Test inject delivers pydantic model instance."""

        class InjectedService(m.BaseModel):
            value: str

        container.bind("inject_model", InjectedService(value="model-value"))

        @d.inject(service="inject_model")
        def read(*, service: InjectedService | None = None) -> str:
            return service.value if service is not None else "none"

        tm.that(read(), eq="model-value")

    # --- d.log_operation -------------------------------------------------
    @staticmethod
    @pytest.mark.parametrize("payload", ["success", "", "multi word result"])
    def test_log_operation_returns_wrapped_value_unchanged(payload: str) -> None:
        """Test log operation returns wrapped value unchanged."""

        @d.log_operation("payload_op")
        def produce() -> str:
            return payload

        tm.that(produce(), eq=payload)

    @staticmethod
    def test_log_operation_with_perf_tracking_returns_value() -> None:
        """Test log operation with perf tracking returns value."""

        @d.log_operation("timed_op", track_perf=True)
        def compute() -> int:
            return 42

        tm.that(compute(), eq=42)

    @staticmethod
    @pytest.mark.parametrize(
        ("exc_type", "message"),
        [
            (ValueError, "bad value"),
            (RuntimeError, "runtime boom"),
            (KeyError, "missing"),
        ],
    )
    def test_log_operation_propagates_wrapped_exception(
        exc_type: type[Exception],
        message: str,
    ) -> None:
        """Test log operation propagates wrapped exception."""

        @d.log_operation("failing_op")
        def boom() -> None:
            raise exc_type(message)

        with pytest.raises(exc_type, match=message):
            boom()

    @staticmethod
    def test_log_operation_emits_operation_name_on_failure(
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test log operation emits operation name on failure."""

        @d.log_operation("named_failure_op")
        def boom() -> None:
            error_msg = "kaboom"
            raise ValueError(error_msg)

        original_settings, original_options = u.logging_settings(), u.logging_options()
        try:
            selected = u.resolve_logging_settings(log_level=c.LogLevel.ERROR)
            u.configure_structlog(
                settings=m.StructlogOptions(async_logging=False),
                runtime_settings=selected,
            )
            with pytest.raises(ValueError, match="kaboom"):
                boom()
            captured = capsys.readouterr()
            tm.that("named_failure_op" in captured.err, truth=True)
            tm.that(captured.out, eq="")
        finally:
            u.configure_structlog(
                settings=original_options, runtime_settings=original_settings
            )

    # --- d.with_correlation ----------------------------------------------
    @staticmethod
    def test_with_correlation_establishes_correlation_id_during_call() -> None:
        """Test with correlation establishes correlation id during call."""

        @d.with_correlation()
        def inside() -> str:
            return FlextContext.ensure_correlation_id()

        tm.that(inside(), truth=True)

    @staticmethod
    def test_with_correlation_returns_wrapped_value() -> None:
        """Test with correlation returns wrapped value."""

        @d.with_correlation()
        def produce() -> str:
            return "wrapped"

        tm.that(produce(), eq="wrapped")

    # --- d.deprecated ----------------------------------------------------
    @staticmethod
    def test_deprecated_warns_and_returns_value() -> None:
        """Test deprecated warns and returns value."""

        @d.deprecated("use new_api")
        def old_api() -> int:
            return 7

        with pytest.warns(DeprecationWarning, match="deprecated"):
            tm.that(old_api(), eq=7)
