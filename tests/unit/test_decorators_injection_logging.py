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
from flext_tests import d

from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from tests.models import m
from tests.unit._decorators_support import capture_stdout


class TestsFlextCoreDecoratorsInjectionLogging:
    """Tests for ``FlextCoreDecoratorsInjectionLogging``."""

    @pytest.fixture
    def container(self) -> FlextContainer:
        """Provide ``container``.

        Returns:
            The resulting ``FlextContainer``.
        """
        return FlextContainer.shared()

    # --- d.inject --------------------------------------------------------
    def test_inject_resolves_registered_service_from_container(
        self,
        container: FlextContainer,
    ) -> None:
        """Test inject resolves registered service from container."""
        container.bind("inject_greeter", "HELLO")

        @d.inject(greeter="inject_greeter")
        def greet(name: str, *, greeter: str = "DEFAULT") -> str:
            return f"{greeter}:{name}"

        assert greet("bob") == "HELLO:bob"

    def test_inject_explicit_kwarg_overrides_container_value(
        self,
        container: FlextContainer,
    ) -> None:
        """Test inject explicit kwarg overrides container value."""
        container.bind("inject_override", "CONTAINER")

        @d.inject(greeter="inject_override")
        def greet(*, greeter: str = "DEFAULT") -> str:
            return greeter

        assert greet(greeter="EXPLICIT") == "EXPLICIT"

    def test_inject_missing_key_preserves_default_argument(self) -> None:
        """Test inject missing key preserves default argument."""

        @d.inject(dependency="unregistered_key_xyz")
        def use(*, dependency: str = "fallback") -> str:
            return dependency

        assert use() == "fallback"

    def test_inject_delivers_pydantic_model_instance(
        self,
        container: FlextContainer,
    ) -> None:
        """Test inject delivers pydantic model instance."""

        class InjectedService(m.BaseModel):
            value: str

        container.bind("inject_model", InjectedService(value="model-value"))

        @d.inject(service="inject_model")
        def read(*, service: InjectedService | None = None) -> str:
            return service.value if service is not None else "none"

        assert read() == "model-value"

    # --- d.log_operation -------------------------------------------------
    @pytest.mark.parametrize("payload", ["success", "", "multi word result"])
    def test_log_operation_returns_wrapped_value_unchanged(self, payload: str) -> None:
        """Test log operation returns wrapped value unchanged."""

        @d.log_operation("payload_op")
        def produce() -> str:
            return payload

        assert produce() == payload

    def test_log_operation_with_perf_tracking_returns_value(self) -> None:
        """Test log operation with perf tracking returns value."""

        @d.log_operation("timed_op", track_perf=True)
        def compute() -> int:
            return 42

        assert compute() == 42

    @pytest.mark.parametrize(
        ("exc_type", "message"),
        [
            (ValueError, "bad value"),
            (RuntimeError, "runtime boom"),
            (KeyError, "missing"),
        ],
    )
    def test_log_operation_propagates_wrapped_exception(
        self,
        exc_type: type[Exception],
        message: str,
    ) -> None:
        """Test log operation propagates wrapped exception."""

        @d.log_operation("failing_op")
        def boom() -> None:
            raise exc_type(message)

        with pytest.raises(exc_type, match=message):
            boom()

    def test_log_operation_emits_operation_name_on_failure(self) -> None:
        """Test log operation emits operation name on failure."""

        @d.log_operation("named_failure_op")
        def boom() -> None:
            error_msg = "kaboom"
            raise ValueError(error_msg)

        def emit() -> None:
            with pytest.raises(ValueError, match="kaboom"):
                boom()

        _ = capture_stdout(emit, contains="named_failure_op")

    # --- d.with_correlation ----------------------------------------------
    def test_with_correlation_establishes_correlation_id_during_call(self) -> None:
        """Test with correlation establishes correlation id during call."""

        @d.with_correlation()
        def inside() -> str:
            return FlextContext.ensure_correlation_id()

        assert inside()

    def test_with_correlation_returns_wrapped_value(self) -> None:
        """Test with correlation returns wrapped value."""

        @d.with_correlation()
        def produce() -> str:
            return "wrapped"

        assert produce() == "wrapped"

    # --- d.deprecated ----------------------------------------------------
    def test_deprecated_warns_and_returns_value(self) -> None:
        """Test deprecated warns and returns value."""

        @d.deprecated("use new_api")
        def old_api() -> int:
            return 7

        with pytest.warns(DeprecationWarning, match="deprecated"):
            assert old_api() == 7
