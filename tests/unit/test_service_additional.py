"""Additional coverage for flext_core.service using real executions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, cast

import pytest

from flext_core import e, p, r, s, t
from flext_tests import u


class AutoSuccessService(s[str]):
    """Service that auto-executes and returns a value."""

    auto_execute: ClassVar[bool] = True

    def execute(self) -> r[str]:
        return r[str].ok("auto_ok")


class AutoFailService(s[str]):
    """Service that auto-executes and raises BaseError on failure."""

    auto_execute: ClassVar[bool] = True

    def execute(self) -> r[str]:
        return r[str].fail("auto_fail")


class RuntimeCloneService(s[str]):
    """Service exposing runtime cloning for testing."""

    flag: bool = True

    def execute(self) -> r[str]:
        return r[str].ok("run")

    def validate_business_rules(self) -> r[bool]:
        if not self.flag:
            return r[bool].fail("bad flag")
        return r[bool].ok(True)


def test_auto_execute_returns_value() -> None:
    """auto_execute=True should return the execution value directly."""
    result_value = AutoSuccessService()
    assert result_value == "auto_ok"


def test_auto_execute_failure_raises_base_error() -> None:
    """Failed auto_execute should raise BaseError with message."""
    with pytest.raises(e.BaseError, match="auto_fail"):
        _ = AutoFailService()


def test_is_valid_handles_validation_exception() -> None:
    """is_valid should return False when validation raises exceptions."""

    class RaisingValidationService(s[str]):
        def validate_business_rules(self) -> r[bool]:
            msg = "boom"
            raise RuntimeError(msg)

        def execute(self) -> r[str]:
            return r[str].ok("x")

    service = RaisingValidationService()
    assert service.is_valid() is False


def test_clone_runtime_creates_isolated_scope() -> None:
    """_clone_runtime should produce new config/context/container."""
    service = RuntimeCloneService()
    # Ensure runtime is initialized by accessing it
    # This will trigger _create_initial_runtime() if needed
    base_runtime = service.runtime
    # Get base config app_name before cloning
    # Access via service.config property which is properly initialized
    base_app_name = service.config.app_name

    cloned = service._clone_runtime(
        config_overrides={"app_name": "cloned"},
        subproject="sub",
        container_services={"val": "data"},
    )

    assert cloned is not base_runtime
    assert cloned.config.app_name == "cloned"
    assert cloned.config.app_name != base_app_name
    # Ensure new container resolves injected service
    resolved_result: p.Result[t.GeneralValueType] = cloned.container.get(
        "val",
    )
    # Type narrowing: assert_result_success accepts r[TResult], protocol Result is compatible
    # Cast to r[t.GeneralValueType] for type compatibility
    resolved_result_typed: r[t.GeneralValueType] = cast(
        "r[t.GeneralValueType]",
        resolved_result,
    )
    u.Tests.Result.assert_result_success(resolved_result_typed)
    assert resolved_result.value == "data"


def test_access_facade_exposes_components() -> None:
    """Service access facade should expose registry/config/context/result."""
    service = RuntimeCloneService()
    # Type narrowing: access returns _ServiceAccess, not Callable
    # Access is a property that returns _ServiceAccess instance
    access = service.access
    # Type annotation: access is _ServiceAccess (not Callable)
    # Use type: ignore[attr-defined] because mypy doesn't recognize _ServiceAccess attributes
    # but they exist at runtime (computed_field properties)
    assert access.cqrs is not None
    assert access.config is service.config
    assert access.context is service.context
    assert access.result is r
    registry = access.registry
    assert registry is not None
    info: Mapping[str, t.FlexibleValue] = service.get_service_info()
    assert info["service_type"] == "RuntimeCloneService"


def test_result_property_raises_on_failure() -> None:
    """Result property should raise BaseError on failed execution."""

    class FailingOnResultService(s[str]):
        def execute(self) -> r[str]:
            return r[str].fail("fail_exec")

    service = FailingOnResultService()
    with pytest.raises(e.BaseError, match="fail_exec"):
        _ = service.result
