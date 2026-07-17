"""Behavioral tests for the FlextContainer facade lifecycle contract.

Complements the split registration/resolution suites by exercising the
container's facade-level public behaviors: singleton identity, drop, clear,
scope isolation, fluent apply, and snapshot. Every assertion targets the
observable public surface (return values, ``r[T]`` outcomes, and public
model state) rather than internal structures.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from flext_core.container import FlextContainer
from tests import u

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreContainer:
    def test_singleton_returns_same_instance(self) -> None:
        """Constructing the container twice yields the same shared instance."""
        first = FlextContainer()
        second = FlextContainer()
        tm.that(
            first is second,
            eq=True,
            msg="FlextContainer() must return the process-wide singleton",
        )

    def test_reset_for_testing_replaces_singleton(self) -> None:
        """reset_for_testing drops the cached singleton so a fresh one is built."""
        original = FlextContainer()
        FlextContainer.reset_for_testing()
        replacement = FlextContainer()
        tm.that(
            replacement is not original,
            eq=True,
            msg="reset_for_testing must force a brand-new singleton instance",
        )

    def test_bind_then_resolve_roundtrip(self, clean_container: p.Container) -> None:
        """A bound value is resolvable as a successful result carrying that value."""
        clean_container.bind("greeting", "hello")
        u.Tests.assert_success(
            clean_container.resolve("greeting"),
            expected_value="hello",
        )

    def test_drop_removes_registration_and_reports_success(
        self,
        clean_container: p.Container,
    ) -> None:
        """Drop returns ok(True) and the service is no longer resolvable."""
        clean_container.bind("temp", "value")
        outcome = clean_container.drop("temp")
        tm.that(
            u.Tests.assert_success(outcome),
            eq=True,
            msg="drop of an existing service must succeed with True",
        )
        tm.that(
            clean_container.has("temp"),
            eq=False,
            msg="Service must be absent after drop",
        )
        _ = u.Tests.assert_failure(
            clean_container.resolve("temp"),
            expected_error="not found",
        )

    def test_drop_unknown_service_fails_with_not_found(
        self,
        clean_container: p.Container,
    ) -> None:
        """Dropping an unregistered name yields a not-found failure result."""
        _ = u.Tests.assert_failure(
            clean_container.drop("ghost"),
            expected_error="not found",
        )

    def test_logger_requires_explicit_module_name(
        self,
        clean_container: p.Container,
    ) -> None:
        params = inspect.signature(type(clean_container).logger).parameters
        module_param = params["module_name"]
        tm.that(
            module_param.default is inspect.Parameter.empty,
            eq=True,
            msg="module_name must be required, not optional",
        )

    def test_clear_empties_all_registrations(
        self,
        clean_container: p.Container,
    ) -> None:
        """Clear removes every registered name, leaving an empty container."""
        clean_container.bind("a", "1")
        clean_container.factory("b", u.Tests.create_factory("2"))
        clean_container.clear()
        tm.that(
            clean_container.names(),
            empty=True,
            msg="clear must remove all registered services and factories",
        )

    def test_clear_is_idempotent(self, clean_container: p.Container) -> None:
        """Clearing an already-empty container remains empty (no error)."""
        clean_container.clear()
        clean_container.clear()
        tm.that(
            clean_container.names(),
            empty=True,
            msg="Repeated clear must keep the container empty",
        )

    def test_scope_inherits_parent_registrations(
        self,
        clean_container: p.Container,
    ) -> None:
        """A derived scope can see services registered on its parent."""
        clean_container.bind("shared_svc", "value")
        scoped = clean_container.scope()
        tm.that(
            scoped.has("shared_svc"),
            eq=True,
            msg="Scope must inherit the parent's registrations",
        )

    def test_scope_registrations_do_not_leak_to_parent(
        self,
        clean_container: p.Container,
    ) -> None:
        """Bindings made inside a scope stay isolated from the parent container."""
        scoped = clean_container.scope()
        scoped.bind("scoped_only", "value")
        tm.that(
            clean_container.has("scoped_only"),
            eq=False,
            msg="Scope-local registrations must not leak back to the parent",
        )

    @pytest.mark.parametrize(
        "overrides",
        [None, {"custom_key": "custom_value"}],
        ids=["none", "mapping"],
    )
    def test_apply_returns_self_for_fluent_chaining(
        self,
        overrides: dict[str, str] | None,
        clean_container: p.Container,
    ) -> None:
        """Apply returns the same container so callers can chain fluently."""
        tm.that(
            clean_container.apply(overrides) is clean_container,
            eq=True,
            msg="apply must return self for fluent configuration",
        )

    def test_snapshot_exposes_config_via_model_dump(
        self,
        clean_container: p.Container,
    ) -> None:
        """Snapshot exposes merged settings as a mapping via its public model API."""
        dumped = clean_container.snapshot().model_dump()
        tm.that(
            dumped,
            is_=dict,
            msg="snapshot().model_dump() must return a plain mapping of settings",
        )
