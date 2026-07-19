"""Behavioral tests for FlextContainer lifecycle: clear, drop, scope, resolve.

Every assertion targets the public container contract (bind/factory/resource,
has/names/resolve/drop/clear/scope + settings/context) and the ``r[T]`` outcome
of fallible operations. No private attribute or internal-collaborator is touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import tm

from tests.models import m
from tests.utilities import u

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextContainerLifecycle:
    def test_clear_removes_every_registration(
        self,
        clean_container: p.Container,
    ) -> None:
        """After clear the container exposes no user registrations."""
        container = clean_container
        _ = container.bind("service1", "value1")
        _ = container.bind("service2", "value2")
        _ = container.factory("factory1", u.Tests.create_factory("value3"))
        tm.that(
            len(container.names()),
            eq=3,
            msg="Three registrations must be visible before clear",
        )

        container.clear()

        tm.that(
            container.names(),
            empty=True,
            msg="names() must be empty after clear",
        )
        for name in ("service1", "service2", "factory1"):
            tm.that(
                container.has(name),
                eq=False,
                msg=f"has({name}) must be False after clear",
            )
            _ = tm.fail(
                container.resolve(name),
                msg=f"resolve({name}) must fail after clear",
            )

    def test_clear_on_empty_container_is_noop(
        self,
        clean_container: p.Container,
    ) -> None:
        """Clearing an already-empty container leaves it empty (no error)."""
        container = clean_container

        container.clear()

        tm.that(
            container.names(),
            empty=True,
            msg="Empty container must stay empty after clear",
        )

    def test_clear_is_idempotent(self, clean_container: p.Container) -> None:
        """Clearing twice yields the same empty observable state."""
        container = clean_container
        _ = container.bind("svc", "v")

        container.clear()
        first = list(container.names())
        container.clear()
        second = list(container.names())

        tm.that(first, eq=second, msg="Repeated clear must be idempotent")
        tm.that(second, empty=True, msg="Container must remain empty")

    def test_bind_returns_container_for_fluent_chaining(
        self,
        clean_container: p.Container,
    ) -> None:
        """Bind returns the same container so registrations can be chained."""
        container = clean_container

        returned = container.bind("chained", "value")

        assert returned is container
        tm.that(
            container.has("chained"),
            eq=True,
            msg="Chained bind must register the service",
        )

    def test_registered_services_resolve_to_their_bound_values(
        self,
        clean_container: p.Container,
    ) -> None:
        """Resolve returns the exact value/instance that was registered."""
        container = clean_container
        _ = container.bind("cache", "redis")
        _ = container.factory("logger", u.Tests.create_factory("logger-instance"))

        tm.ok(
            container.resolve("cache", type_cls=str),
            eq="redis",
            msg="Bound value must resolve unchanged",
        )
        tm.ok(
            container.resolve("logger", type_cls=str),
            eq="logger-instance",
            msg="Factory product must resolve to the produced value",
        )
        tm.that(
            len(container.names()),
            eq=2,
            msg="Both registrations must be counted",
        )

    def test_factory_that_raises_surfaces_a_failure_result(
        self,
        clean_container: p.Container,
    ) -> None:
        """A raising factory yields a failing r[T] carrying the error text."""
        container = clean_container
        error_msg = "Factory failed"

        def failing_factory() -> str:
            raise RuntimeError(error_msg)

        _ = container.factory("failing", failing_factory)

        _ = tm.fail(
            container.resolve("failing"),
            has=error_msg,
            msg="Factory exception must surface as a failure result",
        )

    def test_resolve_unknown_service_fails(
        self,
        clean_container: p.Container,
    ) -> None:
        """Resolving a name that was never registered fails."""
        container = clean_container

        _ = tm.fail(
            container.resolve("missing"),
            msg="Unknown service must not resolve to a value",
        )

    def test_drop_removes_a_registered_service(
        self,
        clean_container: p.Container,
    ) -> None:
        """Drop succeeds for a known name and the service disappears."""
        container = clean_container
        _ = container.bind("temp", "value")

        tm.ok(
            container.drop("temp"),
            eq=True,
            msg="Dropping a registered service must succeed with True",
        )
        tm.that(
            container.has("temp"),
            eq=False,
            msg="Dropped service must no longer be present",
        )

    def test_drop_unknown_service_fails(
        self,
        clean_container: p.Container,
    ) -> None:
        """Drop reports a failure when the name was never registered."""
        container = clean_container

        _ = tm.fail(
            container.drop("never-registered"),
            msg="Dropping an unknown service must fail",
        )

    def test_scope_creates_isolated_child_without_polluting_parent(
        self,
        clean_container: p.Container,
    ) -> None:
        """A scoped container sees its own service and derives its settings.

        The parent container must not observe the child's registration.
        """
        scoped = clean_container.scope(
            subproject="unit",
            registration=u.normalize_service_registration_spec(
                m.ServiceRegistrationSpec(
                    services={"scoped_service": "scoped-value"},
                )
            ),
        )

        tm.that(
            scoped.has("scoped_service"),
            eq=True,
            msg="Scoped container must expose its own service",
        )
        tm.ok(
            scoped.resolve("scoped_service", type_cls=str),
            eq="scoped-value",
            msg="Scoped service must resolve to its registered value",
        )
        tm.that(
            clean_container.has("scoped_service"),
            eq=False,
            msg="Parent container must not see the scoped registration",
        )

        ctx_result = scoped.context.get("subproject")
        scoped_settings = scoped.settings.model_dump()
        base_settings = clean_container.settings.model_dump()
        tm.ok(
            ctx_result,
            eq="unit",
            msg="Scoped context must carry the subproject value",
        )
        tm.that(
            scoped_settings["log_level"],
            eq=base_settings["log_level"],
            msg="Scoped settings must derive app_name from the parent namespace",
        )
