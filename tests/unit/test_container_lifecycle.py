"""Container clearing and scoped lifecycle tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import tm

from tests.constants import c
from tests.models import m
from tests.utilities import u

if TYPE_CHECKING:
    from tests.protocols import p
    from tests.typings import t


class TestsFlextContainerLifecycle:
    def test_clear_all_services(self, clean_container: p.Container) -> None:
        """Test clearing all services and factories using fixtures."""
        container = clean_container
        _ = container.bind("service1", "value1")
        _ = container.bind("service2", "value2")
        factory = u.Tests.create_factory("value3")
        _ = container.factory("factory1", factory)
        tm.that(
            len(container.names()),
            eq=3,
            msg="Container must have 3 services before clear_all",
        )
        container.clear()
        tm.that(
            len(container.names()),
            eq=0,
            msg="Container must have 0 services after clear_all",
        )
        tm.that(
            container.names(),
            empty=True,
            msg="Container services list must be empty after clear_all",
        )

    def test_clear_all_empty(self, clean_container: p.Container) -> None:
        """Test clearing when no services exist using fixtures."""
        container = clean_container
        container.clear()
        tm.that(
            len(container.names()),
            eq=0,
            msg="Empty container must have 0 services after clear_all",
        )
        tm.that(
            container.names(),
            empty=True,
            msg="Empty container services list must be empty after clear_all",
        )

    def test_full_workflow(self, clean_container: p.Container) -> None:
        """Test complete container workflow using fixtures."""
        container = clean_container
        _ = container.bind("db_connection", {"host": c.LOCALHOST})
        _ = container.bind("cache", {"type": "redis"})
        factory = u.Tests.create_factory({"logger": "instance"})
        _ = container.factory("logger", factory)
        required_services = ["db_connection", "cache", "logger"]
        for service_name in required_services:
            tm.that(
                container.has(service_name),
                eq=True,
                msg=f"Container must have {service_name} after registration",
            )
        for name in required_services:
            result: p.Result[t.RegisterableService] = container.resolve(name)
            _ = u.Tests.assert_success(result)
        tm.that(
            len(container.names()),
            eq=3,
            msg="Container must have 3 services in full workflow",
        )
        container.clear()
        tm.that(
            len(container.names()),
            eq=0,
            msg="Container must have 0 services after clear_all in workflow",
        )

    def test_factory_exception_handling(self, clean_container: p.Container) -> None:
        """Test handling of factory exceptions using fixtures."""
        container = clean_container
        error_msg = "Factory failed"

        def failing_factory() -> str:
            raise RuntimeError(error_msg)

        _ = container.factory("failing", failing_factory)
        result: p.Result[t.RegisterableService] = container.resolve("failing")
        _ = u.Tests.assert_failure(result)

    def test_scoped_container_with_context(
        self,
        clean_container: p.Container,
    ) -> None:
        """Test scoped container creation with FlextContext."""
        scoped = clean_container.scope(
            subproject="unit",
            registration=m.ServiceRegistrationSpec.model_validate({
                "services": {"scoped_service": "scoped-value"},
            }),
        )
        tm.that(scoped.has("scoped_service"), eq=True)
        tm.ok(scoped.resolve("scoped_service", type_cls=str), eq="scoped-value")
        ctx_result = scoped.context.get("subproject")
        scoped_settings = scoped.settings.model_dump()
        base_settings = clean_container.settings.model_dump()

        assert ctx_result.success
        assert ctx_result.value == "unit"
        assert scoped_settings["app_name"] == f"{base_settings['app_name']}.unit"
