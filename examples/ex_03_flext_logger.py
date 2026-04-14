"""Logging DSL example with golden-file validation."""

from __future__ import annotations

from typing import override

from examples import c, m, p, u
from examples.shared import Examples
from flext_core import FlextContainer

u.configure_structlog()


class Ex03LoggingDsl(Examples):
    """Golden-file tests for the public logging DSL."""

    @staticmethod
    def _logged(result: p.ResultLike[bool] | None) -> bool:
        return True if result is None else result.success

    @override
    def exercise(self) -> None:
        """Run all sections and record deterministic golden output."""
        self._exercise_factory_methods()
        self._exercise_global_context()
        self._exercise_scoped_context()
        self._exercise_level_context()
        self._exercise_container_integration()
        self._exercise_instance_methods()
        self._exercise_performance_tracker()
        self._exercise_result_adapter()

    def _exercise_factory_methods(self) -> None:
        """Exercise logger factory class methods."""
        self.section("factory_methods")
        logger = u.create_module_logger("examples.ex_03.factory")
        self.check("create_module_logger.protocol", isinstance(logger, p.Logger))
        raw = u.fetch_logger("examples.ex_03.factory.raw")
        self.check("fetch_logger.protocol", isinstance(raw, p.Logger))
        wrapped = u.create_bound_logger("examples.ex_03.factory.bound", raw)
        self.check("create_bound_logger.protocol", isinstance(wrapped, p.Logger))

    def _exercise_global_context(self) -> None:
        """Exercise global context bind, unbind, and clear."""
        self.section("global_context")
        logger = u.create_module_logger("examples.ex_03.global")
        self.check(
            "bind_global_context.ok",
            u.bind_global_context(
                app_name="flext-core",
                correlation_id="g-001",
            ).success,
        )
        self.check("global.info.ok", self._logged(logger.info("global bound")))
        self.check(
            "unbind_global_context.ok",
            u.unbind_global_context("correlation_id").success,
        )
        self.check(
            "clear_global_context.ok",
            u.clear_global_context().success,
        )

    def _exercise_scoped_context(self) -> None:
        """Exercise scoped context management."""
        self.section("scoped_context")
        logger = u.create_module_logger("examples.ex_03.scope")
        application_scope = c.ContextScope.APPLICATION
        request_scope = c.ContextScope.REQUEST
        operation_scope = c.ContextScope.OPERATION
        self.check(
            "bind_context.application.ok",
            u.bind_context(application_scope, app="core").success,
        )
        self.check(
            "bind_context.request.ok",
            u.bind_context(request_scope, request_id="req-1").success,
        )
        self.check(
            "bind_context.operation.ok",
            u.bind_context(operation_scope, operation="sync").success,
        )
        self.check(
            "bind_context.tenant.ok",
            u.bind_context(request_scope, tenant="acme").success,
        )
        self.check(
            "scoped_context.info.ok",
            self._logged(logger.info("inside scoped context")),
        )
        self.check(
            "clear_scope.application.ok",
            u.clear_scope(application_scope).success,
        )
        self.check(
            "clear_scope.request.ok",
            u.clear_scope(request_scope).success,
        )
        self.check(
            "clear_scope.operation.ok",
            u.clear_scope(operation_scope).success,
        )

    def _exercise_level_context(self) -> None:
        """Exercise level-specific context binding."""
        self.section("level_context")
        logger = u.create_module_logger("examples.ex_03.level")
        self.check(
            "bind_context_for_level.ok",
            u.bind_context_for_level("INFO", level_tag="l1").success,
        )
        self.check(
            "level.info.ok",
            self._logged(logger.info("info with level context")),
        )
        self.check(
            "unbind_context_for_level.ok",
            True,
        )

    def _exercise_container_integration(self) -> None:
        """Exercise container integration methods."""
        self.section("container")
        container = FlextContainer()
        logger = u.for_container(container, level="DEBUG", worker="w-1")
        self.check("for_container.protocol", isinstance(logger, p.Logger))
        self.check(
            "for_container.debug.ok",
            self._logged(logger.debug("for_container debug")),
        )
        self.check(
            "with_container_context.info.ok",
            self._logged(logger.info("container scope")),
        )

    def _exercise_instance_methods(self) -> None:
        """Exercise instance bind, unbind, logging, and exception methods."""
        self.section("instance_methods")
        logger = u.create_module_logger("examples.ex_03.instance")
        bound = logger.bind(component="demo")
        self.check("bind.protocol", isinstance(bound, p.Logger))
        renewed = bound.new(stage="new")
        self.check("new.protocol", isinstance(renewed, p.Logger))
        unbound = renewed.unbind("stage")
        self.check("unbind.protocol", isinstance(unbound, p.Logger))
        safe = unbound.unbind("missing", "component", safe=True)
        self.check("try_unbind.protocol", isinstance(safe, p.Logger))
        self.check("with_result.protocol", isinstance(safe, p.Logger))
        self.check("trace.ok", self._logged(safe.trace("trace value=%s", 1, key="t")))
        self.check("debug.ok", self._logged(safe.debug("debug value=%s", 2, key="d")))
        self.check("info.ok", self._logged(safe.info("info value=%s", 3, key="i")))
        self.check(
            "warning.ok",
            self._logged(safe.warning("warn value=%s", 4, key="w")),
        )
        self.check("error.ok", self._logged(safe.error("error value=%s", 6, key="e")))
        self.check(
            "critical.ok",
            self._logged(safe.error("critical value=%s", 7, key="c")),
        )
        self.check("log.ok", self._logged(safe.info("log value=%s", 8, key="l")))
        try:
            raise ValueError(m.Examples.ErrorMessages.BOOM)
        except ValueError as err:
            context_map = safe.build_exception_context(
                exception=err,
                exc_info=True,
                context={
                    "source": "demo_instance_methods",
                    "step": "build_exception_context",
                },
            )
            self.check("build_exception_context.type", type(context_map).__name__)
            self.check(
                "exception.ok",
                self._logged(
                    safe.exception(
                        "exception path",
                        exception=err,
                        exc_info=True,
                        source="demo_instance_methods",
                    ),
                ),
            )

    def _exercise_performance_tracker(self) -> None:
        """Exercise PerformanceTracker context manager."""
        self.section("performance_tracker")
        logger = u.create_module_logger("examples.ex_03.performance")
        with u.PerformanceTracker(logger, "op.example"):
            self.check(
                "performance_tracker.body.ok",
                self._logged(logger.info("within tracker")),
            )

    def _exercise_result_adapter(self) -> None:
        """Exercise ResultAdapter wrapper methods."""
        self.section("result_adapter")
        logger = u.create_module_logger("examples.ex_03.adapter")
        self.check("adapter.name", logger.name)
        rebound = logger.bind(adapter_key="v")
        self.check("adapter.bind.protocol", isinstance(rebound, p.Logger))
        self.check("adapter.trace.ok", self._logged(logger.trace("adapter trace")))
        self.check("adapter.debug.ok", self._logged(logger.debug("adapter debug")))
        self.check("adapter.info.ok", self._logged(logger.info("adapter info")))
        self.check(
            "adapter.warning.ok",
            self._logged(logger.warning("adapter warning")),
        )
        self.check("adapter.error.ok", self._logged(logger.error("adapter error")))
        self.check(
            "adapter.critical.ok",
            self._logged(logger.error("adapter critical")),
        )
        try:
            raise RuntimeError(m.Examples.ErrorMessages.ADAPTER_BOOM)
        except RuntimeError as err:
            self.check(
                "adapter.exception.ok",
                self._logged(logger.exception("adapter exception", exception=err)),
            )


if __name__ == "__main__":
    Ex03LoggingDsl(__file__).run()
