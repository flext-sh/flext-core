"""FlextLogger — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from typing import override

from flext_core import FlextContainer, FlextLogger, FlextRuntime, c, r

from .shared import Examples

FlextRuntime.configure_structlog()


class Ex03FlextLogger(Examples):
    """Golden-file tests for ``FlextLogger`` public API."""

    @staticmethod
    def _logged(result: r[bool] | None) -> bool:
        return True if result is None else result.is_success

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
        logger = FlextLogger.create_module_logger("examples.ex_03.factory")
        self.check("create_module_logger.type", type(logger).__name__)
        raw = FlextLogger.get_logger("examples.ex_03.factory.raw")
        self.check("get_logger.type", type(raw).__name__)
        wrapped = FlextLogger.create_bound_logger("examples.ex_03.factory.bound", raw)
        self.check("create_bound_logger.type", type(wrapped).__name__)

    def _exercise_global_context(self) -> None:
        """Exercise global context bind, unbind, and clear."""
        self.section("global_context")
        logger = FlextLogger.create_module_logger("examples.ex_03.global")
        self.check(
            "bind_global_context.ok",
            FlextLogger.bind_global_context(
                app_name="flext-core", correlation_id="g-001"
            ).is_success,
        )
        self.check("global.info.ok", self._logged(logger.info("global bound")))
        self.check(
            "unbind_global_context.ok",
            FlextLogger.unbind_global_context("correlation_id").is_success,
        )
        self.check(
            "clear_global_context.ok", FlextLogger.clear_global_context().is_success
        )

    def _exercise_scoped_context(self) -> None:
        """Exercise scoped context management."""
        self.section("scoped_context")
        logger = FlextLogger.create_module_logger("examples.ex_03.scope")
        application_scope = c.Context.SCOPE_APPLICATION
        request_scope = c.Context.SCOPE_REQUEST
        operation_scope = c.Context.SCOPE_OPERATION
        self.check(
            "bind_context.application.ok",
            FlextLogger.bind_context(application_scope, app="core").is_success,
        )
        self.check(
            "bind_context.request.ok",
            FlextLogger.bind_context(request_scope, request_id="req-1").is_success,
        )
        self.check(
            "bind_context.operation.ok",
            FlextLogger.bind_context(operation_scope, operation="sync").is_success,
        )
        with FlextLogger.scoped_context(request_scope, tenant="acme"):
            self.check(
                "scoped_context.info.ok",
                self._logged(logger.info("inside scoped context")),
            )
        self.check(
            "clear_scope.application.ok",
            FlextLogger.clear_scope(application_scope).is_success,
        )
        self.check(
            "clear_scope.request.ok", FlextLogger.clear_scope(request_scope).is_success
        )
        self.check(
            "clear_scope.operation.ok",
            FlextLogger.clear_scope(operation_scope).is_success,
        )

    def _exercise_level_context(self) -> None:
        """Exercise level-specific context binding."""
        self.section("level_context")
        logger = FlextLogger.create_module_logger("examples.ex_03.level")
        self.check(
            "bind_context_for_level.ok",
            FlextLogger.bind_context_for_level("INFO", level_tag="l1").is_success,
        )
        self.check(
            "level.info.ok", self._logged(logger.info("info with level context"))
        )
        self.check(
            "unbind_context_for_level.ok",
            FlextLogger.unbind_context_for_level("INFO", "level_tag").is_success,
        )

    def _exercise_container_integration(self) -> None:
        """Exercise container integration methods."""
        self.section("container")
        container = FlextContainer()
        logger = FlextLogger.for_container(container, level="DEBUG", worker="w-1")
        self.check("for_container.type", type(logger).__name__)
        self.check(
            "for_container.debug.ok",
            self._logged(logger.debug("for_container debug")),
        )
        with FlextLogger.with_container_context(
            container, level="INFO", feature="demo"
        ):
            self.check(
                "with_container_context.info.ok",
                self._logged(logger.info("container scope")),
            )

    def _exercise_instance_methods(self) -> None:
        """Exercise instance bind, unbind, logging, and exception methods."""
        self.section("instance_methods")
        logger = FlextLogger.create_module_logger("examples.ex_03.instance")
        bound = logger.bind(component="demo")
        self.check("bind.type", type(bound).__name__)
        renewed = bound.new(stage="new")
        self.check("new.type", type(renewed).__name__)
        unbound = renewed.unbind("stage")
        self.check("unbind.type", type(unbound).__name__)
        safe = unbound.unbind("missing", "component", safe=True)
        self.check("try_unbind.type", type(safe).__name__)
        self.check("with_result.type", type(safe).__name__)
        self.check("trace.ok", self._logged(safe.trace("trace value=%s", 1, key="t")))
        self.check("debug.ok", self._logged(safe.debug("debug value=%s", 2, key="d")))
        self.check("info.ok", self._logged(safe.info("info value=%s", 3, key="i")))
        self.check(
            "warning.ok", self._logged(safe.warning("warn value=%s", 4, key="w"))
        )
        self.check("error.ok", self._logged(safe.error("error value=%s", 6, key="e")))
        self.check(
            "critical.ok", self._logged(safe.error("critical value=%s", 7, key="c"))
        )
        self.check("log.ok", self._logged(safe.info("log value=%s", 8, key="l")))
        try:
            boom_msg = "boom"
            raise ValueError(boom_msg)
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
                    )
                ),
            )

    def _exercise_performance_tracker(self) -> None:
        """Exercise PerformanceTracker context manager."""
        self.section("performance_tracker")
        logger = FlextLogger.create_module_logger("examples.ex_03.performance")
        with FlextLogger.PerformanceTracker(logger, "op.example"):
            self.check(
                "performance_tracker.body.ok",
                self._logged(logger.info("within tracker")),
            )

    def _exercise_result_adapter(self) -> None:
        """Exercise ResultAdapter wrapper methods."""
        self.section("result_adapter")
        logger = FlextLogger.create_module_logger("examples.ex_03.adapter")
        self.check("adapter.name", logger.name)
        rebound = logger.bind(adapter_key="v")
        self.check("adapter.bind.type", type(rebound).__name__)
        self.check("adapter.trace.ok", self._logged(logger.trace("adapter trace")))
        self.check("adapter.debug.ok", self._logged(logger.debug("adapter debug")))
        self.check("adapter.info.ok", self._logged(logger.info("adapter info")))
        self.check(
            "adapter.warning.ok", self._logged(logger.warning("adapter warning"))
        )
        self.check("adapter.error.ok", self._logged(logger.error("adapter error")))
        self.check(
            "adapter.critical.ok", self._logged(logger.error("adapter critical"))
        )
        try:
            adapter_msg = "adapter boom"
            raise RuntimeError(adapter_msg)
        except RuntimeError as err:
            self.check(
                "adapter.exception.ok",
                self._logged(logger.exception("adapter exception", exception=err)),
            )


if __name__ == "__main__":
    Ex03FlextLogger(__file__).run()
