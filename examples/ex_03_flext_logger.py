"""FlextLogger — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from typing import override

from flext_core import FlextContainer, FlextLogger, FlextRuntime, c

from .shared import Examples

FlextRuntime.configure_structlog()


class Ex03FlextLogger(Examples):
    """Golden-file tests for ``FlextLogger`` public API."""

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
        result_logger = logger.with_result()
        self.check(
            "bind_global_context.ok",
            FlextLogger.bind_global_context(
                app_name="flext-core", correlation_id="g-001"
            ).is_success,
        )
        self.check("global.info.ok", result_logger.info("global bound").is_success)
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
        result_logger = logger.with_result()
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
                result_logger.info("inside scoped context").is_success,
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
        result_logger = logger.with_result()
        self.check(
            "bind_context_for_level.ok",
            FlextLogger.bind_context_for_level("INFO", level_tag="l1").is_success,
        )
        self.check(
            "level.info.ok", result_logger.info("info with level context").is_success
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
        result_logger = logger.with_result()
        self.check("for_container.type", type(logger).__name__)
        self.check(
            "for_container.debug.ok",
            result_logger.debug("for_container debug").is_success,
        )
        with FlextLogger.with_container_context(
            container, level="INFO", feature="demo"
        ):
            self.check(
                "with_container_context.info.ok",
                result_logger.info("container scope").is_success,
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
        adapter = safe.with_result()
        self.check("with_result.type", type(adapter).__name__)
        self.check("trace.ok", adapter.trace("trace value=%s", 1, key="t").is_success)
        self.check("debug.ok", adapter.debug("debug value=%s", 2, key="d").is_success)
        self.check("info.ok", adapter.info("info value=%s", 3, key="i").is_success)
        self.check(
            "warning.ok", adapter.warning("warn value=%s", 4, key="w").is_success
        )
        self.check("error.ok", adapter.error("error value=%s", 6, key="e").is_success)
        self.check(
            "critical.ok", adapter.critical("critical value=%s", 7, key="c").is_success
        )
        self.check("log.ok", adapter.info("log value=%s", 8, key="l").is_success)
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
                adapter.exception(
                    "exception path",
                    exception=err,
                    exc_info=True,
                    source="demo_instance_methods",
                ).is_success,
            )

    def _exercise_performance_tracker(self) -> None:
        """Exercise PerformanceTracker context manager."""
        self.section("performance_tracker")
        logger = FlextLogger.create_module_logger("examples.ex_03.performance")
        result_logger = logger.with_result()
        with FlextLogger.PerformanceTracker(logger, "op.example"):
            self.check(
                "performance_tracker.body.ok",
                result_logger.info("within tracker").is_success,
            )

    def _exercise_result_adapter(self) -> None:
        """Exercise ResultAdapter wrapper methods."""
        self.section("result_adapter")
        logger = FlextLogger.create_module_logger("examples.ex_03.adapter")
        adapter = logger.with_result()
        self.check("adapter.name", adapter.name)
        self.check("adapter.with_result.idempotent", adapter.with_result() is adapter)
        rebound = adapter.bind(adapter_key="v")
        self.check("adapter.bind.type", type(rebound).__name__)
        self.check("adapter.trace.ok", adapter.trace("adapter trace").is_success)
        self.check("adapter.debug.ok", adapter.debug("adapter debug").is_success)
        self.check("adapter.info.ok", adapter.info("adapter info").is_success)
        self.check("adapter.warning.ok", adapter.warning("adapter warning").is_success)
        self.check("adapter.error.ok", adapter.error("adapter error").is_success)
        self.check(
            "adapter.critical.ok", adapter.critical("adapter critical").is_success
        )
        try:
            adapter_msg = "adapter boom"
            raise RuntimeError(adapter_msg)
        except RuntimeError as err:
            self.check(
                "adapter.exception.ok",
                adapter.exception(
                    "adapter exception", exception=err, exc_info=True
                ).is_success,
            )


if __name__ == "__main__":
    Ex03FlextLogger(__file__).run()
