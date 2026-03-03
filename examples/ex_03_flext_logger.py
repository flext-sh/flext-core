"""FlextLogger — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from shared import Examples

from flext_core import FlextContainer, FlextLogger, FlextRuntime, c, p

FlextRuntime.configure_structlog()


class Ex03FlextLogger(Examples):
    """Exercise FlextLogger public API."""

    def exercise(self) -> None:
        """Run all logger demonstrations and verify golden file."""
        self.demo_factory_methods()
        self.demo_global_context()
        self.demo_scoped_context()
        self.demo_level_context()
        self.demo_container_integration()
        self.demo_instance_methods()
        self.demo_performance_tracker()
        self.demo_result_adapter()

    def demo_factory_methods(self) -> None:
        """Exercise logger factory class methods."""
        self.section("factory_methods")

        module_name = f"examples.ex_03.{self.rand_str(6)}"
        raw_name = f"{module_name}.raw"
        logger = FlextLogger.create_module_logger(module_name)
        self.check("create_module_logger.type", type(logger).__name__)
        self.check(
            "create_module_logger.type_matches",
            type(logger).__name__ == "FlextLogger",
        )

        raw = FlextLogger.get_logger(raw_name)
        self.check("get_logger.type", type(raw).__name__)
        self.check(
            "get_logger.type_matches",
            type(raw).__name__ == "BoundLoggerFilteringAtInfo",
        )

        wrapped = FlextLogger.create_bound_logger(module_name, raw)
        self.check("create_bound_logger.type", type(wrapped).__name__)
        self.check(
            "create_bound_logger.type_matches",
            type(wrapped).__name__ == "FlextLogger",
        )

    def demo_global_context(self) -> None:
        """Exercise global context bind, unbind, and clear."""
        self.section("global_context")

        logger = FlextLogger.create_module_logger(f"examples.ex_03.{self.rand_str(6)}")
        app_name = self.rand_str(10)
        correlation_id = self.rand_str(12)
        log_message = self.rand_str(14)
        self.check(
            "bind_global_context.ok",
            FlextLogger.bind_global_context(
                app_name=app_name,
                correlation_id=correlation_id,
            ).is_success,
        )
        self.check("global.info.ok", logger.info(log_message).is_success)
        self.check(
            "unbind_global_context.ok",
            FlextLogger.unbind_global_context("correlation_id").is_success,
        )
        self.check(
            "clear_global_context.ok", FlextLogger.clear_global_context().is_success
        )

    def demo_scoped_context(self) -> None:
        """Exercise scoped context management."""
        self.section("scoped_context")

        logger = FlextLogger.create_module_logger(f"examples.ex_03.{self.rand_str(6)}")
        application_scope = c.Context.SCOPE_APPLICATION
        request_scope = c.Context.SCOPE_REQUEST
        operation_scope = c.Context.SCOPE_OPERATION
        app_tag = self.rand_str(7)
        request_id = self.rand_str(9)
        operation_name = self.rand_str(8)
        tenant = self.rand_str(6)
        scoped_msg = self.rand_str(16)

        self.check(
            "bind_context.application.ok",
            FlextLogger.bind_context(application_scope, app=app_tag).is_success,
        )
        self.check(
            "bind_context.request.ok",
            FlextLogger.bind_context(request_scope, request_id=request_id).is_success,
        )
        self.check(
            "bind_context.operation.ok",
            FlextLogger.bind_context(
                operation_scope, operation=operation_name
            ).is_success,
        )
        with FlextLogger.scoped_context(request_scope, tenant=tenant):
            self.check(
                "scoped_context.info.ok",
                logger.info(scoped_msg).is_success,
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

    def demo_level_context(self) -> None:
        """Exercise level-specific context binding."""
        self.section("level_context")

        logger = FlextLogger.create_module_logger(f"examples.ex_03.{self.rand_str(6)}")
        level_tag = self.rand_str(5)
        info_msg = self.rand_str(15)
        self.check(
            "bind_context_for_level.ok",
            FlextLogger.bind_context_for_level("INFO", level_tag=level_tag).is_success,
        )
        self.check("level.info.ok", logger.info(info_msg).is_success)
        self.check(
            "unbind_context_for_level.ok",
            FlextLogger.unbind_context_for_level("INFO", "level_tag").is_success,
        )

    def demo_container_integration(self) -> None:
        """Exercise container integration methods."""
        self.section("container")

        container_level = "DEBUG" if self.rand_bool() else "INFO"
        container_di: p.DI = FlextContainer()
        worker = self.rand_str(5)
        logger = FlextLogger.for_container(
            container_di,
            level=container_level,
            worker=worker,
        )
        self.check("for_container.type", type(logger).__name__)
        self.check("for_container.type_matches", type(logger).__name__ == "FlextLogger")
        debug_msg = self.rand_str(17)
        self.check("for_container.debug.ok", logger.debug(debug_msg).is_success)
        feature = self.rand_str(7)
        info_msg = self.rand_str(13)
        with FlextLogger.with_container_context(
            container_di,
            level="INFO",
            feature=feature,
        ):
            self.check(
                "with_container_context.info.ok",
                logger.info(info_msg).is_success,
            )

    def demo_instance_methods(self) -> None:
        """Exercise instance bind, unbind, logging, and exception methods."""
        self.section("instance_methods")

        logger = FlextLogger.create_module_logger(f"examples.ex_03.{self.rand_str(6)}")
        component = self.rand_str(8)
        bound = logger.bind(component=component)
        self.check("bind.type", type(bound).__name__)
        self.check("bind.type_matches", type(bound).__name__ == "FlextLogger")

        stage = self.rand_str(5)
        renewed = bound.new(stage=stage)
        self.check("new.type", type(renewed).__name__)
        self.check("new.type_matches", type(renewed).__name__ == "FlextLogger")

        unbound = renewed.unbind("stage")
        self.check("unbind.type", type(unbound).__name__)
        self.check("unbind.type_matches", type(unbound).__name__ == "FlextLogger")

        safe = unbound.try_unbind(self.rand_str(6), "component")
        self.check("try_unbind.type", type(safe).__name__)
        self.check("try_unbind.type_matches", type(safe).__name__ == "FlextLogger")

        adapter = safe.with_result()
        self.check("with_result.type", type(adapter).__name__)
        self.check(
            "with_result.type_matches", type(adapter).__name__ == "ResultAdapter"
        )

        trace_arg = self.rand_int(1, 99)
        debug_arg = self.rand_int(1, 99)
        info_arg = self.rand_int(1, 99)
        warning_arg = self.rand_int(1, 99)
        warn_arg = self.rand_int(1, 99)
        error_arg = self.rand_int(1, 99)
        critical_arg = self.rand_int(1, 99)
        log_arg = self.rand_int(1, 99)
        trace_key = self.rand_str(3)
        debug_key = self.rand_str(3)
        info_key = self.rand_str(3)
        warning_key = self.rand_str(3)
        warn_key = self.rand_str(3)
        error_key = self.rand_str(3)
        critical_key = self.rand_str(3)
        log_key = self.rand_str(3)
        self.check(
            "trace.ok",
            safe.trace("trace value=%s", trace_arg, key=trace_key).is_success,
        )
        self.check(
            "debug.ok",
            safe.debug("debug value=%s", debug_arg, key=debug_key).is_success,
        )
        self.check(
            "info.ok",
            safe.info("info value=%s", info_arg, key=info_key).is_success,
        )
        self.check(
            "warning.ok",
            safe.warning("warn value=%s", warning_arg, key=warning_key).is_success,
        )
        self.check(
            "warn.ok",
            safe.warn("warn alias value=%s", warn_arg, key=warn_key).is_success,
        )
        self.check(
            "error.ok",
            safe.error("error value=%s", error_arg, key=error_key).is_success,
        )
        self.check(
            "critical.ok",
            safe.critical(
                "critical value=%s", critical_arg, key=critical_key
            ).is_success,
        )
        self.check(
            "log.ok",
            safe.log("INFO", "log value=%s", log_arg, key=log_key).is_success,
        )

        try:
            boom_msg = self.rand_str(8)
            raise ValueError(boom_msg)
        except ValueError as err:
            source = self.rand_str(7)
            step = self.rand_str(6)
            exception_message = self.rand_str(15)
            context_map = safe.build_exception_context(
                exception=err,
                exc_info=True,
                context={
                    "source": source,
                    "step": step,
                },
            )
            self.check("build_exception_context.type", type(context_map).__name__)
            self.check(
                "build_exception_context.type_matches",
                type(context_map).__name__ == "dict",
            )
            self.check(
                "exception.ok",
                safe.exception(
                    exception_message,
                    exception=err,
                    exc_info=True,
                    source=source,
                ).is_success,
            )

    def demo_performance_tracker(self) -> None:
        """Exercise PerformanceTracker context manager."""
        self.section("performance_tracker")

        logger = FlextLogger.create_module_logger(f"examples.ex_03.{self.rand_str(6)}")
        op_name = self.rand_str(10)
        info_msg = self.rand_str(14)
        with FlextLogger.PerformanceTracker(logger, op_name):
            self.check("performance_tracker.body.ok", logger.info(info_msg).is_success)

    def demo_result_adapter(self) -> None:
        """Exercise ResultAdapter wrapper methods."""
        self.section("result_adapter")

        logger = FlextLogger.create_module_logger(f"examples.ex_03.{self.rand_str(6)}")
        adapter = logger.with_result()
        self.check("adapter.name", adapter.name)
        self.check("adapter.with_result.idempotent", adapter.with_result() is adapter)

        adapter_key = self.rand_str(5)
        rebound = adapter.bind(adapter_key=adapter_key)
        self.check("adapter.bind.type", type(rebound).__name__)
        self.check(
            "adapter.bind.type_matches", type(rebound).__name__ == "ResultAdapter"
        )

        self.check("adapter.trace.ok", adapter.trace(self.rand_str(13)).is_success)
        self.check("adapter.debug.ok", adapter.debug(self.rand_str(13)).is_success)
        self.check("adapter.info.ok", adapter.info(self.rand_str(13)).is_success)
        self.check("adapter.warning.ok", adapter.warning(self.rand_str(13)).is_success)
        self.check("adapter.error.ok", adapter.error(self.rand_str(13)).is_success)
        self.check(
            "adapter.critical.ok", adapter.critical(self.rand_str(13)).is_success
        )

        try:
            adapter_msg = self.rand_str(10)
            raise RuntimeError(adapter_msg)
        except RuntimeError as err:
            self.check(
                "adapter.exception.ok",
                adapter.exception(
                    self.rand_str(12),
                    exception=err,
                    exc_info=True,
                ).is_success,
            )


if __name__ == "__main__":
    Ex03FlextLogger(__file__).run()
