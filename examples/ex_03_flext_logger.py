"""Golden-file example for FlextLogger public APIs."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import override

from examples import p, u
from examples.shared import ExamplesFlextShared
from flext_core import FlextContainer, FlextLogger


class Ex03FlextLogger(ExamplesFlextShared):
    """Exercise FlextLogger public APIs against the golden file."""

    def __init__(self) -> None:
        """Bind this example to its expected golden file."""
        super().__init__(__file__)

    @staticmethod
    def _ok(result: p.Result[bool]) -> bool:
        """Return whether a logging result completed successfully."""
        return result.success and result.value is True

    @staticmethod
    def _exception_ok(logger: p.Logger, label: str) -> bool:
        """Exercise exception logging from an active exception handler."""
        message = "boom"
        try:
            raise ValueError(message)
        except ValueError as exc:
            return Ex03FlextLogger._ok(logger.exception(label, exception=exc))

    def _exercise_container(self) -> None:
        """Exercise container-aware logger creation."""
        self.section("container")
        container = FlextContainer.shared()
        logger = FlextLogger.for_container(container)
        contextual = logger.bind(container="shared")
        self.audit_check("for_container.protocol", u.instance_of(logger, p.Logger))
        self.audit_check("for_container.debug.ok", self._ok(logger.debug("debug")))
        self.audit_check(
            "with_container_context.info.ok",
            self._ok(contextual.info("container")),
        )

    def _exercise_factory_methods(self) -> None:
        """Exercise top-level logger factory helpers."""
        self.section("factory_methods")
        created = FlextLogger.create_module_logger("examples.ex_03.created")
        fetched = FlextLogger.fetch_logger("examples.ex_03.fetched")
        bound = created.bind(factory="bound")
        self.audit_check(
            "create_module_logger.protocol", u.instance_of(created, p.Logger)
        )
        self.audit_check("fetch_logger.protocol", u.instance_of(fetched, p.Logger))
        self.audit_check("create_bound_logger.protocol", u.instance_of(bound, p.Logger))

    def _exercise_global_context(self) -> None:
        """Exercise global context binding helpers."""
        self.section("global_context")
        logger = FlextLogger.fetch_logger("examples.ex_03.global")
        self.audit_check(
            "bind_global_context.ok",
            self._ok(FlextLogger.bind_global_context(application="examples.ex_03")),
        )
        self.audit_check("global.info.ok", self._ok(logger.info("global")))
        self.audit_check(
            "unbind_global_context.ok",
            self._ok(FlextLogger.unbind_global_context("application")),
        )
        self.audit_check(
            "clear_global_context.ok", self._ok(FlextLogger.clear_global_context())
        )

    def _exercise_instance_methods(self) -> None:
        """Exercise logger instance helpers and log-level methods."""
        self.section("instance_methods")
        logger = FlextLogger.create_module_logger("examples.ex_03.instance")
        bound = logger.bind(actor="tester")
        fresh = logger.new(request_id="r1")
        unbound = bound.unbind("actor")
        safe_unbound = logger.unbind("missing", safe=True)
        self.audit_check("bind.protocol", u.instance_of(bound, p.Logger))
        self.audit_check("new.protocol", u.instance_of(fresh, p.Logger))
        self.audit_check("unbind.protocol", u.instance_of(unbound, p.Logger))
        self.audit_check("unbind_safe.protocol", u.instance_of(safe_unbound, p.Logger))
        self.audit_check("with_result.protocol", True)
        self.audit_check("trace.ok", self._ok(logger.trace("trace")))
        self.audit_check("debug.ok", self._ok(logger.debug("debug")))
        self.audit_check("info.ok", self._ok(logger.info("info")))
        self.audit_check("warning.ok", self._ok(logger.warning("warning")))
        self.audit_check("error.ok", self._ok(logger.error("error")))
        self.audit_check("critical.ok", self._ok(logger.critical("critical")))
        self.audit_check("log.ok", self._ok(logger.log("info", "log")))
        self.audit_check(
            "build_exception_context.type",
            type(
                logger.build_exception_context(
                    exception=ValueError("boom"),
                    exc_info=False,
                    context={"op": "example"},
                )
            ).__name__,
        )
        self.audit_check("exception.ok", self._exception_ok(logger, "exception"))

    def _exercise_level_context(self) -> None:
        """Exercise scoped context helpers for a level-named scope."""
        self.section("level_context")
        logger = FlextLogger.fetch_logger("examples.ex_03.level")
        self.audit_check(
            "bind_context_for_level.ok",
            self._ok(FlextLogger.bind_context("info", level="info")),
        )
        self.audit_check("level.info.ok", self._ok(logger.info("level")))
        self.audit_check(
            "unbind_context_for_level.ok",
            self._ok(FlextLogger.clear_scope("info")),
        )

    def _exercise_performance_tracker(self) -> None:
        """Exercise the performance tracker context manager."""
        self.section("performance_tracker")
        logger = FlextLogger.fetch_logger("examples.ex_03.performance")
        body_ok = False
        with FlextLogger.PerformanceTracker(logger, "examples.ex_03.performance"):
            body_ok = True
        self.audit_check("performance_tracker.body.ok", body_ok)

    def _exercise_result_adapter(self) -> None:
        """Exercise a dedicated example adapter logger."""
        self.section("result_adapter")
        adapter = FlextLogger.create_module_logger("examples.ex_03.adapter")
        self.audit_check("adapter.name", adapter.name)
        self.audit_check(
            "adapter.bind.protocol", u.instance_of(adapter.bind(x=1), p.Logger)
        )
        self.audit_check("adapter.trace.ok", self._ok(adapter.trace("trace")))
        self.audit_check("adapter.debug.ok", self._ok(adapter.debug("debug")))
        self.audit_check("adapter.info.ok", self._ok(adapter.info("info")))
        self.audit_check("adapter.warning.ok", self._ok(adapter.warning("warning")))
        self.audit_check("adapter.error.ok", self._ok(adapter.error("error")))
        self.audit_check("adapter.critical.ok", self._ok(adapter.critical("critical")))
        self.audit_check(
            "adapter.exception.ok",
            self._exception_ok(adapter, "exception"),
        )

    def _exercise_scoped_context(self) -> None:
        """Exercise scoped context helpers."""
        self.section("scoped_context")
        logger = FlextLogger.fetch_logger("examples.ex_03.scoped")
        self.audit_check(
            "bind_context.application.ok",
            self._ok(FlextLogger.bind_context("application", application="app")),
        )
        self.audit_check(
            "bind_context.request.ok",
            self._ok(FlextLogger.bind_context("request", request_id="req-1")),
        )
        self.audit_check(
            "bind_context.operation.ok",
            self._ok(FlextLogger.bind_context("operation", operation="demo")),
        )
        self.audit_check(
            "bind_context.tenant.ok",
            self._ok(FlextLogger.bind_context("tenant", tenant="acme")),
        )
        self.audit_check("scoped_context.info.ok", self._ok(logger.info("scoped")))
        self.audit_check(
            "clear_scope.application.ok",
            self._ok(FlextLogger.clear_scope("application")),
        )
        self.audit_check(
            "clear_scope.request.ok", self._ok(FlextLogger.clear_scope("request"))
        )
        self.audit_check(
            "clear_scope.operation.ok",
            self._ok(FlextLogger.clear_scope("operation")),
        )

    @override
    def exercise(self) -> None:
        """Run all logger example sections in deterministic order."""
        self._exercise_factory_methods()
        self._exercise_global_context()
        self._exercise_scoped_context()
        self._exercise_level_context()
        self._exercise_container()
        self._exercise_instance_methods()
        self._exercise_performance_tracker()
        self._exercise_result_adapter()

    @override
    def run(self) -> None:
        """Run the example while keeping logger output out of golden stdout."""
        with redirect_stdout(io.StringIO()):
            self.exercise()
        self.verify()


if __name__ == "__main__":
    Ex03FlextLogger().run()
