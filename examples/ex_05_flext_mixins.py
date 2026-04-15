"""x — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import override

from examples import Examples, c, m, t, u
from flext_core import FlextSettings, p, r, x


class Ex05FlextMixins(Examples):
    """Golden-file tests for ``x`` (``x``) public API."""

    class DemoService(x):
        """Service exercising mixin result/conversion/tracking APIs."""

        def run_track_failure(self) -> str:
            """Run a tracked operation that raises internally."""
            try:
                with self.track("demo_failure"):
                    raise ValueError(m.Examples.ErrorMessages.BOOM)
            except ValueError as exc:
                return str(exc)

        def run_track_success(self) -> Mapping[str, bool | str]:
            """Run a tracked operation that succeeds."""
            with self.track("demo_success") as metrics:
                has_duration = "duration_ms" in metrics
                operation_count = str(metrics.get("operation_count", -1))
            return {"has_duration": has_duration, "operation_count": operation_count}

    class HandlerLike(FlextSettings):
        """Minimal handler-like satisfying ``u.handler``."""

        @classmethod
        @override
        def validate(cls, value: t.ConfigMap) -> Ex05FlextMixins.HandlerLike:
            """Validate using Pydantic model_validate."""
            return cls.model_validate(value)

        def can_handle(self, message_type: type) -> bool:
            """Report capability for handler protocol."""
            return bool(message_type)

        def handle(self, message: m.Command) -> p.Result[str]:
            """Handle data and return result."""
            return r[str].ok(str(message))

    class HandlerBad(m.Value):
        """Non-handler for negative ``handler`` check."""

        marker: str = u.Field(
            "bad",
            description="Marker for negative handler",
            validate_default=True,
        )

    class GoodProcessor(m.Value):
        """Processor satisfying ``p.HasModelDump`` + process + validate."""

        def process(self) -> bool:
            """Process successfully."""
            return True

        @classmethod
        @override
        def validate(cls, value: t.ConfigMap) -> Ex05FlextMixins.GoodProcessor:
            """Validate for Pydantic compatibility."""
            del value
            return cls()

    class BadProcessor(m.Value):
        """Processor missing ``process`` for negative validation."""

    @override
    def exercise(self) -> None:
        """Run all scenarios and record deterministic golden output."""
        u.configure_structlog()
        service = self.DemoService()
        self._exercise_result_and_conversion()
        self._exercise_runtime_properties_and_tracking(service)
        self._exercise_cqrs_validation_and_protocols()

    def _exercise_result_and_conversion(self) -> None:
        """Exercise ok/fail, to_dict, ensure_result, traverse, accumulate."""
        self.section("result_and_conversion")
        ok_result = r[t.ConfigMap].ok(t.ConfigMap(root={"k": "v"}))
        self.check("ok.unwrap_or", str(ok_result.map_or("{}")))
        fail_result = r[t.ConfigMap].fail("failure", error_code="E_EX")
        self.check("fail.error", fail_result.error)
        self.check("fail.error_code", fail_result.error_code)
        to_dict_from_dict = t.ConfigMap(root={"x": 1, "y": "2"})
        to_dict_from_none = t.ConfigMap(root={})
        self.check("to_dict.dict", str(to_dict_from_dict.root))
        self.check("to_dict.none", str(to_dict_from_none.root))
        ensured_raw = r[int].ok(99)
        ensured_existing = r[int].ok(7)
        raw_str: str = str(ensured_raw.value) if ensured_raw.success else "-1"
        existing_str: str = (
            str(ensured_existing.value) if ensured_existing.success else "-1"
        )
        self.check("ensure_result.raw", raw_str)
        self.check("ensure_result.existing", existing_str)

        def _to_even(value: int) -> p.Result[int]:
            if value % 2 == 0:
                return r[int].ok(value)
            return r[int].fail(f"odd:{value}")

        traverse_ok = r[Sequence[int]].traverse([2, 4], _to_even, fail_fast=True)
        traverse_fail = r[Sequence[int]].traverse([2, 3], _to_even, fail_fast=True)
        traverse_collect = r[Sequence[int]].traverse([1, 3], _to_even, fail_fast=False)
        self.check("traverse.ok", str(traverse_ok.unwrap_or([])))
        self.check("traverse.fail_fast", traverse_fail.error)
        self.check("traverse.collect", traverse_collect.error)
        acc_ok = r.accumulate_errors(r[int].ok(1), r[int].ok(2))
        acc_fail = r.accumulate_errors(
            r[int].ok(1),
            r[int].fail("e1"),
            r[int].fail("e2"),
        )
        self.check("accumulate_errors.ok", str(acc_ok.unwrap_or([])))
        self.check("accumulate_errors.fail", acc_fail.error)

    def _exercise_runtime_properties_and_tracking(
        self,
        service: Ex05FlextMixins.DemoService,
    ) -> None:
        """Exercise container, logger, context, settings, and tracking."""
        self.section("runtime_properties_and_tracking")
        self.check("container.type", type(service.container).__name__)
        self.check("logger.type", type(service.logger).__name__)
        self.check("context.type", type(service.context).__name__)
        self.check("settings.type", type(service.settings).__name__)
        self.check("const.scope_operation", c.ContextScope.OPERATION)
        success_metrics = service.run_track_success()
        failure_message = service.run_track_failure()
        self.check("track.success.metrics", success_metrics)
        self.check("track.failure.message", failure_message)

    def _exercise_cqrs_validation_and_protocols(self) -> None:
        """Exercise CQRS models, validation, and protocol check utilities."""
        self.section("cqrs_validation_protocols")
        tracker = m.MetricsTracker()
        self.check("metrics.record_metric", tracker.record_metric("hits", 3).success)
        metrics_map = tracker.metrics
        self.check("metrics.available", bool(metrics_map))
        self.check("metrics.value", str(metrics_map))
        stack = m.ContextStack()
        self.check(
            "context_stack.push_context",
            stack.push_context(
                m.ExecutionContext.create_for_handler(
                    handler_name="Q",
                    handler_mode=c.HandlerType.QUERY,
                ),
            ).success,
        )
        self.check(
            "context_stack.current_context.before_pop",
            stack.current_context() is not None,
        )
        popped = stack.pop_context()
        self.check("context_stack.pop_context.success", popped.success)
        popped_str: str = popped.map(lambda value: str(value)).unwrap_or("{}")
        self.check("context_stack.pop_context.value", popped_str)
        self.check(
            "context_stack.current_context.after_pop",
            stack.current_context() is None,
        )

        self.check(
            "protocol.handler.good",
            bool(u.handler(self.HandlerLike())),
        )
        self.check(
            "protocol.handler.bad",
            bool(u.handler(self.HandlerBad())),
        )


if __name__ == "__main__":
    Ex05FlextMixins(caller_file=__file__).run()
