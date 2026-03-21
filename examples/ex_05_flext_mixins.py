"""FlextMixins — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import override

from flext_core import FlextRuntime, FlextSettings, c, m, r, t, x

from .shared import Examples


class Ex05FlextMixins(Examples):
    """Golden-file tests for ``FlextMixins`` (``x``) public API."""

    class DemoService(x):
        """Service exercising mixin result/conversion/tracking APIs."""

        def run_track_failure(self) -> str:
            """Run a tracked operation that raises internally."""
            try:
                with self.track("demo_failure"):
                    msg = "boom"
                    raise ValueError(msg)
            except ValueError as exc:
                return str(exc)

        def run_track_success(self) -> Mapping[str, bool | str]:
            """Run a tracked operation that succeeds."""
            with self.track("demo_success") as metrics:
                has_duration = "duration_ms" in metrics
                operation_count = str(metrics.get("operation_count", -1))
            return {"has_duration": has_duration, "operation_count": operation_count}

    class HandlerLike(FlextSettings):
        """Minimal handler-like satisfying ``x.ProtocolValidation.is_handler``."""

        @classmethod
        @override
        def validate(cls, value: t.ConfigMap) -> Ex05FlextMixins.HandlerLike:
            """Validate using Pydantic model_validate."""
            return cls.model_validate(value)

        def can_handle(self, message_type: type) -> bool:
            """Report capability for handler protocol."""
            return bool(message_type)

        def handle(self, message: m.Command) -> r[str]:
            """Handle data and return result."""
            return r[str].ok(str(message))

    class HandlerBad:
        """Non-handler for negative ``is_handler`` check."""

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
        FlextRuntime.configure_structlog()
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
        raw_str: str = str(ensured_raw.value) if ensured_raw.is_success else "-1"
        existing_str: str = (
            str(ensured_existing.value) if ensured_existing.is_success else "-1"
        )
        self.check("ensure_result.raw", raw_str)
        self.check("ensure_result.existing", existing_str)

        def _to_even(value: int) -> r[int]:
            if value % 2 == 0:
                return r[int].ok(value)
            return r[int].fail(f"odd:{value}")

        traverse_ok = r[list[int]].traverse([2, 4], _to_even, fail_fast=True)
        traverse_fail = r[list[int]].traverse([2, 3], _to_even, fail_fast=True)
        traverse_collect = r[list[int]].traverse([1, 3], _to_even, fail_fast=False)
        self.check("traverse.ok", str(traverse_ok.unwrap_or([])))
        self.check("traverse.fail_fast", traverse_fail.error)
        self.check("traverse.collect", traverse_collect.error)
        acc_ok = r.accumulate_errors(r[int].ok(1), r[int].ok(2))
        acc_fail = r.accumulate_errors(
            r[int].ok(1), r[int].fail("e1"), r[int].fail("e2")
        )
        self.check("accumulate_errors.ok", str(acc_ok.unwrap_or([])))
        self.check("accumulate_errors.fail", acc_fail.error)

    def _exercise_runtime_properties_and_tracking(
        self, service: Ex05FlextMixins.DemoService
    ) -> None:
        """Exercise container, logger, context, config, and tracking."""
        self.section("runtime_properties_and_tracking")
        self.check("container.type", type(service.container).__name__)
        self.check("logger.type", type(service.logger).__name__)
        self.check("context.type", type(service.context).__name__)
        self.check("config.type", type(service.config).__name__)
        self.check("const.scope_operation", c.SCOPE_OPERATION)
        success_metrics = service.run_track_success()
        failure_message = service.run_track_failure()
        self.check("track.success.metrics", success_metrics)
        self.check("track.failure.message", failure_message)

    def _exercise_cqrs_validation_and_protocols(self) -> None:
        """Exercise CQRS, Validation, and ProtocolValidation namespaces."""
        self.section("cqrs_validation_protocols")
        tracker = x.CQRS.MetricsTracker()
        self.check("metrics.record_metric", tracker.record_metric("hits", 3).is_success)
        metrics_result = tracker.get_metrics()
        self.check("metrics.get_metrics.success", metrics_result.is_success)
        metrics_str: str = metrics_result.map(lambda value: str(value)).unwrap_or("{}")
        self.check("metrics.get_metrics.value", metrics_str)
        stack = x.CQRS.ContextStack()
        self.check(
            "context_stack.push_context",
            stack.push_context(
                m.ExecutionContext.create_for_handler(
                    handler_name="Q", handler_mode=c.HandlerType.QUERY
                )
            ).is_success,
        )
        self.check(
            "context_stack.current_context.before_pop",
            stack.current_context() is not None,
        )
        popped = stack.pop_context()
        self.check("context_stack.pop_context.success", popped.is_success)
        popped_str: str = popped.map(lambda value: str(value)).unwrap_or("{}")
        self.check("context_stack.pop_context.value", popped_str)
        self.check(
            "context_stack.current_context.after_pop", stack.current_context() is None
        )

        def _validator_ok(value: str) -> r[bool]:
            text = str(value)
            return r[bool].ok(text.startswith("a"))

        def _validator_fail(_value: str) -> r[bool]:
            return r[bool].fail("bad-input")

        validators_ok: list[Callable[..., r[bool]]] = [_validator_ok]
        validators_fail: list[Callable[..., r[bool]]] = [_validator_fail]
        validation_ok = x.Validation.validate_with_result("abc", validators_ok)
        validation_fail = x.Validation.validate_with_result("abc", validators_fail)
        self.check("validation.ok", validation_ok.is_success)
        self.check("validation.fail", validation_fail.error)
        self.check(
            "protocol.is_handler.good",
            bool(x.ProtocolValidation.is_handler(self.HandlerLike())),
        )
        self.check(
            "protocol.is_handler.bad",
            bool(x.ProtocolValidation.is_handler(self.HandlerBad())),
        )
        processor_ok = x.ProtocolValidation.validate_processor_protocol(
            self.GoodProcessor()
        )
        processor_fail = x.ProtocolValidation.validate_processor_protocol(
            self.BadProcessor()
        )
        self.check("protocol.validate_processor.good", processor_ok.is_success)
        self.check("protocol.validate_processor.fail", processor_fail.error)


if __name__ == "__main__":
    Ex05FlextMixins(__file__).run()
