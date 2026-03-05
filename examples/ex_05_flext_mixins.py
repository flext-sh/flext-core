"""Golden-file example for FlextMixins public APIs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast, override

from flext_core import FlextRuntime, FlextSettings, c, r, t, u, x

_RESULTS: list[str] = []


def _check(label: str, value: t.Serializable) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: t.Serializable) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int | float):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if u.is_list(v):
        return repr(v)
    if u.is_dict_like(v):
        return repr(v)
    return repr(v)


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    checks = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))

    if not expected_path.exists():
        expected_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(f"PASS: {me.stem} ({checks} checks, generated)\n")
        return

    expected = expected_path.read_text(encoding="utf-8")
    if actual == expected:
        sys.stdout.write(f"PASS: {me.stem} ({checks} checks)\n")
        return

    actual_path = me.with_suffix(".actual")
    actual_path.write_text(actual, encoding="utf-8")
    sys.stdout.write(f"FAIL: {me.stem} ({expected_path.name} != {actual_path.name})\n")
    sys.exit(1)


class _DemoService(x):
    def run_track_success(self) -> dict[str, t.Serializable]:
        with self.track("demo_success") as metrics:
            has_duration = "duration_ms" in metrics
            operation_count = str(metrics.get("operation_count", -1))
        return {
            "has_duration": has_duration,
            "operation_count": operation_count,
        }

    def run_track_failure(self) -> str:
        try:
            with self.track("demo_failure"):
                msg = "boom"
                raise ValueError(msg)
        except ValueError as exc:
            return str(exc)


class _HandlerLike(FlextSettings):
    def handle(self, _data: object) -> object:
        return _data

    @classmethod
    @override
    def validate(cls, value: object) -> _HandlerLike:
        return cls.model_validate(value)


class _HandlerBad(FlextSettings):
    pass


class _ProtocolService:
    def _protocol_name(self) -> str:
        return "Service"

    def execute(self) -> r[dict[str, str]]:
        return r[dict[str, str]].ok({"ok": "yes"})

    def validate_business_rules(self) -> r[bool]:
        return r[bool].ok(True)

    def is_valid(self) -> bool:
        return True

    def get_service_info(self) -> dict[str, str]:
        return {"name": "protocol-service"}


class _GoodProcessor:
    def model_dump(self) -> dict[str, str]:
        return {"status": "ok"}

    def process(self) -> bool:
        return True

    def validate(self) -> bool:
        return True

    def _protocol_name(self) -> str:
        return "HasModelDump"


class _BadProcessor:
    def model_dump(self) -> dict[str, str]:
        return {"status": "bad"}

    def _protocol_name(self) -> str:
        return "HasModelDump"


def _exercise_result_and_conversion_apis(service: _DemoService) -> None:
    _section("result_and_conversion")

    ok_result = service.ok({"k": "v"})
    _check("ok.unwrap_or", str(ok_result.map_or("{}")))

    fail_result = service.fail(
        "failure",
        error_code="E_EX",
        error_data=service.to_dict({"step": 1}),
    )
    _check("fail.error", fail_result.error)
    _check("fail.error_code", fail_result.error_code)

    to_dict_from_dict = service.to_dict({"x": 1, "y": "2"})
    to_dict_from_none = service.to_dict(None)
    _check("to_dict.dict", str(to_dict_from_dict.root))
    _check("to_dict.none", str(to_dict_from_none.root))

    ensured_raw = service.ensure_result(99)
    ensured_existing = service.ensure_result(r[int].ok(7))
    raw_str: str = str(ensured_raw.value) if ensured_raw.is_success else "-1"
    existing_str: str = (
        str(ensured_existing.value) if ensured_existing.is_success else "-1"
    )
    _check("ensure_result.raw", raw_str)
    _check("ensure_result.existing", existing_str)

    def _to_even(value: int) -> r[int]:
        if value % 2 == 0:
            return r[int].ok(value)
        return r[int].fail(f"odd:{value}")

    traverse_ok = service.traverse([2, 4], _to_even, fail_fast=True)
    traverse_fail_fast = service.traverse([2, 3], _to_even, fail_fast=True)
    traverse_collect = service.traverse([1, 3], _to_even, fail_fast=False)
    _check("traverse.ok", str(traverse_ok.unwrap_or([])))
    _check("traverse.fail_fast", traverse_fail_fast.error)
    _check("traverse.collect", traverse_collect.error)

    acc_ok = service.accumulate_errors(r[int].ok(1), r[int].ok(2))
    acc_fail = service.accumulate_errors(
        r[int].ok(1), r[int].fail("e1"), r[int].fail("e2")
    )
    _check("accumulate_errors.ok", str(acc_ok.unwrap_or([])))
    _check("accumulate_errors.fail", acc_fail.error)


def _exercise_runtime_properties_and_tracking(service: _DemoService) -> None:
    _section("runtime_properties_and_tracking")

    _check("container.type", type(service.container).__name__)
    _check("logger.type", type(service.logger).__name__)
    _check("context.type", type(service.context).__name__)
    _check("config.is_flext_settings", isinstance(service.config, FlextSettings))
    _check("const.scope_operation", c.Context.SCOPE_OPERATION)

    success_metrics = service.run_track_success()
    failure_message = service.run_track_failure()
    _check("track.success.metrics", success_metrics)
    _check("track.failure.message", failure_message)


def _exercise_cqrs_validation_and_protocols(_service: _DemoService) -> None:
    _section("cqrs_validation_protocols")

    tracker = x.CQRS.MetricsTracker()
    _check("metrics.record_metric", tracker.record_metric("hits", 3).is_success)
    metrics_result = tracker.get_metrics()
    _check("metrics.get_metrics.success", metrics_result.is_success)
    metrics_value_str: str = (
        str(metrics_result.value.root) if metrics_result.is_success else "{}"
    )
    _check(
        "metrics.get_metrics.value",
        metrics_value_str,
    )

    stack = x.CQRS.ContextStack()
    _check(
        "context_stack.push_context",
        stack.push_context(
            cast("t.Container", {"handler_name": "Q", "handler_mode": "query"})
        ).is_success,
    )
    _check(
        "context_stack.current_context.before_pop", stack.current_context() is not None
    )
    popped = stack.pop_context()
    _check("context_stack.pop_context.success", popped.is_success)
    popped_value_str: str = (
        str(dict(popped.value.items()) if hasattr(popped.value, "items") else {})
        if popped.is_success
        else "{}"
    )
    _check(
        "context_stack.pop_context.value",
        popped_value_str,
    )
    _check("context_stack.current_context.after_pop", stack.current_context() is None)

    def _validator_ok(value: str) -> r[bool]:
        return r[bool].ok(value.startswith("a"))

    def _validator_fail(_value: str) -> r[bool]:
        return r[bool].fail("bad-input")

    validators_ok = [_validator_ok]
    validators_fail = [_validator_fail]
    validate_with_result_fn = getattr(x.Validation, "validate_with_result")
    validation_ok = validate_with_result_fn("abc", validators_ok)
    validation_fail = validate_with_result_fn("abc", validators_fail)
    _check("validation.ok", validation_ok.is_success)
    _check("validation.fail", validation_fail.error)

    protocol_service = _ProtocolService()
    is_handler_fn = getattr(x.ProtocolValidation, "is_handler")
    is_service_fn = getattr(x.ProtocolValidation, "is_service")
    validate_compliance_fn = getattr(
        x.ProtocolValidation, "validate_protocol_compliance"
    )
    validate_processor_fn = getattr(x.ProtocolValidation, "validate_processor_protocol")

    _check("protocol.is_handler.good", bool(is_handler_fn(_HandlerLike())))
    _check("protocol.is_handler.bad", bool(is_handler_fn(_HandlerBad())))
    _check("protocol.is_service", bool(is_service_fn(protocol_service)))

    compliance_ok = validate_compliance_fn(protocol_service, "Service")
    compliance_fail = validate_compliance_fn(protocol_service, "Unknown")
    _check("protocol.validate_compliance.ok", compliance_ok.is_success)
    _check("protocol.validate_compliance.fail", compliance_fail.error)

    processor_ok = validate_processor_fn(_GoodProcessor())
    processor_fail = validate_processor_fn(_BadProcessor())
    _check("protocol.validate_processor.good", processor_ok.is_success)
    _check("protocol.validate_processor.fail", processor_fail.error)


def main() -> None:
    """Run the full mixins golden-file exercise."""
    FlextRuntime.configure_structlog()
    service = _DemoService()

    _exercise_result_and_conversion_apis(service)
    _exercise_runtime_properties_and_tracking(service)
    _exercise_cqrs_validation_and_protocols(service)
    _verify()


if __name__ == "__main__":
    main()
