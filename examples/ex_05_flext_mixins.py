"""Golden-file example for FlextMixins public APIs."""

from __future__ import annotations

from shared import Examples

from flext_core import FlextRuntime, FlextSettings, c, r, t, x


class _DemoService(x):
    def run_track_success(self) -> dict[str, t.ContainerValue]:
        with self.track("demo_success") as metrics:
            has_duration = "duration_ms" in metrics
            operation_count = metrics.get("operation_count", -1)
        return {
            "has_duration": has_duration,
            "operation_count": operation_count,
        }

    def run_track_failure(self, message: str) -> str:
        try:
            with self.track("demo_failure"):
                raise ValueError(message)
        except ValueError as exc:
            return str(exc)


class _HandlerLike(FlextSettings):
    def handle(self, _data: t.ContainerValue) -> t.ContainerValue:
        return _data

    @classmethod
    def validate(cls, value: t.ContainerValue) -> _HandlerLike:
        return cls.model_validate(value)


class _HandlerBad(FlextSettings):
    pass


class _ProtocolService:
    def __init__(
        self,
        execute_payload: dict[str, str],
        valid: bool,
        service_name: str,
    ) -> None:
        self._execute_payload = execute_payload
        self._valid = valid
        self._service_name = service_name

    def _protocol_name(self) -> str:
        return "Service"

    def execute(self) -> r[dict[str, str]]:
        return r[dict[str, str]].ok(self._execute_payload)

    def validate_business_rules(self) -> r[bool]:
        return r[bool].ok(self._valid)

    def is_valid(self) -> bool:
        return self._valid

    def get_service_info(self) -> dict[str, str]:
        return {"name": self._service_name}


class _GoodProcessor:
    def __init__(self, status: str, valid: bool) -> None:
        self._status = status
        self._valid = valid

    def model_dump(self) -> dict[str, str]:
        return {"status": self._status}

    def process(self) -> bool:
        return self._valid

    def validate(self) -> bool:
        return self._valid

    def _protocol_name(self) -> str:
        return "HasModelDump"


class _BadProcessor:
    def __init__(self, status: str) -> None:
        self._status = status

    def model_dump(self) -> dict[str, str]:
        return {"status": self._status}

    def _protocol_name(self) -> str:
        return "HasModelDump"


class Ex05FlextMixins(Examples):
    """Exercise FlextMixins public APIs."""

    def exercise(self) -> None:
        """Run the full mixins golden-file exercise."""
        FlextRuntime.configure_structlog()
        service = _DemoService()
        self._exercise_result_and_conversion_apis(service)
        self._exercise_runtime_properties_and_tracking(service)
        self._exercise_cqrs_validation_and_protocols(service)

    def _exercise_result_and_conversion_apis(self, service: _DemoService) -> None:
        self.section("result_and_conversion")

        ok_key = self.rand_str(4)
        ok_value = self.rand_str(6)
        ok_payload = {ok_key: ok_value}
        ok_result = service.ok(ok_payload)
        self.check("ok.unwrap_or", ok_result.unwrap_or({}))
        self.check("ok.value_matches", ok_result.value == ok_payload)

        fail_message = self.rand_str(8)
        error_code = self.rand_str(6)
        error_data = service.to_dict(self.rand_dict())
        fail_result = service.fail(
            fail_message,
            error_code=error_code,
            error_data=error_data,
        )
        self.check("fail.error", fail_result.error)
        self.check("fail.error_code", fail_result.error_code)
        self.check("fail.error_matches", fail_result.error == fail_message)
        self.check("fail.error_code_matches", fail_result.error_code == error_code)

        dict_key_a = self.rand_str(4)
        dict_key_b = self.rand_str(4)
        dict_payload: dict[str, t.ContainerValue] = {
            dict_key_a: self.rand_int(1, 999),
            dict_key_b: self.rand_str(5),
        }
        to_dict_from_dict = service.to_dict(dict_payload)
        to_dict_from_none = service.to_dict(None)
        self.check("to_dict.dict", dict(to_dict_from_dict.root))
        self.check("to_dict.none", dict(to_dict_from_none.root))
        self.check("to_dict.dict_matches", dict(to_dict_from_dict.root) == dict_payload)
        self.check("to_dict.none_is_empty", dict(to_dict_from_none.root) == {})

        raw_number = self.rand_int(1, 999)
        existing_number = self.rand_int(1, 999)
        ensured_raw = service.ensure_result(raw_number)
        ensured_existing = service.ensure_result(r[int].ok(existing_number))
        self.check("ensure_result.raw", ensured_raw.unwrap_or(-1))
        self.check("ensure_result.existing", ensured_existing.unwrap_or(-1))
        self.check("ensure_result.raw_matches", ensured_raw.unwrap_or(-1) == raw_number)
        self.check(
            "ensure_result.existing_matches",
            ensured_existing.unwrap_or(-1) == existing_number,
        )

        def _to_even(value: int) -> r[int]:
            if value % 2 == 0:
                return r[int].ok(value)
            return r[int].fail(f"odd:{value}")

        even_a = self.rand_int(1, 250) * 2
        even_b = self.rand_int(1, 250) * 2
        odd_a = self.rand_int(0, 249) * 2 + 1
        odd_b = self.rand_int(0, 249) * 2 + 1
        traverse_ok = service.traverse([even_a, even_b], _to_even, fail_fast=True)
        traverse_fail_fast = service.traverse([even_a, odd_a], _to_even, fail_fast=True)
        traverse_collect = service.traverse([odd_a, odd_b], _to_even, fail_fast=False)
        self.check("traverse.ok", list(traverse_ok.unwrap_or([])))
        self.check("traverse.fail_fast", traverse_fail_fast.error)
        self.check("traverse.collect", traverse_collect.error)
        self.check(
            "traverse.ok_matches",
            list(traverse_ok.unwrap_or([])) == [even_a, even_b],
        )
        self.check(
            "traverse.fail_fast_matches",
            traverse_fail_fast.error == f"odd:{odd_a}",
        )
        self.check(
            "traverse.collect_has_both_errors",
            traverse_collect.error == [f"odd:{odd_a}", f"odd:{odd_b}"],
        )

        acc_a = self.rand_int(1, 99)
        acc_b = self.rand_int(1, 99)
        acc_e1 = self.rand_str(5)
        acc_e2 = self.rand_str(5)
        acc_ok = service.accumulate_errors(r[int].ok(acc_a), r[int].ok(acc_b))
        acc_fail = service.accumulate_errors(
            r[int].ok(acc_a), r[int].fail(acc_e1), r[int].fail(acc_e2)
        )
        self.check("accumulate_errors.ok", list(acc_ok.unwrap_or([])))
        self.check("accumulate_errors.fail", acc_fail.error)
        self.check(
            "accumulate_errors.ok_matches",
            list(acc_ok.unwrap_or([])) == [acc_a, acc_b],
        )
        self.check(
            "accumulate_errors.fail_matches",
            acc_fail.error == [acc_e1, acc_e2],
        )

    def _exercise_runtime_properties_and_tracking(self, service: _DemoService) -> None:
        self.section("runtime_properties_and_tracking")

        self.check("container.type", type(service.container).__name__)
        self.check(
            "logger.is_flext_logger", isinstance(service.logger, type(service.logger))
        )
        self.check("context.type", type(service.context).__name__)
        self.check(
            "config.is_flext_settings", isinstance(service.config, FlextSettings)
        )
        self.check("const.scope_operation", c.Context.SCOPE_OPERATION)

        success_metrics = service.run_track_success()
        failure_message_expected = self.rand_str(8)
        failure_message = service.run_track_failure(failure_message_expected)
        self.check("track.success.metrics", success_metrics)
        self.check("track.failure.message", failure_message)
        self.check("track.failure.matches", failure_message == failure_message_expected)

    def _exercise_cqrs_validation_and_protocols(self, _service: _DemoService) -> None:
        self.section("cqrs_validation_protocols")

        tracker = x.CQRS.MetricsTracker()
        metric_name = self.rand_str(5)
        metric_value = self.rand_int(1, 50)
        self.check(
            "metrics.record_metric",
            tracker.record_metric(metric_name, metric_value).is_success,
        )
        metrics_result = tracker.get_metrics()
        self.check("metrics.get_metrics.success", metrics_result.is_success)
        metrics_val = (
            metrics_result.value
            if metrics_result.is_success and metrics_result.value is not None
            else _service.to_dict({})
        )
        self.check(
            "metrics.get_metrics.value",
            dict(metrics_val.root)
            if hasattr(metrics_val, "root")
            else dict(metrics_val),
        )
        metrics_dict = (
            dict(metrics_val.root)
            if hasattr(metrics_val, "root")
            else dict(metrics_val)
        )
        self.check(
            "metrics.get_metrics.matches",
            metrics_dict.get(metric_name) == metric_value,
        )

        stack = x.CQRS.ContextStack()
        handler_name = self.rand_str(4)
        self.check(
            "context_stack.push_context",
            stack.push_context({
                "handler_name": handler_name,
                "handler_mode": "query",
            }).is_success,
        )
        self.check(
            "context_stack.current_context.before_pop",
            stack.current_context() is not None,
        )
        popped = stack.pop_context()
        self.check("context_stack.pop_context.success", popped.is_success)
        popped_val: t.ContainerValue = (
            popped.value if popped.is_success and popped.value is not None else {}
        )
        self.check(
            "context_stack.pop_context.value",
            dict(popped_val.root) if hasattr(popped_val, "root") else dict(popped_val),
        )
        popped_dict = (
            dict(popped_val.root) if hasattr(popped_val, "root") else dict(popped_val)
        )
        self.check(
            "context_stack.pop_context.matches",
            popped_dict == {"handler_name": handler_name, "handler_mode": "query"},
        )
        self.check(
            "context_stack.current_context.after_pop", stack.current_context() is None
        )

        starts_prefix = self.rand_str(1)

        def _validator_ok(value: str) -> r[bool]:
            return r[bool].ok(value.startswith(starts_prefix))

        bad_input_message = self.rand_str(9)

        def _validator_fail(_value: str) -> r[bool]:
            return r[bool].fail(bad_input_message)

        validators_ok = [_validator_ok]
        validators_fail = [_validator_fail]
        validate_with_result_fn = getattr(x.Validation, "validate_with_result")
        validated_value = starts_prefix + self.rand_str(4)
        validation_ok = validate_with_result_fn(validated_value, validators_ok)
        validation_fail = validate_with_result_fn(validated_value, validators_fail)
        self.check("validation.ok", validation_ok.is_success)
        self.check("validation.fail", validation_fail.error)
        self.check("validation.ok_matches", validation_ok.is_success)
        self.check(
            "validation.fail_matches", validation_fail.error == bad_input_message
        )

        protocol_service = _ProtocolService(
            execute_payload={self.rand_str(3): self.rand_str(3)},
            valid=self.rand_bool(),
            service_name=self.rand_str(6),
        )
        is_handler_fn = getattr(x.ProtocolValidation, "is_handler")
        is_service_fn = getattr(x.ProtocolValidation, "is_service")
        validate_compliance_fn = getattr(
            x.ProtocolValidation, "validate_protocol_compliance"
        )
        validate_processor_fn = getattr(
            x.ProtocolValidation, "validate_processor_protocol"
        )

        self.check("protocol.is_handler.good", bool(is_handler_fn(_HandlerLike())))
        self.check("protocol.is_handler.bad", bool(is_handler_fn(_HandlerBad())))
        self.check("protocol.is_service", bool(is_service_fn(protocol_service)))

        compliance_ok = validate_compliance_fn(protocol_service, "Service")
        compliance_fail = validate_compliance_fn(protocol_service, "Unknown")
        self.check("protocol.validate_compliance.ok", compliance_ok.is_success)
        self.check("protocol.validate_compliance.fail", compliance_fail.error)

        processor_ok = validate_processor_fn(
            _GoodProcessor(status=self.rand_str(4), valid=True)
        )
        processor_fail = validate_processor_fn(_BadProcessor(status=self.rand_str(4)))
        self.check("protocol.validate_processor.good", processor_ok.is_success)
        self.check("protocol.validate_processor.fail", processor_fail.error)


if __name__ == "__main__":
    Ex05FlextMixins(__file__).run()
