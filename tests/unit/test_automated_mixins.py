"""Real API tests for flext_core.mixins through FlextService."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import pytest
from flext_tests import tm, tt
from hypothesis import given, settings, strategies as st

from flext_core import FlextMixins, r, s


class TestAutomatedFlextMixins:
    class _MixinTestService(s[str]):
        __test__ = False

        @override
        def execute(self) -> r[str]:
            return r[str].ok("mixin_test")

    def test_service_exposes_mixins_properties(self) -> None:
        svc = self._MixinTestService()
        tm.ok(svc.execute(), eq="mixin_test")
        tm.that(svc.config.version, none=False)
        tm.that(callable(svc.container.register), eq=True)
        tm.that(str(svc.context), none=False)
        tm.that(callable(svc.logger.info), eq=True)

    @pytest.mark.parametrize(
        "operation_name",
        [("track_a", "track_a"), ("track_b", "track_b")],
        ids=lambda case: case[0],
    )
    def test_track_context_manager(self, operation_name: str) -> None:
        svc = self._MixinTestService()
        with svc.track(operation_name) as metrics:
            tm.that(metrics, is_=dict)
            tm.that(metrics, has="operation_count")

    def test_cqrs_metrics_tracker(self) -> None:
        tracker = FlextMixins.CQRS.MetricsTracker()
        tm.ok(tracker.record_metric("latency_ms", 15), eq=True)
        metrics = tm.ok(tracker.get_metrics())
        tm.that(metrics.root, kv=("latency_ms", 15))

    def test_cqrs_context_stack_push_pop(self) -> None:
        stack = FlextMixins.CQRS.ContextStack()
        tm.ok(
            stack.push_context({"handler_name": "h1", "handler_mode": "command"}),
            eq=True,
        )
        popped = tm.ok(stack.pop_context())
        tm.that(popped, kv=("handler_name", "h1"))

    def test_validate_with_result(self) -> None:
        def not_empty(value: str) -> r[bool]:
            if value.strip() == "":
                return r[bool].fail("empty")
            return r[bool].ok(value=True)

        result = FlextMixins.Validation.validate_with_result("valid", [not_empty])
        tm.ok(result, eq="valid")
        tm.fail(
            FlextMixins.Validation.validate_with_result("", [not_empty]), has="empty"
        )

    @given(
        data=st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=40)
    def test_hypothesis_validation_roundtrip(self, data: str) -> None:
        def has_text(value: str) -> r[bool]:
            if value.strip() == "":
                return r[bool].fail("blank")
            return r[bool].ok(value=True)

        tm.ok(FlextMixins.Validation.validate_with_result(data, [has_text]), eq=data)

    @pytest.mark.performance
    @pytest.mark.parametrize(
        "mode",
        [("raw", "raw"), ("track", "track")],
        ids=lambda case: case[0],
    )
    def test_track_benchmark(self, mode: str, benchmark: Callable[..., object]) -> None:
        svc = self._MixinTestService()
        simple = tt.op("simple")

        if mode == "raw":
            raw_value = simple()
            if isinstance(raw_value, str):
                tm.that(raw_value, eq="success")
            else:
                tm.that(False, eq=True)
            _ = benchmark(simple)
            return

        def tracked_call() -> str:
            with svc.track("bench"):
                return "success"

        tm.that(tracked_call(), eq="success")
        _ = benchmark(tracked_call)
