"""Decorator injection, logging, and tracking tests."""

from __future__ import annotations

import time

import pytest
from flext_tests import d

from tests.models import m
from tests.unit._decorators_support import (
    TestsFlextDecoratorsLegacy,
    capture_stdout,
)

INJECT_SCENARIOS = TestsFlextDecoratorsLegacy.INJECT_SCENARIOS
LOG_SCENARIOS = TestsFlextDecoratorsLegacy.LOG_SCENARIOS
TRACK_SCENARIOS = TestsFlextDecoratorsLegacy.TRACK_SCENARIOS


class TestsFlextDecoratorsInjectionLogging(TestsFlextDecoratorsLegacy):
    @pytest.mark.parametrize("test_case", INJECT_SCENARIOS, ids=lambda case: case.name)
    def test_inject_decorator(
        self,
        test_case: TestsFlextDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.INJECT_BASIC:

            @d.inject(test_service="test_service")
            def process_data_basic(
                data: str,
                *,
                test_service: TestsFlextDecoratorsLegacy.TestService | None = None,
            ) -> str:
                if test_service is not None:
                    return f"{data}_{test_service.get_value()}"
                return f"{data}_default"

            assert process_data_basic("input") == "input_default"
        elif test_case.operation == self.DecoratorOperationType.INJECT_MISSING:

            @d.inject(missing_service="missing_service")
            def process_data_missing(*, missing_service: str = "default") -> str:
                return missing_service

            assert process_data_missing() == "default"
        elif test_case.operation == self.DecoratorOperationType.INJECT_PROVIDED:

            class TestServiceTyped(m.BaseModel):
                value: str

            @d.inject(service="service")
            def process(*, service: TestServiceTyped) -> str:
                return service.value

            explicit_service = TestServiceTyped.model_validate({"value": "explicit"})
            assert process(service=explicit_service) == "explicit"

    @pytest.mark.parametrize("test_case", LOG_SCENARIOS, ids=lambda case: case.name)
    def test_log_operation_decorator(
        self,
        test_case: TestsFlextDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.LOG_OPERATION_BASIC:

            @d.log_operation("test_operation")
            def simple_function() -> str:
                return "success"

            assert simple_function() == "success"
        elif test_case.operation == self.DecoratorOperationType.LOG_OPERATION_EXCEPTION:

            @d.log_operation("failing_operation")
            def failing_function() -> None:
                error_msg = "Test error"
                raise ValueError(error_msg)

            def emit() -> None:
                with pytest.raises(ValueError, match="Test error"):
                    failing_function()

            _ = capture_stdout(emit, contains="failing_operation")

    @pytest.mark.parametrize("test_case", TRACK_SCENARIOS, ids=lambda case: case.name)
    def test_track_performance_decorator(
        self,
        test_case: TestsFlextDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.TRACK_PERFORMANCE_BASIC:

            @d.log_operation("timed_operation")
            def timed_function() -> str:
                time.sleep(0.01)
                return "completed"

            assert timed_function() == "completed"
        elif (
            test_case.operation
            == self.DecoratorOperationType.TRACK_PERFORMANCE_EXCEPTION
        ):

            @d.log_operation("failing_operation")
            def failing_function() -> None:
                error_msg = "Timed failure"
                raise RuntimeError(error_msg)

            def emit() -> None:
                with pytest.raises(RuntimeError, match="Timed failure"):
                    failing_function()

            _ = capture_stdout(emit, contains="failing_operation")
