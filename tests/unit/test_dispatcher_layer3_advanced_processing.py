"""Comprehensive Layer 3 Advanced Processing tests for FlextDispatcher.

Module: flext_core.dispatcher
Scope: FlextDispatcher - processor registration, message processing, batch/parallel execution

Tests advanced processing capabilities:
- ProcessorRegistration: Register and manage message processors
- MessageProcessing: Route and execute processors
- BatchProcessing: Process multiple items efficiently
- ParallelProcessing: Concurrent execution with ThreadPoolExecutor
- TimeoutEnforcement: Execution timeout management
- MetricsCollection: Per-processor and global metrics
- Integration: Full workflow with all layers

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, cast

import pytest
from pydantic import BaseModel

from flext_core import FlextDispatcher, FlextResult, p, t
from flext_tests import u


class DoubleProcessor:
    """Simple processor that doubles a number."""

    def process(
        self,
        data: (
            t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]
        ),
    ) -> t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]:
        """Double the input number."""
        if not isinstance(data, int):
            return FlextResult[t.GeneralValueType].fail(
                f"Expected int, got {type(data)}",
            )
        return FlextResult[t.GeneralValueType].ok(data * 2)


class SquareProcessor:
    """Processor that squares a number."""

    def process(
        self,
        data: (
            t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]
        ),
    ) -> t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]:
        """Square the input number."""
        if not isinstance(data, int):
            return FlextResult[t.GeneralValueType].fail(
                f"Expected int, got {type(data)}",
            )
        return FlextResult[t.GeneralValueType].ok(data * data)


class FailingProcessor:
    """Processor that always fails."""

    def process(
        self,
        data: (
            t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]
        ),
    ) -> t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]:
        """Always return failure."""
        return FlextResult[t.GeneralValueType].fail("Processor intentionally failed")


class SlowProcessor:
    """Processor that takes time to execute."""

    def __init__(self, delay_seconds: float = 0.1) -> None:
        """Initialize slow processor."""
        self.delay_seconds = delay_seconds

    def process(
        self,
        data: (
            t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]
        ),
    ) -> t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]:
        """Sleep then return result."""
        time.sleep(self.delay_seconds)
        # Cast data to GeneralValueType for FlextResult.ok()
        # The actual data may be any of the union types, but we wrap it in result
        return FlextResult[t.GeneralValueType].ok(cast("t.GeneralValueType", data))


class CallableProcessor:
    """Processor as callable function.

    Implements Processor protocol by providing process() method.
    """

    def process(
        self,
        data: t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType],
    ) -> t.GeneralValueType | BaseModel | p.Foundation.Result[t.GeneralValueType]:
        """Process data and return result."""
        if isinstance(data, int):
            return FlextResult[t.GeneralValueType].ok(data + 10)
        return FlextResult[t.GeneralValueType].fail("Expected int")


@dataclass(frozen=True, slots=True)
class ProcessorScenario:
    """Processor test scenario definition."""

    name: str
    processor_name: str
    input_data: t.GeneralValueType
    expected_success: bool
    expected_value: t.GeneralValueType | None = None


class DispatcherScenarios:
    """Centralized dispatcher test scenarios using FlextConstants."""

    PROCESSOR_SCENARIOS: ClassVar[list[ProcessorScenario]] = [
        ProcessorScenario("double_success", "double", 5, True, 10),
        ProcessorScenario("square_success", "square", 5, True, 25),
        ProcessorScenario("callable_success", "callable", 10, True, 20),
        ProcessorScenario("failing_processor", "failing", 5, False),
        ProcessorScenario("invalid_input", "double", "not-a-number", False),
    ]

    BATCH_INPUTS: ClassVar[list[list[t.GeneralValueType]]] = [
        [1, 2, 3, 4, 5],
        [],
        [10, 20, 30],
    ]

    PARALLEL_INPUTS: ClassVar[list[list[t.GeneralValueType]]] = [
        [1, 2, 3, 4, 5],
        [],
        [10, 20, 30],
    ]


# Helper function to create test dispatcher with default processors
def create_test_dispatcher_with_defaults() -> FlextDispatcher:
    """Create test dispatcher with common processors using FlextTestsUtilities."""
    # Type annotation: processors dict accepts any object implementing Processor protocol
    # Use object type and cast to avoid mypy valid-type error with Protocol
    processors_raw: dict[str, object] = {
        "double": DoubleProcessor(),
        "square": SquareProcessor(),
        "failing": FailingProcessor(),
        "callable": CallableProcessor(),
    }
    # Cast to dict[str, p.Application.Processor] for type compatibility
    processors: dict[str, p.Application.Processor] = cast(
        "dict[str, p.Application.Processor]",
        processors_raw,
    )
    return u.Tests.DispatcherHelpers.create_test_dispatcher(
        processors,
    )


class TestLayer3MessageProcessing:
    """Test processor registration and execution using FlextTestsUtilities."""

    def test_register_processor_success(self) -> None:
        """Test successful processor registration."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        result = dispatcher.register_processor("double", processor)
        u.Tests.TestUtilities.assert_result_success(result)
        assert dispatcher.processor_metrics["double"]["executions"] == 0

    def test_register_processor_with_config(self) -> None:
        """Test processor registration with configuration."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()

        config: dict[str, t.GeneralValueType] = {
            "timeout": 5.0,
            "retries": 3,
        }
        result = dispatcher.register_processor("double", processor, config)
        u.Tests.TestUtilities.assert_result_success(result)

    def test_register_callable_processor(self) -> None:
        """Test registering callable as processor."""
        dispatcher = FlextDispatcher()
        processor = CallableProcessor()
        result = dispatcher.register_processor("callable", processor)
        u.Tests.TestUtilities.assert_result_success(result)

    @pytest.mark.parametrize(
        "scenario",
        DispatcherScenarios.PROCESSOR_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_process_registered_processor(self, scenario: ProcessorScenario) -> None:
        """Test processing through registered processor."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process(scenario.processor_name, scenario.input_data)
        u.Tests.DispatcherHelpers.assert_processor_result(
            result,
            expected_success=scenario.expected_success,
            expected_value=scenario.expected_value,
        )

    def test_process_unregistered_processor_fails(self) -> None:
        """Test processing with unregistered processor returns error."""
        dispatcher = FlextDispatcher()
        result = dispatcher.process("nonexistent", 5)
        u.Tests.Result.assert_failure_with_error(
            result,
            expected_error="not registered",
        )


class TestLayer3BatchProcessing:
    """Test batch operation correctness using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "items",
        DispatcherScenarios.BATCH_INPUTS,
        ids=lambda x: f"items_{len(x) if isinstance(x, list) else 0}",
    )
    def test_batch_process(self, items: list[t.GeneralValueType]) -> None:
        """Test batch processing with various input sizes."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process_batch("double", items)
        u.Tests.TestUtilities.assert_result_success(result)
        assert len(result.value) == len(items)

    def test_batch_process_custom_batch_size(self) -> None:
        """Test batch processing with custom batch size."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process_batch("double", [1, 2, 3, 4, 5], batch_size=2)
        u.Tests.TestUtilities.assert_result_success(result)

    def test_batch_process_unregistered_processor(self) -> None:
        """Test batch processing unregistered processor fails."""
        dispatcher = FlextDispatcher()
        result = dispatcher.process_batch("nonexistent", [1, 2, 3])
        u.Tests.TestUtilities.assert_result_failure(result)

    def test_batch_metrics_updated(self) -> None:
        """Test batch operation metrics are updated."""
        dispatcher = create_test_dispatcher_with_defaults()
        dispatcher.process_batch("double", [1, 2, 3])
        assert isinstance(dispatcher.batch_performance["batch_operations"], int)
        assert dispatcher.batch_performance["batch_operations"] >= 1


class TestLayer3ParallelProcessing:
    """Test parallel execution and threading."""

    @pytest.mark.parametrize(
        "items",
        DispatcherScenarios.PARALLEL_INPUTS,
        ids=lambda x: f"items_{len(x) if isinstance(x, list) else 0}",
    )
    def test_parallel_process(self, items: list[t.GeneralValueType]) -> None:
        """Test parallel processing with various input sizes."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process_parallel("double", items)
        u.Tests.TestUtilities.assert_result_success(result)
        assert len(result.value) == len(items)

    def test_parallel_process_custom_workers(self) -> None:
        """Test parallel processing with custom worker count."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process_parallel("double", [1, 2, 3, 4, 5], max_workers=2)
        u.Tests.TestUtilities.assert_result_success(result)

    def test_parallel_process_faster_than_sequential(self) -> None:
        """Test parallel processing is faster for slow operations."""
        dispatcher = FlextDispatcher()
        slow_processor = SlowProcessor(0.05)
        dispatcher.register_processor("slow", slow_processor)
        start = time.time()
        result = dispatcher.process_parallel("slow", [1, 2, 3, 4])
        parallel_time = time.time() - start
        u.Tests.TestUtilities.assert_result_success(result)
        assert parallel_time < 0.3

    def test_parallel_metrics_updated(self) -> None:
        """Test parallel operation metrics are updated."""
        dispatcher = create_test_dispatcher_with_defaults()
        dispatcher.process_parallel("double", [1, 2, 3])
        assert isinstance(dispatcher.parallel_performance["parallel_operations"], int)
        assert dispatcher.parallel_performance["parallel_operations"] >= 1


class TestLayer3FastFailExecution:
    """Test fast fail execution - no fallback patterns."""

    def test_process_success(self) -> None:
        """Test successful processor execution."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process("double", 5)
        value = u.Tests.Result.assert_success(result)
        assert isinstance(value, int)
        assert value == 10

    def test_process_failure_fast_fail(self) -> None:
        """Test processor failure returns error immediately (fast fail)."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process("failing", 5)
        u.Tests.TestUtilities.assert_result_failure(result)

    def test_process_unregistered_processor(self) -> None:
        """Test unregistered processor returns error immediately."""
        dispatcher = FlextDispatcher()
        result = dispatcher.process("nonexistent", 5)
        u.Tests.TestUtilities.assert_result_failure(result)


class TestLayer3TimeoutEnforcement:
    """Test timeout enforcement."""

    def test_timeout_success_within_time(self) -> None:
        """Test successful execution within timeout."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.execute_with_timeout("double", 5, timeout=5.0)
        value = u.Tests.Result.assert_success(result)
        assert isinstance(value, int)
        assert value == 10

    def test_timeout_failure_exceeds_time(self) -> None:
        """Test timeout when execution exceeds time limit."""
        dispatcher = FlextDispatcher()
        slow_processor = SlowProcessor(0.5)
        dispatcher.register_processor("slow", slow_processor)
        result = dispatcher.execute_with_timeout("slow", 5, timeout=0.1)
        u.Tests.Result.assert_failure_with_error(
            result,
            expected_error="timeout",
        )

    def test_timeout_with_reasonable_timeout(self) -> None:
        """Test timeout with reasonable time limit."""
        dispatcher = FlextDispatcher()
        slow_processor = SlowProcessor(0.05)
        dispatcher.register_processor("slow", slow_processor)
        result = dispatcher.execute_with_timeout("slow", 5, timeout=1.0)
        u.Tests.TestUtilities.assert_result_success(result)


class TestLayer3MetricsCollection:
    """Test metrics collection and auditing."""

    def test_processor_metrics_created(self) -> None:
        """Test processor metrics are created on registration."""
        dispatcher = create_test_dispatcher_with_defaults()
        assert "double" in dispatcher.processor_metrics

    def test_processor_metrics_execution_count(self) -> None:
        """Test processor execution count increases."""
        dispatcher = create_test_dispatcher_with_defaults()
        initial_executions = dispatcher.processor_metrics["double"]["executions"]
        dispatcher.process("double", 5)
        assert dispatcher.processor_metrics["double"]["executions"] > initial_executions

    def test_batch_performance_property(self) -> None:
        """Test batch performance property returns metrics."""
        dispatcher = create_test_dispatcher_with_defaults()
        performance = dispatcher.batch_performance
        # Type narrowing: performance is a dict at runtime
        assert isinstance(performance, dict)
        assert all(
            key in performance for key in ["batch_operations", "average_batch_size"]
        )

    def test_parallel_performance_property(self) -> None:
        """Test parallel performance property returns metrics."""
        dispatcher = create_test_dispatcher_with_defaults()
        performance = dispatcher.parallel_performance
        # Type narrowing: performance is a dict at runtime
        assert isinstance(performance, dict)
        assert all(key in performance for key in ["parallel_operations", "max_workers"])

    def test_audit_log_retrieval(self) -> None:
        """Test audit log can be retrieved."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.get_process_audit_log()
        u.Tests.TestUtilities.assert_result_success(result)
        assert isinstance(result.value, list)

    def test_performance_analytics(self) -> None:
        """Test comprehensive performance analytics."""
        dispatcher = create_test_dispatcher_with_defaults()
        dispatcher.process("double", 5)
        result = dispatcher.get_performance_analytics()
        u.Tests.TestUtilities.assert_result_success(result)
        analytics = result.value
        # Type narrowing: analytics is a dict at runtime
        assert isinstance(analytics, dict)
        assert all(
            key in analytics
            for key in [
                "global_metrics",
                "processor_metrics",
                "batch_performance",
                "parallel_performance",
            ]
        )


class TestLayer3Integration:
    """Test full workflow integration with all layers."""

    def test_full_workflow_single_processor(self) -> None:
        """Test complete workflow with single processor."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process("double", 5)
        value = u.Tests.Result.assert_success(result)
        assert isinstance(value, int)
        assert value == 10

    def test_full_workflow_batch_then_process(self) -> None:
        """Test workflow combining batch and single processing."""
        dispatcher = create_test_dispatcher_with_defaults()
        batch_result = dispatcher.process_batch("double", [1, 2, 3])
        u.Tests.TestUtilities.assert_result_success(batch_result)
        single_result = dispatcher.process("double", 5)
        value = u.Tests.Result.assert_success(single_result)
        assert isinstance(value, int)
        assert value == 10

    def test_full_workflow_parallel_with_fast_fail(self) -> None:
        """Test workflow combining parallel with fast fail (no fallback)."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process("failing", 5)
        u.Tests.TestUtilities.assert_result_failure(result)
        result2 = dispatcher.process("double", 5)
        value = u.Tests.Result.assert_success(
            result2,
        )
        assert isinstance(value, int)
        assert value == 10

    def test_full_workflow_all_features(self) -> None:
        """Test workflow using all Layer 3 features."""
        dispatcher = FlextDispatcher()
        dispatcher.register_processor("double", DoubleProcessor())
        dispatcher.register_processor("square", SquareProcessor())
        dispatcher.register_processor("slow", SlowProcessor(0.01))
        u.Tests.Result.assert_success_with_value(
            dispatcher.process("double", 5),
            10,
        )
        u.Tests.TestUtilities.assert_result_success(
            dispatcher.process_batch("double", [1, 2, 3]),
        )
        u.Tests.TestUtilities.assert_result_success(
            dispatcher.process_parallel("double", [4, 5, 6]),
        )
        u.Tests.TestUtilities.assert_result_success(
            dispatcher.execute_with_timeout("slow", 5, timeout=1.0),
        )
        u.Tests.Result.assert_success_with_value(
            dispatcher.process("double", 7),
            14,
        )
        analytics = dispatcher.get_performance_analytics()
        u.Tests.TestUtilities.assert_result_success(analytics)

    def test_multiple_processors_independence(self) -> None:
        """Test multiple processors operate independently."""
        dispatcher = FlextDispatcher()
        dispatcher.register_processor("double", DoubleProcessor())
        dispatcher.register_processor("square", SquareProcessor())
        r1 = dispatcher.process("double", 5)
        r2 = dispatcher.process("square", 5)
        u.Tests.Result.assert_success_with_value(r1, 10)
        u.Tests.Result.assert_success_with_value(r2, 25)

    def test_error_handling_propagates(self) -> None:
        """Test errors propagate through Layer 3."""
        dispatcher = create_test_dispatcher_with_defaults()
        result = dispatcher.process("double", "not-a-number")
        u.Tests.Result.assert_failure_with_error(
            result,
            expected_error="Expected int",
        )


__all__ = [
    "TestLayer3BatchProcessing",
    "TestLayer3FastFailExecution",
    "TestLayer3Integration",
    "TestLayer3MessageProcessing",
    "TestLayer3MetricsCollection",
    "TestLayer3ParallelProcessing",
    "TestLayer3TimeoutEnforcement",
]
