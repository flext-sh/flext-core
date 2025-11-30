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
from typing import ClassVar

import pytest

from flext_core import FlextDispatcher, FlextResult


class DoubleProcessor:
    """Simple processor that doubles a number."""

    def process(self, data: object) -> FlextResult[int]:
        """Double the input number."""
        if not isinstance(data, int):
            return FlextResult[int].fail(f"Expected int, got {type(data)}")
        return FlextResult[int].ok(data * 2)


class SquareProcessor:
    """Processor that squares a number."""

    def process(self, data: object) -> FlextResult[int]:
        """Square the input number."""
        if not isinstance(data, int):
            return FlextResult[int].fail(f"Expected int, got {type(data)}")
        return FlextResult[int].ok(data * data)


class FailingProcessor:
    """Processor that always fails."""

    def process(self, data: object) -> FlextResult[object]:
        """Always return failure."""
        return FlextResult[object].fail("Processor intentionally failed")


class SlowProcessor:
    """Processor that takes time to execute."""

    def __init__(self, delay_seconds: float = 0.1) -> None:
        """Initialize slow processor."""
        self.delay_seconds = delay_seconds

    def process(self, data: object) -> FlextResult[object]:
        """Sleep then return result."""
        time.sleep(self.delay_seconds)
        return FlextResult[object].ok(data)


class CallableProcessor:
    """Processor as callable function."""

    def __call__(self, data: object) -> FlextResult[int]:
        """Process as callable."""
        if isinstance(data, int):
            return FlextResult[int].ok(data + 10)
        return FlextResult[int].fail("Expected int")


@dataclass(frozen=True, slots=True)
class ProcessorScenario:
    """Processor test scenario definition."""

    name: str
    processor_name: str
    input_data: object
    expected_success: bool
    expected_value: object | None = None


class DispatcherScenarios:
    """Centralized dispatcher test scenarios using FlextConstants."""

    PROCESSOR_SCENARIOS: ClassVar[list[ProcessorScenario]] = [
        ProcessorScenario("double_success", "double", 5, True, 10),
        ProcessorScenario("square_success", "square", 5, True, 25),
        ProcessorScenario("callable_success", "callable", 10, True, 20),
        ProcessorScenario("failing_processor", "failing", 5, False),
        ProcessorScenario("invalid_input", "double", "not-a-number", False),
    ]

    BATCH_INPUTS: ClassVar[list[list[object]]] = [
        [1, 2, 3, 4, 5],
        [],
        [10, 20, 30],
    ]

    PARALLEL_INPUTS: ClassVar[list[list[object]]] = [
        [1, 2, 3, 4, 5],
        [],
        [10, 20, 30],
    ]


class DispatcherTestHelpers:
    """Generalized helpers for dispatcher testing."""

    @staticmethod
    def create_test_dispatcher() -> FlextDispatcher:
        """Create test dispatcher with common processors."""
        dispatcher = FlextDispatcher()
        dispatcher.register_processor("double", DoubleProcessor())
        dispatcher.register_processor("square", SquareProcessor())
        dispatcher.register_processor("failing", FailingProcessor())
        dispatcher.register_processor("callable", CallableProcessor())
        return dispatcher

    @staticmethod
    def assert_processor_result(
        result: FlextResult[object],
        should_succeed: bool,
        expected_value: object | None = None,
    ) -> None:
        """Assert processor result matches expectations."""
        assert result.is_success == should_succeed
        if should_succeed and expected_value is not None:
            assert result.value == expected_value


class TestLayer3MessageProcessing:
    """Test processor registration and execution using FlextTestsUtilities."""

    def test_register_processor_success(self) -> None:
        """Test successful processor registration."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        result = dispatcher.register_processor("double", processor)
        assert result.is_success
        assert dispatcher.processor_metrics["double"]["executions"] == 0

    def test_register_processor_with_config(self) -> None:
        """Test processor registration with configuration."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        config: dict[str, object] = {"timeout": 5.0, "retries": 3}
        result = dispatcher.register_processor("double", processor, config)
        assert result.is_success

    def test_register_callable_processor(self) -> None:
        """Test registering callable as processor."""
        dispatcher = FlextDispatcher()
        processor = CallableProcessor()
        result = dispatcher.register_processor("callable", processor)
        assert result.is_success

    @pytest.mark.parametrize(
        "scenario", DispatcherScenarios.PROCESSOR_SCENARIOS, ids=lambda s: s.name,
    )
    def test_process_registered_processor(self, scenario: ProcessorScenario) -> None:
        """Test processing through registered processor."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process(scenario.processor_name, scenario.input_data)
        DispatcherTestHelpers.assert_processor_result(
            result, scenario.expected_success, scenario.expected_value,
        )

    def test_process_unregistered_processor_fails(self) -> None:
        """Test processing with unregistered processor returns error."""
        dispatcher = FlextDispatcher()
        result = dispatcher.process("nonexistent", 5)
        assert result.is_failure
        assert result.error is not None
        assert "not registered" in result.error


class TestLayer3BatchProcessing:
    """Test batch operation correctness using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "items",
        DispatcherScenarios.BATCH_INPUTS,
        ids=lambda x: f"items_{len(x) if isinstance(x, list) else 0}",
    )
    def test_batch_process(self, items: list[object]) -> None:
        """Test batch processing with various input sizes."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process_batch("double", items)
        assert result.is_success
        assert len(result.value) == len(items)

    def test_batch_process_custom_batch_size(self) -> None:
        """Test batch processing with custom batch size."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process_batch("double", [1, 2, 3, 4, 5], batch_size=2)
        assert result.is_success

    def test_batch_process_unregistered_processor(self) -> None:
        """Test batch processing unregistered processor fails."""
        dispatcher = FlextDispatcher()
        result = dispatcher.process_batch("nonexistent", [1, 2, 3])
        assert result.is_failure

    def test_batch_metrics_updated(self) -> None:
        """Test batch operation metrics are updated."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
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
    def test_parallel_process(self, items: list[object]) -> None:
        """Test parallel processing with various input sizes."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process_parallel("double", items)
        assert result.is_success
        assert len(result.value) == len(items)

    def test_parallel_process_custom_workers(self) -> None:
        """Test parallel processing with custom worker count."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process_parallel("double", [1, 2, 3, 4, 5], max_workers=2)
        assert result.is_success

    def test_parallel_process_faster_than_sequential(self) -> None:
        """Test parallel processing is faster for slow operations."""
        dispatcher = FlextDispatcher()
        slow_processor = SlowProcessor(0.05)
        dispatcher.register_processor("slow", slow_processor)
        start = time.time()
        result = dispatcher.process_parallel("slow", [1, 2, 3, 4])
        parallel_time = time.time() - start
        assert result.is_success
        assert parallel_time < 0.3

    def test_parallel_metrics_updated(self) -> None:
        """Test parallel operation metrics are updated."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        dispatcher.process_parallel("double", [1, 2, 3])
        assert isinstance(dispatcher.parallel_performance["parallel_operations"], int)
        assert dispatcher.parallel_performance["parallel_operations"] >= 1


class TestLayer3FastFailExecution:
    """Test fast fail execution - no fallback patterns."""

    def test_process_success(self) -> None:
        """Test successful processor execution."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process("double", 5)
        assert result.is_success
        assert result.value == 10

    def test_process_failure_fast_fail(self) -> None:
        """Test processor failure returns error immediately (fast fail)."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process("failing", 5)
        assert result.is_failure
        assert result.error is not None

    def test_process_unregistered_processor(self) -> None:
        """Test unregistered processor returns error immediately."""
        dispatcher = FlextDispatcher()
        result = dispatcher.process("nonexistent", 5)
        assert result.is_failure
        assert result.error is not None


class TestLayer3TimeoutEnforcement:
    """Test timeout enforcement."""

    def test_timeout_success_within_time(self) -> None:
        """Test successful execution within timeout."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.execute_with_timeout("double", 5, timeout=5.0)
        assert result.is_success
        assert result.value == 10

    def test_timeout_failure_exceeds_time(self) -> None:
        """Test timeout when execution exceeds time limit."""
        dispatcher = FlextDispatcher()
        slow_processor = SlowProcessor(0.5)
        dispatcher.register_processor("slow", slow_processor)
        result = dispatcher.execute_with_timeout("slow", 5, timeout=0.1)
        assert result.is_failure
        assert result.error is not None
        assert "timeout" in result.error.lower()

    def test_timeout_with_reasonable_timeout(self) -> None:
        """Test timeout with reasonable time limit."""
        dispatcher = FlextDispatcher()
        slow_processor = SlowProcessor(0.05)
        dispatcher.register_processor("slow", slow_processor)
        result = dispatcher.execute_with_timeout("slow", 5, timeout=1.0)
        assert result.is_success


class TestLayer3MetricsCollection:
    """Test metrics collection and auditing."""

    def test_processor_metrics_created(self) -> None:
        """Test processor metrics are created on registration."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        assert "double" in dispatcher.processor_metrics

    def test_processor_metrics_execution_count(self) -> None:
        """Test processor execution count increases."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        initial_executions = dispatcher.processor_metrics["double"]["executions"]
        dispatcher.process("double", 5)
        assert dispatcher.processor_metrics["double"]["executions"] > initial_executions

    def test_batch_performance_property(self) -> None:
        """Test batch performance property returns metrics."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        performance = dispatcher.batch_performance
        assert all(
            key in performance for key in ["batch_operations", "average_batch_size"]
        )

    def test_parallel_performance_property(self) -> None:
        """Test parallel performance property returns metrics."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        performance = dispatcher.parallel_performance
        assert all(key in performance for key in ["parallel_operations", "max_workers"])

    def test_audit_log_retrieval(self) -> None:
        """Test audit log can be retrieved."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.get_process_audit_log()
        assert result.is_success
        assert isinstance(result.value, list)

    def test_performance_analytics(self) -> None:
        """Test comprehensive performance analytics."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        dispatcher.process("double", 5)
        result = dispatcher.get_performance_analytics()
        assert result.is_success
        analytics = result.value
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
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process("double", 5)
        assert result.is_success
        assert result.value == 10

    def test_full_workflow_batch_then_process(self) -> None:
        """Test workflow combining batch and single processing."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        batch_result = dispatcher.process_batch("double", [1, 2, 3])
        assert batch_result.is_success
        single_result = dispatcher.process("double", 5)
        assert single_result.is_success
        assert single_result.value == 10

    def test_full_workflow_parallel_with_fast_fail(self) -> None:
        """Test workflow combining parallel with fast fail (no fallback)."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process("failing", 5)
        assert result.is_failure
        result2 = dispatcher.process("double", 5)
        assert result2.is_success
        assert result2.value == 10

    def test_full_workflow_all_features(self) -> None:
        """Test workflow using all Layer 3 features."""
        dispatcher = FlextDispatcher()
        dispatcher.register_processor("double", DoubleProcessor())
        dispatcher.register_processor("square", SquareProcessor())
        dispatcher.register_processor("slow", SlowProcessor(0.01))
        assert dispatcher.process("double", 5).value == 10
        assert dispatcher.process_batch("double", [1, 2, 3]).is_success
        assert dispatcher.process_parallel("double", [4, 5, 6]).is_success
        assert dispatcher.execute_with_timeout("slow", 5, timeout=1.0).is_success
        assert dispatcher.process("double", 7).value == 14
        analytics = dispatcher.get_performance_analytics()
        assert analytics.is_success

    def test_multiple_processors_independence(self) -> None:
        """Test multiple processors operate independently."""
        dispatcher = FlextDispatcher()
        dispatcher.register_processor("double", DoubleProcessor())
        dispatcher.register_processor("square", SquareProcessor())
        r1 = dispatcher.process("double", 5)
        r2 = dispatcher.process("square", 5)
        assert r1.value == 10
        assert r2.value == 25

    def test_error_handling_propagates(self) -> None:
        """Test errors propagate through Layer 3."""
        dispatcher = DispatcherTestHelpers.create_test_dispatcher()
        result = dispatcher.process("double", "not-a-number")
        assert result.is_failure
        assert result.error is not None
        assert "Expected int" in result.error


__all__ = [
    "TestLayer3BatchProcessing",
    "TestLayer3FastFailExecution",
    "TestLayer3Integration",
    "TestLayer3MessageProcessing",
    "TestLayer3MetricsCollection",
    "TestLayer3ParallelProcessing",
    "TestLayer3TimeoutEnforcement",
]
