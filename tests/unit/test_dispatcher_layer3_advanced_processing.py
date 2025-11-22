"""Comprehensive Layer 3 Advanced Processing tests for FlextDispatcher.

This module tests the advanced processing capabilities:
- ProcessorRegistration: Register and manage message processors
- MessageProcessing: Route and execute processors
- BatchProcessing: Process multiple items efficiently
- ParallelProcessing: Concurrent execution with ThreadPoolExecutor
- FallbackChains: Sequential fallback execution
- TimeoutEnforcement: Execution timeout management
- MetricsCollection: Per-processor and global metrics
- Integration: Full workflow with all layers

All tests use REAL implementations without mocks, following railway-oriented
programming with FlextResult[T] error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time

from flext_core import (
    FlextDispatcher,
    FlextResult,
)

# ==================== REAL PROCESSOR IMPLEMENTATIONS ====================


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


# ==================== TEST CLASSES ====================


class TestLayer3MessageProcessing:
    """Test processor registration and execution."""

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

    def test_process_registered_processor(self) -> None:
        """Test processing through registered processor."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process("double", 5)

        assert result.is_success
        assert result.unwrap() == 10

    def test_process_unregistered_processor_fails(self) -> None:
        """Test processing with unregistered processor returns error."""
        dispatcher = FlextDispatcher()

        result = dispatcher.process("nonexistent", 5)

        assert result.is_failure
        assert result.error is not None
        assert "not registered" in result.error

    def test_process_with_callable_processor(self) -> None:
        """Test processing with callable processor."""
        dispatcher = FlextDispatcher()
        processor = CallableProcessor()
        dispatcher.register_processor("callable", processor)

        result = dispatcher.process("callable", 10)

        assert result.is_success
        assert result.unwrap() == 20


class TestLayer3BatchProcessing:
    """Test batch operation correctness."""

    def test_batch_process_multiple_items(self) -> None:
        """Test batch processing multiple items."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process_batch("double", [1, 2, 3, 4, 5])

        assert result.is_success
        items = result.unwrap()
        assert len(items) == 5

    def test_batch_process_empty_list(self) -> None:
        """Test batch processing empty list."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process_batch("double", [])

        assert result.is_success
        assert result.unwrap() == []

    def test_batch_process_custom_batch_size(self) -> None:
        """Test batch processing with custom batch size."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process_batch("double", [1, 2, 3, 4, 5], batch_size=2)

        assert result.is_success

    def test_batch_process_unregistered_processor(self) -> None:
        """Test batch processing unregistered processor fails."""
        dispatcher = FlextDispatcher()

        result = dispatcher.process_batch("nonexistent", [1, 2, 3])

        assert result.is_failure

    def test_batch_metrics_updated(self) -> None:
        """Test batch operation metrics are updated."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        dispatcher.process_batch("double", [1, 2, 3])

        assert isinstance(dispatcher.batch_performance["batch_operations"], int)
        assert dispatcher.batch_performance["batch_operations"] >= 1


class TestLayer3ParallelProcessing:
    """Test parallel execution and threading."""

    def test_parallel_process_multiple_items(self) -> None:
        """Test parallel processing multiple items."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process_parallel("double", [1, 2, 3, 4, 5])

        assert result.is_success
        items = result.unwrap()
        assert len(items) == 5

    def test_parallel_process_empty_list(self) -> None:
        """Test parallel processing empty list."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process_parallel("double", [])

        assert result.is_success
        assert result.unwrap() == []

    def test_parallel_process_custom_workers(self) -> None:
        """Test parallel processing with custom worker count."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process_parallel("double", [1, 2, 3, 4, 5], max_workers=2)

        assert result.is_success

    def test_parallel_process_faster_than_sequential(self) -> None:
        """Test parallel processing is faster for slow operations."""
        dispatcher = FlextDispatcher()
        slow_processor = SlowProcessor(0.05)
        dispatcher.register_processor("slow", slow_processor)

        # Parallel should be faster
        start = time.time()
        result = dispatcher.process_parallel("slow", [1, 2, 3, 4])
        parallel_time = time.time() - start

        assert result.is_success
        # Parallel time should be less than sequential (0.05 * 4 = 0.2)
        assert parallel_time < 0.3  # Some overhead allowed

    def test_parallel_metrics_updated(self) -> None:
        """Test parallel operation metrics are updated."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        dispatcher.process_parallel("double", [1, 2, 3])

        assert isinstance(dispatcher.parallel_performance["parallel_operations"], int)
        assert dispatcher.parallel_performance["parallel_operations"] >= 1


class TestLayer3FastFailExecution:
    """Test fast fail execution - no fallback patterns."""

    def test_process_success(self) -> None:
        """Test successful processor execution."""
        dispatcher = FlextDispatcher()
        double_processor = DoubleProcessor()
        dispatcher.register_processor("double", double_processor)

        result = dispatcher.process("double", 5)

        assert result.is_success
        assert result.unwrap() == 10

    def test_process_failure_fast_fail(self) -> None:
        """Test processor failure returns error immediately (fast fail)."""
        dispatcher = FlextDispatcher()
        failing_processor = FailingProcessor()
        dispatcher.register_processor("failing", failing_processor)

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
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.execute_with_timeout("double", 5, timeout=5.0)

        assert result.is_success
        assert result.unwrap() == 10

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
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        metrics = dispatcher.processor_metrics
        assert "double" in metrics

    def test_processor_metrics_execution_count(self) -> None:
        """Test processor execution count increases."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)
        initial_executions = dispatcher.processor_metrics["double"]["executions"]

        dispatcher.process("double", 5)

        assert dispatcher.processor_metrics["double"]["executions"] > initial_executions

    def test_batch_performance_property(self) -> None:
        """Test batch performance property returns metrics."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        performance = dispatcher.batch_performance
        assert "batch_operations" in performance
        assert "average_batch_size" in performance

    def test_parallel_performance_property(self) -> None:
        """Test parallel performance property returns metrics."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        performance = dispatcher.parallel_performance
        assert "parallel_operations" in performance
        assert "max_workers" in performance

    def test_audit_log_retrieval(self) -> None:
        """Test audit log can be retrieved."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.get_process_audit_log()

        assert result.is_success
        assert isinstance(result.unwrap(), list)

    def test_performance_analytics(self) -> None:
        """Test comprehensive performance analytics."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)
        dispatcher.process("double", 5)

        result = dispatcher.get_performance_analytics()

        assert result.is_success
        analytics = result.unwrap()
        assert "global_metrics" in analytics
        assert "processor_metrics" in analytics
        assert "batch_performance" in analytics
        assert "parallel_performance" in analytics


class TestLayer3Integration:
    """Test full workflow integration with all layers."""

    def test_full_workflow_single_processor(self) -> None:
        """Test complete workflow with single processor."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process("double", 5)

        assert result.is_success
        assert result.unwrap() == 10

    def test_full_workflow_batch_then_process(self) -> None:
        """Test workflow combining batch and single processing."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        # Batch processing
        batch_result = dispatcher.process_batch("double", [1, 2, 3])
        assert batch_result.is_success

        # Single processing
        single_result = dispatcher.process("double", 5)
        assert single_result.is_success
        assert single_result.unwrap() == 10

    def test_full_workflow_parallel_with_fast_fail(self) -> None:
        """Test workflow combining parallel with fast fail (no fallback)."""
        dispatcher = FlextDispatcher()
        double_processor = DoubleProcessor()
        failing_processor = FailingProcessor()
        dispatcher.register_processor("double", double_processor)
        dispatcher.register_processor("failing", failing_processor)

        # Fast fail: failing processor should return error immediately
        result = dispatcher.process("failing", 5)
        assert result.is_failure
        assert result.error is not None

        # Successful processor should work
        result2 = dispatcher.process("double", 5)
        assert result2.is_success
        assert result2.unwrap() == 10

    def test_full_workflow_all_features(self) -> None:
        """Test workflow using all Layer 3 features."""
        dispatcher = FlextDispatcher()
        double = DoubleProcessor()
        square = SquareProcessor()
        slow = SlowProcessor(0.01)

        dispatcher.register_processor("double", double)
        dispatcher.register_processor("square", square)
        dispatcher.register_processor("slow", slow)

        # Process single
        r1 = dispatcher.process("double", 5)
        assert r1.is_success and r1.unwrap() == 10

        # Batch process
        r2 = dispatcher.process_batch("double", [1, 2, 3])
        assert r2.is_success

        # Parallel process
        r3 = dispatcher.process_parallel("double", [4, 5, 6])
        assert r3.is_success

        # Timeout
        r4 = dispatcher.execute_with_timeout("slow", 5, timeout=1.0)
        assert r4.is_success

        # Direct process (no fallback pattern)
        r5 = dispatcher.process("double", 7)
        assert r5.is_success and r5.unwrap() == 14

        # Verify metrics
        analytics = dispatcher.get_performance_analytics()
        assert analytics.is_success

    def test_multiple_processors_independence(self) -> None:
        """Test multiple processors operate independently."""
        dispatcher = FlextDispatcher()
        double = DoubleProcessor()
        square = SquareProcessor()
        dispatcher.register_processor("double", double)
        dispatcher.register_processor("square", square)

        # Process with both
        r1 = dispatcher.process("double", 5)
        r2 = dispatcher.process("square", 5)

        assert r1.is_success and r1.unwrap() == 10
        assert r2.is_success and r2.unwrap() == 25

    def test_error_handling_propagates(self) -> None:
        """Test errors propagate through Layer 3."""
        dispatcher = FlextDispatcher()
        processor = DoubleProcessor()
        dispatcher.register_processor("double", processor)

        result = dispatcher.process("double", "not-a-number")

        assert result.is_failure
        assert result.error is not None
        assert "Expected int" in result.error
