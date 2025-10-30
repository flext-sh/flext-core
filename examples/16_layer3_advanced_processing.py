"""Layer 3 Advanced Processing Integration Examples.

This example demonstrates the NEW Layer 3 capabilities for batch, parallel,
and fault-tolerant processing. Layer 3 integrates on top of Layer 1 CQRS
and Layer 2 Reliability Patterns.

Key Features:
- Processor registration and execution
- Batch processing for multiple items
- Parallel processing with ThreadPoolExecutor
- Timeout enforcement
- Fallback chains for resilience
- Comprehensive metrics and auditing

All operations use FlextResult[T] for composable error handling.
"""

import time

from flext_core import FlextDispatcher, FlextResult

# ==================== PROCESSOR IMPLEMENTATIONS ====================


class DataDoubler:
    """Simple processor that doubles numeric data."""

    def process(self, data: int) -> FlextResult[int]:
        """Double the input value."""
        if not isinstance(data, int):
            return FlextResult[int].fail(f"Expected int, got {type(data)}")
        return FlextResult[int].ok(data * 2)


class DataSquarer:
    """Processor that squares numeric data."""

    def process(self, data: int) -> FlextResult[int]:
        """Square the input value."""
        if not isinstance(data, int):
            return FlextResult[int].fail(f"Expected int, got {type(data)}")
        return FlextResult[int].ok(data * data)


class SlowProcessor:
    """Processor simulating slow I/O operation."""

    def __init__(self, delay: float = 0.1) -> None:
        """Initialize with delay in seconds."""
        self.delay = delay

    def process(self, data: int) -> FlextResult[int]:
        """Simulate slow operation then return result."""
        time.sleep(self.delay)
        return FlextResult[int].ok(data)


class DataValidator:
    """Processor validating data integrity."""

    def process(self, data: int) -> FlextResult[int]:
        """Validate data is positive."""
        if data < 0:
            return FlextResult[int].fail("Data must be positive")
        return FlextResult[int].ok(data)


# ==================== EXAMPLE 1: BASIC PROCESSOR REGISTRATION ====================


def example_1_basic_registration() -> None:
    """Example 1: Register and execute a simple processor."""
    print("\n=== EXAMPLE 1: Basic Processor Registration ===\n")

    dispatcher = FlextDispatcher()

    # Register processor
    result = dispatcher.register_processor("doubler", DataDoubler())
    assert result.is_success
    print("✅ Registered 'doubler' processor")

    # Execute processor
    process_result = dispatcher.process("doubler", 5)
    assert process_result.is_success
    assert process_result.unwrap() == 10
    print(f"✅ Processed 5 → {process_result.unwrap()}")


# ==================== EXAMPLE 2: BATCH PROCESSING ====================


def example_2_batch_processing() -> None:
    """Example 2: Process multiple items in batch."""
    print("\n=== EXAMPLE 2: Batch Processing ===\n")

    dispatcher = FlextDispatcher()
    dispatcher.register_processor("doubler", DataDoubler())

    # Process batch of items
    data_list = [1, 2, 3, 4, 5]
    result = dispatcher.process_batch("doubler", data_list, batch_size=2)

    assert result.is_success
    processed = result.unwrap()
    assert processed == [2, 4, 6, 8, 10]
    print(f"✅ Batch processed {data_list} → {processed}")

    # Check batch performance metrics
    batch_perf = dispatcher.batch_performance
    print(f"✅ Batch operations: {batch_perf['batch_operations']}")


# ==================== EXAMPLE 3: PARALLEL PROCESSING ====================


def example_3_parallel_processing() -> None:
    """Example 3: Process items concurrently."""
    print("\n=== EXAMPLE 3: Parallel Processing ===\n")

    dispatcher = FlextDispatcher()
    dispatcher.register_processor("doubler", DataDoubler())

    # Process items in parallel
    data_list = list(range(1, 11))
    start = time.time()
    result = dispatcher.process_parallel("doubler", data_list, max_workers=4)
    elapsed = time.time() - start

    assert result.is_success
    processed = result.unwrap()
    print(f"✅ Parallel processed {data_list}")
    print(f"✅ Result: {processed}")
    print(f"✅ Time: {elapsed:.3f}s with 4 workers")

    # Check parallel performance
    parallel_perf = dispatcher.parallel_performance
    print(f"✅ Parallel operations: {parallel_perf['parallel_operations']}")


# ==================== EXAMPLE 4: TIMEOUT ENFORCEMENT ====================


def example_4_timeout_enforcement() -> None:
    """Example 4: Execute with timeout protection."""
    print("\n=== EXAMPLE 4: Timeout Enforcement ===\n")

    dispatcher = FlextDispatcher()

    # Register slow processor
    slow = SlowProcessor(delay=0.05)
    dispatcher.register_processor("slow", slow)

    # Execute with reasonable timeout - should succeed
    result = dispatcher.execute_with_timeout("slow", 10, timeout=1.0)
    assert result.is_success
    print("✅ Slow operation completed within timeout")

    # Execute with tight timeout - should fail
    very_slow = SlowProcessor(delay=0.5)
    dispatcher.register_processor("very_slow", very_slow)
    result = dispatcher.execute_with_timeout("very_slow", 10, timeout=0.1)
    assert result.is_failure
    assert "timeout" in result.error.lower()
    print(f"✅ Timeout error caught: {result.error}")


# ==================== EXAMPLE 5: FALLBACK CHAINS ====================


def example_5_fallback_chains() -> None:
    """Example 5: Execute with fallback chain."""
    print("\n=== EXAMPLE 5: Fallback Chains ===\n")

    dispatcher = FlextDispatcher()

    # Register processors
    doubler = DataDoubler()
    squarer = DataSquarer()
    dispatcher.register_processor("doubler", doubler)
    dispatcher.register_processor("squarer", squarer)

    # Try primary, fall back to secondary
    result = dispatcher.execute_with_fallback("doubler", 5, fallback_names=["squarer"])
    assert result.is_success
    assert result.unwrap() == 10  # From doubler, not squarer
    print(f"✅ Primary processor succeeded: 5 → {result.unwrap()}")

    # Now test with validator that fails - falls back
    validator = DataValidator()
    dispatcher.register_processor("validator", validator)

    result = dispatcher.execute_with_fallback(
        "validator",
        -5,  # Negative - will fail validation
        fallback_names=["doubler"],
    )
    assert result.is_success
    assert result.unwrap() == -10  # Fell back to doubler
    print(
        f"✅ Fallback executed: validator failed, doubler succeeded: -5 → {result.unwrap()}"
    )


# ==================== EXAMPLE 6: METRICS AND AUDITING ====================


def example_6_metrics_auditing() -> None:
    """Example 6: Access comprehensive metrics and audit trail."""
    print("\n=== EXAMPLE 6: Metrics & Auditing ===\n")

    dispatcher = FlextDispatcher()
    dispatcher.register_processor("doubler", DataDoubler())

    # Execute several operations
    dispatcher.process("doubler", 1)
    dispatcher.process("doubler", 2)
    dispatcher.process_batch("doubler", [3, 4, 5])

    # Get processor metrics
    metrics = dispatcher.processor_metrics
    doubler_metrics = metrics["doubler"]
    print(f"✅ Processor metrics: {doubler_metrics}")
    assert doubler_metrics["executions"] == 5  # 1 + 1 + 3 items

    # Get batch performance
    batch_perf = dispatcher.batch_performance
    print(f"✅ Batch performance: {batch_perf}")

    # Get comprehensive analytics
    analytics_result = dispatcher.get_performance_analytics()
    assert analytics_result.is_success
    analytics = analytics_result.unwrap()
    print(f"✅ Analytics keys: {list(analytics.keys())}")

    # Get audit log
    audit_result = dispatcher.get_process_audit_log()
    assert audit_result.is_success
    audit_log = audit_result.unwrap()
    print(f"✅ Audit log entries: {len(audit_log)}")


# ==================== EXAMPLE 7: INTEGRATED WORKFLOW ====================


def example_7_integrated_workflow() -> None:
    """Example 7: Complete integrated workflow."""
    print("\n=== EXAMPLE 7: Integrated Workflow ===\n")

    dispatcher = FlextDispatcher()

    # Register multiple processors
    dispatcher.register_processor("doubler", DataDoubler())
    dispatcher.register_processor("squarer", DataSquarer())
    dispatcher.register_processor("validator", DataValidator())

    # Step 1: Validate input
    data = 5
    result = dispatcher.process("validator", data)
    if result.is_failure:
        print(f"❌ Validation failed: {result.error}")
        return
    print(f"✅ Validated input: {data}")

    # Step 2: Process batch with doubler
    batch_result = dispatcher.process_batch("doubler", [1, 2, 3, 4, 5])
    assert batch_result.is_success
    batch_processed = batch_result.unwrap()
    print(f"✅ Batch doubled: {batch_processed}")

    # Step 3: Process with parallel squarer
    parallel_result = dispatcher.process_parallel(
        "squarer", [1, 2, 3, 4, 5], max_workers=2
    )
    assert parallel_result.is_success
    parallel_processed = parallel_result.unwrap()
    print(f"✅ Parallel squared: {parallel_processed}")

    # Step 4: Fallback pattern
    fallback_result = dispatcher.execute_with_fallback(
        "doubler", 10, fallback_names=["squarer"]
    )
    assert fallback_result.is_success
    print(f"✅ Fallback execution: 10 → {fallback_result.unwrap()}")

    # Step 5: View comprehensive metrics
    analytics_result = dispatcher.get_performance_analytics()
    if analytics_result.is_success:
        analytics = analytics_result.unwrap()
        print("\n✅ Final Analytics:")
        print(f"   Global metrics: {analytics['global_metrics']}")
        print(f"   Processor metrics: {analytics['processor_metrics']}")
        print(f"   Batch performance: {analytics['batch_performance']}")
        print(f"   Parallel performance: {analytics['parallel_performance']}")


# ==================== MAIN EXECUTION ====================


def main() -> None:
    """Run all examples."""
    print("=" * 70)
    print("Layer 3 Advanced Processing Examples")
    print("=" * 70)

    try:
        example_1_basic_registration()
        example_2_batch_processing()
        example_3_parallel_processing()
        example_4_timeout_enforcement()
        example_5_fallback_chains()
        example_6_metrics_auditing()
        example_7_integrated_workflow()

        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
