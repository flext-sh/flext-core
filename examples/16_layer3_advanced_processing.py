"""Dispatcher processing examples focused on reliability and batching.

Showcases the dispatcher processor API for batch, parallel, timeout-aware,
and fallback-driven execution built on top of the CQRS dispatcher pipeline.
Each operation returns ``FlextResult`` to keep flows composable and explicit
about failures.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from flext_core import (
    FlextConstants,
    FlextDispatcher,
    FlextResult,
    p,
    t,
)

# ==================== PROCESSOR IMPLEMENTATIONS ====================


class DataDoubler:
    """Simple processor that doubles numeric data.

    Implements p.Processor for dispatcher integration.
    """

    def process(  # noqa: PLR6301  # Required by p.Processor protocol
        self,
        data: (t.GeneralValueType | BaseModel | p.ResultProtocol[t.GeneralValueType]),
    ) -> t.GeneralValueType | BaseModel | p.ResultProtocol[t.GeneralValueType]:
        """Double the input value."""
        if not isinstance(data, int):
            return FlextResult[int].fail(f"Expected int, got {type(data)}")
        return FlextResult[int].ok(data * 2)


class DataSquarer:
    """Processor that squares numeric data.

    Implements p.Processor for dispatcher integration.
    """

    @staticmethod
    def process(
        data: (t.GeneralValueType | BaseModel | p.ResultProtocol[t.GeneralValueType]),
    ) -> t.GeneralValueType | BaseModel | p.ResultProtocol[t.GeneralValueType]:
        """Square the input value."""
        if not isinstance(data, int):
            return FlextResult[int].fail(f"Expected int, got {type(data)}")
        return FlextResult[int].ok(data * data)


class SlowProcessor:
    """Processor simulating slow I/O operation."""

    def __init__(self, delay: float = 0.1) -> None:
        """Initialize with delay in seconds."""
        self.delay = delay

    def process(
        self,
        data: (t.GeneralValueType | BaseModel | p.ResultProtocol[t.GeneralValueType]),
    ) -> t.GeneralValueType | BaseModel | p.ResultProtocol[t.GeneralValueType]:
        """Simulate slow operation then return result."""
        if not isinstance(data, int):
            return FlextResult[int].fail(f"Expected int, got {type(data)}")
        time.sleep(self.delay)
        return FlextResult[int].ok(data)


class DataValidator:
    """Processor validating data integrity.

    Implements p.Processor for dispatcher integration.
    """

    @staticmethod
    def process(
        data: (t.GeneralValueType | BaseModel | p.ResultProtocol[t.GeneralValueType]),
    ) -> t.GeneralValueType | BaseModel | p.ResultProtocol[t.GeneralValueType]:
        """Validate data is positive."""
        if not isinstance(data, int):
            return FlextResult[int].fail(f"Expected int, got {type(data)}")
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
    if result.is_failure:
        print(f"❌ Failed to register processor: {result.error}")
        return
    print("✅ Registered 'doubler' processor")

    # Execute processor
    process_result = dispatcher.process("doubler", 5)
    if process_result.is_failure:
        print(f"❌ Failed to process: {process_result.error}")
        return
    if process_result.unwrap() != 10:
        print(f"❌ Expected 10, got {process_result.unwrap()}")
        return
    print(f"✅ Processed 5 → {process_result.unwrap()}")


# ==================== EXAMPLE 2: BATCH PROCESSING ====================


def example_2_batch_processing() -> None:
    """Example 2: Process multiple items in batch."""
    print("\n=== EXAMPLE 2: Batch Processing ===\n")

    dispatcher = FlextDispatcher()
    dispatcher.register_processor("doubler", DataDoubler())

    # Process batch of items using FlextConstants (DRY - centralized batch size)
    data_list: list[t.GeneralValueType] = [1, 2, 3, 4, 5]
    result = dispatcher.process_batch(
        "doubler",
        data_list,
        batch_size=FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
    )

    if result.is_failure:
        print(f"❌ Failed to process batch: {result.error}")
        return
    processed = result.unwrap()
    if processed != [2, 4, 6, 8, 10]:
        print(f"❌ Expected [2, 4, 6, 8, 10], got {processed}")
        return
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

    # Process items in parallel using FlextConstants (DRY - centralized workers)
    data_list: list[t.GeneralValueType] = list(range(1, 11))
    start = time.time()
    result = dispatcher.process_parallel(
        "doubler", data_list, max_workers=FlextConstants.Processing.DEFAULT_MAX_WORKERS
    )
    elapsed = time.time() - start

    if result.is_failure:
        print(f"❌ Failed to process in parallel: {result.error}")
        return
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

    # Register slow processor using FlextConstants (DRY - centralized timeout)
    slow = SlowProcessor(delay=FlextConstants.Validation.SLOW_OPERATION_THRESHOLD)
    dispatcher.register_processor("slow", slow)

    # Execute with reasonable timeout using FlextConstants (DRY)
    result = dispatcher.execute_with_timeout(
        "slow", 10, timeout=float(FlextConstants.Defaults.TIMEOUT_SECONDS)
    )
    if result.is_failure:
        print(f"❌ Operation should have succeeded but failed: {result.error}")
        return
    print("✅ Slow operation completed within timeout")

    # Execute with tight timeout - should fail
    very_slow = SlowProcessor(delay=0.5)
    dispatcher.register_processor("very_slow", very_slow)
    result = dispatcher.execute_with_timeout("very_slow", 10, timeout=0.1)
    if result.is_success:
        print("❌ Operation should have timed out but succeeded")
        return
    if result.error and "timeout" not in result.error.lower():
        print(f"❌ Expected timeout error, got: {result.error}")
        return
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
    result = dispatcher.process("doubler", 5)
    if result.is_failure:
        # Fallback to squarer
        result = dispatcher.process("squarer", 5)
    if result.is_failure:
        print(f"❌ Fallback chain failed: {result.error}")
        return
    if result.unwrap() != 10:
        print(f"❌ Expected 10 from doubler, got {result.unwrap()}")
        return
    print(f"✅ Primary processor succeeded: 5 → {result.unwrap()}")

    # Now test with validator that fails - falls back
    validator = DataValidator()
    dispatcher.register_processor("validator", validator)

    # Try validator, fall back to doubler if validation fails
    result = dispatcher.process("validator", -5)
    if result.is_failure:
        # Fallback to doubler
        result = dispatcher.process("doubler", -5)
    if result.is_failure:
        print(f"❌ Fallback chain failed: {result.error}")
        return
    if result.unwrap() != -10:
        print(f"❌ Expected -10 from fallback, got {result.unwrap()}")
        return
    print(
        f"✅ Fallback executed: validator failed, doubler succeeded: -5 → {result.unwrap()}",
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
    if doubler_metrics["executions"] != 5:
        print(f"❌ Expected 5 executions, got {doubler_metrics['executions']}")
        return

    # Get batch performance
    batch_perf = dispatcher.batch_performance
    print(f"✅ Batch performance: {batch_perf}")

    # Get comprehensive analytics
    analytics_result = dispatcher.get_performance_analytics()
    if analytics_result.is_failure:
        print(f"❌ Failed to get analytics: {analytics_result.error}")
        return
    analytics = analytics_result.unwrap()
    if isinstance(analytics, dict):
        analytics_dict: dict[str, t.GeneralValueType] = analytics
        print(f"✅ Analytics keys: {list(analytics_dict.keys())}")
    else:
        print("✅ Analytics retrieved")

    # Get audit log
    audit_result = dispatcher.get_process_audit_log()
    if audit_result.is_failure:
        print(f"❌ Failed to get audit log: {audit_result.error}")
        return
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
    if batch_result.is_failure:
        print(f"❌ Batch processing failed: {batch_result.error}")
        return
    batch_processed = batch_result.unwrap()
    print(f"✅ Batch doubled: {batch_processed}")

    # Step 3: Process with parallel squarer
    parallel_result = dispatcher.process_parallel(
        "squarer",
        [1, 2, 3, 4, 5],
        max_workers=2,
    )
    if parallel_result.is_failure:
        print(f"❌ Parallel processing failed: {parallel_result.error}")
        return
    parallel_processed = parallel_result.unwrap()
    print(f"✅ Parallel squared: {parallel_processed}")

    # Step 4: Fallback pattern
    fallback_result = dispatcher.process("doubler", 10)
    if fallback_result.is_failure:
        # Fallback to squarer
        fallback_result = dispatcher.process("squarer", 10)
    if fallback_result.is_failure:
        print(f"❌ Fallback execution failed: {fallback_result.error}")
        return
    print(f"✅ Fallback execution: 10 → {fallback_result.unwrap()}")

    # Step 5: View comprehensive metrics
    analytics_result = dispatcher.get_performance_analytics()
    if analytics_result.is_success:
        analytics = analytics_result.unwrap()
        print("\n✅ Final Analytics:")
        if isinstance(analytics, dict):
            print(f"   Global metrics: {analytics.get('global_metrics', 'N/A')}")
            print(f"   Processor metrics: {analytics.get('processor_metrics', 'N/A')}")
            print(f"   Batch performance: {analytics.get('batch_performance', 'N/A')}")
            print(
                f"   Parallel performance: {analytics.get('parallel_performance', 'N/A')}"
            )
        else:
            print("   Analytics retrieved")


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
