"""Comprehensive tests for FlextProcessors - Data Processing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time

from flext_core import FlextProcessors, FlextResult


class TestFlextProcessors:
    """Test suite for FlextProcessors data processing functionality."""

    def test_processors_initialization(self) -> None:
        """Test processors initialization."""
        processors = FlextProcessors()
        assert processors is not None
        assert isinstance(processors, FlextProcessors)

    def test_processors_with_custom_config(self) -> None:
        """Test processors initialization with custom configuration."""
        config = {"max_retries": 3, "timeout": 30}
        processors = FlextProcessors(config=config)
        assert processors is not None

    def test_processors_register_processor(self) -> None:
        """Test processor registration."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        result = processors.register("test_processor", test_processor)
        assert result.is_success

    def test_processors_register_processor_invalid(self) -> None:
        """Test processor registration with invalid parameters."""
        processors = FlextProcessors()

        result = processors.register("", None)  # type: ignore[arg-type]
        assert result.is_failure

    def test_processors_unregister_processor(self) -> None:
        """Test processor unregistration."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        result = processors.unregister("test_processor", test_processor)
        assert result.is_success

    def test_processors_unregister_nonexistent_processor(self) -> None:
        """Test unregistering non-existent processor."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        result = processors.unregister("nonexistent_processor", test_processor)
        assert result.is_failure

    def test_processors_process_data(self) -> None:
        """Test data processing."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        result = processors.process("test_processor", "test_data")
        assert result.is_success
        assert "processed_test_data" in result.data

    def test_processors_process_nonexistent_processor(self) -> None:
        """Test processing with non-existent processor."""
        processors = FlextProcessors()

        result = processors.process("nonexistent_processor", "test_data")
        assert result.is_failure
        assert "No processor found" in result.error

    def test_processors_process_with_failing_processor(self) -> None:
        """Test processing with failing processor."""
        processors = FlextProcessors()

        def failing_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].fail("Processor failed")

        processors.register("test_processor", failing_processor)
        result = processors.process("test_processor", "test_data")
        assert result.is_failure
        assert "Processor failed" in result.error

    def test_processors_process_with_exception(self) -> None:
        """Test processing with exception."""
        processors = FlextProcessors()

        def exception_processor(_data: object) -> FlextResult[str]:
            msg = "Processor exception"
            raise ValueError(msg)

        processors.register("test_processor", exception_processor)
        result = processors.process("test_processor", "test_data")
        assert result.is_failure
        assert "Processor exception" in result.error

    def test_processors_process_with_retry(self) -> None:
        """Test processing with retry mechanism."""
        processors = FlextProcessors(config={"max_retries": 3, "retry_delay": 0.01})

        retry_count = 0

        def retry_processor(_data: object) -> FlextResult[str]:
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok(f"success_after_{retry_count}_retries")

        processors.register("test_processor", retry_processor)
        result = processors.process("test_processor", "test_data")
        assert result.is_success
        assert "success_after_3_retries" in result.data

    def test_processors_process_with_timeout(self) -> None:
        """Test processing with timeout."""
        processors = FlextProcessors(config={"timeout": 0.1})

        def timeout_processor(_data: object) -> FlextResult[str]:
            time.sleep(0.2)  # Exceed timeout
            return FlextResult[str].ok("should_not_reach_here")

        processors.register("test_processor", timeout_processor)
        result = processors.process("test_processor", "test_data")
        assert result.is_failure
        assert "timeout" in result.error.lower()

    def test_processors_process_with_validation(self) -> None:
        """Test processing with validation."""
        processors = FlextProcessors()

        def validated_processor(_data: object) -> FlextResult[str]:
            if not _data:
                return FlextResult[str].fail("Data is required")
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", validated_processor)

        # Valid data
        result = processors.process("test_processor", "valid_data")
        assert result.is_success

        # Invalid data
        result = processors.process("test_processor", "")
        assert result.is_failure
        assert "Data is required" in result.error

    def test_processors_process_with_middleware(self) -> None:
        """Test processing with middleware."""
        processors = FlextProcessors()

        middleware_called = False

        def middleware(data: object) -> object:
            nonlocal middleware_called
            middleware_called = True
            return data

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.add_middleware(middleware)
        processors.register("test_processor", test_processor)
        result = processors.process("test_processor", "test_data")
        assert result.is_success
        assert middleware_called is True

    def test_processors_process_with_logging(self) -> None:
        """Test processing with logging."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        result = processors.process("test_processor", "test_data")
        assert result.is_success

    def test_processors_process_with_metrics(self) -> None:
        """Test processing with metrics."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        result = processors.process("test_processor", "test_data")
        assert result.is_success

        # Check metrics
        metrics = processors.get_metrics()
        assert "test_processor" in metrics
        assert metrics["test_processor"]["processings"] >= 1

    def test_processors_process_with_correlation_id(self) -> None:
        """Test processing with correlation ID."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        result = processors.process(
            "test_processor", "test_data", correlation_id="test_corr_123"
        )
        assert result.is_success

    def test_processors_process_with_batch(self) -> None:
        """Test processing with batch processing."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        data_list = ["test1", "test2", "test3"]
        results = processors.process_batch("test_processor", data_list)
        assert len(results) == 3
        assert all(result.is_success for result in results)

    def test_processors_process_with_parallel(self) -> None:
        """Test processing with parallel processing."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            time.sleep(0.1)  # Simulate work
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        data_list = ["test1", "test2", "test3"]

        start_time = time.time()
        results = processors.process_parallel("test_processor", data_list)
        end_time = time.time()

        assert len(results) == 3
        assert all(result.is_success for result in results)
        # Should complete faster than sequential execution
        assert end_time - start_time < 0.3

    def test_processors_process_with_circuit_breaker(self) -> None:
        """Test processing with circuit breaker."""
        processors = FlextProcessors(config={"circuit_breaker_threshold": 3})

        def failing_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].fail("Service unavailable")

        processors.register("test_processor", failing_processor)

        # Execute failing processings to trigger circuit breaker
        for _ in range(5):
            result = processors.process("test_processor", "test_data")
            assert result.is_failure

        # Circuit breaker should be open
        assert processors.is_circuit_breaker_open("test_processor") is True

    def test_processors_process_with_rate_limiting(self) -> None:
        """Test processing with rate limiting."""
        processors = FlextProcessors(config={"rate_limit": 2, "rate_limit_window": 1})

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        # Execute processings within rate limit
        for i in range(2):
            result = processors.process("test_processor", f"test{i}")
            assert result.is_success

        # Exceed rate limit
        result = processors.process("test_processor", "test3")
        assert result.is_failure
        assert "rate limit" in result.error.lower()

    def test_processors_process_with_caching(self) -> None:
        """Test processing with caching."""
        processors = FlextProcessors(config={"cache_ttl": 60})

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        # First processing should cache result
        result1 = processors.process("test_processor", "test_data")
        assert result1.is_success

        # Second processing should use cache
        result2 = processors.process("test_processor", "test_data")
        assert result2.is_success
        assert result1.data == result2.data

    def test_processors_process_with_audit(self) -> None:
        """Test processing with audit logging."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        result = processors.process("test_processor", "test_data")
        assert result.is_success

        # Check audit log
        audit_log = processors.get_audit_log()
        assert len(audit_log) >= 1
        assert audit_log[0]["processor_name"] == "test_processor"

    def test_processors_process_with_performance_monitoring(self) -> None:
        """Test processing with performance monitoring."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            time.sleep(0.1)  # Simulate work
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        result = processors.process("test_processor", "test_data")
        assert result.is_success

        # Check performance metrics
        performance = processors.get_performance_metrics()
        assert "test_processor" in performance
        assert performance["test_processor"]["avg_processing_time"] >= 0.1

    def test_processors_process_with_error_handling(self) -> None:
        """Test processing with error handling."""
        processors = FlextProcessors()

        def error_processor(_data: object) -> FlextResult[str]:
            msg = "Processor error"
            raise RuntimeError(msg)

        processors.register("test_processor", error_processor)

        result = processors.process("test_processor", "test_data")
        assert result.is_failure
        assert "Processor error" in result.error

    def test_processors_process_with_cleanup(self) -> None:
        """Test processing with cleanup."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        result = processors.process("test_processor", "test_data")
        assert result.is_success

        # Cleanup
        processors.cleanup()

        # After cleanup, processors should be cleared
        result = processors.process("test_processor", "test_data")
        assert result.is_failure
        assert "No processor found" in result.error

    def test_processors_get_registered_processors(self) -> None:
        """Test getting registered processors."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        registered_processors = processors.get_processors("test_processor")
        assert len(registered_processors) == 1
        assert test_processor in registered_processors

    def test_processors_get_processors_for_nonexistent_processor(self) -> None:
        """Test getting processors for non-existent processor."""
        processors = FlextProcessors()

        registered_processors = processors.get_processors("nonexistent_processor")
        assert len(registered_processors) == 0

    def test_processors_clear_processors(self) -> None:
        """Test clearing all processors."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        processors.clear_processors()

        registered_processors = processors.get_processors("test_processor")
        assert len(registered_processors) == 0

    def test_processors_statistics(self) -> None:
        """Test processors statistics tracking."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        processors.process("test_processor", "test_data")

        stats = processors.get_statistics()
        assert "test_processor" in stats
        assert stats["test_processor"]["processings"] >= 1

    def test_processors_thread_safety(self) -> None:
        """Test processors thread safety."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        results = []

        def process_data(thread_id: int) -> None:
            result = processors.process("test_processor", f"data_{thread_id}")
            results.append(result)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_data, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.is_success for result in results)

    def test_processors_performance(self) -> None:
        """Test processors performance characteristics."""
        processors = FlextProcessors()

        def fast_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", fast_processor)

        start_time = time.time()

        # Perform many operations
        for i in range(100):
            processors.process("test_processor", f"data_{i}")

        end_time = time.time()

        # Should complete 100 operations in reasonable time
        assert end_time - start_time < 1.0

    def test_processors_error_handling(self) -> None:
        """Test processors error handling mechanisms."""
        processors = FlextProcessors()

        def error_processor(_data: object) -> FlextResult[str]:
            msg = "Processor error"
            raise ValueError(msg)

        processors.register("test_processor", error_processor)

        result = processors.process("test_processor", "test_data")
        assert result.is_failure
        assert "Processor error" in result.error

    def test_processors_validation(self) -> None:
        """Test processors validation."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        result = processors.validate()
        assert result.is_success

    def test_processors_export_import(self) -> None:
        """Test processors export/import."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)

        # Export processors configuration
        config = processors.export_config()
        assert isinstance(config, dict)
        assert "test_processor" in config

        # Create new processors and import configuration
        new_processors = FlextProcessors()
        result = new_processors.import_config(config)
        assert result.is_success

        # Verify processor is available in new processors
        result = new_processors.process("test_processor", "test_data")
        assert result.is_success
        assert "processed_test_data" in result.data

    def test_processors_cleanup(self) -> None:
        """Test processors cleanup."""
        processors = FlextProcessors()

        def test_processor(_data: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_data}")

        processors.register("test_processor", test_processor)
        processors.process("test_processor", "test_data")

        processors.cleanup()

        # After cleanup, processors should be cleared
        registered_processors = processors.get_processors("test_processor")
        assert len(registered_processors) == 0
