"""Comprehensive tests for FlextProtocols - Protocol Definitions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

from flext_core import FlextProtocols


class TestFlextProtocols:
    """Test suite for FlextProtocols protocol functionality."""

    def test_protocols_initialization(self) -> None:
        """Test protocols initialization."""
        protocols = FlextProtocols()
        assert protocols is not None
        assert isinstance(protocols, FlextProtocols)

    def test_protocols_with_custom_config(self) -> None:
        """Test protocols initialization with custom configuration."""
        config = {"max_retries": 3, "timeout": 30}
        protocols = FlextProtocols(config=config)
        assert protocols is not None

    def test_protocols_register_protocol(self) -> None:
        """Test protocol registration."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        result = protocols.register("test_protocol", TestProtocol)
        assert result.is_success

    def test_protocols_register_protocol_invalid(self) -> None:
        """Test protocol registration with invalid parameters."""
        protocols = FlextProtocols()

        result = protocols.register("", None)  # type: ignore[arg-type]
        assert result.is_failure

    def test_protocols_unregister_protocol(self) -> None:
        """Test protocol unregistration."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        protocols.register("test_protocol", TestProtocol)
        result = protocols.unregister("test_protocol", TestProtocol)
        assert result.is_success

    def test_protocols_unregister_nonexistent_protocol(self) -> None:
        """Test unregistering non-existent protocol."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        result = protocols.unregister("nonexistent_protocol", TestProtocol)
        assert result.is_failure

    def test_protocols_validate_implementation(self) -> None:
        """Test protocol implementation validation."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_success

    def test_protocols_validate_nonexistent_protocol(self) -> None:
        """Test validation with non-existent protocol."""
        protocols = FlextProtocols()

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        result = protocols.validate_implementation(
            "nonexistent_protocol", TestImplementation
        )
        assert result.is_failure
        assert "No protocol found" in result.error

    def test_protocols_validate_invalid_implementation(self) -> None:
        """Test validation with invalid implementation."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class InvalidImplementation:
            def wrong_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation(
            "test_protocol", InvalidImplementation
        )
        assert result.is_failure
        assert "Implementation validation failed" in result.error

    def test_protocols_validate_with_retry(self) -> None:
        """Test validation with retry mechanism."""
        protocols = FlextProtocols(config={"max_retries": 3, "retry_delay": 0.01})

        retry_count = 0

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class RetryImplementation:
            def test_method(self) -> str:
                nonlocal retry_count
                retry_count += 1
                if retry_count < 3:
                    msg = "Temporary failure"
                    raise ValueError(msg)
                return f"success_after_{retry_count}_retries"

        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation("test_protocol", RetryImplementation)
        assert result.is_success
        assert "success_after_3_retries" in result.data

    def test_protocols_validate_with_timeout(self) -> None:
        """Test validation with timeout."""
        protocols = FlextProtocols(config={"timeout": 0.1})

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TimeoutImplementation:
            def test_method(self) -> str:
                time.sleep(0.2)  # Exceed timeout
                return "should_not_reach_here"

        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation(
            "test_protocol", TimeoutImplementation
        )
        assert result.is_failure
        assert "timeout" in result.error.lower()

    def test_protocols_validate_with_validation(self) -> None:
        """Test validation with validation."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class ValidatedImplementation:
            def test_method(self) -> str:
                return "validated_result"

        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation(
            "test_protocol", ValidatedImplementation
        )
        assert result.is_success

    def test_protocols_validate_with_middleware(self) -> None:
        """Test validation with middleware."""
        protocols = FlextProtocols()

        middleware_called = False

        def middleware(
            protocol_class: object, implementation_class: object
        ) -> tuple[object, object]:
            nonlocal middleware_called
            middleware_called = True
            return protocol_class, implementation_class

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.add_middleware(middleware)
        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_success
        assert middleware_called is True

    def test_protocols_validate_with_logging(self) -> None:
        """Test validation with logging."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_success

    def test_protocols_validate_with_metrics(self) -> None:
        """Test validation with metrics."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_success

        # Check metrics
        metrics = protocols.get_metrics()
        assert "test_protocol" in metrics
        assert metrics["test_protocol"]["validations"] >= 1

    def test_protocols_validate_with_correlation_id(self) -> None:
        """Test validation with correlation ID."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)
        result = protocols.validate_implementation(
            "test_protocol", TestImplementation, correlation_id="test_corr_123"
        )
        assert result.is_success

    def test_protocols_validate_with_batch(self) -> None:
        """Test validation with batch processing."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation1:
            def test_method(self) -> str:
                return "test_result_1"

        class TestImplementation2:
            def test_method(self) -> str:
                return "test_result_2"

        class TestImplementation3:
            def test_method(self) -> str:
                return "test_result_3"

        protocols.register("test_protocol", TestProtocol)

        implementations = [
            TestImplementation1,
            TestImplementation2,
            TestImplementation3,
        ]
        results = protocols.validate_batch("test_protocol", implementations)
        assert len(results) == 3
        assert all(result.is_success for result in results)

    def test_protocols_validate_with_parallel(self) -> None:
        """Test validation with parallel processing."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation1:
            def test_method(self) -> str:
                time.sleep(0.1)  # Simulate work
                return "test_result_1"

        class TestImplementation2:
            def test_method(self) -> str:
                time.sleep(0.1)  # Simulate work
                return "test_result_2"

        class TestImplementation3:
            def test_method(self) -> str:
                time.sleep(0.1)  # Simulate work
                return "test_result_3"

        protocols.register("test_protocol", TestProtocol)

        implementations = [
            TestImplementation1,
            TestImplementation2,
            TestImplementation3,
        ]

        start_time = time.time()
        results = protocols.validate_parallel("test_protocol", implementations)
        end_time = time.time()

        assert len(results) == 3
        assert all(result.is_success for result in results)
        # Should complete faster than sequential execution
        assert end_time - start_time < 0.3

    def test_protocols_validate_with_circuit_breaker(self) -> None:
        """Test validation with circuit breaker."""
        protocols = FlextProtocols(config={"circuit_breaker_threshold": 3})

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class FailingImplementation:
            def test_method(self) -> str:
                msg = "Service unavailable"
                raise ValueError(msg)

        protocols.register("test_protocol", TestProtocol)

        # Execute failing validations to trigger circuit breaker
        for _ in range(5):
            result = protocols.validate_implementation(
                "test_protocol", FailingImplementation
            )
            assert result.is_failure

        # Circuit breaker should be open
        assert protocols.is_circuit_breaker_open("test_protocol") is True

    def test_protocols_validate_with_rate_limiting(self) -> None:
        """Test validation with rate limiting."""
        protocols = FlextProtocols(config={"rate_limit": 2, "rate_limit_window": 1})

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)

        # Execute validations within rate limit
        for _i in range(2):
            result = protocols.validate_implementation(
                "test_protocol", TestImplementation
            )
            assert result.is_success

        # Exceed rate limit
        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_failure
        assert "rate limit" in result.error.lower()

    def test_protocols_validate_with_caching(self) -> None:
        """Test validation with caching."""
        protocols = FlextProtocols(config={"cache_ttl": 60})

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)

        # First validation should cache result
        result1 = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result1.is_success

        # Second validation should use cache
        result2 = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result2.is_success
        assert result1.data == result2.data

    def test_protocols_validate_with_audit(self) -> None:
        """Test validation with audit logging."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)

        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_success

        # Check audit log
        audit_log = protocols.get_audit_log()
        assert len(audit_log) >= 1
        assert audit_log[0]["protocol_name"] == "test_protocol"

    def test_protocols_validate_with_performance_monitoring(self) -> None:
        """Test validation with performance monitoring."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                time.sleep(0.1)  # Simulate work
                return "test_result"

        protocols.register("test_protocol", TestProtocol)

        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_success

        # Check performance metrics
        performance = protocols.get_performance_metrics()
        assert "test_protocol" in performance
        assert performance["test_protocol"]["avg_validation_time"] >= 0.1

    def test_protocols_validate_with_error_handling(self) -> None:
        """Test validation with error handling."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class ErrorImplementation:
            def test_method(self) -> str:
                msg = "Implementation error"
                raise RuntimeError(msg)

        protocols.register("test_protocol", TestProtocol)

        result = protocols.validate_implementation("test_protocol", ErrorImplementation)
        assert result.is_failure
        assert "Implementation error" in result.error

    def test_protocols_validate_with_cleanup(self) -> None:
        """Test validation with cleanup."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)

        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_success

        # Cleanup
        protocols.cleanup()

        # After cleanup, protocols should be cleared
        result = protocols.validate_implementation("test_protocol", TestImplementation)
        assert result.is_failure
        assert "No protocol found" in result.error

    def test_protocols_get_registered_protocols(self) -> None:
        """Test getting registered protocols."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        protocols.register("test_protocol", TestProtocol)
        registered_protocols = protocols.get_protocols("test_protocol")
        assert len(registered_protocols) == 1
        assert TestProtocol in registered_protocols

    def test_protocols_get_protocols_for_nonexistent_protocol(self) -> None:
        """Test getting protocols for non-existent protocol."""
        protocols = FlextProtocols()

        registered_protocols = protocols.get_protocols("nonexistent_protocol")
        assert len(registered_protocols) == 0

    def test_protocols_clear_protocols(self) -> None:
        """Test clearing all protocols."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        protocols.register("test_protocol", TestProtocol)
        protocols.clear_protocols()

        registered_protocols = protocols.get_protocols("test_protocol")
        assert len(registered_protocols) == 0

    def test_protocols_statistics(self) -> None:
        """Test protocols statistics tracking."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)
        protocols.validate_implementation("test_protocol", TestImplementation)

        stats = protocols.get_statistics()
        assert "test_protocol" in stats
        assert stats["test_protocol"]["validations"] >= 1

    def test_protocols_thread_safety(self) -> None:
        """Test protocols thread safety."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)

        results = []

        def validate_implementation(_thread_id: int) -> None:
            result = protocols.validate_implementation(
                "test_protocol", TestImplementation
            )
            results.append(result)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=validate_implementation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.is_success for result in results)

    def test_protocols_performance(self) -> None:
        """Test protocols performance characteristics."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class FastImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.register("test_protocol", TestProtocol)

        start_time = time.time()

        # Perform many operations
        for _i in range(100):
            protocols.validate_implementation("test_protocol", FastImplementation)

        end_time = time.time()

        # Should complete 100 operations in reasonable time
        assert end_time - start_time < 1.0

    def test_protocols_error_handling(self) -> None:
        """Test protocols error handling mechanisms."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        class ErrorImplementation:
            def test_method(self) -> str:
                msg = "Implementation error"
                raise ValueError(msg)

        protocols.register("test_protocol", TestProtocol)

        result = protocols.validate_implementation("test_protocol", ErrorImplementation)
        assert result.is_failure
        assert "Implementation error" in result.error

    def test_protocols_validation(self) -> None:
        """Test protocols validation."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        protocols.register("test_protocol", TestProtocol)

        result = protocols.validate()
        assert result.is_success

    def test_protocols_export_import(self) -> None:
        """Test protocols export/import."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        protocols.register("test_protocol", TestProtocol)

        # Export protocols configuration
        config = protocols.export_config()
        assert isinstance(config, dict)
        assert "test_protocol" in config

        # Create new protocols and import configuration
        new_protocols = FlextProtocols()
        result = new_protocols.import_config(config)
        assert result.is_success

        # Verify protocol is available in new protocols
        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        result = new_protocols.validate_implementation(
            "test_protocol", TestImplementation
        )
        assert result.is_success

    def test_protocols_cleanup(self) -> None:
        """Test protocols cleanup."""
        protocols = FlextProtocols()

        class TestProtocol(Protocol):
            def test_method(self) -> str: ...

        protocols.register("test_protocol", TestProtocol)

        class TestImplementation:
            def test_method(self) -> str:
                return "test_result"

        protocols.validate_implementation("test_protocol", TestImplementation)

        protocols.cleanup()

        # After cleanup, protocols should be cleared
        registered_protocols = protocols.get_protocols("test_protocol")
        assert len(registered_protocols) == 0
