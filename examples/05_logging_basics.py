# !/usr/bin/env python3
"""05 - FlextLogger Fundamentals: Complete Structured Logging.

This example demonstrates the COMPLETE FlextLogger API - the foundation
for structured logging across the entire FLEXT ecosystem. FlextLogger provides
context-aware, structured logging with correlation tracking and performance metrics.

Key Concepts Demonstrated:
- Logger Creation: Direct instantiation and configuration
- Log Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Structured Logging: Key-value pairs and context
- Context Management: bind(), unbind(), contextualize()
- Correlation IDs: Request tracking across services
- Performance Tracking: Timing and metrics
- Exception Logging: Structured error capture
- Child Loggers: Hierarchical logging contexts
- Global Configuration: Configure all loggers at once

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import warnings
from datetime import UTC, datetime
from uuid import uuid4

from flext_core import FlextCore

from .example_scenarios import ExampleScenarios


class ComprehensiveLoggingService(FlextCore.Service[FlextCore.Types.StringDict]):
    """Service demonstrating ALL FlextLogger patterns and methods."""

    _logger: FlextCore.Logger
    _container: FlextCore.Container
    _config: FlextCore.Config

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        manager = FlextCore.Container.ensure_global_manager()
        self._container = manager.get_or_create()
        self._logger = FlextCore.Logger(__name__)
        self._config = FlextCore.Config()
        self._scenarios = ExampleScenarios
        self._metadata = self._scenarios.metadata(tags=["logging", "demo"])
        self._user = self._scenarios.user()
        self._payload = self._scenarios.payload()

    def execute(self) -> FlextCore.Result[FlextCore.Types.StringDict]:
        """Execute method required by FlextService."""
        # This is a demonstration service, logs and returns status
        self._logger.info(
            "Executing logging demonstration",
            extra={"data": {"demo": "logging"}},
        )
        return FlextCore.Result[FlextCore.Types.StringDict].ok({
            "status": "completed",
            "logged": "True",
        })

    # ========== BASIC LOGGING ==========

    def demonstrate_basic_logging(self) -> None:
        """Show basic logging patterns."""
        print("\n=== Basic Logging ===")

        # Create logger
        logger = FlextCore.Logger(__name__)
        print(f"✅ Logger created: {logger}")

        # Log at different levels
        logger.debug("Debug message - detailed information")
        logger.info("Info message - general information")
        logger.warning("Warning message - something to watch")
        logger.error("Error message - something went wrong")
        logger.critical("Critical message - system failure")

        # Log level information
        print("All log levels demonstrated successfully")

    # ========== STRUCTURED LOGGING ==========

    def demonstrate_structured_logging(self) -> None:
        """Show structured logging with key-value pairs."""
        print("\n=== Structured Logging ===")

        logger = FlextCore.Logger(__name__)

        logger.info(
            "User action",
            extra={
                "user_id": self._user["id"],
                "action": "login",
                "ip_address": "192.168.1.1",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        logger.info(
            "Request processed",
            extra={
                "request_id": str(uuid4()),
                "method": "POST",
                "path": "/api/users",
                "status_code": 200,
                "response_time_ms": 45.2,
                "scenario": self._metadata,
            },
        )

        logger.info(
            "Order completed",
            extra={
                "order": self._payload,
                "customer": {
                    "id": self._user["id"],
                    "tier": "premium",
                },
            },
        )

    # ========== CONTEXT MANAGEMENT ==========

    def demonstrate_context_management(self) -> None:
        """Show context binding and management."""
        print("\n=== Context Management ===")

        logger = FlextCore.Logger(__name__)

        logger.bind(request_id=str(uuid4()), user_id=self._user["id"])
        logger.info("Starting process")

        logger.bind(session_id="session-456", tags=self._metadata["tags"])
        logger.info("Process step 1")

        logger.info("Process step 2", extra={"step": 2, "progress": 0.5})

        logger.info("Process completed")
        logger.info("Clean slate")  # No context

    # ========== CONTEXTUALIZE DECORATOR ==========

    def demonstrate_contextualize(self) -> None:
        """Show contextualize context manager."""
        print("\n=== Contextualize Context Manager ===")

        logger = FlextCore.Logger(__name__)

        # Normal log without context
        logger.info("Before context")

        # Log processing steps
        logger.info("Processing started")

        # Simulate processing steps
        for i in range(3):
            logger.info("Processing item", extra={"item_index": i})

        logger.info("Processing completed")
        logger.info("After context")

    # ========== PERFORMANCE TRACKING ==========

    def demonstrate_performance_tracking(self) -> None:
        """Show performance and timing features."""
        print("\n=== Performance Tracking ===")

        logger = FlextCore.Logger(__name__)

        # Manual timing
        start_time = time.time()

        # Simulate work
        time.sleep(0.1)

        elapsed = time.time() - start_time
        logger.info(
            "Operation completed",
            extra={
                "duration_seconds": elapsed,
                "duration_ms": elapsed * 1000,
            },
        )

        # Log with performance metrics
        logger.info(
            "Database query",
            extra={
                "query": "SELECT * FROM users",
                "rows_returned": 150,
                "execution_time_ms": 23.5,
                "cache_hit": False,
            },
        )

        # Memory and resource metrics
        logger.info(
            "Resource usage",
            extra={
                "memory_mb": 256.5,
                "cpu_percent": 45.2,
                "threads": 10,
                "connections": 5,
            },
        )

    # ========== EXCEPTION LOGGING ==========

    def demonstrate_exception_logging(self) -> None:
        """Show exception logging patterns."""
        print("\n=== Exception Logging ===")

        logger = FlextCore.Logger(__name__)

        # Log exceptions with traceback using FlextResult pattern
        def risky_division() -> int:
            return 1 // 0  # Will raise ZeroDivisionError

        result: FlextCore.Result[int] = FlextCore.Result[int].safe_call(risky_division)
        if result.is_failure:
            logger.error(
                "Division failed",
                extra={
                    "operation": "division",
                    "numerator": 1,
                    "denominator": 0,
                    "error": result.error,
                },
            )

        # Log exception without traceback
        try:
            {"key": "value"}["missing_key"]
        except KeyError as e:
            logger.warning("Key not found: %s", e, extra={"available_keys": ["key"]})

        # Log with custom error details
        error_details = {
            "error_code": FlextCore.Constants.Errors.VALIDATION_ERROR,
            "field": "email",
            "value": "invalid-email",
            "reason": "Missing @ symbol",
        }
        logger.error("Validation failed", extra=error_details)

    # ========== CORRELATION TRACKING ==========

    def demonstrate_correlation_tracking(self) -> None:
        """Show correlation ID tracking across operations."""
        print("\n=== Correlation Tracking ===")

        # Create correlation ID for request
        correlation_id = str(uuid4())

        # Create logger with correlation ID
        logger = FlextCore.Logger(__name__)
        logger.bind(correlation_id=correlation_id)

        # All logs now have correlation ID
        logger.info("Request received")

        # Simulate service calls
        self._call_service_a(correlation_id)
        self._call_service_b(correlation_id)

        logger.info("Request completed")

    def _call_service_a(self, correlation_id: str) -> None:
        """Simulate service A call with correlation."""
        logger = FlextCore.Logger("service_a")
        logger.bind(correlation_id=correlation_id, service="service_a")
        logger.info("Service A processing")
        time.sleep(0.05)
        logger.info("Service A completed")

    def _call_service_b(self, correlation_id: str) -> None:
        """Simulate service B call with correlation."""
        logger = FlextCore.Logger("service_b")
        logger.bind(correlation_id=correlation_id, service="service_b")
        logger.info("Service B processing")
        time.sleep(0.05)
        logger.info("Service B completed")

    # ========== CHILD LOGGERS ==========

    def demonstrate_child_loggers(self) -> None:
        """Show child logger hierarchy."""
        print("\n=== Child Loggers ===")

        # Create parent logger
        parent_logger = FlextCore.Logger("parent")
        parent_logger.bind(app="main_app")

        # Create child loggers
        child1 = FlextCore.Logger("parent.child1")
        child1.bind(module="module1")

        child2 = FlextCore.Logger("parent.child2")
        child2.bind(module="module2")

        # Log from different levels
        parent_logger.info("Parent operation")
        child1.info("Child 1 operation")
        child2.info("Child 2 operation")

        # Grandchild logger
        grandchild = FlextCore.Logger("parent.child1.grandchild")
        grandchild.bind(component="component1")
        grandchild.info("Grandchild operation")

    # ========== GLOBAL CONFIGURATION ==========

    def demonstrate_global_configuration(self) -> None:
        """Show global logger configuration."""
        print("\n=== Global Logger Configuration ===")

        # Configure all loggers globally
        result = FlextCore.Logger.configure(
            log_level=FlextCore.Constants.Logging.INFO,
            json_output=False,
            include_source=True,
            structured_output=True,
            log_verbosity=FlextCore.Constants.Logging.VERBOSITY,
        )

        if result.is_success:
            print("✅ Global logging configured")
        else:
            print(f"❌ Configuration failed: {result.error}")

        # Check current configuration
        config = FlextCore.Config()
        print(f"Current log level: {config.log_level}")
        print(f"JSON output: {config.json_output}")
        print(f"Include source: {config.include_source}")
        print(f"Log verbosity: {config.log_verbosity}")

    # ========== LOG FILTERING ==========

    def demonstrate_log_filtering(self) -> None:
        """Show log filtering and conditional logging."""
        print("\n=== Log Filtering ===")

        logger = FlextCore.Logger(__name__)

        # Conditional logging based on level
        # Note: In production, you would check if debug level is enabled
        debug_data = self._compute_expensive_debug_data()
        logger.debug("Debug data", extra=debug_data)

        # Log sampling for high-frequency events
        for i in range(100):
            if i % 10 == 0:  # Log every 10th event
                logger.info("Sampled event", extra={"index": i, "sampled": True})

        # Log with tags for filtering
        logger.info("User event", extra={"tags": ["user", "action", "important"]})
        logger.debug("System event", extra={"tags": ["system", "background"]})

    def _compute_expensive_debug_data(self) -> dict[str, str | list[int]]:
        """Simulate expensive debug data computation."""
        return {
            "expensive_computation": "result",
            "debug_metrics": [1, 2, 3, 4, 5],
        }

    # ========== CUSTOM FIELDS ==========

    def demonstrate_custom_fields(self) -> None:
        """Show custom field addition."""
        print("\n=== Custom Fields ===")

        logger = FlextCore.Logger(__name__)

        # Add custom fields that persist
        logger.bind(
            environment=self._config.environment,
            version="1.0.0",
            instance_id=str(uuid4()),
        )

        # Log with persistent custom fields
        logger.info("Application started")

        # Add temporary fields for specific log
        logger.info(
            "Feature used",
            extra={
                "feature": "data_export",
                "format": "csv",
                "rows": 1000,
            },
        )

        # Business-specific fields
        logger.info(
            "Payment processed",
            extra={
                "payment_id": "PAY-123",
                "amount": 99.99,
                "currency": "USD",
                "method": "credit_card",
                "merchant": "ACME Corp",
            },
        )

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated logging patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Print statements (DEPRECATED)
        warnings.warn(
            "Print statements are DEPRECATED! Use FlextCore.Logger.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (print):")
        print("print('Debug:', variable)")
        print("print(f'Error: {error}')")

        print("\n✅ CORRECT WAY (FlextLogger):")
        print("logger = FlextCore.Logger(__name__)")
        print("logger.debug('Debug info', extra={'variable': variable})")

        # OLD: Basic logging module (DEPRECATED)
        warnings.warn(
            "Basic logging module is DEPRECATED! Use FlextLogger for structured logging.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (logging module):")
        print("import logging")
        print("logging.info('Message')")

        print("\n✅ CORRECT WAY (FlextLogger):")
        print("from flext_core import FlextLogger")
        print("logger = FlextCore.Logger(__name__)")
        print("logger.info('Message', extra={'context': 'data'})")

        # OLD: String formatting in logs (DEPRECATED)
        warnings.warn(
            "String formatting in logs is DEPRECATED! Use structured logging.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (string formatting):")
        print("logger.info(f'User {user_id} logged in from {ip}')")

        print("\n✅ CORRECT WAY (structured):")
        print("logger.info('User login', extra={'user_id': user_id, 'ip': ip})")


def main() -> None:
    """Main entry point demonstrating all FlextLogger capabilities."""
    service = ComprehensiveLoggingService()

    print("=" * 60)
    print("FLEXTLOGGER COMPLETE API DEMONSTRATION")
    print("Foundation for Structured Logging in FLEXT Ecosystem")
    print("=" * 60)

    # Core patterns
    service.demonstrate_basic_logging()
    service.demonstrate_structured_logging()

    # Context management
    service.demonstrate_context_management()
    service.demonstrate_contextualize()

    # Advanced features
    service.demonstrate_performance_tracking()
    service.demonstrate_exception_logging()
    service.demonstrate_correlation_tracking()

    # Professional patterns
    service.demonstrate_child_loggers()
    service.demonstrate_global_configuration()
    service.demonstrate_log_filtering()
    service.demonstrate_custom_fields()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextLogger methods demonstrated!")
    print("🎯 Next: See 06_*.py for intermediate patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
