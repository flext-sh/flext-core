# !/usr/bin/env python3
"""05 - FlextCore.Logger Fundamentals: Complete Structured Logging.

This example demonstrates the COMPLETE FlextCore.Logger API - the foundation
for structured logging across the entire FLEXT ecosystem. FlextCore.Logger provides
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


class ComprehensiveLoggingService(FlextCore.Service[FlextCore.Types.Dict]):
    """Service demonstrating ALL FlextCore.Logger patterns with FlextMixins.Service infrastructure.

    This service inherits from FlextCore.Service to demonstrate:
    - Inherited container property (FlextCore.Container singleton)
    - Inherited logger property (FlextCore.Logger with service context)
    - Inherited context property (FlextCore.Context for request tracking)
    - Inherited config property (FlextCore.Config with logging settings)
    - Inherited metrics property (FlextMetrics for observability)

    The focus is on demonstrating FlextCore.Logger structured logging patterns
    with FlextCore.Exceptions integration, while leveraging complete FlextMixins.Service
    infrastructure for service orchestration.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins.Service infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextCore.Service base class:
        - self.logger: FlextCore.Logger with service context (ALREADY CONFIGURED!)
        - self.container: FlextCore.Container global singleton
        - self.context: FlextCore.Context for request tracking
        - self.config: FlextCore.Config with logging configuration
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        self._scenarios = ExampleScenarios()
        self._metadata = self._scenarios.metadata(tags=["logging", "demo"])
        self._user = self._scenarios.user()
        self._payload = self._scenarios.payload()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "ComprehensiveLoggingService initialized with inherited infrastructure",
            extra={
                "service_type": "FlextCore.Logger demonstration",
                "logger_name": str(self.logger),
                "config_log_level": self.config.log_level,
                "structured_logging": True,
            },
        )

    def execute(self) -> FlextCore.Result[FlextCore.Types.Dict]:
        """Execute all FlextCore.Logger demonstrations and return summary.

        Demonstrates inherited logger property alongside other infrastructure
        components from FlextMixins.Service. This is the PRIMARY demonstration
        of structured logging across the FLEXT ecosystem.
        """
        self.logger.info("Starting comprehensive FlextCore.Logger demonstration")

        try:
            # Run all demonstrations
            self.demonstrate_basic_logging()
            self.demonstrate_structured_logging()
            self.demonstrate_context_management()
            self.demonstrate_contextualize()
            self.demonstrate_performance_tracking()
            self.demonstrate_exception_logging()
            self.demonstrate_correlation_tracking()
            self.demonstrate_child_loggers()
            self.demonstrate_log_filtering()
            self.demonstrate_custom_fields()
            self.demonstrate_flext_constants_logging()
            self.demonstrate_flext_runtime_integration()
            self.demonstrate_flext_exceptions_integration()
            self.demonstrate_from_callable()
            self.demonstrate_flow_through()
            self.demonstrate_lash()
            self.demonstrate_alt()
            self.demonstrate_value_or_call()
            self.demonstrate_deprecated_patterns()

            # Summary using inherited logger and config
            summary: dict[str, object] = {
                "demonstrations_completed": 19,
                "status": "completed",
                "logged": "True",
                "infrastructure": {
                    "logger": type(self.logger).__name__,
                    "container": type(self.container).__name__,
                    "context": type(self.context).__name__,
                    "config": type(self.config).__name__,
                },
                "logging_config": {
                    "log_level": self.config.log_level,
                    "json_output": self.config.json_output,
                    "include_source": self.config.include_source,
                },
            }

            self.logger.info(
                "FlextCore.Logger demonstration completed successfully",
                extra={"demonstrations": summary["demonstrations_completed"]},
            )

            return FlextCore.Result[FlextCore.Types.Dict].ok(summary)

        except Exception as e:
            error_msg = f"Logging demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextCore.Result[FlextCore.Types.Dict].fail(
                error_msg, error_code=FlextCore.Constants.Errors.VALIDATION_ERROR
            )

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

        # Log exceptions with traceback using FlextCore.Result pattern
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
        # result = FlextCore.Logger.configure(
        #     log_level=FlextCore.Constants.Logging.INFO,
        #     json_output=False,
        #     include_source=True,
        #     structured_output=True,
        #     log_verbosity=FlextCore.Constants.Logging.VERBOSITY,
        # )

        # if result.is_success:
        #     print("✅ Global logging configured")
        # else:
        #     print(f"❌ Configuration failed: {result.error}")

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

    def _compute_expensive_debug_data(
        self,
    ) -> dict[str, str | list[int]]:
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
            environment=self.config.environment,
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

    # ========== NEW FlextCore.Result METHODS (v0.9.9+) ==========

    def demonstrate_from_callable(self) -> None:
        """Show from_callable for safe logging operations."""
        print("\n=== from_callable(): Safe Logging Operations ===")

        # Safe log file initialization
        def risky_log_setup() -> str:
            """Simulate risky log file setup that might raise."""
            if not self.config.log_file:
                msg = "Log file path not configured"
                raise ValueError(msg)
            return self.config.log_file

        log_path_result: FlextCore.Result[str] = FlextCore.Result.from_callable(
            risky_log_setup
        )
        if log_path_result.is_success:
            print(f"✅ Log file configured: {log_path_result.unwrap()}")
        else:
            print(f"✅ Caught configuration error safely: {log_path_result.error}")

        # Safe logger validation
        def validate_logger_config() -> str:
            """Validate logger configuration is complete."""
            if self.config.log_level not in FlextCore.Constants.Logging.VALID_LEVELS:
                msg = f"Invalid log level: {self.config.log_level}"
                raise ValueError(msg)
            return f"Logger configured with level {self.config.log_level}"

        validation_result: FlextCore.Result[str] = FlextCore.Result.from_callable(
            validate_logger_config
        )
        if validation_result.is_success:
            print(f"✅ {validation_result.unwrap()}")

    def demonstrate_flow_through(self) -> None:
        """Show pipeline composition for multi-step logging operations."""
        print("\n=== flow_through(): Logging Pipeline ===")

        def init_logger(_: object) -> FlextCore.Result[FlextCore.Logger]:
            """Step 1: Initialize logger."""
            return FlextCore.Result[FlextCore.Logger].ok(FlextCore.Logger(__name__))

        def configure_context(
            logger: FlextCore.Logger,
        ) -> FlextCore.Result[FlextCore.Logger]:
            """Step 2: Configure logger context."""
            logger.bind(
                service="logging-demo",
                environment=self.config.environment,
            )
            return FlextCore.Result[FlextCore.Logger].ok(logger)

        def add_correlation(
            logger: FlextCore.Logger,
        ) -> FlextCore.Result[FlextCore.Logger]:
            """Step 3: Add correlation ID."""
            correlation_id = str(uuid4())
            logger.bind(correlation_id=correlation_id)
            return FlextCore.Result[FlextCore.Logger].ok(logger)

        def log_initialization(
            logger: FlextCore.Logger,
        ) -> FlextCore.Result[FlextCore.Logger]:
            """Step 4: Log successful initialization."""
            logger.info("Logger pipeline initialized successfully")
            return FlextCore.Result[FlextCore.Logger].ok(logger)

        # Pipeline: init → configure → correlate → log
        result = (
            init_logger(True)  # Start with init_logger
            .flat_map(configure_context)
            .flat_map(add_correlation)
            .flat_map(log_initialization)
        )

        if result.is_success:
            logger = result.unwrap()
            print(f"✅ Logger pipeline success: {logger}")

    def demonstrate_lash(self) -> None:
        """Show error recovery in logging operations."""
        print("\n=== lash(): Logging Error Recovery ===")

        def try_complex_logging() -> FlextCore.Result[str]:
            """Attempt complex logging operation (might fail)."""
            return FlextCore.Result[str].fail("Complex logging operation failed")

        def recover_with_simple_log(error: str) -> FlextCore.Result[str]:
            """Recover by using simple console logging."""
            print(f"  Recovering from: {error}")
            logger = FlextCore.Logger(__name__)
            logger.info("Using simple console logging as fallback")
            return FlextCore.Result[str].ok("Simple logging active")

        result = try_complex_logging().lash(recover_with_simple_log)
        if result.is_success:
            print(f"✅ Recovered: {result.unwrap()}")

    def demonstrate_alt(self) -> None:
        """Show fallback pattern for logging destinations."""
        print("\n=== alt(): Logging Destination Fallback ===")

        # Primary: File logging (simulated failure)
        primary = FlextCore.Result[str].fail("File logging not available")

        # Fallback: Console logging
        fallback_logger = FlextCore.Logger(__name__)
        fallback_logger.info("Using console logging")
        fallback = FlextCore.Result[str].ok("Console logging active")

        result = primary.alt(fallback)
        if result.is_success:
            print(f"✅ Got fallback logging: {result.unwrap()}")

    def demonstrate_value_or_call(self) -> None:
        """Show lazy default evaluation for logging operations."""
        print("\n=== value_or_call(): Lazy Logger Initialization ===")

        # Success case - logger already configured
        success_logger = "preconfigured-logger"
        success = FlextCore.Result[str].ok(success_logger)

        expensive_created = False

        def expensive_logger_init() -> str:
            """Expensive logger initialization (only if needed)."""
            nonlocal expensive_created
            expensive_created = True
            print("  Creating expensive logger with file handlers...")
            return "expensive-file-logger"

        # Success case - expensive init NOT called
        logger_id = success.value_or_call(expensive_logger_init)
        print(f"✅ Success: logger={logger_id}, expensive_created={expensive_created}")

        # Failure case - expensive init IS called
        expensive_created = False
        failure = FlextCore.Result[str].fail("Logger config failed")
        logger_id = failure.value_or_call(expensive_logger_init)
        print(
            f"✅ Failure recovered: logger={logger_id}, expensive_created={expensive_created}"
        )

    # ========== FOUNDATION LAYER INTEGRATION (Layer 0.5 - 2) ==========

    def demonstrate_flext_constants_logging(self) -> None:
        """Show FlextCore.Constants.Logging integration with FlextCore.Logger."""
        print("\n=== FlextCore.Constants.Logging Integration (Layer 1) ===")

        logger = FlextCore.Logger(__name__)

        # Log level constants from FlextCore.Constants
        print("FlextCore.Constants.Logging levels:")
        print(f"  DEFAULT_LEVEL: {FlextCore.Constants.Logging.DEFAULT_LEVEL}")
        print(f"  VALID_LEVELS: {FlextCore.Constants.Logging.VALID_LEVELS}")

        # Validate log level using FlextCore.Constants
        log_level = "INFO"
        if log_level in FlextCore.Constants.Logging.VALID_LEVELS:
            print(f"✅ Log level '{log_level}' is valid")

        # Log format constants
        print(f"  DEFAULT_FORMAT: {FlextCore.Constants.Logging.DEFAULT_FORMAT}")
        print(
            f"  JSON_OUTPUT_DEFAULT: {FlextCore.Constants.Logging.JSON_OUTPUT_DEFAULT}"
        )
        print(f"  INCLUDE_SOURCE: {FlextCore.Constants.Logging.INCLUDE_SOURCE}")

        # Structured logging configuration
        print("  STRUCTURED_OUTPUT:", FlextCore.Constants.Logging.STRUCTURED_OUTPUT)
        print("  TRACK_PERFORMANCE:", FlextCore.Constants.Logging.TRACK_PERFORMANCE)
        print("  TRACK_TIMING:", FlextCore.Constants.Logging.TRACK_TIMING)

        # Log with FlextCore.Constants configuration
        logger.info(
            "Logging with FlextCore.Constants configuration",
            extra={
                "log_level": FlextCore.Constants.Logging.DEFAULT_LEVEL,
                "structured": FlextCore.Constants.Logging.STRUCTURED_OUTPUT,
                "performance_tracking": FlextCore.Constants.Logging.TRACK_PERFORMANCE,
            },
        )

        print("✅ FlextCore.Constants.Logging integration demonstrated")

    def demonstrate_flext_runtime_integration(self) -> None:
        """Show FlextCore.Runtime (Layer 0.5) logging defaults."""
        print("\n=== FlextCore.Runtime Integration (Layer 0.5) ===")

        logger = FlextCore.Logger(__name__)

        # FlextCore.Runtime logging defaults
        print("FlextCore.Runtime logging configuration defaults:")
        print(f"  DEFAULT_LOG_LEVEL: {FlextCore.Constants.Logging.DEFAULT_LEVEL}")
        print(f"  LOG_LEVEL_DEBUG: {FlextCore.Constants.Logging.DEBUG}")
        print(f"  LOG_LEVEL_INFO: {FlextCore.Constants.Logging.INFO}")
        print(f"  LOG_LEVEL_WARNING: {FlextCore.Constants.Logging.WARNING}")
        print(f"  LOG_LEVEL_ERROR: {FlextCore.Constants.Logging.ERROR}")
        print(f"  LOG_LEVEL_CRITICAL: {FlextCore.Constants.Logging.CRITICAL}")

        # Numeric logging levels
        print(f"  LOG_LEVEL_NUM_INFO: {20}")  # INFO level
        print(f"  LOG_LEVEL_NUM_ERROR: {40}")  # ERROR level

        # Log with FlextCore.Runtime defaults
        logger.info(
            "Logging with FlextCore.Runtime defaults",
            extra={
                "default_level": FlextCore.Constants.Logging.DEFAULT_LEVEL,
                "level_numeric": 20,  # INFO level
            },
        )

        # Validate log level using FlextCore.Runtime
        test_level = "INFO"
        level_valid = test_level in {
            FlextCore.Constants.Logging.DEBUG,
            FlextCore.Constants.Logging.INFO,
            FlextCore.Constants.Logging.WARNING,
            FlextCore.Constants.Logging.ERROR,
            FlextCore.Constants.Logging.CRITICAL,
        }
        print(f"✅ Log level '{test_level}' is valid: {level_valid}")

    def demonstrate_flext_exceptions_integration(self) -> None:
        """Show FlextCore.Exceptions (Layer 2) with logging."""
        print("\n=== FlextCore.Exceptions Integration (Layer 2) ===")

        logger = FlextCore.Logger(__name__)

        # Log ValidationError with structured exception info
        try:
            invalid_level = "INVALID_LEVEL"
            if invalid_level not in FlextCore.Constants.Logging.VALID_LEVELS:
                error_message = "Invalid log level configuration"
                raise FlextCore.Exceptions.ValidationError(
                    error_message,
                    field="log_level",
                    value=invalid_level,
                    error_code=FlextCore.Constants.Errors.VALIDATION_ERROR,
                )
        except FlextCore.Exceptions.ValidationError as e:
            logger.exception(
                "Validation error in logging configuration",
                extra={
                    "error_code": e.error_code,
                    "field": e.field,
                    "invalid_value": e.value,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ ValidationError logged: {e.error_code}")
            print(f"   Field: {e.field}, Value: {e.value}")

        # Log ConfigurationError with logging setup failure
        try:
            error_message = (
                "Failed to configure file logging handler: /invalid/path/log.txt"
            )
            raise FlextCore.Exceptions.ConfigurationError(
                error_message,
                config_key="log_file",
                config_source=".env",
            )
        except FlextCore.Exceptions.ConfigurationError as e:
            logger.exception(
                "Configuration error in logging setup",
                extra={
                    "error_code": e.error_code,
                    "config_key": e.config_key,
                    "config_source": e.config_source,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ ConfigurationError logged: {e.error_code}")
            print(f"   Config key: {e.config_key}, Source: {e.config_source}")

        # Log with comprehensive error context
        try:
            error_message = "Complex logging operation failed"
            raise FlextCore.Exceptions.BaseError(
                error_message,
                error_code=FlextCore.Constants.Errors.UNKNOWN_ERROR,
                metadata={
                    "operation": "file_logging",
                    "handler": "RotatingFileHandler",
                    "max_bytes": 10485760,
                },
            )
        except FlextCore.Exceptions.BaseError as e:
            logger.critical(
                "Critical logging failure",
                extra={
                    "error_code": e.error_code,
                    "correlation_id": e.correlation_id,
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                },
            )
            print("✅ Critical error logged with metadata")
            print(f"   Correlation ID: {e.correlation_id}")

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

        print("\n✅ CORRECT WAY (FlextCore.Logger):")
        print("logger = FlextCore.Logger(__name__)")
        print("logger.debug('Debug info', extra={'variable': variable})")

        # OLD: Basic logging module (DEPRECATED)
        warnings.warn(
            "Basic logging module is DEPRECATED! Use FlextCore.Logger for structured logging.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (logging module):")
        print("import logging")
        print("logging.info('Message')")

        print("\n✅ CORRECT WAY (FlextCore.Logger):")
        print("\n✅ CORRECT WAY (FlextCore.Logger):")
        print("from flext_core import FlextCore.Constants")
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
    """Main entry point demonstrating all FlextCore.Logger capabilities."""
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
    # service.demonstrate_global_configuration()  # Commented out for demo
    service.demonstrate_log_filtering()
    service.demonstrate_custom_fields()

    # Foundation layer integration (NEW in Phase 1)
    service.demonstrate_flext_constants_logging()
    service.demonstrate_flext_runtime_integration()
    service.demonstrate_flext_exceptions_integration()

    # New FlextCore.Result methods (v0.9.9+)
    service.demonstrate_from_callable()
    service.demonstrate_flow_through()
    service.demonstrate_lash()
    service.demonstrate_alt()
    service.demonstrate_value_or_call()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextCore.Logger methods demonstrated!")
    print(
        "✨ Including new v0.9.9+ methods: from_callable, flow_through, lash, alt, value_or_call"
    )
    print(
        "🔧 Including foundation integration: FlextCore.Constants.Logging, FlextCore.Runtime (Layer 0.5), FlextCore.Exceptions (Layer 2)"
    )
    print("🎯 Next: See 06_*.py for intermediate patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
