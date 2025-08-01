#!/usr/bin/env python3
"""FLEXT Logging Structured System Example.

Comprehensive demonstration of FlextLoggerFactory system showing enterprise-grade
structured logging with context management, level filtering, and observability.

Features demonstrated:
    - Core structured logging with FlextLogger
    - Logger factory pattern with caching and global configuration
    - Context management with scoped logging and automatic cleanup
    - Level-based filtering with performance optimization
    - Global log store for testing and observability
    - Context inheritance and hierarchical logging
    - Exception logging with automatic traceback capture
    - Enterprise logging patterns for production applications

Key Components:
    - FlextLogger: Core structured logger with context management
    - FlextLoggerFactory: Centralized logger creation with caching
    - FlextLogContext: Context manager for scoped logging
    - FlextLoggerFactory: Unified public API for all logging operations
    - Global log store: In-memory storage for testing and observability

This example shows real-world enterprise logging scenarios
demonstrating the power and flexibility of the FlextLoggerFactory system.
"""

import time
import traceback
from types import TracebackType

from flext_core.loggings import (
    FlextLogger,
    FlextLoggerFactory,
    create_log_context,
    get_logger,
)

# Constants for magic numbers
MAX_PAYMENT_AMOUNT = 10000
MAX_STATEMENTS_THRESHOLD = 50
MAX_VALUE_DISPLAY_LENGTH = 50


def _raise_validation_error(message: str) -> None:
    """Raise validation error with proper message handling."""
    raise ValueError(message)


def _raise_gateway_error(message: str) -> None:
    """Raise gateway error with proper message handling."""
    raise ConnectionError(message)


def _raise_amount_error() -> None:
    """Raise amount validation error."""
    msg = "Payment amount must be positive"
    raise ValueError(msg)


def _raise_method_error(payment_method: str) -> None:
    """Raise payment method error."""
    error_msg = f"Unsupported payment method: {payment_method}"
    raise ValueError(error_msg)


def _raise_timeout_error() -> None:
    """Raise payment gateway timeout error."""
    msg = "Payment gateway timeout"
    raise ConnectionError(msg)


def demonstrate_basic_logging() -> None:
    """Demonstrate basic structured logging with FlextLogger."""
    print("\n" + "=" * 80)
    print("📝 BASIC STRUCTURED LOGGING")
    print("=" * 80)

    # 1. Create basic logger
    print("\n1. Creating and using basic logger:")
    logger = FlextLogger("myapp.service", "DEBUG")

    # Basic logging at different levels
    logger.trace("Application startup trace", phase="initialization", step=1)
    logger.debug("Debug information", module="config", settings_loaded=True)
    logger.info("Service started successfully", port=FlextConstants.Platform.FLEXCORE_PORT, version=FlextConstants.VERSION)
    logger.warning("High memory usage detected", memory_usage_percent=85)
    logger.error("Database connection failed", database="users", retry_count=3)
    logger.critical("System overload detected", cpu_usage_percent=95)

    print("✅ Basic logging completed - check stderr for log entries")

    # 2. Logger with different level filtering
    print("\n2. Level filtering demonstration:")

    # Create logger with WARNING level - should filter out DEBUG and INFO
    warning_logger = FlextLogger("myapp.critical", "WARNING")

    print("   Logging with WARNING level filter:")
    warning_logger.debug("This debug message should be filtered out")
    warning_logger.info("This info message should be filtered out")
    warning_logger.warning("This warning message should appear")
    warning_logger.error("This error message should appear")

    print("✅ Level filtering completed - only WARNING and ERROR should appear")

    # 3. Context-aware logging
    print("\n3. Context-aware logging:")

    context_logger = FlextLogger("myapp.context", "INFO")
    context_logger.set_context(
        {
            "user_id": "user_123",
            "session_id": "sess_abc",
            "request_id": "req_456",
        },
    )

    context_logger.info(
        "User authentication successful",
        method="oauth",
        provider="google",
    )
    context_logger.info("User profile loaded", profile_size_kb=45, cache_hit=True)
    context_logger.warning("Rate limit approaching", requests_remaining=5)

    print("✅ Context logging completed - context should be included in all entries")


def demonstrate_logger_factory() -> None:
    """Demonstrate logger factory pattern with caching and global configuration."""
    print("\n" + "=" * 80)
    print("🏭 LOGGER FACTORY AND GLOBAL CONFIGURATION")
    print("=" * 80)

    # 1. Factory logger creation with caching
    print("\n1. Factory pattern with caching:")

    # Create loggers using factory
    service_logger = FlextLoggerFactory.get_logger("myapp.service", "DEBUG")
    database_logger = FlextLoggerFactory.get_logger("myapp.database", "INFO")
    api_logger = FlextLoggerFactory.get_logger("myapp.api", "DEBUG")

    # Create same logger again - should return cached instance
    service_logger_2 = FlextLoggerFactory.get_logger("myapp.service", "DEBUG")

    print(f"   Service logger instances same: {service_logger is service_logger_2}")
    print("✅ Logger factory caching working correctly")

    # 2. Global level configuration
    print("\n2. Global level configuration:")

    # Log at different levels before global change
    print("   Before global level change (individual levels):")
    service_logger.debug("Service debug message")  # Should appear (DEBUG level)
    database_logger.debug("Database debug message")  # Should not appear (INFO level)
    api_logger.debug("API debug message")  # Should appear (DEBUG level)

    # Change global level to WARNING
    print("   Changing global level to WARNING...")
    FlextLoggerFactory.set_global_level("WARNING")

    print("   After global level change (WARNING for all):")
    service_logger.debug("Service debug after global change")  # Should not appear
    service_logger.info("Service info after global change")  # Should not appear
    service_logger.warning("Service warning after global change")  # Should appear

    database_logger.info("Database info after global change")  # Should not appear
    database_logger.error("Database error after global change")  # Should appear

    api_logger.debug("API debug after global change")  # Should not appear
    api_logger.critical("API critical after global change")  # Should appear

    print("✅ Global level configuration completed")

    # 3. Convenience function usage
    print("\n3. Convenience function usage:")

    # Reset global level for this demo
    FlextLoggerFactory.set_global_level("DEBUG")

    # Use convenience function
    convenience_logger = get_logger("myapp.convenience", "INFO")
    convenience_logger.info("Logger created with convenience function", easy=True)

    print("✅ Convenience function working correctly")


def demonstrate_context_management() -> None:
    """Demonstrate context management with scoped logging."""
    print("\n" + "=" * 80)
    print("🔄 CONTEXT MANAGEMENT AND SCOPED LOGGING")
    print("=" * 80)

    # 1. Basic context management
    print("\n1. Basic context management:")

    base_logger = FlextLogger("myapp.context", "INFO")
    base_logger.set_context({"service": "user_service", "version": "0.9.0"})

    base_logger.info("Service started with base context")

    # Temporarily add request context
    print("   Adding temporary request context:")
    with create_log_context(base_logger, request_id="req_789", user_id="user_456"):
        base_logger.info("Processing user request", action="get_profile")
        base_logger.info("Database query executed", table="users", duration_ms=45)

        # Nested context
        print("   Adding nested operation context:")
        with create_log_context(
            base_logger,
            operation="profile_enrichment",
            source="external_api",
        ):
            base_logger.info("Enriching profile data", api_endpoint="/api/v1/profiles")
            base_logger.warning("API response delayed", expected_ms=100, actual_ms=250)

    # Context should be restored
    base_logger.info("Request processing completed", success=True)

    print("✅ Context management completed - contexts should be properly scoped")

    # 2. Context inheritance patterns
    print("\n2. Context inheritance patterns:")

    parent_logger = FlextLogger("myapp.parent", "DEBUG")
    parent_logger.set_context({"environment": "production", "datacenter": "us-east-1"})

    # Create child logger with additional context
    child_logger = parent_logger.with_context(
        component="payment_processor",
        correlation_id="corr_123",
    )

    parent_logger.info("Parent logger with base context")
    child_logger.info(
        "Child logger with inherited + additional context",
        transaction_id="tx_456",
        amount=99.99,
    )

    print("✅ Context inheritance working correctly")

    # 3. Convenience context manager
    print("\n3. Convenience context manager:")

    convenience_logger = get_logger("myapp.convenience_context", "INFO")
    convenience_logger.set_context({"application": "ecommerce"})

    with create_log_context(
        convenience_logger,
        order_id="order_789",
        customer_id="cust_123",
    ):
        convenience_logger.info("Order processing started")
        convenience_logger.info("Payment validation", payment_method="credit_card")
        convenience_logger.info("Inventory check", items_count=3, available=True)
        convenience_logger.info("Order processing completed", status="success")

    convenience_logger.info("Post-order cleanup completed")

    print("✅ Convenience context manager working correctly")


def demonstrate_exception_logging() -> None:  # noqa: PLR0915
    """Demonstrate exception logging with automatic traceback capture."""
    print("\n" + "=" * 80)
    print("🚨 EXCEPTION LOGGING AND ERROR HANDLING")
    print("=" * 80)

    # 1. Basic exception logging
    print("\n1. Basic exception logging:")

    error_logger = FlextLogger("myapp.errors", "DEBUG")
    error_logger.set_context({"module": "payment_processor"})

    # Simulate various exceptions
    try:
        # Simulate division by zero
        result = 10 / 0
        print(f"Result: {result}")
    except ZeroDivisionError:
        error_logger.exception(
            "Division by zero error",
            operation="calculate_fee",
            numerator=10,
            denominator=0,
        )

    try:
        # Simulate key error
        data = {"name": "John"}
        email = data["email"]  # Key doesn't exist
        print(f"Email: {email}")
    except KeyError:
        error_logger.exception(
            "Missing required field",
            operation="user_validation",
            required_field="email",
            available_fields=list(data.keys()),
        )

    try:
        # Simulate type error
        number = "not_a_number"
        result = number + 10
        print(f"Result: {result}")
    except TypeError:
        error_logger.exception(
            "Type conversion error",
            operation="data_processing",
            expected_type="int",
            actual_type=type(number).__name__,
        )

    print("✅ Exception logging completed - tracebacks should be captured")

    # 2. Contextual exception logging
    print("\n2. Contextual exception logging:")

    def process_payment(amount: float, payment_method: str) -> dict[str, object]:
        """Simulate payment processing with potential errors."""
        contextual_logger = get_logger("myapp.payment", "INFO")

        with create_log_context(
            contextual_logger,
            amount=amount,
            payment_method=payment_method,
            transaction_id=f"tx_{int(time.time())}",
        ):
            try:
                contextual_logger.info("Payment processing started")

                if amount <= 0:
                    _raise_amount_error()

                if payment_method not in {"credit_card", "debit_card", "paypal"}:
                    _raise_method_error(payment_method)

                # Simulate payment processing
                if amount > MAX_PAYMENT_AMOUNT:
                    _raise_timeout_error()

                contextual_logger.info("Payment authorization successful")
                return {"status": "success", "transaction_id": f"tx_{int(time.time())}"}

            except ValueError as e:
                contextual_logger.exception(
                    "Payment validation failed",
                    error_type="validation_error",
                    error_details=str(e),
                )
                raise
            except ConnectionError as e:
                contextual_logger.exception(
                    "Payment gateway error",
                    error_type="gateway_error",
                    error_details=str(e),
                )
                raise
            except (OSError, RuntimeError) as e:
                contextual_logger.exception(
                    "Unexpected payment error",
                    error_type="unknown_error",
                    error_details=str(e),
                )
                raise

    # Test different error scenarios
    test_cases = [
        {"amount": -50, "method": "credit_card", "expected": "validation"},
        {"amount": 100, "method": "bitcoin", "expected": "validation"},
        {"amount": 15000, "method": "credit_card", "expected": "gateway"},
        {"amount": 99.99, "method": "credit_card", "expected": "success"},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"   Test case {i}: ${test_case['amount']} via {test_case['method']}")
        try:
            result = process_payment(test_case["amount"], test_case["method"])
            print(f"     Success: {result}")
        except (ValueError, ConnectionError, OSError, RuntimeError) as e:
            print(f"     Expected error: {type(e).__name__}: {e}")

    print("✅ Contextual exception logging completed")


def demonstrate_unified_api() -> None:  # noqa: PLR0915
    """Demonstrate unified FlextLoggerFactory API and observability features."""
    print("\n" + "=" * 80)
    print("🎛️ UNIFIED API AND OBSERVABILITY")
    print("=" * 80)

    # 1. Unified API usage
    print("\n1. Unified FlextLoggerFactory API:")

    # Create loggers through unified API
    api_logger = FlextLoggerFactory.get_logger("myapp.unified_api", "DEBUG")
    metrics_logger = FlextLoggerFactory.get_logger("myapp.metrics", "INFO")

    api_logger.info("API server starting", port=FlextConstants.Platform.FLEXCORE_PORT, workers=4)
    metrics_logger.info("Metrics collection enabled", interval_seconds=30)

    # Global configuration through unified API
    print("   Setting global level through unified API...")
    FlextLoggerFactory.set_global_level("WARNING")

    # These should not appear (below WARNING level)
    api_logger.debug("Debug message after global change")
    api_logger.info("Info message after global change")

    # These should appear (WARNING level and above)
    api_logger.warning("Warning message after global change")
    metrics_logger.error("Error message after global change")

    print("✅ Unified API working correctly")

    # 2. Log store observability
    print("\n2. Log store and observability:")

    # Clear log store for clean demo
    FlextLoggerFactory.clear_log_store()

    # Reset level for logging
    FlextLoggerFactory.set_global_level("INFO")

    # Generate some log entries
    observability_logger = FlextLoggerFactory.get_logger("myapp.observability", "INFO")

    with create_log_context(observability_logger, session_id="obs_session_123"):
        observability_logger.info("Session started", user_agent="Mozilla/5.0")
        observability_logger.info("Page viewed", page="/dashboard", load_time_ms=245)
        observability_logger.warning(
            "Slow query detected",
            query_time_ms=1500,
            table="analytics",
        )
        observability_logger.error(
            "Cache miss",
            cache_key="user_preferences",
            fallback="database",
        )

    # Retrieve and analyze log store
    log_entries = FlextLoggerFactory.get_log_store()

    print(f"   Total log entries captured: {len(log_entries)}")

    # Analyze log entries
    levels: dict[str, int] = {}
    loggers: dict[str, int] = {}
    has_context = 0

    for entry in log_entries:
        level = entry.get("level", "UNKNOWN")
        logger_name = entry.get("logger", "unknown")
        context = entry.get("context", {})

        levels[level] = levels.get(level, 0) + 1
        loggers[logger_name] = loggers.get(logger_name, 0) + 1

        if context:
            has_context += 1

    print(f"   Entries by level: {levels}")
    print(f"   Entries by logger: {dict(list(loggers.items())[:5])}")  # Show first 5
    print(f"   Entries with context: {has_context}/{len(log_entries)}")

    # Show sample log entry structure
    if log_entries:
        sample_entry = log_entries[-1]  # Get last entry
        print("   Sample entry structure:")
        for key, value in sample_entry.items():
            if key == "context" and isinstance(value, dict) and value:
                print(
                    f"     {key}: {dict(list(value.items())[:3])}...",
                )  # Show first 3 context items
            else:
                value_str = (
                    str(value)[:MAX_VALUE_DISPLAY_LENGTH] + "..."
                    if len(str(value)) > MAX_VALUE_DISPLAY_LENGTH
                    else str(value)
                )
                print(f"     {key}: {value_str}")

    print("✅ Log store observability completed")

    # 3. Testing utilities
    print("\n3. Testing utilities:")

    # Clear specific logger cache
    print("   Clearing logger cache...")
    initial_log_count = len(FlextLoggerFactory.get_log_store())
    FlextLoggerFactory.clear_loggers()

    # Create new logger after cache clear
    fresh_logger = FlextLoggerFactory.get_logger("myapp.fresh", "DEBUG")
    fresh_logger.info("Fresh logger after cache clear", cache_cleared=True)

    final_log_count = len(FlextLoggerFactory.get_log_store())
    print(f"   Log entries before clear: {initial_log_count}")
    print(f"   Log entries after new logger: {final_log_count}")

    # Clear log store for clean state
    print("   Clearing log store for clean state...")
    FlextLoggerFactory.clear_log_store()
    empty_store = FlextLoggerFactory.get_log_store()
    print(f"   Log store entries after clear: {len(empty_store)}")

    print("✅ Testing utilities working correctly")


def demonstrate_enterprise_patterns() -> None:  # noqa: PLR0915
    """Demonstrate enterprise logging patterns and best practices."""
    print("\n" + "=" * 80)
    print("🏢 ENTERPRISE LOGGING PATTERNS")
    print("=" * 80)

    # 1. Request correlation and tracing
    print("\n1. Request correlation and distributed tracing:")

    def process_user_request(
        user_id: str,
        request_id: str,
        operation: str,
    ) -> dict[str, object]:
        """Simulate enterprise request processing with correlation."""
        # Create service-specific loggers
        gateway_logger = get_logger("enterprise.api_gateway", "INFO")
        auth_logger = get_logger("enterprise.auth_service", "INFO")
        user_logger = get_logger("enterprise.user_service", "INFO")
        db_logger = get_logger("enterprise.database", "INFO")

        # Common correlation context
        correlation_context = {
            "correlation_id": request_id,
            "user_id": user_id,
            "operation": operation,
            "trace_id": f"trace_{int(time.time())}",
        }

        try:
            # API Gateway
            with create_log_context(
                gateway_logger,
                **correlation_context,
                service="api_gateway",
            ):
                gateway_logger.info(
                    "Request received",
                    endpoint=f"/api/users/{user_id}/{operation}",
                    method="GET",
                    source_ip="192.168.1.100",
                )

                # Authentication Service
                with create_log_context(
                    auth_logger,
                    **correlation_context,
                    service="auth",
                ):
                    auth_logger.info("Authentication started", auth_method="jwt")
                    time.sleep(0.01)  # Simulate processing
                    auth_logger.info("Authentication successful", token_valid=True)

                # User Service
                with create_log_context(
                    user_logger,
                    **correlation_context,
                    service="user",
                ):
                    user_logger.info("User service processing started")

                    # Database operations
                    with create_log_context(
                        db_logger,
                        **correlation_context,
                        service="database",
                    ):
                        db_logger.info(
                            "Database query started",
                            table="users",
                            query_type="SELECT",
                        )
                        time.sleep(0.005)  # Simulate query
                        db_logger.info(
                            "Database query completed",
                            rows_returned=1,
                            duration_ms=5,
                        )

                    user_logger.info(
                        "User service processing completed",
                        result_size_kb=2.5,
                    )

                gateway_logger.info(
                    "Request completed successfully",
                    status_code=200,
                    response_time_ms=16,
                )

                return {
                    "status": "success",
                    "correlation_id": request_id,
                    "data": {"user_id": user_id, "operation": operation},
                }

        except (RuntimeError, ValueError, TypeError) as e:
            gateway_logger.exception(
                "Request processing failed",
                **correlation_context,
                error_type=type(e).__name__,
            )
            raise

    # Process multiple requests
    requests = [
        {"user_id": "user_001", "request_id": "req_001", "operation": "profile"},
        {"user_id": "user_002", "request_id": "req_002", "operation": "settings"},
        {"user_id": "user_003", "request_id": "req_003", "operation": "preferences"},
    ]

    for request in requests:
        try:
            result = process_user_request(**request)
            print(f"   Request {request['request_id']}: {result['status']}")
        except (ValueError, OSError, RuntimeError) as e:
            print(f"   Request {request['request_id']}: Failed - {e}")

    print("✅ Request correlation completed - check logs for trace correlation")

    # 2. Performance monitoring and metrics
    print("\n2. Performance monitoring and metrics:")

    perf_logger = get_logger("enterprise.performance", "INFO")

    class PerformanceMonitor:
        """Context manager for performance monitoring."""

        def __init__(
            self,
            logger: FlextLogger,
            operation: str,
            **context: object,
        ) -> None:
            self.logger = logger
            self.operation = operation
            self.context = context
            self.start_time = 0.0

        def __enter__(self) -> "PerformanceMonitor":
            self.start_time = time.time()
            self.logger.info(
                "Operation started",
                operation=self.operation,
                **self.context,
            )
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            duration_ms = (time.time() - self.start_time) * 1000

            if exc_type is None:
                self.logger.info(
                    "Operation completed successfully",
                    operation=self.operation,
                    duration_ms=round(duration_ms, 2),
                    **self.context,
                )
            else:
                self.logger.error(
                    "Operation failed",
                    operation=self.operation,
                    duration_ms=round(duration_ms, 2),
                    error_type=exc_type.__name__ if exc_type else "Unknown",
                    **self.context,
                )

    # Use performance monitoring
    operations = [
        {"op": "database_query", "table": "users", "complexity": "simple"},
        {"op": "cache_lookup", "cache_type": "redis", "key_pattern": "user:*"},
        {"op": "external_api_call", "service": "payment_processor", "timeout_ms": 5000},
        {"op": "data_transformation", "input_size_mb": 10.5, "output_format": "json"},
    ]

    for op_config in operations:
        if not isinstance(op_config, dict):
            msg = "op_config deve ser um dicionário"
            raise TypeError(msg)
        operation = op_config.pop("op")

        with PerformanceMonitor(perf_logger, operation, **op_config):
            # Simulate different operation durations
            if operation == "database_query":
                time.sleep(0.005)  # 5ms
            elif operation == "cache_lookup":
                time.sleep(0.001)  # 1ms
            elif operation == "external_api_call":
                time.sleep(0.050)  # 50ms
            elif operation == "data_transformation":
                time.sleep(0.020)  # 20ms

    print("✅ Performance monitoring completed")

    # 3. Business metrics and analytics
    print("\n3. Business metrics and analytics:")

    business_logger = get_logger("enterprise.business", "INFO")

    # Simulate business events
    business_events = [
        {
            "event": "user_registration",
            "user_id": "new_user_001",
            "source": "web_app",
            "plan": "premium",
            "revenue_impact": 99.99,
        },
        {
            "event": "subscription_renewal",
            "user_id": "user_123",
            "plan": "enterprise",
            "renewal_period": "yearly",
            "revenue_impact": 1199.99,
        },
        {
            "event": "feature_usage",
            "user_id": "user_456",
            "feature": "advanced_analytics",
            "usage_count": 1,
            "session_duration_minutes": 45,
        },
        {
            "event": "support_ticket_created",
            "user_id": "user_789",
            "priority": "high",
            "category": "billing",
            "agent_assigned": "agent_001",
        },
    ]

    for event in business_events:
        if not isinstance(event, dict):
            msg = "event deve ser um dicionário"
            raise TypeError(msg)
        event_type = event.pop("event")
        message = f"Business event: {event_type}"
        business_logger.info(
            message,
            event_type=event_type,
            timestamp=time.time(),
            **event,
        )

    print("✅ Business metrics logging completed")


def main() -> None:
    """Execute all FlextLoggerFactory demonstrations."""
    print("🚀 FLEXT LOGGING - STRUCTURED SYSTEM EXAMPLE")
    print("Demonstrating comprehensive logging patterns for enterprise applications")

    try:
        demonstrate_basic_logging()
        demonstrate_logger_factory()
        demonstrate_context_management()
        demonstrate_exception_logging()
        demonstrate_unified_api()
        demonstrate_enterprise_patterns()

        print("\n" + "=" * 80)
        print("✅ ALL FLEXT LOGGING DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Summary of capabilities demonstrated:")
        print("   📝 Basic structured logging with level filtering")
        print("   🏭 Logger factory pattern with caching and global configuration")
        print("   🔄 Context management with scoped logging and inheritance")
        print("   🚨 Exception logging with automatic traceback capture")
        print("   🎛️ Unified API with log store observability")
        print("   🏢 Enterprise patterns for distributed tracing and monitoring")
        print("\n💡 FlextLoggerFactory provides enterprise-grade structured logging")
        print(
            "   with context management, observability, and performance optimization!",
        )

        # Final log store summary
        final_logs = FlextLoggerFactory.get_log_store()
        print(f"\n📈 Total log entries generated during demo: {len(final_logs)}")

    except (OSError, RuntimeError, ValueError) as e:
        print(f"\n❌ Error during FlextLoggerFactory demonstration: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
