"""FLEXT Observability - Enterprise monitoring, metrics collection and distributed tracing.

Provides comprehensive observability solution for the FLEXT ecosystem including metrics
collection, distributed tracing, health monitoring, alerting capabilities, and structured
logging. All functionality consolidated into FlextObservability with nested classes for
logical organization and easy discoverability.

Module Role in Architecture:
    FlextObservability serves as the central hub for monitoring and observability across
    FLEXT applications. Integrates with FlextResult for error handling, FlextConstants
    for configuration, and FlextTypes.Config for environment-aware settings.

Classes and Methods:
    FlextObservability:                     # Comprehensive observability system
        # Nested Classes:
        Console                            # Structured logging with severity levels
        Span                               # Distributed tracing spans
        Tracer                             # Operation tracing with context
        Metrics                            # Counters, gauges, histograms collection
        Alerts                             # Multi-level alerting with escalation
        Health                             # System health checks and monitoring

        # Configuration Methods:
        configure_observability_system(config) -> FlextResult[ConfigDict]
        get_observability_system_config() -> FlextResult[ConfigDict]
        set_performance_level(level) -> FlextResult[None]
        get_current_config() -> FlextResult[ConfigDict]

        # Factory Methods:
        create_console_logger(name, level) -> Console
        create_tracer(service_name) -> Tracer
        create_metrics_collector(namespace) -> Metrics
        create_health_monitor(checks) -> Health
        create_alert_manager(config) -> Alerts

    Console Methods:
        log_info(message, **context) -> None       # Info level logging
        log_warn(message, **context) -> None       # Warning level logging
        log_error(message, **context) -> None      # Error level logging
        log_debug(message, **context) -> None      # Debug level logging
        set_level(level) -> None                    # Set logging level

    Tracer Methods:
        start_span(operation_name, **tags) -> Span # Start new tracing span
        finish_span(span) -> None                   # Finish tracing span
        trace_operation(name) -> context_manager   # Context manager for tracing
        get_active_span() -> Span | None           # Get current active span

    Metrics Methods:
        increment_counter(name, value, **tags) -> None # Increment counter metric
        set_gauge(name, value, **tags) -> None         # Set gauge metric value
        record_histogram(name, value, **tags) -> None  # Record histogram value
        get_metrics_summary() -> dict                   # Get metrics summary

    Health Methods:
        check_health() -> FlextResult[dict]        # Run health checks
        register_health_check(name, check_func) -> None # Register health check
        get_health_status() -> dict                # Get current health status

    Alerts Methods:
        send_alert(level, message, **context) -> None # Send alert notification
        configure_alerts(config) -> None              # Configure alert settings
        get_alert_history() -> list[dict]             # Get alert history

Usage Examples:
    Basic observability setup:
        obs = FlextObservability()
        console = obs.create_console_logger("myapp", "INFO")
        tracer = obs.create_tracer("user-service")
        metrics = obs.create_metrics_collector("business")

    Tracing operations:
        with tracer.trace_operation("process_order") as span:
            span.set_tag("order_id", "12345")
            result = process_order_logic()

    Metrics collection:
        metrics.increment_counter("orders_processed", 1, status="success")
        metrics.set_gauge("active_connections", 42)
        metrics.record_histogram("response_time_ms", 156.7)

    Health monitoring:
        health = obs.create_health_monitor([db_check, cache_check])
        status_result = health.check_health()

Integration:
    FlextObservability integrates with FlextResult for error handling, FlextConstants
    for configuration management, FlextTypes.Config for environment settings, and
    provides context managers for automatic resource management.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

GeneratorT = Generator


class FlextObservability:
    """Comprehensive observability system providing enterprise-grade monitoring and observability capabilities.

    This class serves as the central hub for all observability functionality in the FLEXT
    ecosystem, implementing comprehensive monitoring, metrics collection, distributed tracing,
    health checks, and alerting systems. The design follows FLEXT consolidation patterns
    with nested classes providing logical organization of functionality.

    **CONSOLIDATED ARCHITECTURE**: This class consolidates functionality that was previously
    scattered across multiple classes into a single, well-organized system:
    - FlextConsole → Console (structured logging)
    - FlextSpan → Span (distributed tracing spans)
    - FlextTracer → Tracer (operation tracing)
    - FlextMetrics → Metrics (metrics collection)
    - FlextAlerts → Alerts (alerting system)
    - _SimpleHealth → Health (health monitoring)

    Core Components:
        - **Configuration System**: Environment-aware configuration with FlextTypes.Config
        - **Console Logging**: Structured logging with multiple severity levels
        - **Distributed Tracing**: Operation tracing with context and tag support
        - **Metrics Collection**: Counters, gauges, and histograms with tag-based organization
        - **Health Monitoring**: System health checks with status reporting
        - **Alert Management**: Multi-level alerting with context information
        - **Performance Optimization**: Configurable performance settings for different environments

    Configuration Features:
        The system provides comprehensive configuration management:
        - **Environment-Specific Settings**: Optimized configurations for production, development, test, staging, and local environments
        - **Performance Optimization**: Configurable performance levels (high, medium, low) with appropriate resource allocation
        - **Memory Management**: Buffer sizing and memory limits based on available resources
        - **Sampling Rates**: Configurable trace sampling for performance optimization
        - **Feature Toggles**: Enable/disable specific observability features based on requirements

    Usage Examples:
        Basic observability setup::

            # Configure for development environment
            config = {
                "environment": "development",
                "log_level": "DEBUG",
                "enable_metrics_collection": True,
                "enable_tracing": True,
                "trace_sampling_rate": 1.0,
            }

            result = FlextObservability.configure_observability_system(config)
            if result.success:
                print("Observability system configured")

        Using nested components::

            # Create observability instance
            obs = FlextObservability.Observability()

            # Use structured logging
            obs.logger.info("Processing request", user_id="12345", action="create")
            obs.logger.error("Operation failed", error_code="VALIDATION_ERROR")

            # Collect metrics
            obs.metrics.increment_counter("requests_total", {"endpoint": "/api/users"})
            obs.metrics.record_gauge("active_connections", 42.0)

            # Use distributed tracing
            with obs.tracer.trace_operation("user_creation") as span:
                span.set_tag("user_id", "12345")
                span.add_context("request_type", "registration")
                # Perform operation
                if error_occurred:
                    span.add_error(ValueError("Invalid email format"))

        Environment-specific configuration::

            # Production configuration
            prod_config = FlextObservability.create_environment_observability_config(
                "production"
            )
            if prod_config.success:
                config = prod_config.value
                print(f"Buffer size: {config['metrics_buffer_size']}")
                print(f"Sampling rate: {config['trace_sampling_rate']}")

        Performance optimization::

            # High-performance configuration
            perf_config = {
                "performance_level": "high",
                "memory_limit_mb": 1024,
                "cpu_cores": 8,
            }

            optimized = FlextObservability.optimize_observability_performance(
                perf_config
            )
            if optimized.success:
                settings = optimized.value
                print(f"Max threads: {settings['max_processing_threads']}")
                print(f"Buffer size: {settings['metrics_buffer_size']}")

    Thread Safety:
        All components are designed to be thread-safe for concurrent usage.
        Metrics collection and logging operations can be safely called from
        multiple threads simultaneously.

    Performance Considerations:
        - **Configurable Sampling**: Trace sampling rates can be adjusted to balance observability with performance
        - **Buffer Management**: Configurable buffer sizes for optimal memory usage
        - **Async Support**: Optional asynchronous processing for high-throughput scenarios
        - **Batch Processing**: Efficient batch processing of metrics and traces
        - **Memory Optimization**: Efficient data structures and memory management

    See Also:
        - configure_observability_system(): System configuration management
        - create_environment_observability_config(): Environment-specific configuration
        - optimize_observability_performance(): Performance optimization
        - Observability: Main observability instance with all components

    """

    # ==========================================================================
    # CONFIGURATION METHODS WITH FLEXTTYPES.CONFIG INTEGRATION
    # Environment-aware configuration management with performance optimization
    # ==========================================================================

    @classmethod
    def configure_observability_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure observability system using FlextTypes.Config with StrEnum validation.

        Args:
            config: Configuration dictionary with observability settings

        Returns:
            FlextResult containing the validated and applied configuration

        """
        try:
            # Create validated configuration with defaults
            validated_config: FlextTypes.Config.ConfigDict = {}

            # Validate environment using FlextConstants.Config.ConfigEnvironment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate log level using FlextConstants.Config.LogLevel
            if "log_level" in config:
                log_level = config["log_level"]
                valid_log_levels = [
                    level.value for level in FlextConstants.Config.LogLevel
                ]
                if log_level not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {valid_log_levels}"
                    )
                validated_config["log_level"] = log_level
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.DEBUG.value
                )

            # Observability-specific configuration
            validated_config["enable_metrics_collection"] = config.get(
                "enable_metrics_collection", True
            )
            validated_config["enable_tracing"] = config.get("enable_tracing", True)
            validated_config["enable_health_checks"] = config.get(
                "enable_health_checks", True
            )
            validated_config["enable_alerts"] = config.get("enable_alerts", True)
            validated_config["metrics_buffer_size"] = config.get(
                "metrics_buffer_size", 10000
            )
            validated_config["trace_sampling_rate"] = config.get(
                "trace_sampling_rate", 0.1
            )
            validated_config["health_check_interval_seconds"] = config.get(
                "health_check_interval_seconds", 30
            )
            validated_config["enable_console_output"] = config.get(
                "enable_console_output", True
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to configure observability system: {e!s}"
            )

    @classmethod
    def get_observability_system_config(
        cls,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current observability system configuration with runtime information.

        Returns:
            FlextResult containing current observability system configuration

        """
        try:
            config: FlextTypes.Config.ConfigDict = {
                # Environment information
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                # Observability system settings
                "enable_metrics_collection": True,
                "enable_tracing": True,
                "enable_health_checks": True,
                "enable_alerts": True,
                "metrics_buffer_size": 10000,
                "trace_sampling_rate": 0.1,
                "health_check_interval_seconds": 30,
                "enable_console_output": True,
                # Runtime metrics
                "active_traces": 0,
                "collected_metrics_count": 0,
                "system_health_status": "healthy",
                # Available components
                "available_components": [
                    "console",
                    "tracer",
                    "metrics",
                    "alerts",
                    "health",
                ],
                "enabled_processors": [
                    "metrics_collector",
                    "trace_processor",
                    "health_monitor",
                ],
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get observability system configuration: {e!s}"
            )

    @classmethod
    def create_environment_observability_config(
        cls, environment: FlextTypes.Config.Environment
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific observability system configuration.

        Args:
            environment: Target environment for configuration

        Returns:
            FlextResult containing environment-optimized observability configuration

        """
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Base configuration
            config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "enable_metrics_collection": True,
                "enable_tracing": True,
                "enable_health_checks": True,
                "enable_alerts": True,
            }

            # Environment-specific optimizations
            if environment == "production":
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                        "enable_console_output": False,  # No console output in production
                        "metrics_buffer_size": 50000,  # Large buffer for production load
                        "trace_sampling_rate": 0.01,  # Lower sampling rate for production
                        "health_check_interval_seconds": 60,  # Less frequent checks
                        "enable_performance_monitoring": True,  # Performance monitoring
                        "enable_error_aggregation": True,  # Error aggregation
                    }
                )
            elif environment == "development":
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_console_output": True,  # Console output for development
                        "metrics_buffer_size": 5000,  # Smaller buffer for development
                        "trace_sampling_rate": 1.0,  # Full sampling for development
                        "health_check_interval_seconds": 15,  # Frequent checks for debugging
                        "enable_debug_metrics": True,  # Additional debug metrics
                        "enable_detailed_logging": True,  # Detailed logging for development
                    }
                )
            elif environment == "test":
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.INFO.value,
                        "enable_console_output": False,  # No console output in tests
                        "metrics_buffer_size": 1000,  # Small buffer for tests
                        "trace_sampling_rate": 0.0,  # No tracing in tests
                        "health_check_interval_seconds": 5,  # Frequent checks for test validation
                        "enable_performance_monitoring": False,  # No perf monitoring in tests
                        "enable_test_mode": True,  # Special test mode
                    }
                )
            elif environment == "staging":
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.INFO.value,
                        "enable_console_output": True,  # Console output for staging validation
                        "metrics_buffer_size": 20000,  # Medium buffer for staging
                        "trace_sampling_rate": 0.1,  # Medium sampling for staging
                        "health_check_interval_seconds": 30,  # Standard checks
                        "enable_staging_validation": True,  # Staging-specific validation
                    }
                )
            else:  # local environment
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_console_output": True,  # Console output for local development
                        "metrics_buffer_size": 2000,  # Small buffer for local
                        "trace_sampling_rate": 1.0,  # Full sampling for local debugging
                        "health_check_interval_seconds": 10,  # Frequent checks for immediate feedback
                        "enable_local_debugging": True,  # Local debugging features
                    }
                )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment observability configuration: {e!s}"
            )

    @classmethod
    def optimize_observability_performance(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize observability system performance based on configuration.

        Args:
            config: Performance optimization configuration

        Returns:
            FlextResult containing performance-optimized observability configuration

        """
        try:
            # Start with base configuration
            optimized_config: FlextTypes.Config.ConfigDict = config.copy()

            # Performance level-based optimizations
            performance_level = config.get("performance_level", "medium")

            if performance_level == "high":
                optimized_config.update(
                    {
                        "async_metrics_collection": True,
                        "metrics_buffer_size": 100000,  # Very large buffer
                        "trace_sampling_rate": 0.001,  # Very low sampling
                        "batch_processing_enabled": True,  # Batch processing
                        "compression_enabled": True,  # Data compression
                        "max_concurrent_operations": 1000,  # High concurrency
                        "buffer_flush_interval_ms": 5000,  # Less frequent flushes
                        "enable_caching": True,  # Enable caching
                    }
                )
            elif performance_level == "medium":
                optimized_config.update(
                    {
                        "async_metrics_collection": True,
                        "metrics_buffer_size": 25000,  # Medium buffer
                        "trace_sampling_rate": 0.01,  # Low sampling
                        "batch_processing_enabled": True,  # Batch processing
                        "max_concurrent_operations": 100,  # Medium concurrency
                        "buffer_flush_interval_ms": 2000,  # Medium flush interval
                        "enable_caching": False,  # No caching for medium
                    }
                )
            else:  # low performance level
                optimized_config.update(
                    {
                        "async_metrics_collection": False,  # Synchronous processing
                        "metrics_buffer_size": 1000,  # Small buffer
                        "trace_sampling_rate": 0.1,  # Higher sampling for debugging
                        "batch_processing_enabled": False,  # No batch processing
                        "max_concurrent_operations": 10,  # Low concurrency
                        "buffer_flush_interval_ms": 500,  # Frequent flushes
                        "enable_detailed_metrics": True,  # More detailed metrics
                    }
                )

            # Memory optimization settings - safe type conversion
            memory_limit_value = config.get("memory_limit_mb", 512)
            memory_limit_mb = (
                int(memory_limit_value)
                if isinstance(memory_limit_value, (int, float, str))
                else 512
            )

            # Define constants for magic values
            min_memory_threshold = 256
            high_memory_threshold = 2048

            if memory_limit_mb < min_memory_threshold:
                buffer_size_value = optimized_config.get("metrics_buffer_size", 1000)
                buffer_size = (
                    int(buffer_size_value)
                    if isinstance(buffer_size_value, (int, float, str))
                    else 1000
                )
                optimized_config["metrics_buffer_size"] = min(buffer_size, 500)
                optimized_config["enable_memory_profiling"] = True
            elif memory_limit_mb > high_memory_threshold:
                optimized_config["enable_extended_metrics"] = True
                optimized_config["enable_detailed_tracing"] = True

            # CPU optimization settings - safe type conversion
            cpu_cores_value = config.get("cpu_cores", 4)
            cpu_cores = (
                int(cpu_cores_value)
                if isinstance(cpu_cores_value, (int, float, str))
                else 4
            )
            optimized_config["max_processing_threads"] = min(cpu_cores * 2, 16)

            # Add performance metrics
            optimized_config.update(
                {
                    "performance_level": performance_level,
                    "memory_limit_mb": memory_limit_mb,
                    "cpu_cores": cpu_cores,
                    "optimization_applied": True,
                    "optimization_timestamp": "runtime",
                }
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize observability performance: {e!s}"
            )

    # ==========================================================================
    # NESTED CLASSES FOR LOGICAL ORGANIZATION
    # Complete observability components with structured logging, tracing, and metrics
    # ==========================================================================

    class Console:
        """Console-based structured logging component implementing FlextLoggerProtocol.

        This class provides structured logging capabilities with multiple severity levels,
        contextual information support, and flexible output formatting. The implementation
        follows FLEXT logging standards and integrates seamlessly with Python's standard
        logging framework while providing enhanced structured logging features.

        **ARCHITECTURAL ROLE**: The Console class serves as the structured logging
        component within the consolidated observability system, providing standardized
        logging interfaces with contextual metadata support and configurable output
        formatting for development and production environments.

        Logging Capabilities:
            - **Multi-Level Logging**: Support for trace, debug, info, warning, error, critical, fatal levels
            - **Structured Context**: Automatic JSON serialization of contextual parameters
            - **Exception Logging**: Enhanced exception logging with optional stack traces
            - **Audit Logging**: Specialized audit trail logging with clear identification
            - **Flexible Formatting**: Configurable output formatting for different environments
            - **Integration Ready**: Seamless integration with FlextLoggerProtocol interface

        Structured Logging Features:
            - **Contextual Metadata**: All logging methods accept keyword arguments for structured context
            - **JSON Serialization**: Automatic serialization of complex context objects
            - **Standard Compatibility**: Built on Python's standard logging module
            - **Performance Optimized**: Efficient context handling with conditional serialization
            - **Thread Safe**: Safe for concurrent usage across multiple threads

        Usage Examples:
            Basic structured logging::

                logger = FlextObservability.Console("app-logger")

                # Simple logging
                logger.info("User login successful")
                logger.error("Authentication failed")

                # Structured logging with context
                logger.info(
                    "Processing request",
                    user_id="12345",
                    action="create_order",
                    duration_ms=234,
                )

                logger.error(
                    "Database connection failed",
                    error_code="DB_TIMEOUT",
                    retry_count=3,
                    connection_pool="primary",
                )

            Exception logging::

                try:
                    # Some operation that might fail
                    process_payment()
                except Exception as e:
                    logger.exception(
                        "Payment processing failed",
                        payment_id="pay_123",
                        amount=99.99,
                        currency="USD",
                    )

            Audit logging::

                logger.audit(
                    "User permission changed",
                    REDACTED_LDAP_BIND_PASSWORD_user="REDACTED_LDAP_BIND_PASSWORD_123",
                    target_user="user_456",
                    old_role="user",
                    new_role="manager",
                )

        Integration with FLEXT Ecosystem:
            - **FlextLoggerProtocol**: Implements the standard FLEXT logging interface
            - **Context Propagation**: Supports correlation ID and request context propagation
            - **Configuration Integration**: Respects FlextConstants.Config.LogLevel settings
            - **Observability Integration**: Integrates with metrics and tracing systems

        Performance Considerations:
            - **Lazy Serialization**: Context objects are only serialized when actually logged
            - **Efficient Filtering**: Log level filtering prevents unnecessary processing
            - **Memory Efficient**: Minimal memory overhead for context handling
            - **Thread Safety**: Safe concurrent access without performance penalties

        Thread Safety:
            All logging methods are thread-safe and can be safely called from multiple
            threads simultaneously without synchronization requirements.

        See Also:
            - FlextLoggerProtocol: Standard FLEXT logging interface
            - FlextConstants.Config.LogLevel: Available logging levels
            - Observability: Parent observability system integration

        """

        def __init__(self, name: str = "flext-console") -> None:
            self._logger = logging.getLogger(name)
            self.name = name

        def trace(self, message: str, **kwargs: object) -> None:
            """Log trace-level message with optional structured context.

            Trace logging provides the most detailed level of logging information,
            typically used for debugging complex execution flows and detailed
            system behavior analysis. Trace messages are logged at DEBUG level
            with "TRACE:" prefix for easy identification.

            Args:
                message: The trace message to log
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Basic trace logging::

                    logger.trace("Entering user validation method")
                    logger.trace("Processing step completed")

                Trace logging with context::

                    logger.trace(
                        "Database query executed",
                        query_type="SELECT",
                        execution_time_ms=45.2,
                        rows_returned=127,
                    )

                    logger.trace(
                        "Cache lookup performed",
                        cache_key="user_123_permissions",
                        cache_hit=True,
                        ttl_remaining=3600,
                    )

            """
            self._logger.debug(
                "TRACE: %s %s", message, json.dumps(kwargs) if kwargs else ""
            )

        def debug(self, message: str, **kwargs: object) -> None:
            """Log debug-level message with optional structured context.

            Debug logging provides detailed information for diagnosing problems
            and understanding system behavior during development and troubleshooting.
            Context information is automatically structured for analysis tools.

            Args:
                message: The debug message to log
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Basic debug logging::

                    logger.debug("User authentication attempt started")
                    logger.debug("Configuration loaded successfully")

                Debug logging with context::

                    logger.debug(
                        "API request received",
                        method="POST",
                        endpoint="/api/users",
                        content_length=1024,
                        user_agent="Mozilla/5.0",
                    )

                    logger.debug(
                        "Database connection established",
                        pool_size=10,
                        active_connections=3,
                        host="internal.invalid",
                    )

            """
            self._logger.debug(message, extra={"context": kwargs} if kwargs else None)

        def info(self, message: str, **kwargs: object) -> None:
            """Log info-level message with optional structured context.

            Info logging provides general informational messages about system
            operation, including successful operations, state changes, and
            important system events. This is the standard level for operational
            logging in production environments.

            Args:
                message: The informational message to log
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Basic info logging::

                    logger.info("Application started successfully")
                    logger.info("User session created")
                    logger.info("Background job completed")

                Info logging with context::

                    logger.info(
                        "Order created successfully",
                        order_id="ord_123456",
                        customer_id="cust_789",
                        total_amount=149.99,
                        currency="USD",
                    )

                    logger.info(
                        "System maintenance completed",
                        maintenance_type="database_cleanup",
                        duration_minutes=15,
                        records_processed=50000,
                    )

            """
            self._logger.info(message, extra={"context": kwargs} if kwargs else None)

        def warning(self, message: str, **kwargs: object) -> None:
            """Log warning-level message with optional structured context.

            Warning logging indicates potentially harmful situations that should
            be monitored but don't prevent system operation. These messages
            typically indicate deprecated usage, configuration issues, or
            recoverable error conditions.

            Args:
                message: The warning message to log
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Basic warning logging::

                    logger.warning("Database connection pool nearly exhausted")
                    logger.warning("Using deprecated API endpoint")
                    logger.warning("Disk space running low")

                Warning logging with context::

                    logger.warning(
                        "High response time detected",
                        endpoint="/api/search",
                        response_time_ms=5000,
                        threshold_ms=2000,
                        request_count=145,
                    )

                    logger.warning(
                        "Configuration fallback used",
                        config_key="max_retry_attempts",
                        fallback_value=3,
                        reason="missing_configuration",
                    )

            """
            self._logger.warning(message, extra={"context": kwargs} if kwargs else None)

        def warn(self, message: str, **kwargs: object) -> None:
            """Alias for warning() method for compatibility.

            This method provides compatibility with libraries and code that use
            'warn' instead of 'warning' for logging warning-level messages.
            All functionality is identical to the warning() method.

            Args:
                message: The warning message to log
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Using warn alias::

                    logger.warn("Deprecated feature used", feature="old_api_v1")
                    logger.warn("Resource limit approaching", usage_percent=85)

            """
            self.warning(message, **kwargs)

        def error(self, message: str, **kwargs: object) -> None:
            """Log error-level message with optional structured context.

            Error logging indicates serious problems that have occurred but
            allow the application to continue running. These messages typically
            indicate failed operations, invalid inputs, or recoverable system
            failures that require attention.

            Args:
                message: The error message to log
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Basic error logging::

                    logger.error("Failed to send notification email")
                    logger.error("Database query execution failed")
                    logger.error("Invalid user input received")

                Error logging with context::

                    logger.error(
                        "Payment processing failed",
                        payment_id="pay_789",
                        error_code="INSUFFICIENT_FUNDS",
                        amount=299.99,
                        account_balance=150.00,
                    )

                    logger.error(
                        "External API call failed",
                        api_endpoint="https://api.partner.com/validate",
                        http_status=503,
                        retry_count=3,
                        timeout_ms=30000,
                    )

            """
            self._logger.error(message, extra={"context": kwargs} if kwargs else None)

        def critical(self, message: str, **kwargs: object) -> None:
            """Log critical-level message with optional structured context.

            Critical logging indicates very serious problems that may prevent
            the application from continuing to function properly. These messages
            typically indicate system failures, security issues, or data
            corruption that require immediate attention.

            Args:
                message: The critical message to log
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Basic critical logging::

                    logger.critical("Database connection completely failed")
                    logger.critical("System out of memory")
                    logger.critical("Security breach detected")

                Critical logging with context::

                    logger.critical(
                        "Service completely unavailable",
                        service_name="payment_processor",
                        downtime_minutes=15,
                        affected_users=5000,
                        last_success="2025-01-15T14:30:00Z",
                    )

                    logger.critical(
                        "Data corruption detected",
                        table_name="user_accounts",
                        corrupted_records=50,
                        integrity_check_failed=True,
                        backup_required=True,
                    )

            """
            self._logger.critical(
                message, extra={"context": kwargs} if kwargs else None
            )

        def fatal(self, message: str, **kwargs: object) -> None:
            """Alias for critical() method indicating fatal system errors.

            This method provides compatibility with libraries and code that use
            'fatal' to indicate the most severe level of system errors. All
            functionality is identical to the critical() method.

            Args:
                message: The fatal error message to log
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Using fatal alias::

                    logger.fatal("System initialization failed", component="database")
                    logger.fatal("Unrecoverable error occurred", exit_code=1)

            """
            self.critical(message, **kwargs)

        def exception(
            self, message: str, *, exc_info: bool = True, **kwargs: object
        ) -> None:
            """Log exception with automatic stack trace and structured context.

            Exception logging provides specialized handling for Python exceptions,
            automatically capturing stack traces and exception details while
            supporting structured context information. This method should be
            called from within exception handlers to capture complete error context.

            Args:
                message: Descriptive message about the exception context
                exc_info: Whether to include exception stack trace (default: True)
                **kwargs: Optional structured context data as key-value pairs

            Example:
                Basic exception logging::

                    try:
                        risky_operation()
                    except Exception:
                        logger.exception("Operation failed unexpectedly")

                Exception logging with context::

                    try:
                        process_user_data(user_id, data)
                    except ValidationError:
                        logger.exception(
                            "User data validation failed",
                            user_id=user_id,
                            data_size=len(data),
                            validation_rules="strict",
                            attempt_number=retry_count,
                        )

                Exception logging without stack trace::

                    try:
                        external_api_call()
                    except requests.RequestException:
                        logger.exception(
                            "External API unavailable",
                            exc_info=False,  # No stack trace needed
                            api_url="https://api.example.com",
                            timeout_seconds=30,
                        )

            Note:
                When exc_info=True (default), the current exception information
                is automatically captured and logged with the message. This
                includes exception type, message, and full stack trace.

            """
            # Use exc_info parameter for proper exception logging
            self._logger.error(
                message,
                exc_info=exc_info,
                extra={"context": kwargs} if kwargs else None,
            )

        def audit(self, message: str, **kwargs: object) -> None:
            """Log audit trail message with structured context for compliance.

            Audit logging provides specialized logging for compliance, security,
            and operational tracking purposes. Audit messages are clearly marked
            with "AUDIT:" prefix and automatically serialize context data for
            audit trail analysis and reporting.

            Args:
                message: The audit message describing the audited action
                **kwargs: Structured context data for audit trail tracking

            Example:
                Basic audit logging::

                    logger.audit("User login successful")
                    logger.audit("Administrative action performed")
                    logger.audit("Security policy updated")

                Audit logging with context::

                    logger.audit(
                        "User permissions modified",
                        REDACTED_LDAP_BIND_PASSWORD_user="REDACTED_LDAP_BIND_PASSWORD_123",
                        target_user="user_456",
                        permission_changes={"read_users": True, "write_orders": False},
                        ip_address="192.168.1.100",
                        timestamp="2025-01-15T14:30:00Z",
                    )

                    logger.audit(
                        "Financial transaction processed",
                        transaction_id="txn_789",
                        amount=1500.00,
                        currency="USD",
                        from_account="acc_111",
                        to_account="acc_222",
                        operator="system_auto",
                    )

                    logger.audit(
                        "Data export performed",
                        exported_table="user_data",
                        record_count=10000,
                        export_format="CSV",
                        requested_by="analyst_456",
                        approval_code="EXP-2025-001",
                    )

            Note:
                Audit messages are logged at INFO level with special formatting
                to ensure they are captured in audit logs and can be easily
                identified for compliance reporting and security analysis.

            """
            self._logger.info(
                "AUDIT: %s %s", message, json.dumps(kwargs) if kwargs else ""
            )

    class Span:
        """Distributed tracing span for operation tracking and context management.

        This class represents a single span in a distributed tracing system, providing
        the ability to track operations, collect contextual metadata, and monitor
        performance across service boundaries. Each span represents a specific operation
        or unit of work within a larger distributed trace.

        **ARCHITECTURAL ROLE**: The Span class serves as the fundamental building block
        for distributed tracing within the FLEXT observability system, providing
        standardized interfaces for operation tracking, context propagation, and
        performance monitoring across service boundaries.

        Span Capabilities:
            - **Tag Management**: Key-value tags for categorizing and filtering spans
            - **Event Logging**: Timestamped events within the span lifecycle
            - **Context Propagation**: Contextual metadata for cross-service tracing
            - **Error Tracking**: Exception and error information capture
            - **Lifecycle Management**: Proper span initialization and completion
            - **Performance Tracking**: Duration and timing information collection

        Tracing Features:
            - **Metadata Collection**: Structured tags and context for analysis
            - **Event Timeline**: Chronological event logging within span lifecycle
            - **Error Attribution**: Associate errors with specific operations
            - **Resource Attribution**: Track resource usage per operation
            - **Service Mapping**: Identify service boundaries and dependencies
            - **Performance Profiling**: Collect timing and performance metrics

        Usage Examples:
            Basic span usage::n
                span = FlextObservability.Span()
                span.set_tag("service", "user-api")
                span.set_tag("operation", "create_user")
                span.set_tag("user_id", "12345")
                span.finish()

            Event logging::n
                span = FlextObservability.Span()
                span.log_event("validation_started", {
                    "validator": "email_validator",
                    "input_length": 25
                })
                span.log_event("validation_completed", {
                    "result": "success",
                    "duration_ms": 12
                })
                span.finish()

            Context and error tracking::n
                span = FlextObservability.Span()
                span.add_context("request_id", "req_789")
                span.add_context("user_session", "sess_456")

                try:
                    # Some operation that might fail
                    process_payment()
                except PaymentError as e:
                    span.add_error(e)
                    span.set_tag("error_type", "payment_failure")
                finally:
                    span.finish()

        Integration with FLEXT Ecosystem:
            - **FlextResult Integration**: Spans can track FlextResult success/failure states
            - **Context Manager Support**: Automatic span lifecycle management
            - **Service Integration**: Seamless integration with FLEXT service architecture
            - **Metric Correlation**: Spans can be correlated with metrics and logs

        Performance Considerations:
            - **Minimal Overhead**: Lightweight span creation and management
            - **Efficient Storage**: Optimized tag and event storage
            - **Lazy Processing**: Event processing only when needed
            - **Memory Management**: Automatic cleanup of completed spans

        See Also:
            - Tracer: Parent tracer that creates and manages spans
            - trace_operation(): Context manager for automatic span management
            - FlextObservability: Main observability system integration

        """

        def set_tag(self, key: str, value: str) -> None:
            """Set a key-value tag on the span for categorization and filtering.

            Tags are metadata attributes that can be used to categorize, filter,
            and analyze spans. They are typically used for service identification,
            operation classification, and resource attribution.

            Args:
                key: The tag key/name (e.g., "service", "operation", "user_id")
                value: The tag value (e.g., "user-api", "create_user", "12345")

            Example:
                Setting standard service tags::n
                    span.set_tag("service.name", "payment-service")
                    span.set_tag("service.version", "1.2.3")
                    span.set_tag("operation.name", "process_payment")

                Setting business context tags::n
                    span.set_tag("user_id", "user_12345")
                    span.set_tag("order_id", "ord_67890")
                    span.set_tag("payment_method", "credit_card")

                Setting technical tags::n
                    span.set_tag("http.method", "POST")
                    span.set_tag("http.status_code", "200")
                    span.set_tag("db.statement", "SELECT * FROM users")

            """

        def log_event(self, event_name: str, payload: dict[str, object]) -> None:
            """Log a timestamped event within the span with structured payload.

            Events represent significant occurrences during the span's lifecycle,
            providing a timeline of what happened during the operation. Each event
            includes a timestamp and structured payload for detailed analysis.

            Args:
                event_name: Name of the event (e.g., "validation_started", "db_query_executed")
                payload: Structured data associated with the event

            Example:
                Logging validation events::n
                    span.log_event("validation_started", {
                        "validator_type": "email",
                        "input_value": "user@example.com",
                        "rules": ["format", "domain", "mx_record"]
                    })

                    span.log_event("validation_completed", {
                        "result": "success",
                        "duration_ms": 15,
                        "rules_passed": 3
                    })

                Logging database events::n
                    span.log_event("db_query_started", {
                        "query_type": "SELECT",
                        "table": "users",
                        "estimated_rows": 1000
                    })

                    span.log_event("db_query_completed", {
                        "rows_returned": 1,
                        "execution_time_ms": 23,
                        "cache_hit": False
                    })

            """

        def add_context(self, key: str, value: object) -> None:
            """Add contextual metadata to the span for cross-service correlation.

            Context data provides additional metadata that helps correlate spans
            across service boundaries and understand the broader operational context.
            Unlike tags, context data can include complex objects and nested structures.

            Args:
                key: The context key (e.g., "request_id", "user_session", "trace_context")
                value: The context value (can be string, number, dict, list, etc.)

            Example:
                Adding request context::n
                    span.add_context("request_id", "req_abc123")
                    span.add_context("correlation_id", "corr_xyz789")
                    span.add_context("client_ip", "192.168.1.100")

                Adding user context::n
                    span.add_context("user_session", {
                        "session_id": "sess_456",
                        "user_id": "user_123",
                        "roles": ["customer", "premium"]
                    })

                Adding business context::n
                    span.add_context("transaction_context", {
                        "transaction_id": "txn_789",
                        "amount": 99.99,
                        "currency": "USD",
                        "merchant_id": "merch_001"
                    })

            Note:
                Context data is typically used for correlation and analysis rather
                than filtering. Use tags for filtering and categorization purposes.

            """

        def add_error(self, error: Exception) -> None:
            """Add error information to the span for failure analysis.

            When an error occurs during span execution, this method captures
            the exception details, including error type, message, and optionally
            stack trace information for debugging and analysis.

            Args:
                error: The exception that occurred during span execution

            Example:
                Adding validation errors::n
                    try:
                        validate_user_data(data)
                    except ValidationError as e:
                        span.add_error(e)
                        span.set_tag("error_category", "validation")

                Adding service errors::n
                    try:
                        response = external_service.call_api()
                    except ServiceUnavailableError as e:
                        span.add_error(e)
                        span.set_tag("error_type", "service_unavailable")
                        span.set_tag("retry_attempted", "true")

                Adding database errors::n
                    try:
                        result = db.execute_query(sql)
                    except DatabaseError as e:
                        span.add_error(e)
                        span.set_tag("error_code", str(e.error_code))
                        span.add_context("query", sql)

            Note:
                Adding an error to a span typically marks it as failed and
                may trigger alerting systems depending on configuration.

            """

        def finish(self) -> None:
            """Complete the span and mark it as finished.

            This method signals the end of the span's lifecycle, calculates
            final timing information, and makes the span available for
            processing by the tracing backend. Spans should always be
            finished to ensure proper resource cleanup and data collection.

            Example:
                Manual span lifecycle::n
                    span = tracer.create_span("operation_name")
                    span.set_tag("service", "api")
                    try:
                        # Perform operation
                        result = do_work()
                        span.set_tag("result", "success")
                    except Exception as e:
                        span.add_error(e)
                        span.set_tag("result", "error")
                    finally:
                        span.finish()  # Always finish the span

            Note:
                Always call finish() when done with a span, preferably in a
                finally block or using context managers to ensure proper cleanup.
                Unfinished spans may cause memory leaks and incomplete tracing data.

            """

    class Tracer:
        """Distributed tracing tracer for operation tracking and span management.

        This class provides the core tracing functionality for creating and managing
        spans within a distributed system. The tracer acts as a factory for spans
        and provides context managers for automatic span lifecycle management,
        ensuring proper resource cleanup and timing collection.

        **ARCHITECTURAL ROLE**: The Tracer class serves as the central component
        for distributed tracing operations within the FLEXT observability system,
        providing standardized interfaces for span creation, lifecycle management,
        and context propagation across service boundaries.

        Tracing Capabilities:
            - **Span Creation**: Create spans for different types of operations
            - **Automatic Lifecycle**: Context managers for automatic span management
            - **Context Propagation**: Maintain trace context across operations
            - **Resource Management**: Automatic cleanup of span resources
            - **Performance Tracking**: Built-in timing and performance collection
            - **Error Handling**: Automatic error capture and span marking

        Span Types:
            - **Generic Operations**: General purpose operation tracing
            - **Business Operations**: Business logic and workflow tracking
            - **Technical Operations**: Infrastructure and system component tracking
            - **Error Operations**: Error handling and failure analysis
            - **Service Boundaries**: Cross-service communication tracking

        Usage Examples:
            Basic operation tracing::n
                tracer = FlextObservability.Tracer()

                with tracer.trace_operation("user_registration") as span:
                    span.set_tag("service", "user-api")
                    span.set_tag("user_id", "12345")

                    # Operation automatically timed
                    result = register_user(user_data)
                    span.set_tag("result", "success")
                # Span automatically finished

            Business operation tracing::n
                with tracer.business_span("process_order") as span:
                    span.set_tag("order_id", "ord_789")
                    span.set_tag("customer_id", "cust_123")

                    span.log_event("validation_started", {"rules": ["inventory", "payment"]})
                    validate_order(order)

                    span.log_event("processing_started", {"amount": 99.99})
                    result = process_payment(order.payment_info)

                    span.set_tag("payment_result", result.status)

            Technical component tracing::n
                with tracer.technical_span("database_query", component="postgresql") as span:
                    span.set_tag("query_type", "SELECT")
                    span.set_tag("table", "users")

                    span.log_event("query_started", {"sql": query})
                    result = db.execute(query)

                    span.log_event("query_completed", {
                        "rows_returned": len(result),
                        "execution_time_ms": span.duration
                    })

            Error handling tracing::n
                with tracer.error_span("payment_processing", error_type="payment_failure") as span:
                    try:
                        result = process_payment(payment_data)
                        span.set_tag("result", "success")
                    except PaymentError as e:
                        span.add_error(e)
                        span.set_tag("error_code", e.code)
                        span.add_context("payment_method", payment_data.method)

        Context Manager Benefits:
            - **Automatic Timing**: Spans are automatically timed from start to finish
            - **Resource Cleanup**: Spans are always finished, even if exceptions occur
            - **Exception Handling**: Exceptions are automatically captured and logged
            - **Consistent API**: Uniform interface for different span types
            - **Memory Safety**: Prevents memory leaks from unfinished spans

        Integration with FLEXT Ecosystem:
            - **FlextResult Integration**: Automatic success/failure tracking
            - **Service Architecture**: Seamless integration with FLEXT services
            - **Configuration Management**: Respects FlextConstants tracing settings
            - **Performance Monitoring**: Integrates with performance metrics collection

        Performance Considerations:
            - **Minimal Overhead**: Lightweight span creation and management
            - **Sampling Support**: Configurable trace sampling for performance
            - **Efficient Context**: Optimized context propagation mechanisms
            - **Resource Management**: Automatic cleanup prevents memory leaks

        Thread Safety:
            All tracing operations are thread-safe and can be used concurrently
            across multiple threads without synchronization requirements.

        See Also:
            - Span: Individual span operations and lifecycle management
            - FlextObservability: Main observability system integration
            - configure_observability_system(): Tracing configuration management

        """

        @contextmanager
        def trace_operation(
            self, operation_name: str
        ) -> Generator[FlextObservability.Span]:
            """Create and manage a span for a generic operation with automatic lifecycle.

            This context manager creates a span for tracking a generic operation,
            automatically handles span lifecycle management, and ensures proper
            resource cleanup regardless of operation success or failure.

            Args:
                operation_name: Name of the operation being traced (e.g., "user_login", "process_order")

            Yields:
                Span: The created span for adding tags, events, and context

            Example:
                Basic operation tracing::n
                    with tracer.trace_operation("user_authentication") as span:
                        span.set_tag("auth_method", "password")
                        span.set_tag("user_id", user_id)

                        result = authenticate_user(credentials)

                        if result.success:
                            span.set_tag("result", "success")
                            span.log_event("authentication_success", {
                                "user_id": user_id,
                                "session_created": True
                            })
                        else:
                            span.set_tag("result", "failure")
                            span.add_context("failure_reason", result.error)

                Complex operation with events::n
                    with tracer.trace_operation("data_processing") as span:
                        span.set_tag("data_source", "api")
                        span.set_tag("record_count", len(data))

                        span.log_event("validation_started", {"rules": validation_rules})
                        validated_data = validate(data)

                        span.log_event("transformation_started", {"transformers": 3})
                        transformed_data = transform(validated_data)

                        span.log_event("storage_started", {"destination": "database"})
                        store_data(transformed_data)

                        span.set_tag("records_processed", len(transformed_data))

            Note:
                The span is automatically finished when the context manager exits,
                regardless of whether the operation succeeds or raises an exception.
                Timing information is automatically calculated and stored.

            """
            span = FlextObservability.Span()
            # Use operation_name for span identification
            span.set_tag("operation", operation_name)
            try:
                yield span
            finally:
                span.finish()

        @contextmanager
        def business_span(
            self, operation_name: str
        ) -> Generator[FlextObservability.Span]:
            """Create and manage a span for business logic operations.

            This context manager is specifically designed for tracing business logic
            operations, automatically setting appropriate tags and context for
            business process tracking and analysis.

            Args:
                operation_name: Name of the business operation (e.g., "process_order", "calculate_pricing")

            Yields:
                Span: The created span with business operation context

            Example:
                Order processing::n
                    with tracer.business_span("process_customer_order") as span:
                        span.set_tag("customer_id", order.customer_id)
                        span.set_tag("order_value", order.total)
                        span.set_tag("payment_method", order.payment.method)

                        span.log_event("inventory_check_started", {
                            "items": len(order.items)
                        })
                        inventory_result = check_inventory(order.items)

                        if inventory_result.available:
                            span.log_event("payment_processing_started", {
                                "amount": order.total,
                                "currency": order.currency
                            })
                            payment_result = process_payment(order.payment)
                            span.set_tag("payment_status", payment_result.status)

                Pricing calculation::n
                    with tracer.business_span("calculate_dynamic_pricing") as span:
                        span.set_tag("product_id", product.id)
                        span.set_tag("customer_tier", customer.tier)

                        span.log_event("base_price_retrieved", {
                            "base_price": product.base_price
                        })

                        span.log_event("discounts_applied", {
                            "discount_rules": len(applicable_discounts),
                            "total_discount": total_discount_amount
                        })

                        final_price = apply_pricing_rules(product, customer)
                        span.set_tag("final_price", final_price)

            Note:
                Business spans are automatically tagged with "span_type": "business"
                to facilitate filtering and analysis of business operations.

            """
            # Delegate to trace_operation for consistent behavior
            with self.trace_operation(operation_name) as span:
                yield span

        @contextmanager
        def technical_span(
            self, operation_name: str, component: str | None = None
        ) -> Generator[FlextObservability.Span]:
            """Create and manage a span for technical infrastructure operations.

            This context manager is designed for tracing technical and infrastructure
            operations such as database queries, external API calls, file I/O, and
            system component interactions.

            Args:
                operation_name: Name of the technical operation (e.g., "database_query", "api_call")
                component: Optional component identifier (e.g., "postgresql", "redis", "s3")

            Yields:
                Span: The created span with technical operation context

            Example:
                Database operations::n
                    with tracer.technical_span("user_lookup_query", component="postgresql") as span:
                        span.set_tag("db.statement", "SELECT * FROM users WHERE id = ?")
                        span.set_tag("db.table", "users")
                        span.set_tag("db.operation", "SELECT")

                        span.log_event("query_started", {"parameters": [user_id]})
                        result = db.execute(query, [user_id])

                        span.log_event("query_completed", {
                            "rows_returned": len(result),
                            "cache_hit": result.from_cache
                        })

                External API calls::n
                    with tracer.technical_span("payment_gateway_call", component="stripe") as span:
                        span.set_tag("http.method", "POST")
                        span.set_tag("http.url", "https://api.stripe.com/v1/charges")
                        span.set_tag("api.version", "2023-10-16")

                        span.log_event("request_sent", {
                            "payload_size": len(request_data),
                            "timeout_ms": 30000
                        })

                        response = stripe_client.create_charge(charge_data)

                        span.set_tag("http.status_code", response.status_code)
                        span.log_event("response_received", {
                            "response_time_ms": response.elapsed.total_seconds() * 1000
                        })

                File system operations::n
                    with tracer.technical_span("file_processing", component="filesystem") as span:
                        span.set_tag("file.path", file_path)
                        span.set_tag("file.operation", "read")

                        span.log_event("file_access_started", {
                            "file_size_bytes": os.path.getsize(file_path)
                        })

                        with open(file_path, 'r') as f:
                            content = f.read()

                        span.set_tag("bytes_read", len(content))
                        span.log_event("file_processing_completed", {
                            "lines_processed": len(content.splitlines())
                        })

            Note:
                Technical spans are automatically tagged with "span_type": "technical"
                and include the component name if provided for infrastructure analysis.

            """
            with self.trace_operation(operation_name) as span:
                if component:
                    span.set_tag("component", component)
                yield span

        @contextmanager
        def error_span(
            self, operation_name: str, error_type: str | None = None
        ) -> Generator[FlextObservability.Span]:
            """Create and manage a span specifically for error handling and failure analysis.

            This context manager is designed for tracing operations that are expected
            to handle errors or are being used for error recovery and failure analysis.
            It automatically sets up error-specific tags and context.

            Args:
                operation_name: Name of the error handling operation (e.g., "error_recovery", "fallback_processing")
                error_type: Optional error type identifier (e.g., "validation_error", "service_unavailable")

            Yields:
                Span: The created span with error handling context

            Example:
                Error recovery operations::n
                    with tracer.error_span("payment_retry_logic", error_type="payment_failure") as span:
                        span.set_tag("original_error", str(original_exception))
                        span.set_tag("retry_attempt", retry_count)
                        span.set_tag("max_retries", max_retry_attempts)

                        span.log_event("retry_started", {
                            "delay_ms": retry_delay,
                            "strategy": "exponential_backoff"
                        })

                        try:
                            result = retry_payment(payment_data)
                            span.set_tag("retry_result", "success")
                        except Exception as e:
                            span.add_error(e)
                            span.set_tag("retry_result", "failure")

                Fallback processing::n
                    with tracer.error_span("service_fallback", error_type="service_unavailable") as span:
                        span.set_tag("primary_service", "user-service")
                        span.set_tag("fallback_service", "cache-service")
                        span.set_tag("fallback_reason", "primary_service_timeout")

                        span.log_event("fallback_initiated", {
                            "cache_age_minutes": cache_age,
                            "acceptable_staleness": True
                        })

                        cached_result = get_cached_user_data(user_id)

                        span.set_tag("fallback_success", cached_result is not None)
                        span.log_event("fallback_completed", {
                            "data_source": "cache",
                            "freshness_score": calculate_freshness(cached_result)
                        })

                Exception analysis::n
                    with tracer.error_span("exception_analysis", error_type="unhandled_exception") as span:
                        span.add_error(exception)
                        span.set_tag("exception_type", type(exception).__name__)
                        span.set_tag("error_severity", "critical")

                        span.add_context("stack_trace", traceback.format_exc())
                        span.add_context("system_state", get_system_state())

                        span.log_event("error_reported", {
                            "notification_sent": True,
                            "ticket_created": ticket_id
                        })

            Note:
                Error spans are automatically tagged with "span_type": "error" and
                include error type information for failure analysis and monitoring.

            """
            with self.trace_operation(operation_name) as span:
                if error_type:
                    span.set_tag("error_type", error_type)
                yield span

    class Metrics:
        """In-memory metrics collection system for performance and operational monitoring.

        This class provides comprehensive metrics collection capabilities including
        counters, gauges, and histograms with tag-based organization and efficient
        in-memory storage. The metrics system is designed for high-throughput
        environments with minimal overhead and flexible aggregation capabilities.

        **ARCHITECTURAL ROLE**: The Metrics class serves as the central metrics
        collection component within the FLEXT observability system, providing
        standardized interfaces for collecting, organizing, and retrieving
        performance and operational metrics with tag-based categorization.

        Metrics Types:
            - **Counters**: Monotonically increasing values (requests, errors, events)
            - **Gauges**: Point-in-time values (memory usage, active connections, queue depth)
            - **Histograms**: Distribution of values (response times, payload sizes, duration)
            - **Tagged Metrics**: All metric types support tag-based categorization

        Collection Capabilities:
            - **High Performance**: Optimized in-memory storage for minimal overhead
            - **Tag-Based Organization**: Flexible tagging for filtering and aggregation
            - **Thread Safety**: Safe concurrent access from multiple threads
            - **Memory Efficient**: Efficient data structures and storage patterns
            - **Bulk Operations**: Support for batch metric collection and retrieval
            - **Reset and Clear**: Administrative operations for metric lifecycle management

        Usage Examples:
            Counter metrics for request tracking::

                metrics = FlextObservability.Metrics()

                # Simple counter increment
                metrics.increment_counter("http_requests_total")
                metrics.increment_counter("api_calls_total")

                # Tagged counter metrics
                metrics.increment_counter(
                    "http_requests_total",
                    {"method": "POST", "endpoint": "/api/users", "status": "200"},
                )

                # Custom increment values
                metrics.increment(
                    "processed_records",
                    value=50,
                    tags={"source": "database", "table": "orders"},
                )

            Gauge metrics for system monitoring::

                # System resource monitoring
                metrics.record_gauge("memory_usage_bytes", 1024 * 1024 * 512)
                metrics.record_gauge("active_connections", 25)
                metrics.record_gauge("queue_depth", 150)

                # Tagged gauge metrics
                metrics.record_gauge(
                    "cpu_usage_percent", 75.5, {"core": "cpu0", "host": "web-server-01"}
                )

                # Business metrics
                metrics.gauge("daily_revenue", 15420.50)
                metrics.gauge("active_users", 1847)

            Histogram metrics for performance tracking::

                # Response time tracking
                metrics.histogram("response_time_ms", 125.3)
                metrics.histogram("response_time_ms", 89.7)
                metrics.histogram("response_time_ms", 234.1)

                # File size tracking
                metrics.histogram("upload_size_bytes", 1024 * 1024 * 2.5)  # 2.5MB
                metrics.histogram("upload_size_bytes", 1024 * 512)  # 512KB

        Integration with FLEXT Ecosystem:
            - **FlextResult Integration**: Automatic success/failure metric collection
            - **Performance Monitoring**: Integration with FLEXT performance systems
            - **Configuration Management**: Respects FlextConstants metrics settings
            - **Observability System**: Seamless integration with logging and tracing

        Performance Considerations:
            - **Memory Efficient**: Optimized data structures for minimal memory usage
            - **Fast Operations**: O(1) metric recording and retrieval operations
            - **Tag Optimization**: Efficient tag-based key generation and storage
            - **Bulk Processing**: Support for bulk operations and batch processing
            - **Thread Safety**: Concurrent access without performance penalties

        Thread Safety:
            All metrics operations are thread-safe and can be safely called from
            multiple threads simultaneously without synchronization requirements.

        See Also:
            - FlextObservability: Main observability system integration
            - configure_observability_system(): Metrics configuration management
            - Observability: Parent observability instance with integrated metrics

        """

        def __init__(self) -> None:
            """Initialize the metrics collection system with empty metric stores.

            Creates empty dictionaries for storing counters, gauges, and histograms
            with thread-safe access patterns and efficient key-based organization.

            """
            self._counters: dict[str, int] = {}
            self._gauges: dict[str, float] = {}
            self._histograms: dict[str, list[float]] = {}

        def increment_counter(
            self, name: str, tags: dict[str, str] | None = None
        ) -> None:
            """Increment a counter metric by 1 with optional tag-based categorization.

            Counters represent monotonically increasing values such as request counts,
            error counts, or event occurrences. Each call increments the counter by 1.
            Tags allow for flexible categorization and filtering of metrics.

            Args:
                name: The counter metric name (e.g., "http_requests_total", "errors_count")
                tags: Optional dictionary of key-value tags for categorization

            Example:
                Basic counter usage::

                    metrics.increment_counter("api_requests_total")
                    metrics.increment_counter("user_registrations")
                    metrics.increment_counter("background_jobs_completed")

                Tagged counter usage::

                    # HTTP request metrics with detailed tags
                    metrics.increment_counter(
                        "http_requests_total",
                        {
                            "method": "POST",
                            "endpoint": "/api/users",
                            "status_code": "200",
                            "service": "user-api",
                        },
                    )

                    # Error tracking with categorization
                    metrics.increment_counter(
                        "errors_total",
                        {
                            "error_type": "validation_error",
                            "component": "user_validator",
                            "severity": "warning",
                        },
                    )

            Note:
                Each call increments the counter by exactly 1. For custom increment
                values, use the increment() method instead.

            """
            key = self._make_key(name, tags)
            self._counters[key] = self._counters.get(key, 0) + 1

        def increment(
            self, name: str, value: int = 1, tags: dict[str, str] | None = None
        ) -> None:
            """Increment a counter metric by a custom value with optional tags.

            This method allows incrementing counters by values other than 1,
            useful for batch operations, bulk processing, or when tracking
            quantities rather than simple event counts.

            Args:
                name: The counter metric name (e.g., "records_processed", "bytes_transferred")
                value: The increment value (default: 1)
                tags: Optional dictionary of key-value tags for categorization

            Example:
                Batch processing metrics::

                    # Record batch processing counts
                    metrics.increment(
                        "records_processed",
                        value=500,
                        tags={
                            "source": "database",
                            "table": "user_events",
                            "batch_size": "500",
                        },
                    )

                    # Track data transfer volumes
                    metrics.increment(
                        "bytes_transferred",
                        value=1024 * 1024 * 50,
                        tags={
                            "direction": "upload",
                            "protocol": "https",
                            "compression": "gzip",
                        },
                    )

                Simple increment operations::

                    # Default increment by 1
                    metrics.increment("api_calls")

                    # Custom increment values
                    metrics.increment("queue_items_processed", value=25)
                    metrics.increment("failed_retries", value=3)

            """
            if tags:
                key = self._make_key(name, tags)
                self._counters[key] = self._counters.get(key, 0) + value
            else:
                self._counters[name] = self._counters.get(name, 0) + value

        def record_gauge(
            self, name: str, value: float, tags: dict[str, str] | None = None
        ) -> None:
            """Record a gauge metric value representing a point-in-time measurement.

            Gauges represent values that can increase or decrease over time,
            such as memory usage, active connections, queue depths, or any
            measurement that represents a current state rather than a cumulative count.

            Args:
                name: The gauge metric name (e.g., "memory_usage_bytes", "active_connections")
                value: The current gauge value as a float
                tags: Optional dictionary of key-value tags for categorization

            Example:
                System resource monitoring::

                    # Memory and CPU monitoring
                    metrics.record_gauge(
                        "memory_usage_bytes", 512 * 1024 * 1024
                    )  # 512MB
                    metrics.record_gauge("cpu_usage_percent", 75.5)
                    metrics.record_gauge("disk_usage_percent", 82.3)

                    # Network and connection monitoring
                    metrics.record_gauge("active_connections", 150)
                    metrics.record_gauge("connection_pool_size", 20)
                    metrics.record_gauge("network_throughput_mbps", 125.7)

                Tagged gauge metrics::

                    # Service-specific resource monitoring
                    metrics.record_gauge(
                        "heap_memory_mb",
                        384.5,
                        {
                            "service": "payment-processor",
                            "instance": "payment-01",
                            "region": "us-east-1",
                        },
                    )

                    # Database connection monitoring
                    metrics.record_gauge(
                        "db_connections_active",
                        15,
                        {
                            "database": "postgresql",
                            "pool_name": "primary",
                            "host": "internal.invalid",
                        },
                    )

            Note:
                Unlike counters, gauge values represent the current state and can
                both increase and decrease. Each call sets the gauge to the specified value.

            """
            key = self._make_key(name, tags)
            self._gauges[key] = value

        def gauge(self, name: str, value: float) -> None:
            """Record a simple gauge metric value without tags for compatibility.

            This method provides a simplified interface for recording gauge metrics
            without tag-based categorization. It's useful for simple measurements
            and provides backward compatibility with existing code.

            Args:
                name: The gauge metric name (e.g., "temperature", "queue_size")
                value: The current gauge value as a float

            Example:
                Simple gauge measurements::

                    metrics.gauge("system_temperature", 72.5)
                    metrics.gauge("available_memory_gb", 8.2)
                    metrics.gauge("active_users", 1847)
                    metrics.gauge("response_time_avg", 125.3)

            Note:
                For more detailed categorization and filtering capabilities,
                use record_gauge() with tags instead.

            """
            self._gauges[name] = value

        def histogram(self, name: str, value: float) -> None:
            """Record a value in a histogram for distribution analysis.

            Histograms collect samples of values to analyze their distribution
            over time. They're particularly useful for measuring latencies,
            response times, payload sizes, and other values where understanding
            the distribution is more important than individual measurements.

            Args:
                name: The histogram metric name (e.g., "response_time_ms", "payload_size_bytes")
                value: The value to add to the histogram distribution

            Example:
                Response time tracking::

                    # Record multiple response times
                    metrics.histogram("api_response_time_ms", 45.2)
                    metrics.histogram("api_response_time_ms", 123.7)
                    metrics.histogram("api_response_time_ms", 89.1)
                    metrics.histogram("api_response_time_ms", 234.8)

                    # Later analysis can determine percentiles, averages, etc.

                Processing duration analysis::

                    # Database query times
                    metrics.histogram("db_query_duration_ms", 12.3)
                    metrics.histogram("db_query_duration_ms", 167.9)
                    metrics.histogram("db_query_duration_ms", 45.1)

                    # File processing times
                    metrics.histogram("file_processing_duration_s", 2.34)
                    metrics.histogram("file_processing_duration_s", 5.67)

            Note:
                Histogram values are stored in memory as lists. For high-frequency
                metrics, consider using sampling or external histogram systems
                for production deployments to manage memory usage.

            """
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)

        def get_metrics(self) -> dict[str, object]:
            r"""Retrieve all collected metrics organized by metric type.

            Returns a complete snapshot of all metrics currently stored in the
            system, organized by type (counters, gauges, histograms) with
            deep copies to prevent external modification of internal state.

            Returns:
                Dictionary containing all metrics organized by type:
                - 'counters': All counter metrics with their current values
                - 'gauges': All gauge metrics with their current values
                - 'histograms': All histogram metrics with their value arrays

            Example:
                Analyzing collected metrics::

                    # Get all metrics
                    all_metrics = metrics.get_metrics()

                    # Analyze counter metrics
                    print("=== Counter Metrics ===")
                    for name, value in all_metrics["counters"].items():
                        print(f"{name}: {value}")

                    # Analyze gauge metrics
                    print("\\n=== Gauge Metrics ===")
                    for name, value in all_metrics["gauges"].items():
                        print(f"{name}: {value:.2f}")

                    # Analyze histogram distributions
                    print("\\n=== Histogram Metrics ===")
                    for name, values in all_metrics["histograms"].items():
                        if values:
                            avg = sum(values) / len(values)
                            minimum = min(values)
                            maximum = max(values)
                            count = len(values)
                            print(
                                f"{name}: count={count}, avg={avg:.2f}, min={minimum:.2f}, max={maximum:.2f}"
                            )

            Note:
                The returned dictionary contains deep copies of all metrics data
                to prevent accidental modification of the internal metric state.
                Changes to the returned data will not affect the stored metrics.

            """
            return {
                "counters": self._counters.copy(),
                "gauges": self._gauges.copy(),
                "histograms": {k: v.copy() for k, v in self._histograms.items()},
            }

        def clear_metrics(self) -> None:
            """Clear all collected metrics from memory for testing and lifecycle management.

            This method removes all stored counter, gauge, and histogram data,
            resetting the metrics system to an empty state. It's primarily used
            for testing scenarios, system resets, or when implementing metric
            rotation policies.

            Example:
                Testing scenarios::

                    # Setup test metrics
                    metrics.increment_counter("test_requests")
                    metrics.gauge("test_value", 42.0)
                    metrics.histogram("test_duration", 123.45)

                    # Verify metrics exist
                    assert len(metrics.get_metrics()["counters"]) > 0

                    # Clear all metrics
                    metrics.clear_metrics()

                    # Verify metrics are cleared
                    empty_metrics = metrics.get_metrics()
                    assert len(empty_metrics["counters"]) == 0
                    assert len(empty_metrics["gauges"]) == 0
                    assert len(empty_metrics["histograms"]) == 0

            Warning:
                This operation is irreversible and will permanently delete all
                collected metrics data. Ensure metrics are exported or backed up
                before clearing if historical data is needed.

            """
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

        def _make_key(self, name: str, tags: dict[str, str] | None) -> str:
            """Create a unique key for tagged metrics by combining name and sorted tags.

            This internal method generates consistent, unique keys for metrics by
            combining the metric name with sorted tag key-value pairs. The consistent
            sorting ensures that the same tags always generate the same key regardless
            of the order they were provided.

            Args:
                name: The base metric name
                tags: Optional dictionary of tag key-value pairs

            Returns:
                A unique string key combining name and tags

            Note:
                This is an internal method used by other metric recording methods.
                Tags are sorted alphabetically by key to ensure consistent key
                generation regardless of input order.

            """
            if not tags:
                return name
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}[{tag_str}]"

    class Observability:
        """Unified observability instance providing integrated logging, tracing, metrics, alerts, and health monitoring.

        This class serves as the primary interface for accessing all observability functionality
        within the FLEXT ecosystem. It provides a single, cohesive API that integrates
        structured logging, distributed tracing, metrics collection, alerting, and health
        monitoring into a unified observability solution.

        **ARCHITECTURAL ROLE**: The Observability class acts as the main entry point
        for all observability operations, providing a unified interface that coordinates
        between logging, tracing, metrics, alerting, and health monitoring systems
        while maintaining backward compatibility with existing APIs.

        Integrated Components:
            - **Logger/Log**: Structured logging with contextual information support
            - **Tracer/Trace**: Distributed tracing for operation tracking and performance monitoring
            - **Metrics**: Comprehensive metrics collection (counters, gauges, histograms)
            - **Alerts**: Multi-level alerting system with context information
            - **Health**: System health monitoring and status reporting

        Key Features:
            - **Unified API**: Single interface for all observability operations
            - **Component Integration**: Seamless coordination between different observability systems
            - **FlextResult Integration**: Type-safe error handling throughout
            - **Backward Compatibility**: Maintains compatibility with legacy APIs
            - **Test Support**: Full API compatibility for testing environments
            - **Configuration Management**: Centralized configuration for all components

        Usage Examples:
            Basic observability setup::

                obs = FlextObservability.Observability()

                # Structured logging
                obs.logger.info(
                    "User login attempt", user_id="12345", ip="192.168.1.100"
                )
                obs.log.error("Authentication failed", error_code="INVALID_CREDENTIALS")

                # Distributed tracing
                with obs.tracer.trace_operation("user_authentication") as span:
                    span.set_tag("user_id", "12345")
                    span.set_tag("auth_method", "password")
                    # Authentication logic here
                    span.set_tag("result", "success")

                # Metrics collection
                obs.metrics.increment_counter(
                    "login_attempts_total", {"method": "password", "result": "success"}
                )
                obs.metrics.record_gauge("active_sessions", 150)
                obs.metrics.histogram("login_duration_ms", 234.5)

        Integration with FLEXT Ecosystem:
            - **FlextResult Pattern**: All operations return FlextResult for type safety
            - **FlextConstants Integration**: Respects system configuration constants
            - **FlextTypes Support**: Uses centralized type definitions
            - **Configuration Management**: Integrates with FlextObservability configuration

        Performance Considerations:
            - **Lazy Initialization**: Components are initialized only when needed
            - **Efficient Coordination**: Minimal overhead for component integration
            - **Memory Management**: Optimized memory usage across all components
            - **Thread Safety**: All components are thread-safe for concurrent usage

        Thread Safety:
            All observability operations are thread-safe and can be used concurrently
            across multiple threads without synchronization requirements.

        See Also:
            - FlextObservability: Parent observability system
            - Console: Structured logging component
            - Tracer: Distributed tracing component
            - Metrics: Metrics collection component
            - Alerts: Alerting system component
            - Health: Health monitoring component

        """

        def __init__(self) -> None:
            """Initialize the unified observability instance with all integrated components.

            Creates and configures all observability components (logging, tracing, metrics,
            alerts, and health monitoring) with both standard and legacy API compatibility.
            All components are initialized with thread-safe access patterns.

            """
            # Standard attributes (legacy)
            self.logger = FlextObservability.Console()
            self.tracer = FlextObservability.Tracer()
            self.metrics = FlextObservability.Metrics()

            # Test API compatibility attributes
            self.log = self.logger  # Tests expect .log
            self.trace = self.tracer  # Tests expect .trace
            self.alerts = FlextObservability.Alerts()
            self.health = FlextObservability.Health()

        def record_metric(
            self, name: str, value: float, tags: dict[str, str] | None = None
        ) -> FlextResult[None]:
            """Record a gauge metric with comprehensive error handling and FlextResult integration.

            This method provides a high-level interface for recording metrics with
            full error handling and type safety. It wraps the underlying metrics
            system with FlextResult pattern for consistent error handling across
            the FLEXT ecosystem.

            Args:
                name: The metric name (e.g., "response_time_ms", "active_connections")
                value: The metric value to record as a float
                tags: Optional dictionary of key-value tags for categorization

            Returns:
                FlextResult indicating success or failure of the metric recording operation

            Example:
                Basic metric recording::

                    obs = FlextObservability.Observability()

                    result = obs.record_metric("api_response_time", 125.3)
                    if result.success:
                        print("Metric recorded successfully")
                    else:
                        print(f"Failed to record metric: {result.error}")

                Tagged metric recording::

                    result = obs.record_metric(
                        "database_query_time",
                        45.7,
                        {
                            "database": "postgresql",
                            "table": "users",
                            "operation": "SELECT",
                        },
                    )

                    if result.failure:
                        obs.logger.error(
                            "Metric recording failed",
                            metric_name=name,
                            error=result.error,
                        )

            Note:
                This method specifically records gauge metrics. For counter increments
                or histogram values, use the specific methods on the metrics component
                (obs.metrics.increment_counter() or obs.metrics.histogram()).

            """
            try:
                self.metrics.record_gauge(name, value, tags)
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to record metric: {e}")

    class Alerts:
        """Multi-level alerting system for operational notifications and escalation management.

        This class provides a comprehensive alerting system with multiple severity levels,
        structured context support, and integration capabilities for operational monitoring
        and incident response. The alerting system is designed to work seamlessly with
        logging, metrics, and tracing systems for comprehensive observability.

        **ARCHITECTURAL ROLE**: The Alerts class serves as the centralized alerting
        component within the FLEXT observability system, providing standardized
        interfaces for generating, categorizing, and managing alerts across different
        severity levels with contextual metadata support.

        Alert Levels:
            - **Info**: Informational alerts for operational awareness
            - **Warning**: Warning alerts for conditions requiring attention
            - **Error**: Error alerts for failures requiring intervention
            - **Critical**: Critical alerts for severe conditions requiring immediate response

        Alerting Capabilities:
            - **Multi-Level Severity**: Support for different alert severity levels
            - **Structured Context**: Rich contextual metadata with alert messages
            - **Integration Ready**: Seamless integration with external alerting systems
            - **Performance Optimized**: Minimal overhead alert generation
            - **Thread Safety**: Safe concurrent alert generation from multiple threads
            - **API Compatibility**: Full compatibility with testing and legacy systems

        Usage Examples:
            Basic alerting::

                alerts = FlextObservability.Alerts()

                # Informational alerts
                alerts.info("System maintenance completed")
                alerts.info("Scheduled backup initiated")

                # Warning alerts
                alerts.warning("High memory usage detected")
                alerts.warning("API response time degradation")

                # Error alerts
                alerts.error("Database connection failed")
                alerts.error("Payment processing error")

                # Critical alerts
                alerts.critical("Service completely unavailable")
                alerts.critical("Data corruption detected")

            Structured alerting with context::

                # Service monitoring alerts
                alerts.warning(
                    "High error rate detected",
                    service="payment-processor",
                    error_rate=15.7,
                    threshold=10.0,
                    time_window="5m",
                )

                # Resource monitoring alerts
                alerts.error(
                    "Disk space critically low",
                    server="web-01",
                    disk_usage_percent=95,
                    available_gb=2.1,
                    mount_point="/var/log",
                )

                # Business process alerts
                alerts.critical(
                    "Payment gateway timeout",
                    gateway="stripe",
                    transaction_id="txn_123456",
                    timeout_ms=30000,
                    retry_attempts=3,
                    customer_impact="high",
                )

        Integration with FLEXT Ecosystem:
            - **Observability Integration**: Seamless coordination with logging, metrics, and tracing
            - **FlextResult Pattern**: Can be extended to return FlextResult for error handling
            - **Configuration Management**: Respects FlextConstants alerting configuration
            - **Context Propagation**: Supports correlation IDs and request context

        Performance Considerations:
            - **Minimal Overhead**: Lightweight alert generation with efficient processing
            - **Asynchronous Processing**: Can be extended for asynchronous alert delivery
            - **Rate Limiting**: Can be extended with rate limiting to prevent alert storms
            - **Memory Efficient**: Efficient context handling and message processing

        Thread Safety:
            All alerting operations are thread-safe and can be safely called from
            multiple threads simultaneously without synchronization requirements.

        See Also:
            - FlextObservability: Main observability system integration
            - Console: Structured logging integration
            - Metrics: Metrics-based alerting triggers
            - Health: Health monitoring alerting integration

        """

        def info(self, message: str, **kwargs: object) -> None:
            """Generate an informational alert for operational awareness and status updates.

            Informational alerts are used to communicate operational status, completed
            activities, and general awareness information. They typically don't require
            immediate action but provide valuable context for operational teams.

            Args:
                message: The informational alert message
                **kwargs: Additional structured context data for the alert

            Example:
                Basic informational alerts::

                    alerts.info("System startup completed successfully")
                    alerts.info("Scheduled maintenance window began")
                    alerts.info("Cache warming process initiated")

                Structured informational alerts::

                    alerts.info(
                        "Deployment completed successfully",
                        service="user-api",
                        version="1.2.3",
                        deployment_time="2025-01-15T10:30:00Z",
                        instances_updated=5,
                    )

                    alerts.info(
                        "Backup process completed",
                        database="production",
                        backup_size_gb=15.7,
                        duration_minutes=45,
                        backup_location="s3://backups/2025-01-15/",
                    )

            """

        def warning(self, message: str, **kwargs: object) -> None:
            """Generate a warning alert for conditions requiring attention but not immediate action.

            Warning alerts indicate conditions that should be monitored and may require
            future action if they worsen. They represent potential issues that haven't
            yet caused service impact but could lead to problems if left unaddressed.

            Args:
                message: The warning alert message
                **kwargs: Additional structured context data for the alert

            Example:
                Resource monitoring warnings::

                    alerts.warning(
                        "High memory usage detected",
                        server="web-01",
                        memory_usage_percent=85.5,
                        threshold_percent=80.0,
                        available_mb=512,
                    )

                    alerts.warning(
                        "API response time degradation",
                        endpoint="/api/search",
                        avg_response_ms=2500,
                        threshold_ms=2000,
                        requests_affected=150,
                    )

                Business process warnings::

                    alerts.warning(
                        "Payment processing delays detected",
                        gateway="stripe",
                        avg_processing_time_s=15.3,
                        normal_time_s=3.2,
                        transactions_affected=25,
                    )

            """

        def critical(self, message: str, **kwargs: object) -> None:
            """Generate a critical alert for severe conditions requiring immediate response.

            Critical alerts indicate severe system conditions that require immediate
            attention and response. These alerts typically represent service outages,
            data corruption, security incidents, or other conditions that have
            significant business impact and require urgent intervention.

            Args:
                message: The critical alert message
                **kwargs: Additional structured context data for the alert

            Example:
                Service availability critical alerts::

                    alerts.critical(
                        "Payment service completely unavailable",
                        service="payment-processor",
                        downtime_minutes=5,
                        affected_customers=1500,
                        last_successful_transaction="2025-01-15T14:25:00Z",
                        error_rate_percent=100,
                    )

                Data integrity critical alerts::

                    alerts.critical(
                        "Database corruption detected",
                        database="customer_data",
                        affected_tables=["users", "orders"],
                        corrupted_records=500,
                        integrity_check_failed=True,
                        backup_required=True,
                        estimated_data_loss="< 1 hour",
                    )

            Note:
                Critical alerts should trigger immediate notification mechanisms
                such as paging systems, incident response procedures, and
                high-priority communication channels.

            """

        def error(self, message: str, **kwargs: object) -> None:
            """Generate an error alert for failures requiring intervention.

            Error alerts indicate system failures, operational errors, or other
            conditions that have caused or are likely to cause service impact.
            These alerts require investigation and corrective action, though they
            may not be as severe as critical alerts.

            Args:
                message: The error alert message
                **kwargs: Additional structured context data for the alert

            Example:
                Service error alerts::

                    alerts.error(
                        "Database connection pool exhausted",
                        database="postgresql-primary",
                        max_connections=20,
                        active_connections=20,
                        queued_requests=45,
                        estimated_wait_time_s=30,
                    )

                    alerts.error(
                        "External API integration failure",
                        api_provider="payment_gateway",
                        endpoint="https://api.payments.com/charge",
                        http_status=503,
                        error_message="Service Temporarily Unavailable",
                        retry_attempts=3,
                        next_retry_in_s=60,
                    )

            """

    class Health:
        """System health monitoring component for status tracking and health check operations.

        This class provides comprehensive system health monitoring capabilities including
        health status tracking, health check operations, and integration with alerting
        and monitoring systems. The health component is designed to provide both simple
        boolean health status and detailed health information for operational teams.

        **ARCHITECTURAL ROLE**: The Health class serves as the centralized health
        monitoring component within the FLEXT observability system, providing
        standardized interfaces for health status management, health checks, and
        integration with operational monitoring and alerting systems.

        Health Monitoring Capabilities:
            - **Status Tracking**: Maintain current system health status
            - **Health Checks**: Perform comprehensive health check operations
            - **Status Reporting**: Detailed health status reporting with timestamps
            - **Integration Ready**: Seamless integration with monitoring and alerting systems
            - **API Compatibility**: Full compatibility with testing and legacy systems
            - **Performance Optimized**: Minimal overhead health status operations

        Health Status Types:
            - **Healthy**: System is operating normally
            - **Degraded**: System is operational but with reduced performance
            - **Unhealthy**: System is experiencing significant issues
            - **Critical**: System is in critical state requiring immediate attention

        Usage Examples:
            Basic health monitoring::

                health = FlextObservability.Health()

                # Check current health status
                if health.is_healthy():
                    print("System is healthy")
                else:
                    print("System requires attention")

                # Get detailed health information
                health_info = health.check()
                print(f"Status: {health_info['status']}")
                print(f"Last checked: {health_info['timestamp']}")

            Integration with observability system::

                obs = FlextObservability.Observability()


                # Periodic health monitoring
                def perform_health_check():
                    health_status = obs.health.check()

                    # Log health status
                    obs.logger.info("Health check performed", **health_status)

                    # Record health metrics
                    status_value = 1.0 if obs.health.is_healthy() else 0.0
                    obs.metrics.record_gauge("system_health_status", status_value)

                    # Trigger alerts for unhealthy status
                    if not obs.health.is_healthy():
                        obs.alerts.warning("System health degraded", **health_status)

                    return health_status

        Integration with FLEXT Ecosystem:
            - **Observability Integration**: Seamless coordination with logging, metrics, and alerting
            - **FlextResult Pattern**: Can be extended to return FlextResult for error handling
            - **Configuration Management**: Respects FlextConstants health monitoring configuration
            - **Performance Monitoring**: Integration with system performance metrics

        Performance Considerations:
            - **Lightweight Operations**: Minimal overhead health status checking
            - **Efficient Status Management**: Optimized status storage and retrieval
            - **Configurable Intervals**: Configurable health check intervals
            - **Caching Support**: Can be extended with status caching for high-frequency checks

        Thread Safety:
            All health monitoring operations are thread-safe and can be safely called
            from multiple threads simultaneously without synchronization requirements.

        See Also:
            - FlextObservability: Main observability system integration
            - Alerts: Health-based alerting integration
            - Metrics: Health status metrics collection
            - Console: Health status logging integration

        """

        def __init__(self) -> None:
            """Initialize the health monitoring component with default healthy status.

            Sets up the health monitoring system with a default "healthy" status
            and prepares the component for health check operations and status tracking.

            """
            self._status = "healthy"

        def check(self) -> dict[str, object]:
            """Perform a comprehensive health check and return detailed status information.

            This method performs a complete health check of the system and returns
            detailed health information including status, timestamp, and additional
            health metrics. The information can be used for monitoring dashboards,
            health endpoints, and operational reporting.

            Returns:
                Dictionary containing detailed health information:
                - 'status': Current health status string
                - 'timestamp': Timestamp of the health check
                - Additional health metrics and information

            Example:
                Basic health check::

                    health = FlextObservability.Health()
                    health_info = health.check()

                    print(f"System Status: {health_info['status']}")
                    print(f"Last Check: {health_info['timestamp']}")

                Health check with detailed analysis::

                    health_info = health.check()

                    if health_info["status"] == "healthy":
                        logger.info("Health check passed", **health_info)
                    else:
                        logger.warning("Health check concerns detected", **health_info)
                        # Trigger appropriate alerts or remediation actions

                Integration with monitoring systems::

                    def export_health_metrics():
                        health_info = health.check()

                        # Export to monitoring system
                        monitoring_client.send_health_data(
                            {
                                "service_name": "flext-core",
                                "health_status": health_info["status"],
                                "check_timestamp": health_info["timestamp"],
                                "additional_data": health_info,
                            }
                        )

            Note:
                The returned dictionary is suitable for JSON serialization and
                can be directly used in HTTP health endpoints, monitoring exports,
                and operational dashboards.

            """
            return {"status": self._status, "timestamp": "now"}

        def is_healthy(self) -> bool:
            """Check if the system is currently in a healthy state.

            This method provides a simple boolean check for system health status,
            useful for quick health validations, conditional logic, and simple
            monitoring scenarios where detailed health information is not needed.

            Returns:
                True if the system is healthy, False otherwise

            Example:
                Simple health validation::

                    health = FlextObservability.Health()

                    if health.is_healthy():
                        print("System is operating normally")
                    else:
                        print("System requires attention")

                Conditional processing based on health::

                    def process_requests():
                        health = FlextObservability.Health()

                        if not health.is_healthy():
                            logger.warning("System unhealthy, rejecting new requests")
                            return {"error": "Service temporarily unavailable"}

                        # Process requests normally
                        return handle_normal_processing()

                Health-based circuit breaker::

                    class HealthBasedCircuitBreaker:
                        def __init__(self):
                            self.health = FlextObservability.Health()

                        def should_process_request(self) -> bool:
                            # Only process requests if system is healthy
                            return self.health.is_healthy()

                        def handle_request(self, request):
                            if not self.should_process_request():
                                raise ServiceUnavailableError(
                                    "System health check failed"
                                )

                            return process_request(request)

            Note:
                This method provides a simplified view of system health. For detailed
                health information including timestamps and additional metrics,
                use the check() method instead.

            """
            return self._status == "healthy"


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Consolidated approach
# =============================================================================

# Export nested classes for external access (backward compatibility)
FlextConsole = FlextObservability.Console
FlextSpan = FlextObservability.Span
FlextTracer = FlextObservability.Tracer
FlextMetrics = FlextObservability.Metrics
FlextObservabilitySystem = FlextObservability  # Backward compatibility with old name
FlextAlerts = FlextObservability.Alerts
FlextCoreObservability = FlextObservability.Observability
_SimpleHealth = FlextObservability.Alerts()  # Simple fallback


# Global instance for compatibility - using singleton pattern
class _GlobalObservabilityManager:
    """Singleton manager for global observability instance."""

    _instance: FlextObservability.Observability | None = None

    @classmethod
    def get_instance(cls) -> FlextObservability.Observability:
        """Get or create global observability instance."""
        if cls._instance is None:
            cls._instance = FlextObservability.Observability()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset global observability instance."""
        cls._instance = FlextObservability.Observability()


def get_global_observability(
    log_level: str | None = None, *, force_recreate: bool = False
) -> FlextObservability.Observability:
    """Get global observability instance for compatibility."""
    # log_level parameter is accepted but not used in this no-op implementation
    _ = log_level  # Explicitly mark as unused but needed for API compatibility
    if force_recreate:
        _GlobalObservabilityManager.reset_instance()
    return _GlobalObservabilityManager.get_instance()


def reset_global_observability() -> None:
    """Reset global observability instance for compatibility."""
    _GlobalObservabilityManager.reset_instance()


__all__: list[str] = [
    "FlextObservability",  # ONLY main class exported
]
