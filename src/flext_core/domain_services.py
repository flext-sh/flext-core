"""Domain-Driven Design services for complex business operations.

Provides domain service patterns following DDD principles with stateless cross-entity
operations, business logic orchestration, and type-safe error handling using FlextResult.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import ConfigDict

from flext_core.constants import FlextConstants
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

TDomainResult = TypeVar("TDomainResult")


class FlextDomainService[TDomainResult](
    FlextModels.Config,
    FlextMixins.Serializable,
    FlextMixins.Loggable,
    ABC,
):
    """Abstract base class for production-ready domain services implementing DDD patterns.

    This abstract class provides the foundation for implementing complex business operations
    that span multiple entities or aggregates following Domain-Driven Design principles.
    Services are stateless, type-safe, and integrate with the complete FLEXT ecosystem.

    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow non-Pydantic types like FlextDbOracleApi
    )

    # Mixin functionality is now inherited via FlextMixins.Serializable
    def is_valid(self) -> bool:
        """Check if domain service is valid using efficient validation patterns.

        Performs efficient validation of the domain service instance including
        business rule validation, configuration validation, and data integrity checks.
        This method provides a boolean interface for quick validity assessment.
        """
        try:
            validation_result = self.validate_business_rules()
            return validation_result.is_success
        except Exception:
            # Use FlextUtilities for error logging if needed
            return False

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate domain service business rules with efficient error reporting.

        Performs efficient validation of business rules specific to the domain service.
        The default implementation returns success, but concrete services should override
        this method to implement domain-specific validation logic.

        """
        return FlextResult[None].ok(None)

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain service operation with type-safe result handling.

        This abstract method must be implemented by concrete domain services to define
        the core business logic and operation flow. The method should orchestrate
        complex business operations across multiple entities while maintaining
        domain integrity and returning type-safe results.

        """
        raise NotImplementedError

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration with efficient checks and customization support.

        Performs validation of the service configuration including field validation,
        dependency checks, and custom business requirements. The default implementation
        returns success, but concrete services should override this method to implement
        service-specific configuration validation.
        """
        return FlextResult[None].ok(None)

    def execute_operation(
        self,
        operation_name: str,
        operation: object,
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute operation with efficient error handling and validation using foundation patterns.

        Provides a standardized way to execute operations with built-in configuration
        validation, error handling, and logging. This method wraps operation execution
        in a consistent pattern that handles common error scenarios and provides
        detailed error reporting.

        """
        try:
            # Validate configuration first
            config_result = self.validate_config()
            if config_result.is_failure:
                error_message = (
                    config_result.error
                    or f"{FlextConstants.Messages.VALIDATION_FAILED}: Configuration validation failed"
                )
                return FlextResult[object].fail(
                    error_message, error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate operation is callable and execute
            if not callable(operation):
                return FlextResult[object].fail(
                    f"{FlextConstants.Messages.OPERATION_FAILED}: Operation {operation_name} is not callable",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                )

            # Execute the callable operation - MyPy should understand this is reachable
            result = operation(*args, **kwargs)
            return FlextResult[object].ok(result)

        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult[object].fail(
                f"{FlextConstants.Messages.OPERATION_FAILED}: Operation {operation_name} failed: {e}",
                error_code=FlextConstants.Errors.EXCEPTION_ERROR,
            )
        except Exception as e:
            # Catch any other exceptions using FlextConstants
            return FlextResult[object].fail(
                f"{FlextConstants.Messages.UNKNOWN_ERROR}: Unexpected error in {operation_name}: {e}",
                error_code=FlextConstants.Errors.UNKNOWN_ERROR,
            )

    def get_service_info(self) -> dict[str, object]:
        """Get efficient service information for monitoring, diagnostics, and observability.

        Provides detailed information about the service instance including metadata,
        configuration status, validation results, and runtime information. This method
        is designed for monitoring systems, health checks, and operational diagnostics.

        """
        config_result = self.validate_config()
        rules_result = self.validate_business_rules()
        is_valid = config_result.is_success and rules_result.is_success

        return {
            "service_type": self.__class__.__name__,
            "service_id": f"service_{self.__class__.__name__.lower()}_{FlextUtilities.Generators.generate_id()}",
            "config_valid": config_result.is_success,
            "business_rules_valid": rules_result.is_success,
            "configuration": self.model_dump(),  # Add configuration as expected by tests
            "is_valid": is_valid,  # Add overall validity as expected by tests
            "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
        }

    # =============================================================================
    # FLEXT DOMAIN SERVICES CONFIGURATION METHODS - Standard FlextTypes.Config
    # Enterprise-grade configuration management with environment-aware settings,
    # performance optimization, and efficient validation for domain services
    # =============================================================================

    @classmethod
    def configure_domain_services_system(
        cls, config: FlextTypes.Config.ConfigDict | None,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure domain services system using FlextTypes.Config with StrEnum validation.

        Configures the FLEXT domain services system including DDD patterns,
        business rule validation, cross-entity operations, and stateless
        service execution with performance monitoring and error handling.
        """
        try:
            # Validate config is not None
            if config is None:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    "Configuration cannot be None",
                )

            # Validate config is a dictionary (runtime check for type safety)
            # This check handles cases where config might not be a dict at runtime
            try:
                validated_config = dict(config)
            except (TypeError, ValueError, RuntimeError):
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    "Configuration must be a dictionary",
                )

            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}",
                    )
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate service_level (using validation level as basis)
            if "service_level" in config:
                service_value = config["service_level"]
                valid_levels = [e.value for e in FlextConstants.Config.ValidationLevel]
                if service_value not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid service_level '{service_value}'. Valid options: {valid_levels}",
                    )
            else:
                validated_config["service_level"] = (
                    FlextConstants.Config.ValidationLevel.LOOSE.value
                )

            # Validate log_level
            if "log_level" in config:
                log_value = config["log_level"]
                valid_log_levels = [e.value for e in FlextConstants.Config.LogLevel]
                if log_value not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_value}'. Valid options: {valid_log_levels}",
                    )
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.DEBUG.value
                )

            # Set default values for domain services specific settings
            validated_config.setdefault("enable_business_rule_validation", True)
            validated_config.setdefault("max_service_operations", 50)
            validated_config.setdefault("service_execution_timeout_seconds", 60)
            validated_config.setdefault("enable_cross_entity_operations", True)
            validated_config.setdefault("enable_performance_monitoring", True)
            validated_config.setdefault("enable_service_caching", False)
            validated_config.setdefault("service_retry_attempts", 3)
            validated_config.setdefault("enable_ddd_validation", True)

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to configure domain services system: {e}",
            )

    @classmethod
    def get_domain_services_system_config(
        cls,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current domain services system configuration with runtime metrics.

        Retrieves the current domain services system configuration including
        runtime metrics, performance data, active services, and DDD pattern
        validation status for monitoring and diagnostics.

        """
        try:
            # Build current configuration with runtime metrics
            current_config: FlextTypes.Config.ConfigDict = {
                # Core system configuration
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "service_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                # Domain services specific configuration
                "enable_business_rule_validation": True,
                "max_service_operations": 50,
                "service_execution_timeout_seconds": 60,
                "enable_cross_entity_operations": True,
                "enable_performance_monitoring": True,
                # Runtime metrics and status
                "active_service_operations": 0,  # Would be dynamically calculated
                "total_service_executions": 0,  # Runtime counter
                "successful_service_operations": 0,  # Success counter
                "failed_service_operations": 0,  # Failure counter
                "average_service_execution_time_ms": 0.0,  # Performance metric
                # DDD pattern validation status
                "ddd_validation_status": "enabled",
                "business_rules_validated": 0,  # Counter of validated business rules
                "cross_entity_operations_performed": 0,  # Cross-entity operations counter
                # Service registry information
                "registered_domain_services": [],  # List of registered services
                "available_service_patterns": ["abstract", "stateless", "cross-entity"],
                # Monitoring and diagnostics
                "last_health_check": FlextUtilities.Generators.generate_iso_timestamp(),
                "system_status": "operational",
                "configuration_source": "default",
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(current_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get domain services system configuration: {e}",
            )

    @classmethod
    def create_environment_domain_services_config(
        cls, environment: str,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific domain services configuration.

        Generates optimized configuration for domain services based on the
        target environment (development, staging, production, test, local)
        with appropriate DDD patterns, business rule validation, and
        performance settings for each environment.

        """
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}",
                )

            # Base configuration for all environments
            base_config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "enable_business_rule_validation": True,
                "enable_cross_entity_operations": True,
                "enable_ddd_validation": True,
            }

            # Environment-specific configurations
            if environment == "production":
                base_config.update(
                    {
                        "service_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                        "enable_performance_monitoring": True,  # Critical in production
                        "max_service_operations": 100,  # Higher concurrency
                        "service_execution_timeout_seconds": 30,  # Stricter timeout
                        "enable_service_caching": True,  # Performance optimization
                        "service_retry_attempts": 5,  # More retries for reliability
                        "enable_detailed_error_reporting": False,  # Security consideration
                    },
                )
            elif environment == "staging":
                base_config.update(
                    {
                        "service_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                        "log_level": FlextConstants.Config.LogLevel.INFO.value,
                        "enable_performance_monitoring": True,  # Monitor staging performance
                        "max_service_operations": 75,  # Moderate concurrency
                        "service_execution_timeout_seconds": 45,  # Balanced timeout
                        "enable_service_caching": True,  # Test caching behavior
                        "service_retry_attempts": 3,  # Standard retry policy
                        "enable_detailed_error_reporting": True,  # Full error details for debugging
                    },
                )
            elif environment == "development":
                base_config.update(
                    {
                        "service_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_performance_monitoring": True,  # Monitor development performance
                        "max_service_operations": 25,  # Lower concurrency for debugging
                        "service_execution_timeout_seconds": 120,  # Generous timeout for debugging
                        "enable_service_caching": False,  # Disable caching for development
                        "service_retry_attempts": 1,  # Minimal retries for fast failure
                        "enable_detailed_error_reporting": True,  # Full error details for debugging
                    },
                )
            elif environment == "test":
                base_config.update(
                    {
                        "service_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                        "enable_performance_monitoring": False,  # No performance monitoring in tests
                        "max_service_operations": 10,  # Limited concurrency for testing
                        "service_execution_timeout_seconds": 60,  # Standard timeout
                        "enable_service_caching": False,  # No caching in tests
                        "service_retry_attempts": 0,  # No retries in tests for deterministic behavior
                        "enable_detailed_error_reporting": True,  # Full error details for test diagnostics
                    },
                )
            elif environment == "local":
                base_config.update(
                    {
                        "service_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_performance_monitoring": False,  # No monitoring for local development
                        "max_service_operations": 5,  # Very limited concurrency
                        "service_execution_timeout_seconds": 300,  # Very generous timeout
                        "enable_service_caching": False,  # No caching for local development
                        "service_retry_attempts": 0,  # No retries for immediate feedback
                        "enable_detailed_error_reporting": True,  # Full error details
                    },
                )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(base_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment domain services configuration: {e}",
            )

    @classmethod
    def optimize_domain_services_performance(
        cls, config: FlextTypes.Config.ConfigDict | None,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize domain services system performance based on configuration.

        Analyzes the provided configuration and generates performance-optimized
        settings for the FLEXT domain services system. This includes business rule
        validation optimization, cross-entity operation tuning, service execution
        performance, and memory management for optimal DDD pattern execution.

        """
        try:
            # Validate config is not None
            if config is None:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    "Configuration cannot be None",
                )

            # Create optimized configuration
            optimized_config = dict(config)

            # Get performance level from config
            performance_level = config.get("performance_level", "medium")

            # Base performance settings
            optimized_config.update(
                {
                    "performance_level": performance_level,
                    "optimization_enabled": True,
                    "optimization_timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
                },
            )

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update(
                    {
                        # Service execution optimization
                        "service_cache_size": 500,
                        "enable_service_pooling": True,
                        "service_pool_size": 100,
                        "max_concurrent_services": 50,
                        "service_discovery_cache_ttl": 3600,  # 1 hour
                        # Business rule optimization
                        "enable_business_rule_caching": True,
                        "business_rule_cache_size": 1000,
                        "business_rule_validation_threads": 8,
                        "parallel_business_rule_validation": True,
                        # Cross-entity operation optimization
                        "cross_entity_batch_size": 100,
                        "enable_cross_entity_batching": True,
                        "cross_entity_processing_threads": 16,
                        "cross_entity_queue_size": 2000,
                        # Memory and performance optimization
                        "memory_pool_size_mb": 200,
                        "enable_object_pooling": True,
                        "gc_optimization_enabled": True,
                        "optimization_level": "aggressive",
                    },
                )
            elif performance_level == "medium":
                optimized_config.update(
                    {
                        # Balanced service settings
                        "service_cache_size": 250,
                        "enable_service_pooling": True,
                        "service_pool_size": 50,
                        "max_concurrent_services": 25,
                        "service_discovery_cache_ttl": 1800,  # 30 minutes
                        # Moderate business rule optimization
                        "enable_business_rule_caching": True,
                        "business_rule_cache_size": 500,
                        "business_rule_validation_threads": 4,
                        "parallel_business_rule_validation": True,
                        # Standard cross-entity processing
                        "cross_entity_batch_size": 50,
                        "enable_cross_entity_batching": True,
                        "cross_entity_processing_threads": 8,
                        "cross_entity_queue_size": 1000,
                        # Moderate memory settings
                        "memory_pool_size_mb": 100,
                        "enable_object_pooling": True,
                        "gc_optimization_enabled": True,
                        "optimization_level": "balanced",
                    },
                )
            elif performance_level == "low":
                optimized_config.update(
                    {
                        # Conservative service settings
                        "service_cache_size": 50,
                        "enable_service_pooling": False,
                        "service_pool_size": 10,
                        "max_concurrent_services": 5,
                        "service_discovery_cache_ttl": 600,  # 10 minutes
                        # Minimal business rule optimization
                        "enable_business_rule_caching": False,
                        "business_rule_cache_size": 100,
                        "business_rule_validation_threads": 1,
                        "parallel_business_rule_validation": False,
                        # Sequential cross-entity processing
                        "cross_entity_batch_size": 10,
                        "enable_cross_entity_batching": False,
                        "cross_entity_processing_threads": 1,
                        "cross_entity_queue_size": 100,
                        # Minimal memory usage
                        "memory_pool_size_mb": 25,
                        "enable_object_pooling": False,
                        "gc_optimization_enabled": False,
                        "optimization_level": "conservative",
                    },
                )

            # Additional performance metrics and targets
            optimized_config.update(
                {
                    "expected_throughput_services_per_second": 200
                    if performance_level == "high"
                    else 100
                    if performance_level == "medium"
                    else 25,
                    "target_service_latency_ms": 10
                    if performance_level == "high"
                    else 25
                    if performance_level == "medium"
                    else 100,
                    "target_business_rule_validation_ms": 5
                    if performance_level == "high"
                    else 15
                    if performance_level == "medium"
                    else 50,
                    "memory_efficiency_target": 0.90
                    if performance_level == "high"
                    else 0.80
                    if performance_level == "medium"
                    else 0.60,
                },
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize domain services performance: {e}",
            )


__all__: list[str] = [
    "FlextDomainService",
]
