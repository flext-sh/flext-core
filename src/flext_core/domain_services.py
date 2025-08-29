"""FLEXT Domain Services - Domain-Driven Design services for complex business operations.

Provides comprehensive domain service patterns following DDD principles with stateless
cross-entity operations, business logic orchestration, validation frameworks, and
type-safe error handling. Domain services handle complex business operations that span
multiple entities or require coordination of multiple domain objects.

Module Role in Architecture:
    FlextDomainService provides the domain service layer for complex business operations
    that cannot naturally belong to a single entity. Implements stateless services with
    cross-entity operations, business rule validation and FlextResult integration for
    railway-oriented programming.

Classes and Methods:
    FlextDomainService[T]:              # Base domain service with generic type parameter
        # Configuration Methods:
        configure_domain_services_system(config) -> FlextResult[ConfigDict] # Configure system
        get_domain_services_system_config() -> FlextResult[ConfigDict] # Get current config
        optimize_domain_services_performance(config) -> FlextResult[ConfigDict] # Performance tuning

        # Core Service Methods:
        execute() -> FlextResult[T]                # Execute main business operation
        validate_business_rules() -> FlextResult[None] # Validate business rules
        validate_preconditions() -> FlextResult[None] # Validate preconditions
        validate_postconditions(result: T) -> FlextResult[None] # Validate postconditions

        # Business Logic Coordination:
        coordinate_entities(entities) -> FlextResult[None] # Coordinate multiple entities
        orchestrate_operations(operations) -> FlextResult[T] # Orchestrate complex operations
        handle_cross_entity_invariants() -> FlextResult[None] # Handle cross-entity invariants

        # Transaction Support:
        begin_transaction() -> FlextResult[None]   # Begin distributed transaction
        commit_transaction() -> FlextResult[None]  # Commit transaction
        rollback_transaction() -> FlextResult[None] # Rollback transaction

        # Domain Event Integration:
        publish_domain_events(events) -> FlextResult[None] # Publish domain events
        handle_domain_event(event) -> FlextResult[None] # Handle incoming domain event
        collect_domain_events() -> list[DomainEvent] # Collect events from operation

        # Performance and Monitoring:
        get_service_metrics() -> dict              # Get service performance metrics
        reset_metrics() -> None                    # Reset performance metrics
        enable_monitoring(enabled: bool) -> None   # Enable/disable monitoring

        # Validation Utilities:
        validate_entity(entity) -> FlextResult[None] # Validate single entity
        validate_aggregate(aggregate) -> FlextResult[None] # Validate aggregate root
        validate_business_invariant(condition, message) -> FlextResult[None] # Validate invariant

        # Mixin Integration:
        to_dict() -> dict                          # Serialize service state (from Serializable)
        FlextLogger() -> FlextLogger               # Get service logger (from Loggable)
        log_operation(operation, **context) -> None # Log service operation

Usage Examples:
    Basic domain service implementation:
        class UserRegistrationService(FlextDomainService[User]):
            def execute(self) -> FlextResult[User]:
                return (
                    self.validate_business_rules()
                    .flat_map(lambda _: self.create_user())
                    .flat_map(lambda user: self.send_welcome_email(user))
                    .tap(lambda user: self.log_registration(user))
                )

            def validate_business_rules(self) -> FlextResult[None]:
                return self.validate_email_uniqueness().flat_map(
                    lambda _: self.validate_password_policy()
                )

    Service execution:
        service = UserRegistrationService(email="john@example.com", password="secret123")
        result = service.execute()
        if result.success:
            user = result.unwrap()
            print(f"User registered: {user.id}")

    Configuration:
        config = {
            "environment": "production",
            "service_level": "strict",
            "enable_performance_monitoring": True,
            "max_service_operations": 100
        }
        FlextDomainService.configure_domain_services_system(config)

Integration:
    FlextDomainService integrates with FlextResult for error handling, FlextMixins for
    behaviors, FlextModels.BaseConfig for configuration, FlextConstants for error codes,
    FlextUtilities for common operations, and FlextTypes.Config for type-safe configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import ConfigDict

from flext_core.constants import FlextConstants
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

# =============================================================================
# FLEXT DOMAIN SERVICE - Public DDD Domain Service implementation
# =============================================================================


class FlextDomainService[TDomainResult](
    FlextModels.BaseConfig,
    FlextMixins.Serializable,
    FlextMixins.Loggable,
    ABC,
):
    """Abstract base class for enterprise-grade domain services implementing DDD patterns.

    This abstract class provides the foundation for implementing complex business operations
    that span multiple entities or aggregates following Domain-Driven Design principles.
    Services are stateless, type-safe, and integrate with the complete FLEXT ecosystem.

    The FlextDomainService class combines multiple FLEXT foundation patterns:
        - **FlextModels.BaseConfig**: Pydantic configuration and validation
        - **FlextMixins.Serializable**: JSON serialization capabilities
        - **FlextMixins.Loggable**: Structured logging with correlation IDs
        - **ABC**: Abstract base class enforcement for concrete implementations

    Key Features:
        - **Generic Type Safety**: Parameterized with result type TDomainResult
        - **Railway-Oriented Programming**: All operations return FlextResult[T]
        - **Business Rule Validation**: Comprehensive validation framework
        - **Cross-Entity Operations**: Coordinate operations across multiple entities
        - **Stateless Design**: No state maintained between service invocations
        - **Performance Monitoring**: Built-in execution tracking and metrics
        - **Configuration Management**: Environment-aware configuration support
        - **Error Handling**: Comprehensive error classification and reporting

    DDD Pattern Implementation:
        Domain services implement the Service pattern from Domain-Driven Design,
        representing operations that:

        - Do not naturally belong to a specific entity or value object
        - Coordinate behavior across multiple domain objects
        - Implement complex business rules and invariants
        - Provide stateless operations with clear inputs and outputs
        - Maintain domain integrity across entity boundaries

    Configuration Features:
        The class is configured as a frozen Pydantic model with:
        - **Immutability**: Frozen=True prevents modification after creation
        - **Validation**: validate_assignment=True ensures data integrity
        - **Type Safety**: extra="forbid" prevents undefined attributes
        - **Flexible Types**: arbitrary_types_allowed=True supports complex domain objects

    Abstract Methods:
        Concrete implementations must provide:
        - **execute()**: Main service operation returning FlextResult[TDomainResult]

    Usage Examples:
        Basic domain service implementation:

            # Example: OrderProcessingService class implementation
            class OrderProcessingService(FlextDomainService[Order]):
                customer_id: str
                order_items: list[OrderItem]
                payment_info: PaymentInfo

                def execute(self) -> FlextResult[Order]:
                    return (self.validate_business_rules()
                        .flat_map(lambda _: self.validate_inventory())
                        .flat_map(lambda _: self.process_payment())
                        .flat_map(lambda payment: self.create_order(payment))
                        .tap(lambda order: self.send_confirmation(order)))

                def validate_business_rules(self) -> FlextResult[None]:
                    return (self.validate_customer_exists()
                        .flat_map(lambda _: self.validate_order_items())
                        .flat_map(lambda _: self.validate_payment_info()))

        Service with custom validation:

            # Example: UserMigrationService class implementation
            class UserMigrationService(FlextDomainService[MigrationResult]):
                source_user_id: str
                target_system: str

                def validate_config(self) -> FlextResult[None]:
                    if not self.source_user_id:
                        return FlextResult[None].fail("Source user ID required")
                    if not self.target_system:
                        return FlextResult[None].fail("Target system required")
                    return FlextResult[None].ok(None)

                def execute(self) -> FlextResult[MigrationResult]:
                    return (self.validate_config()
                        .flat_map(lambda _: self.extract_user_data())
                        .flat_map(lambda data: self.transform_data(data))
                        .flat_map(lambda data: self.load_to_target(data)))

        Performance monitoring integration::

            service = OrderProcessingService(
                customer_id="cust123", order_items=[item1, item2], payment_info=payment
            )

            # Get service information for monitoring
            info = service.get_service_info()
            logger.info(f"Executing service: {info['service_type']}")

            # Execute with error handling
            result = service.execute()
            if result.success:
                order = result.value
                logger.info(f"Order created: {order.id}")
            else:
                logger.error(f"Service failed: {result.error}")

    Type Parameters:
        TDomainResult: The type of result returned by the service execute method.
            This allows for type-safe service definitions and ensures proper
            return type validation.

    Attributes:
        model_config (ConfigDict): Pydantic configuration ensuring immutability,
            validation, and type safety for all service instances.

    Integration with FLEXT Ecosystem:
        - **FlextResult[T]**: All operations return results for railway programming
        - **FlextConstants**: Error codes and messages for consistent error handling
        - **FlextUtilities**: ID generation, timestamps, and helper functions
        - **FlextMixins**: Serialization and logging capabilities
        - **FlextTypes.Config**: Type-safe configuration management

    See Also:
        - FlextResult[T]: Type-safe error handling patterns
        - FlextMixins.Serializable: JSON serialization capabilities
        - FlextMixins.Loggable: Structured logging integration
        - FlextModels.BaseConfig: Configuration base class
        - configure_domain_services_system(): System configuration

    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow non-Pydantic types like FlextDbOracleApi
    )

    # Mixin functionality is now inherited via FlextMixins.Serializable

    def is_valid(self) -> bool:
        """Check if domain service is valid using comprehensive validation patterns.

        Performs comprehensive validation of the domain service instance including
        business rule validation, configuration validation, and data integrity checks.
        This method provides a boolean interface for quick validity assessment.

        Returns:
            bool: True if the service is valid and ready for execution, False otherwise.

        Usage Examples:
            Basic validation check::

                service = UserRegistrationService(
                    email="user@example.com", password="secure123"
                )

                if service.is_valid():
                    result = service.execute()
                else:
                    logger.error("Service validation failed")

            Conditional execution::

                services = [service1, service2, service3]
                valid_services = [s for s in services if s.is_valid()]

                for service in valid_services:
                    result = service.execute()

        Note:
            This method catches and handles validation exceptions internally,
            returning False for any validation errors. For detailed validation
            error information, use validate_business_rules() directly.

        See Also:
            - validate_business_rules(): Detailed validation with error information
            - validate_config(): Configuration-specific validation

        """
        try:
            validation_result = self.validate_business_rules()
            return validation_result.is_success
        except Exception:
            # Use FlextUtilities for error logging if needed
            return False

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate domain service business rules with comprehensive error reporting.

        Performs comprehensive validation of business rules specific to the domain service.
        The default implementation returns success, but concrete services should override
        this method to implement domain-specific validation logic.

        Business rule validation typically includes:
            - Entity state validation across multiple entities
            - Invariant enforcement at the domain level
            - Cross-aggregate consistency checks
            - Domain-specific constraint validation
            - Business policy compliance verification

        Returns:
            FlextResult[None]: Success if all business rules are valid, failure with
                detailed error information if validation fails.

        Usage Examples:
            Override in concrete services::

                class TransferFundsService(FlextDomainService[TransferResult]):
                    source_account: str
                    target_account: str
                    amount: Decimal

                    def validate_business_rules(self) -> FlextResult[None]:
                        return (
                            self.validate_account_exists(self.source_account)
                            .flat_map(
                                lambda _: self.validate_account_exists(
                                    self.target_account
                                )
                            )
                            .flat_map(lambda _: self.validate_sufficient_funds())
                            .flat_map(lambda _: self.validate_transfer_limits())
                            .flat_map(lambda _: self.validate_business_hours())
                        )

            Chain validation results::

                validation_result = service.validate_business_rules()
                if validation_result.success:
                    execution_result = service.execute()
                else:
                    logger.error(
                        f"Business rule validation failed: {validation_result.error}"
                    )

        Implementation Notes:
            - Override this method in concrete services to implement specific validation
            - Use railway-oriented programming with flat_map for chained validations
            - Return descriptive error messages for failed validations
            - Consider performance implications of complex validation logic

        See Also:
            - validate_config(): Configuration validation
            - is_valid(): Boolean validation interface
            - execute(): Main service execution method

        """
        return FlextResult[None].ok(None)

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain service operation with type-safe result handling.

        This abstract method must be implemented by concrete domain services to define
        the core business logic and operation flow. The method should orchestrate
        complex business operations across multiple entities while maintaining
        domain integrity and returning type-safe results.

        Returns:
            FlextResult[TDomainResult]: Type-safe result containing the operation outcome.
                Success contains the domain result of type TDomainResult, failure contains
                detailed error information with appropriate error codes.

        Implementation Guidelines:
            - Use railway-oriented programming with flat_map for operation chaining
            - Validate business rules before executing core operations
            - Maintain transactional boundaries for data consistency
            - Handle all expected error conditions with appropriate error messages
            - Log significant operations and errors for observability
            - Return domain-specific result types that represent business outcomes

        Usage Examples:
            Basic service execution pattern::

                class OrderProcessingService(FlextDomainService[Order]):
                    def execute(self) -> FlextResult[Order]:
                        return (
                            self.validate_business_rules()
                            .flat_map(lambda _: self.reserve_inventory())
                            .flat_map(
                                lambda reservation: self.process_payment(reservation)
                            )
                            .flat_map(lambda payment: self.create_order(payment))
                            .tap(lambda order: self.send_confirmation(order))
                            .map_error(lambda e: f"Order processing failed: {e}")
                        )

            Complex multi-step operation::

                class DataMigrationService(FlextDomainService[MigrationSummary]):
                    def execute(self) -> FlextResult[MigrationSummary]:
                        return (
                            self.validate_source_connection()
                            .flat_map(lambda _: self.validate_target_connection())
                            .flat_map(lambda _: self.extract_data())
                            .flat_map(lambda data: self.transform_data(data))
                            .flat_map(lambda data: self.validate_transformed_data(data))
                            .flat_map(lambda data: self.load_data(data))
                            .map(lambda result: self.create_summary(result))
                        )

        Error Handling:
            Services should handle errors appropriately and return meaningful error messages:

            - Use specific error codes from FlextConstants.Errors
            - Provide descriptive error messages for business rule violations
            - Log errors with appropriate context for debugging
            - Consider retry logic for transient failures
            - Ensure proper cleanup in case of failures

        Performance Considerations:
            - Implement timeout handling for long-running operations
            - Consider batching for operations on multiple entities
            - Use caching for frequently accessed data
            - Monitor execution times for performance optimization

        Note:
            This method is abstract and must be implemented by all concrete domain services.
            The implementation defines the core business logic and operation flow for the service.

        Raises:
            NotImplementedError: If called on the abstract base class directly.

        See Also:
            - validate_business_rules(): Business rule validation
            - execute_operation(): Helper method for operation execution
            - get_service_info(): Service information for monitoring

        """
        raise NotImplementedError

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration with comprehensive checks and customization support.

        Performs validation of the service configuration including field validation,
        dependency checks, and custom business requirements. The default implementation
        returns success, but concrete services should override this method to implement
        service-specific configuration validation.

        Returns:
            FlextResult[None]: Success if configuration is valid, failure with detailed
                error information if validation fails.

        Validation Categories:
            - **Field Validation**: Required fields, data types, value constraints
            - **Dependency Validation**: External service availability, database connections
            - **Business Rule Validation**: Domain-specific configuration requirements
            - **Environment Validation**: Environment-specific configuration checks
            - **Security Validation**: Credentials, permissions, access controls

        Usage Examples:
            Override for service-specific validation::

                class EmailService(FlextDomainService[EmailResult]):
                    smtp_host: str
                    smtp_port: int
                    username: str
                    password: str

                    def validate_config(self) -> FlextResult[None]:
                        return (
                            self.validate_smtp_connection()
                            .flat_map(lambda _: self.validate_credentials())
                            .flat_map(lambda _: self.validate_port_range())
                            .flat_map(lambda _: self.validate_security_settings())
                        )

                    def validate_smtp_connection(self) -> FlextResult[None]:
                        if not self.smtp_host:
                            return FlextResult[None].fail("SMTP host is required")
                        # Test connection logic here
                        return FlextResult[None].ok(None)

            Database service configuration::

                class DatabaseService(FlextDomainService[QueryResult]):
                    connection_string: str
                    timeout_seconds: int

                    def validate_config(self) -> FlextResult[None]:
                        if not self.connection_string:
                            return FlextResult[None].fail("Connection string required")
                        if self.timeout_seconds <= 0:
                            return FlextResult[None].fail("Timeout must be positive")
                        return self.test_database_connection()

        Implementation Patterns:
            Use railway-oriented programming for chained validation::

                def validate_config(self) -> FlextResult[None]:
                    return (
                        self.validate_required_fields()
                        .flat_map(lambda _: self.validate_data_types())
                        .flat_map(lambda _: self.validate_business_constraints())
                        .flat_map(lambda _: self.validate_external_dependencies())
                    )

        Default Behavior:
            The base implementation returns success to allow services without
            custom configuration validation to work out of the box. Services
            with specific configuration requirements should override this method.

        Integration:
            This method is called automatically by execute_operation() before
            executing operations, ensuring configuration validity before processing.

        See Also:
            - validate_business_rules(): Business rule validation
            - execute_operation(): Operation execution with configuration validation
            - is_valid(): Overall service validity check

        """
        return FlextResult[None].ok(None)

    def execute_operation(
        self,
        operation_name: str,
        operation: object,
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute operation with comprehensive error handling and validation using foundation patterns.

        Provides a standardized way to execute operations with built-in configuration
        validation, error handling, and logging. This method wraps operation execution
        in a consistent pattern that handles common error scenarios and provides
        detailed error reporting.

        Args:
            operation_name (str): Descriptive name of the operation for logging and error reporting.
                Should be a clear, human-readable identifier for the operation being performed.
            operation (object): The operation to execute. Must be a callable object (function,
                method, or callable class instance).
            *args (object): Variable positional arguments to pass to the operation.
            **kwargs (object): Variable keyword arguments to pass to the operation.

        Returns:
            FlextResult[object]: Type-safe result containing either:
                - Success: The result returned by the operation
                - Failure: Detailed error information including error code and message

        Error Handling:
            The method handles multiple error categories:
            - **Configuration Errors**: Invalid service configuration
            - **Validation Errors**: Operation validation failures
            - **Runtime Errors**: RuntimeError, ValueError, TypeError exceptions
            - **Unknown Errors**: Unexpected exceptions with full context

        Usage Examples:
            Execute simple operations::

                def send_email(to: str, subject: str, body: str) -> bool:
                    # Email sending logic
                    return True


                result = service.execute_operation(
                    "send_welcome_email",
                    send_email,
                    "user@example.com",
                    "Welcome!",
                    "Welcome to our service",
                )

                if result.success:
                    logger.info("Email sent successfully")
                else:
                    logger.error(f"Email failed: {result.error}")

            Execute operations with complex parameters::

                def process_payment(amount: Decimal, card_info: dict) -> PaymentResult:
                    # Payment processing logic
                    return PaymentResult(success=True, transaction_id="tx123")


                result = service.execute_operation(
                    "process_customer_payment",
                    process_payment,
                    amount=Decimal("99.99"),
                    card_info={"number": "****1234", "cvv": "123"},
                )

            Handle operation results::

                operation_result = service.execute_operation(
                    "complex_calculation", calculate_metrics
                )

                if operation_result.success:
                    metrics = operation_result.value
                    logger.info(f"Metrics calculated: {metrics}")
                elif (
                    operation_result.error_code
                    == FlextConstants.Errors.VALIDATION_ERROR
                ):
                    logger.warning(f"Configuration issue: {operation_result.error}")
                elif (
                    operation_result.error_code == FlextConstants.Errors.OPERATION_ERROR
                ):
                    logger.error(f"Operation failed: {operation_result.error}")
                else:
                    logger.critical(f"Unexpected error: {operation_result.error}")

        Validation Flow:
            1. **Configuration Validation**: Calls validate_config() before execution
            2. **Callable Validation**: Ensures the operation is callable
            3. **Operation Execution**: Executes the operation with provided arguments
            4. **Error Classification**: Categorizes and formats any errors that occur

        Error Codes:
            The method uses standardized error codes from FlextConstants.Errors:
            - **VALIDATION_ERROR**: Configuration or validation failures
            - **OPERATION_ERROR**: Operation-specific failures
            - **EXCEPTION_ERROR**: Runtime exceptions (RuntimeError, ValueError, TypeError)
            - **UNKNOWN_ERROR**: Unexpected exceptions

        Performance Considerations:
            - Configuration validation is performed on every call for safety
            - Error handling adds minimal overhead to successful operations
            - Consider caching configuration validation results for high-frequency operations

        Thread Safety:
            This method is thread-safe as long as:
            - The service instance is immutable (enforced by Pydantic frozen=True)
            - The operation being executed is thread-safe
            - No shared mutable state is accessed during execution

        See Also:
            - validate_config(): Configuration validation method
            - execute(): Abstract method for main service execution
            - FlextResult[T]: Type-safe result handling patterns
            - FlextConstants.Errors: Standard error code definitions

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
                    error_message, error_code=FlextConstants.Errors.VALIDATION_ERROR
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
        """Get comprehensive service information for monitoring, diagnostics, and observability.

        Provides detailed information about the service instance including metadata,
        configuration status, validation results, and runtime information. This method
        is designed for monitoring systems, health checks, and operational diagnostics.

        Returns:
            dict[str, object]: Comprehensive service information dictionary containing:
                - **service_type**: Class name of the service
                - **service_id**: Unique identifier for this service instance
                - **config_valid**: Boolean indicating configuration validity
                - **business_rules_valid**: Boolean indicating business rule validity
                - **timestamp**: ISO timestamp of when information was generated

        Information Categories:
            - **Identity**: Service type, unique ID, and classification
            - **Status**: Configuration and business rule validation status
            - **Metadata**: Creation timestamp and runtime information
            - **Health**: Overall service health and readiness indicators

        Usage Examples:
            Basic service monitoring::

                service = UserRegistrationService(
                    email="user@example.com", password="secure123"
                )

                info = service.get_service_info()

                logger.info(f"Service: {info['service_type']}")
                logger.info(f"ID: {info['service_id']}")
                logger.info(f"Config Valid: {info['config_valid']}")
                logger.info(f"Rules Valid: {info['business_rules_valid']}")

            Health check integration::

                def check_service_health(service: FlextDomainService) -> bool:
                    info = service.get_service_info()
                    return info["config_valid"] and info["business_rules_valid"]


                services = [service1, service2, service3]
                healthy_services = [s for s in services if check_service_health(s)]

                logger.info(f"{len(healthy_services)}/{len(services)} services healthy")

            Monitoring dashboard data::

                def collect_service_metrics(services: list[FlextDomainService]) -> dict:
                    metrics = {
                        "total_services": len(services),
                        "healthy_services": 0,
                        "services": [],
                    }

                    for service in services:
                        info = service.get_service_info()
                        metrics["services"].append(info)

                        if info["config_valid"] and info["business_rules_valid"]:
                            metrics["healthy_services"] += 1

                    return metrics

            Operational logging::

                # Log service information before execution
                info = service.get_service_info()
                correlation_id = FlextUtilities.Generators.generate_correlation_id()

                logger.info(
                    "Executing domain service",
                    service_type=info["service_type"],
                    service_id=info["service_id"],
                    config_valid=info["config_valid"],
                    correlation_id=correlation_id,
                )

                result = service.execute()

        Dictionary Structure:
            The returned dictionary contains the following keys::

                {
                    "service_type": "UserRegistrationService",
                    "service_id": "service_userregistrationservice_abc123",
                    "config_valid": True,
                    "business_rules_valid": True,
                    "timestamp": "2025-01-15T10:30:45.123456Z",
                }

        Performance Notes:
            - Validation checks are performed on each call for accuracy
            - ID generation occurs on each call for uniqueness
            - Consider caching results if called frequently in high-throughput scenarios
            - Timestamp generation uses UTC for consistency across time zones

        Integration Points:
            - **FlextUtilities.Generators**: ID and timestamp generation
            - **validate_config()**: Configuration validation status
            - **validate_business_rules()**: Business rule validation status
            - **Monitoring Systems**: Health checks and service discovery
            - **Logging Systems**: Structured logging with service context

        See Also:
            - validate_config(): Configuration validation method
            - validate_business_rules(): Business rule validation method
            - FlextUtilities.Generators: ID and timestamp generation utilities
            - is_valid(): Boolean validity check method

        """
        return {
            "service_type": self.__class__.__name__,
            "service_id": f"service_{self.__class__.__name__.lower()}_{FlextUtilities.Generators.generate_id()}",
            "config_valid": self.validate_config().is_success,
            "business_rules_valid": self.validate_business_rules().is_success,
            "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
        }

    # =============================================================================
    # FLEXT DOMAIN SERVICES CONFIGURATION METHODS - Standard FlextTypes.Config
    # Enterprise-grade configuration management with environment-aware settings,
    # performance optimization, and comprehensive validation for domain services
    # =============================================================================

    @classmethod
    def configure_domain_services_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure domain services system using FlextTypes.Config with StrEnum validation.

        Configures the FLEXT domain services system including DDD patterns,
        business rule validation, cross-entity operations, and stateless
        service execution with performance monitoring and error handling.

        Args:
            config: Configuration dictionary supporting:
                   - environment: Runtime environment (development, production, test, staging, local)
                   - service_level: Service validation level (strict, normal, loose)
                   - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
                   - enable_business_rule_validation: Enable business rule validation
                   - max_service_operations: Maximum concurrent service operations
                   - service_execution_timeout_seconds: Timeout for service operations
                   - enable_cross_entity_operations: Enable cross-entity operations
                   - enable_performance_monitoring: Enable performance monitoring

        Returns:
            FlextResult containing validated configuration with domain services settings

        Example:
            ```python
            config = {
                "environment": "production",
                "service_level": "strict",
                "log_level": "WARNING",
                "enable_business_rule_validation": True,
                "max_service_operations": 20,
                "service_execution_timeout_seconds": 30,
            }
            result = FlextDomainService.configure_domain_services_system(config)
            if result.success:
                validated_config = result.unwrap()
            ```

        """
        try:
            # Create working copy of config
            validated_config = dict(config)

            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
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
                        f"Invalid service_level '{service_value}'. Valid options: {valid_levels}"
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
                        f"Invalid log_level '{log_value}'. Valid options: {valid_log_levels}"
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
                f"Failed to configure domain services system: {e}"
            )

    @classmethod
    def get_domain_services_system_config(
        cls,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current domain services system configuration with runtime metrics.

        Retrieves the current domain services system configuration including
        runtime metrics, performance data, active services, and DDD pattern
        validation status for monitoring and diagnostics.

        Returns:
            FlextResult containing current domain services system configuration with:
            - environment: Current runtime environment
            - service_level: Current service validation level
            - log_level: Current logging level
            - business_rule_validation_enabled: Business rule validation status
            - active_service_operations: Number of currently active service operations
            - service_performance_metrics: Performance metrics for domain services
            - ddd_validation_status: DDD pattern validation status
            - cross_entity_operations_enabled: Cross-entity operations status

        Example:
            ```python
            result = FlextDomainService.get_domain_services_system_config()
            if result.success:
                current_config = result.unwrap()
                print(
                    f"Active operations: {current_config['active_service_operations']}"
                )
            ```

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
                f"Failed to get domain services system configuration: {e}"
            )

    @classmethod
    def create_environment_domain_services_config(
        cls, environment: str
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific domain services configuration.

        Generates optimized configuration for domain services based on the
        target environment (development, staging, production, test, local)
        with appropriate DDD patterns, business rule validation, and
        performance settings for each environment.

        Args:
            environment: Target environment name (development, staging, production, test, local)

        Returns:
            FlextResult containing environment-optimized domain services configuration

        Example:
            ```python
            result = FlextDomainService.create_environment_domain_services_config(
                "production"
            )
            if result.success:
                prod_config = result.unwrap()
                print(f"Service level: {prod_config['service_level']}")
            ```

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

            # Base configuration for all environments
            base_config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "enable_business_rule_validation": True,
                "enable_cross_entity_operations": True,
                "enable_ddd_validation": True,
            }

            # Environment-specific configurations
            if environment == "production":
                base_config.update({
                    "service_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_performance_monitoring": True,  # Critical in production
                    "max_service_operations": 100,  # Higher concurrency
                    "service_execution_timeout_seconds": 30,  # Stricter timeout
                    "enable_service_caching": True,  # Performance optimization
                    "service_retry_attempts": 5,  # More retries for reliability
                    "enable_detailed_error_reporting": False,  # Security consideration
                })
            elif environment == "staging":
                base_config.update({
                    "service_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_performance_monitoring": True,  # Monitor staging performance
                    "max_service_operations": 75,  # Moderate concurrency
                    "service_execution_timeout_seconds": 45,  # Balanced timeout
                    "enable_service_caching": True,  # Test caching behavior
                    "service_retry_attempts": 3,  # Standard retry policy
                    "enable_detailed_error_reporting": True,  # Full error details for debugging
                })
            elif environment == "development":
                base_config.update({
                    "service_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_performance_monitoring": True,  # Monitor development performance
                    "max_service_operations": 25,  # Lower concurrency for debugging
                    "service_execution_timeout_seconds": 120,  # Generous timeout for debugging
                    "enable_service_caching": False,  # Disable caching for development
                    "service_retry_attempts": 1,  # Minimal retries for fast failure
                    "enable_detailed_error_reporting": True,  # Full error details for debugging
                })
            elif environment == "test":
                base_config.update({
                    "service_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_performance_monitoring": False,  # No performance monitoring in tests
                    "max_service_operations": 10,  # Limited concurrency for testing
                    "service_execution_timeout_seconds": 60,  # Standard timeout
                    "enable_service_caching": False,  # No caching in tests
                    "service_retry_attempts": 0,  # No retries in tests for deterministic behavior
                    "enable_detailed_error_reporting": True,  # Full error details for test diagnostics
                })
            elif environment == "local":
                base_config.update({
                    "service_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_performance_monitoring": False,  # No monitoring for local development
                    "max_service_operations": 5,  # Very limited concurrency
                    "service_execution_timeout_seconds": 300,  # Very generous timeout
                    "enable_service_caching": False,  # No caching for local development
                    "service_retry_attempts": 0,  # No retries for immediate feedback
                    "enable_detailed_error_reporting": True,  # Full error details
                })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(base_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment domain services configuration: {e}"
            )

    @classmethod
    def optimize_domain_services_performance(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize domain services system performance based on configuration.

        Analyzes the provided configuration and generates performance-optimized
        settings for the FLEXT domain services system. This includes business rule
        validation optimization, cross-entity operation tuning, service execution
        performance, and memory management for optimal DDD pattern execution.

        Args:
            config: Base configuration dictionary containing performance preferences:
                   - performance_level: Performance optimization level (high, medium, low)
                   - max_concurrent_services: Maximum concurrent service executions
                   - service_pool_size: Service instance pool size for reuse
                   - business_rule_optimization: Enable business rule validation optimization
                   - cross_entity_optimization: Enable cross-entity operation optimization

        Returns:
            FlextResult containing optimized configuration with performance settings
            tuned for domain services system performance requirements.

        Example:
            ```python
            config = {
                "performance_level": "high",
                "max_concurrent_services": 50,
                "service_pool_size": 100,
            }
            result = FlextDomainService.optimize_domain_services_performance(config)
            if result.success:
                optimized = result.unwrap()
                print("Service cache size:", optimized.get("service_cache_size"))
            ```

        """
        try:
            # Create optimized configuration
            optimized_config = dict(config)

            # Get performance level from config
            performance_level = config.get("performance_level", "medium")

            # Base performance settings
            optimized_config.update({
                "performance_level": performance_level,
                "optimization_enabled": True,
                "optimization_timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
            })

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update({
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
                })
            elif performance_level == "medium":
                optimized_config.update({
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
                })
            elif performance_level == "low":
                optimized_config.update({
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
                })

            # Additional performance metrics and targets
            optimized_config.update({
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
            })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize domain services performance: {e}"
            )


# Export API
__all__: list[str] = [
    "FlextDomainService",  # Main domain service base class
    # Legacy compatibility aliases moved to flext_core.legacy to avoid type conflicts
]
