"""FLEXT Core Domain Services - Domain Layer Service Implementation.

Domain-Driven Design (DDD) domain service implementation for stateless business
operations that don't naturally belong to entities or value objects across the
32-project FLEXT ecosystem. Foundation for cross-entity business logic, complex
validation, and integration services in data integration domains.

Module Role in Architecture:
    Domain Layer → Domain Services → Cross-Entity Business Logic

    This module provides DDD domain service patterns used throughout FLEXT projects:
    - Stateless business operations spanning multiple entities or aggregates
    - Complex validation and business rule enforcement across boundaries
    - Integration services for external system communication
    - Policy enforcement and business rule evaluation services

Domain Service Architecture Patterns:
    Stateless Operations: Thread-safe services without state management
    Cross-Entity Logic: Business operations spanning multiple domain objects
    Policy Enforcement: Business rule evaluation and validation services
    Integration Patterns: External system communication and coordination

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Abstract base service, validation, serialization
    🚧 Active Development: Service composition patterns (Enhancement 3 - Med)
    📋 TODO Integration: Complex business workflow orchestration (Priority 2)

Domain Service Features:
    FlextDomainService: Abstract base with execute method enforcement
    Stateless Design: Thread-safe operations without instance state
    Validation Integration: Automatic validation through mixin inheritance
    Service Composition: Complex business operations through service coordination

Ecosystem Usage Patterns:
    # FLEXT Service Domain Services
    class UserRegistrationService(FlextDomainService):
        def execute(self, registration_data: dict) -> FlextResult[User]:
            # Cross-entity validation and business logic
            if self.is_email_taken(registration_data["email"]):
                return FlextResult.fail("Email already registered")
            return FlextResult.ok(User(**registration_data))

    # Singer Tap/Target Services
    class OracleConnectionValidationService(FlextDomainService):
        def execute(self, connection_data: dict) -> FlextResult[bool]:
            # Complex validation across connection parameters
            if not self.validate_host_reachability(connection_data["host"]):
                return FlextResult.fail("Host unreachable")
            return FlextResult.ok(data=True)

    # ALGAR Migration Services
    class LdapMigrationService(FlextDomainService):
        def execute(self, migration_request: dict) -> FlextResult[MigrationResult]:
            # Complex migration logic across multiple systems
            source_users = self.extract_users(migration_request["source"])
            return self.migrate_users(source_users, migration_request["target"])

Service Operation Categories:
    - Cross-Entity Operations: Business logic spanning multiple domain objects
    - Complex Validation: Multi-step validation across aggregate boundaries
    - Integration Services: External system communication and data exchange
    - Policy Services: Business rule evaluation and enforcement
    - Calculation Services: Complex business computations and analytics

Quality Standards:
    - All domain services must be stateless for thread safety
    - Services must implement the abstract execute method
    - Cross-entity operations must respect aggregate boundaries
    - Services should coordinate through domain events when possible

See Also:
    docs/TODO.md: Enhancement 3 - Service composition pattern development
    entities.py: Entity patterns that services coordinate
    aggregate_root.py: Aggregate boundaries that services must respect

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Callable

from flext_core.mixins import FlextSerializableMixin, FlextValidatableMixin
from flext_core.result import FlextResult

# =============================================================================
# FLEXT DOMAIN SERVICE - Public DDD Domain Service implementation
# =============================================================================


class FlextDomainService(
    BaseModel,
    FlextValidatableMixin,
    FlextSerializableMixin,
    ABC,
):
    """Abstract Domain-Driven Design service for stateless cross-entity operations.

    Comprehensive domain service implementation providing stateless business logic that
    doesn't naturally belong to any single entity or value object. Combines Pydantic
    validation with DDD principles for complex business operations and workflows.

    Architecture:
        - Abstract base class enforcing execute method implementation
        - Pydantic BaseModel for automatic validation and serialization
        - Composition-based delegation to validation and serialization mixins
        - Frozen configuration for immutability and thread safety
        - Stateless design for scalable and thread-safe operations

    Domain Service Principles:
        - Stateless operations ensuring thread safety and scalability
        - Cross-entity business logic that doesn't belong to single aggregate
        - Complex business rule enforcement across aggregate boundaries
        - Service composition for multi-step business processes
        - Integration point for external system communication
        - Business policy enforcement and rule evaluation

    Service Operation Categories:
        - Calculation services: Complex business computations
        - Validation services: Cross-entity validation and rule enforcement
        - Integration services: External system communication and data exchange
        - Policy services: Business rule evaluation and decision making
        - Orchestration services: Multi-step business process coordination
        - Transformation services: Data transformation and mapping operations

    Usage Patterns:
        # Define calculation domain service
        class OrderTaxCalculationService(FlextDomainService):
            tax_rate: float
            jurisdiction: str

            def execute(self) -> FlextResult[TaxCalculationResult]:
                # Complex tax calculation logic
                if self.tax_rate < 0 or self.tax_rate > 1:
                    return FlextResult.fail("Invalid tax rate")

                # Perform tax calculation
                result = TaxCalculationResult(
                    rate=self.tax_rate,
                    jurisdiction=self.jurisdiction,
                    calculated_at=datetime.utcnow()
                )

                return FlextResult.ok(result)

        # Define integration domain service
        class PaymentProcessingService(FlextDomainService):
            payment_provider: str
            api_key: str

            def execute(self) -> FlextResult[PaymentResult]:
                # External payment system integration
                try:
                    # Validate payment provider configuration
                    if not self.payment_provider or not self.api_key:
                        return FlextResult.fail("Payment configuration incomplete")

                    # Process payment through external service
                    payment_result = self._process_payment()
                    return FlextResult.ok(payment_result)

                except PaymentException as e:
                    return FlextResult.fail(f"Payment processing failed: {e}")

        # Define policy domain service
        class DiscountPolicyService(FlextDomainService):
            customer_tier: str
            order_amount: float

            def execute(self) -> FlextResult[DiscountPolicy]:
                # Business rule evaluation for discount eligibility
                if self.order_amount < 0:
                    return FlextResult.fail("Invalid order amount")

                # Apply business rules
                discount_rate = self._calculate_discount_rate()

                policy = DiscountPolicy(
                    customer_tier=self.customer_tier,
                    discount_rate=discount_rate,
                    minimum_order=self._get_minimum_order(),
                    valid_until=datetime.utcnow() + timedelta(days=30)
                )

                return FlextResult.ok(policy)

        # Use domain services
        tax_service = OrderTaxCalculationService(
            tax_rate=0.085,
            jurisdiction="CA"
        )

        tax_result = tax_service.execute()
        if tax_result.is_success:
            tax_calculation = tax_result.data
            # Apply tax calculation to order

        # Service composition for complex operations
        payment_service = PaymentProcessingService(
            payment_provider="stripe",
            api_key="sk_test_..."
        )

        discount_service = DiscountPolicyService(
            customer_tier="premium",
            order_amount=299.99
        )

        # Execute services in sequence
        discount_result = discount_service.execute()
        if discount_result.is_success:
            payment_result = payment_service.execute()

    Service Composition Patterns:
        - Sequential service execution for multi-step processes
        - Service result chaining with FlextResult pattern
        - Service dependency injection for complex workflows
        - Service orchestration for business process automation
        - Error handling and rollback strategies for failed operations

    Thread Safety and Scalability:
        - Stateless design ensuring safe concurrent execution
        - Immutable service configuration preventing state corruption
        - No shared mutable state between service instances
        - Thread-safe service composition and orchestration
        - Scalable service deployment patterns for distributed systems
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Mixin functionality is now inherited properly:
    # - Validation methods from FlextValidatableMixin
    # - Serialization methods from FlextSerializableMixin

    @abstractmethod
    def execute(self) -> FlextResult[object]:
        """Execute the domain service operation.

        Must be implemented by concrete services.
        """

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration - override in subclasses.

        Default implementation returns success. Override to add custom validation.
        """
        return FlextResult.ok(None)

    def execute_operation(
        self,
        operation_name: str,
        operation: Callable[[object], object],
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute operation with standard error handling and logging.

        Args:
            operation_name: Name of the operation for logging
            operation: Operation to execute
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            Result of the operation

        """
        try:
            # Validate configuration first
            config_result = self.validate_config()
            if config_result.is_failure:
                error_message = config_result.error or "Configuration validation failed"
                return FlextResult.fail(error_message)

            # Execute operation
            result = operation(*args, **kwargs)
            return FlextResult.ok(result)
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"Operation {operation_name} failed: {e}")

    def get_service_info(self) -> dict[str, object]:
        """Get service information for monitoring."""
        return {
            "service_type": self.__class__.__name__,
            "config_valid": self.validate_config().is_success,
        }


# Export API
__all__ = ["FlextDomainService"]
