"""FLEXT Core Domain Services Module.

Comprehensive Domain-Driven Design (DDD) domain service implementation for stateless
business operations that don't naturally belong to entities or value objects.
Implements consolidated architecture with Pydantic validation and mixin inheritance.

Architecture:
    - Domain-Driven Design domain service patterns for cross-entity operations
    - Abstract base class enforcing service operation implementation
    - Pydantic BaseModel integration for automatic validation and serialization
    - Multiple inheritance from validation and serialization mixins
    - Stateless design for thread-safe and scalable service operations
    - FlextResult pattern integration for consistent error handling

Domain Service System Components:
    - FlextDomainService: Abstract base domain service with operation interface
    - Service execution: Abstract execute method enforcing implementation
    - Validation integration: Automatic validation through mixin inheritance
    - Serialization support: Service state serialization for transport and persistence
    - Error handling: FlextResult pattern for type-safe service operations

Maintenance Guidelines:
    - Create domain services by inheriting from FlextDomainService abstract base
    - Implement execute method for service-specific business operations
    - Use domain services for operations involving multiple entities or value objects
    - Keep services stateless to ensure thread safety and scalability
    - Integrate validation through inherited mixin capabilities
    - Follow DDD principles with clear service boundaries and responsibilities

Design Decisions:
    - Abstract base class pattern enforcing execute method implementation
    - Pydantic frozen models for immutability and thread safety
    - Multiple inheritance from validation and serialization mixins
    - Stateless design for scalable service operations
    - FlextResult return type for consistent error handling
    - Service composition patterns for complex business operations

Domain-Driven Design Features:
    - Stateless domain operations for cross-entity business logic
    - Service encapsulation of complex business rules and workflows
    - Clear separation of service responsibilities from entity behaviors
    - Domain service composition for complex business scenarios
    - Integration with aggregate roots and value objects
    - Business process orchestration through service coordination

Service Operation Patterns:
    - Cross-entity business operations that don't belong to single aggregate
    - Complex validation and business rule enforcement across boundaries
    - Integration services for external system communication
    - Calculation services for complex business computations
    - Policy services for business rule evaluation and enforcement
    - Orchestration services for multi-step business processes

Dependencies:
    - pydantic: Data validation and immutable model configuration
    - mixins: Validation and serialization behavior inheritance
    - result: FlextResult pattern for consistent error handling (TYPE_CHECKING)
    - abc: Abstract base class patterns for enforcing implementation contracts

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from flext_core._mixins_base import _BaseSerializableMixin, _BaseValidatableMixin
from flext_core.mixins import FlextSerializableMixin, FlextValidatableMixin

if TYPE_CHECKING:
    from flext_core.result import FlextResult

# =============================================================================
# FLEXT DOMAIN SERVICE - Public DDD Domain Service implementation
# =============================================================================


class FlextDomainService(BaseModel, ABC):
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

    def __init__(self, **data: object) -> None:
        """Initialize domain service with mixin functionality through composition."""
        super().__init__(**data)
        # Initialize mixin functionality through composition
        self._validation_errors: list[str] = []
        self._is_valid: bool | None = None

    # =========================================================================
    # VALIDATION FUNCTIONALITY - Composition-based delegation to _BaseValidatableMixin
    # =========================================================================

    def _add_validation_error(self, error: str) -> None:
        """Add validation error (delegates to base)."""
        return _BaseValidatableMixin._add_validation_error(self, error)

    def _clear_validation_errors(self) -> None:
        """Clear all validation errors (delegates to base)."""
        return _BaseValidatableMixin._clear_validation_errors(self)

    def _mark_valid(self) -> None:
        """Mark as valid and clear errors (delegates to base)."""
        return _BaseValidatableMixin._mark_valid(self)

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors (delegates to base)."""
        return _BaseValidatableMixin.validation_errors.fget(self)  # type: ignore[misc]

    @property
    def is_valid(self) -> bool:
        """Check if object is valid (delegates to base)."""
        return _BaseValidatableMixin.is_valid.fget(self)  # type: ignore[misc]

    def has_validation_errors(self) -> bool:
        """Check if object has validation errors (delegates to base)."""
        return _BaseValidatableMixin.has_validation_errors(self)

    # =========================================================================
    # SERIALIZATION FUNCTIONALITY - Composition-based delegation to _BaseSerializableMixin
    # =========================================================================

    def to_dict_basic(self) -> dict[str, object]:
        """Convert to basic dictionary representation (delegates to base)."""
        return _BaseSerializableMixin.to_dict_basic(self)

    def _serialize_value(self, value: object) -> object | None:
        """Serialize a single value for dict conversion (delegates to base)."""
        return _BaseSerializableMixin._serialize_value(self, value)

    def _serialize_collection(
        self,
        collection: list[object] | tuple[object, ...],
    ) -> list[object]:
        """Serialize list or tuple values (delegates to base)."""
        return _BaseSerializableMixin._serialize_collection(self, collection)

    def _serialize_dict(self, dict_value: dict[str, object]) -> dict[str, object]:
        """Serialize dictionary values (delegates to base)."""
        return _BaseSerializableMixin._serialize_dict(self, dict_value)

    def _from_dict_basic(self, data: dict[str, object]) -> FlextDomainService:
        """Create instance from dictionary (delegates to base)."""
        _BaseSerializableMixin._from_dict_basic(self, data)
        return self

    @abstractmethod
    def execute(self) -> FlextResult[object]:
        """Execute the domain service operation.

        Must be implemented by concrete services.
        """


# Export API
__all__ = ["FlextDomainService"]
