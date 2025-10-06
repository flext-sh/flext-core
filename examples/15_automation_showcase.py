"""Automation Decorators Showcase.

This example demonstrates the context enrichment capabilities introduced in the
Phase 1 architectural enhancement.

KEY FEATURES DEMONSTRATED:
- Automatic context enrichment in FlextService and FlextHandlers
- _with_correlation_id: Distributed tracing support
- _with_user_context: User audit trail
- _with_operation_context: Operation tracking
- _enrich_context: Service metadata enrichment

USAGE PATTERNS:
- Context enrichment best practices
- Integration with FlextService base class
- Structured logging with automatic context

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextResult,
    FlextService,
)

# =============================================================================
# EXAMPLE 1: Service with Context Enrichment
# =============================================================================


class UserService(FlextService[dict[str, object]]):
    """Service demonstrating automatic context enrichment."""

    def __init__(self, **data: object) -> None:
        """Initialize with automatic context enrichment.

        FlextService.__init__ automatically calls:
        - _enrich_context() with service metadata
        """
        super().__init__(**data)
        # Context now includes: service_type, service_module

    def execute(self) -> FlextResult[dict[str, object]]:
        """Required abstract method implementation."""
        return FlextResult[dict[str, object]].ok({"status": "initialized"})

    def create_user(self, username: str, email: str) -> FlextResult[dict[str, object]]:
        """Create user with automatic context enrichment."""
        # Context includes service metadata from __init__
        if self.logger:
            self.logger.info("Creating user", username=username, email=email)

        # Business logic
        user_data: dict[str, object] = {
            "id": "usr_123",
            "username": username,
            "email": email,
        }

        return FlextResult[dict[str, object]].ok(user_data)


# =============================================================================
# EXAMPLE 2: Context Enrichment with Correlation ID
# =============================================================================


class PaymentService(FlextService[dict[str, object]]):
    """Service demonstrating correlation ID tracking."""

    def __init__(self, **data: object) -> None:
        """Initialize with automatic context enrichment."""
        super().__init__(**data)

    def execute(self) -> FlextResult[dict[str, object]]:
        """Required abstract method implementation."""
        return FlextResult[dict[str, object]].ok({"status": "initialized"})

    def process_payment(
        self,
        payment_id: str,
        amount: float,
        user_id: str,
    ) -> FlextResult[dict[str, object]]:
        """Process payment with correlation tracking.

        Demonstrates:
        1. Correlation ID generation for distributed tracing
        2. User context for audit trail
        3. Operation context for tracking
        """
        # Generate correlation ID for distributed tracing
        correlation_id = self._with_correlation_id()

        # Set user context for audit trail
        self._with_user_context(user_id, payment_id=payment_id)

        # Set operation context
        self._with_operation_context("process_payment", amount=amount)

        # All logs now include full context automatically
        if self.logger:
            self.logger.info(
                "Processing payment",
                payment_id=payment_id,
                amount=amount,
                correlation_id=correlation_id,
            )

        # Business logic
        payment_data: dict[str, object] = {
            "payment_id": payment_id,
            "amount": amount,
            "status": "completed",
            "correlation_id": correlation_id,
        }

        # Clean up operation context
        self._clear_operation_context()

        return FlextResult[dict[str, object]].ok(payment_data)


# =============================================================================
# EXAMPLE 3: Using execute_with_context_enrichment Helper
# =============================================================================


class OrderService(FlextService[dict[str, object]]):
    """Service demonstrating context enrichment helper method."""

    def __init__(self, **data: object) -> None:
        """Initialize service."""
        super().__init__(**data)
        self._order_data: dict[str, object] = {}

    def execute(self) -> FlextResult[dict[str, object]]:
        """Process order with business logic."""
        # Implement actual order processing
        self._order_data = {
            "order_id": "ord_123",
            "status": "processed",
        }
        return FlextResult[dict[str, object]].ok(self._order_data)

    def process_order(
        self,
        order_id: str,
        customer_id: str,
        correlation_id: str | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Process order with automatic context enrichment.

        Uses execute_with_context_enrichment() helper that:
        - Sets correlation ID
        - Sets operation context
        - Sets user context
        - Tracks performance
        - Logs operation start/complete/error
        - Cleans up context after operation
        """
        # Store order data for execute() to process
        self._order_data = {
            "order_id": order_id,
            "customer_id": customer_id,
        }

        # Use helper method for automatic context enrichment
        return self.execute_with_context_enrichment(
            operation_name="process_order",
            correlation_id=correlation_id,
            user_id=customer_id,
            order_id=order_id,
        )


# =============================================================================
# DEMONSTRATION
# =============================================================================


def main() -> None:
    """Demonstrate context enrichment in action."""
    print("=" * 80)
    print("FLEXT-CORE CONTEXT ENRICHMENT SHOWCASE")
    print("=" * 80)

    # Example 1: Basic service with automatic context
    print("\n1. BASIC SERVICE WITH CONTEXT ENRICHMENT")
    print("-" * 80)
    user_service = UserService()
    result1 = user_service.create_user("john_doe", "john@example.com")
    print(f"Result: {result1.value if result1.is_success else result1.error}")

    # Example 2: Payment service with correlation ID
    print("\n2. SERVICE WITH CORRELATION ID TRACKING")
    print("-" * 80)
    payment_service = PaymentService()
    result2 = payment_service.process_payment(
        payment_id="pay_123",
        amount=99.99,
        user_id="usr_456",
    )
    print(f"Result: {result2.value if result2.is_success else result2.error}")

    # Example 3: Order service using helper method
    print("\n3. SERVICE USING CONTEXT ENRICHMENT HELPER")
    print("-" * 80)
    order_service = OrderService()
    result3 = order_service.process_order(
        order_id="ord_123",
        customer_id="cust_456",
        correlation_id="corr_abc123",
    )
    print(f"Result: {result3.value if result3.is_success else result3.error}")

    print("\n" + "=" * 80)
    print("KEY BENEFITS DEMONSTRATED:")
    print("=" * 80)
    print("✅ Automatic context enrichment in FlextService")
    print("✅ Correlation ID generation for distributed tracing")
    print("✅ User context enrichment for audit trails")
    print("✅ Operation context tracking")
    print("✅ Automatic context cleanup")
    print("✅ Structured logging with full context")
    print("✅ Helper method for complete automation")
    print("✅ Zero boilerplate infrastructure code")
    print("=" * 80)


if __name__ == "__main__":
    main()
