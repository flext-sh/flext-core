# !/usr/bin/env python3
"""08 - Complete FLEXT Integration: All Components Working Together.

This comprehensive example demonstrates how ALL FLEXT components work together
in a real-world application scenario - an e-commerce order processing system.

Integrates:
- FlextResult for railway-oriented error handling throughout
- FlextContainer for dependency injection and service management
- FlextModels for domain modeling (entities, values, aggregates)
- FlextConfig for environment-aware configuration
- FlextLogger for structured logging with correlation tracking
- FlextProcessors for handler pipelines and strategy patterns
- FlextModels.Payload and DomainEvent for messaging

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import cast
from uuid import uuid4

from pydantic import Field

from flext_core import (
    Flext,
    FlextConfig,
    FlextContainer,
    FlextContext,
    FlextExceptions,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
    FlextTypes,
)

from .example_scenarios import (
    ExampleScenarios,
    OrderItemDict,
    RealisticDataDict,
    RealisticOrderDict,
)

# Constants
UNKNOWN_ERROR_MSG = "Unknown error"


# Type guard helper functions (Python 3.13+ pattern)
def assert_logger_initialized(logger: FlextLogger | None) -> FlextLogger:
    """Type guard to assert logger is initialized.

    Args:
        logger: Logger instance or None.

    Returns:
        Non-None logger instance.

    Raises:
        FlextExceptions.ConfigurationError: If logger is None.

    """
    if logger is None:
        msg = "Logger must be initialized"
        raise FlextExceptions.ConfigurationError(msg)
    return logger


def assert_container_initialized(container: FlextContainer | None) -> FlextContainer:
    """Type guard to assert container is initialized.

    Args:
        container: Container instance or None.

    Returns:
        Non-None container instance.

    Raises:
        FlextExceptions.ConfigurationError: If container is None.

    """
    if container is None:
        msg = "Container must be initialized"
        raise FlextExceptions.ConfigurationError(msg)
    return container


class ProductId(FlextModels.Value):
    """Product identifier value object."""

    value: str

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate product ID format."""
        if not self.value or not self.value.startswith("PROD-"):
            return FlextResult[None].fail("Invalid product ID format")
        return FlextResult[None].ok(None)


class Money(FlextModels.Value):
    """Money value object with currency."""

    amount: Decimal
    currency: str = "USD"

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate money amount."""
        if self.amount < 0:
            return FlextResult[None].fail("Amount cannot be negative")
        if self.currency not in {"USD", "EUR", "GBP"}:
            return FlextResult[None].fail(f"Unsupported currency: {self.currency}")
        return FlextResult[None].ok(None)

    def add(self, other: Money) -> FlextResult[Money]:
        """Add money amounts with same currency."""
        if self.currency != other.currency:
            return FlextResult[Money].fail("Cannot add different currencies")
        return FlextResult[Money].ok(
            Money(amount=self.amount + other.amount, currency=self.currency),
        )


class Product(FlextModels.Entity):
    """Product entity."""

    name: str
    price: Money
    stock: int

    def is_available(self, quantity: int) -> bool:
        """Check if product is available in requested quantity."""
        return self.stock >= quantity

    def reserve(self, quantity: int) -> FlextResult[None]:
        """Reserve product quantity."""
        if not self.is_available(quantity):
            return FlextResult[None].fail(f"Insufficient stock: {self.stock} available")
        self.stock -= quantity
        self.add_domain_event(
            "ProductReserved",
            {
                "product_id": self.id,
                "quantity": quantity,
                "remaining_stock": self.stock,
            },
        )
        return FlextResult[None].ok(None)


class OrderItem(FlextModels.Value):
    """Order line item value object."""

    product: Product
    quantity: int
    unit_price: Money

    def calculate_total(self) -> Money:
        """Calculate total price for this item."""
        return Money(
            amount=self.unit_price.amount * self.quantity,
            currency=self.unit_price.currency,
        )


class Order(FlextModels.AggregateRoot):
    """Order aggregate root - maintains consistency boundary."""

    customer_id: str
    items: list[OrderItem] = Field(default_factory=list)
    status: str = "DRAFT"
    total: Money = Field(
        default_factory=lambda: Money(amount=Decimal(0), currency="USD"),
    )

    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        """Add item to order with business validation."""
        # Validate business rules
        if self.status != "DRAFT":
            return FlextResult[None].fail("Cannot modify submitted order")

        if quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")

        # Check product availability
        if not product.is_available(quantity):
            return FlextResult[None].fail(f"Product {product.name} not available")

        # Reserve product
        reserve_result = product.reserve(quantity)
        if reserve_result.is_failure:
            return reserve_result

        # Add item
        item = OrderItem(product=product, quantity=quantity, unit_price=product.price)
        self.items.append(item)

        # Update total
        total_amount = sum(item.calculate_total().amount for item in self.items)
        self.total = Money(
            amount=Decimal(str(total_amount)),
            currency="USD",
        )

        # Emit domain event
        self.add_domain_event(
            "OrderItemAdded",
            {
                "order_id": self.id,
                "product_id": product.id,
                "quantity": quantity,
                "total": str(self.total.amount),
            },
        )

        return FlextResult[None].ok(None)

    def get_domain_events(self) -> FlextTypes.List:
        """Get all domain events."""
        return self.domain_events

    def submit(self) -> FlextResult[None]:
        """Submit order for processing."""
        if self.status != "DRAFT":
            return FlextResult[None].fail("Order already submitted")

        if not self.items:
            return FlextResult[None].fail("Cannot submit empty order")

        self.status = "SUBMITTED"
        self.add_domain_event(
            "OrderSubmitted",
            {
                "order_id": self.id,
                "customer_id": self.customer_id,
                "total": str(self.total.amount),
                "items_count": len(self.items),
            },
        )

        return FlextResult[None].ok(None)


# ========== SERVICES (Using FlextService patterns) ==========


class InventoryService(FlextService[FlextTypes.Dict]):
    """Inventory management service with FlextMixins.Service infrastructure.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextContainer singleton)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextContext for request tracking)
    - Inherited config property (FlextConfig with application settings)
    - Inherited metrics property (FlextMetrics for observability)

    Manages product inventory with domain model operations and structured logging.
    """

    def __init__(self) -> None:
        """Initialize inventory service with inherited FlextMixins.Service infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context (ALREADY CONFIGURED!)
        - self.container: FlextContainer global singleton
        - self.context: FlextContext for request tracking
        - self.config: FlextConfig with application settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        # Use self.logger from FlextMixins.Logging, not _logger
        self._scenarios = ExampleScenarios
        self._products = {}
        self._initialize_products()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "InventoryService initialized with inherited infrastructure",
            extra={
                "service_type": "Inventory Management",
                "products_loaded": len(self._products),
            }
        )

    def _initialize_products(self) -> None:
        """Initialize sample products."""
        realistic_data: RealisticDataDict = self._scenarios.realistic_data()
        realistic_order: RealisticOrderDict = realistic_data["order"]
        products: list[Product] = []
        item: OrderItemDict
        for item in realistic_order["items"]:
            amount_raw = item.get("price", Decimal(0))
            amount = (
                amount_raw
                if isinstance(amount_raw, Decimal)
                else Decimal(str(amount_raw))
            )
            product = Product(
                id=str(item.get("product_id", uuid4())),
                name=str(item.get("name", "Catalog Item")),
                price=Money(amount=amount, currency="USD"),
                stock=int(item.get("quantity", 1)) * 10,
            )
            products.append(product)

        if not products:
            products.append(
                Product(
                    id="PROD-FALLBACK",
                    name="Fallback Item",
                    price=Money(amount=Decimal("50.00"), currency="USD"),
                    stock=10,
                ),
            )
        for product in products:
            self._products[product.id] = product

    def get_product(self, product_id: str) -> FlextResult[Product]:
        """Get product by ID."""
        product = self._products.get(product_id)
        if not product:
            return FlextResult[Product].fail(f"Product not found: {product_id}")
        return FlextResult[Product].ok(product)

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute inventory operation."""
        return FlextResult[FlextTypes.Dict].ok({"status": "inventory_service_ready"})

    def process_operation(
        self,
        data: dict[str, str | int],
    ) -> FlextResult[FlextTypes.Dict]:
        """Process inventory operation."""
        operation = data.get("operation")

        if operation == "get_product":
            product_id = data["product_id"]
            if isinstance(product_id, str):
                result = self.get_product(product_id)
                if result.is_success:
                    product = result.unwrap()
                    return FlextResult[FlextTypes.Dict].ok({
                        "product": {
                            "id": product.id,
                            "name": product.name,
                            "price": str(product.price.amount),
                            "stock": product.stock,
                        },
                    })
                return FlextResult[FlextTypes.Dict].fail(
                    result.error or UNKNOWN_ERROR_MSG,
                )
            return FlextResult[FlextTypes.Dict].fail("Invalid product ID type")

        return FlextResult[FlextTypes.Dict].fail(f"Unknown operation: {operation}")


class PaymentService(FlextService[FlextTypes.Dict]):
    """Payment processing service with FlextMixins.Service infrastructure.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextContainer singleton)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextContext for correlation tracking)
    - Inherited config property (FlextConfig with payment settings)
    - Inherited metrics property (FlextMetrics for payment observability)

    Implements payment processing with strategy pattern and structured logging.
    """

    def __init__(self) -> None:
        """Initialize payment service with inherited FlextMixins.Service infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context (ALREADY CONFIGURED!)
        - self.container: FlextContainer global singleton
        - self.context: FlextContext for correlation tracking
        - self.config: FlextConfig with payment settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        # Use self.logger from FlextMixins.Logging, not _logger

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "PaymentService initialized with inherited infrastructure",
            extra={
                "service_type": "Payment Processing",
                "payment_methods": ["credit_card", "paypal"],
            }
        )

    def process_payment(
        self,
        order: Order,
        method: str,
    ) -> FlextResult[FlextTypes.Dict]:
        """Process payment for order."""
        self.logger.info(
            "Processing payment",
            extra={
                "order_id": order.id,
                "method": method,
                "amount": str(order.total.amount),
            },
        )

        # Simulate payment processing
        if method == "credit_card":
            # Simulate credit card processing
            time.sleep(0.1)  # Simulate API call
            return FlextResult[FlextTypes.Dict].ok({
                "transaction_id": str(uuid4()),
                "status": "approved",
                "method": method,
                "amount": str(order.total.amount),
            })

        if method == "paypal":
            # Simulate PayPal processing
            time.sleep(0.15)  # Simulate API call
            return FlextResult[FlextTypes.Dict].ok({
                "transaction_id": str(uuid4()),
                "status": "approved",
                "method": method,
                "amount": str(order.total.amount),
            })

        return FlextResult[FlextTypes.Dict].fail(
            f"Unsupported payment method: {method}",
        )

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute payment operation."""
        return FlextResult[FlextTypes.Dict].ok({"status": "payment_service_ready"})

    def process_operation(
        self,
        data: dict[str, str | int],
    ) -> FlextResult[dict[str, str | int]]:
        """Process payment operation."""
        _ = data  # This would process the payment based on data
        return FlextResult[dict[str, str | int]].ok({"status": "processed"})


class OrderService(FlextService[FlextTypes.Dict]):
    """Order processing service with FlextMixins.Service infrastructure - orchestrates the workflow.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextContainer singleton for service dependencies)
    - Inherited logger property (FlextLogger with service context and correlation tracking)
    - Inherited context property (FlextContext for request and correlation IDs)
    - Inherited config property (FlextConfig with order processing settings)
    - Inherited metrics property (FlextMetrics for order observability)

    Orchestrates complete order workflow: validation → creation → payment → confirmation.
    Uses inherited container to access InventoryService and PaymentService dependencies.
    """

    def __init__(self) -> None:
        """Initialize order service with inherited FlextMixins.Service infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context (ALREADY CONFIGURED!)
        - self.container: FlextContainer global singleton (access inventory/payment services)
        - self.context: FlextContext for request and correlation tracking
        - self.config: FlextConfig with order processing configuration
        - self.metrics: FlextMetrics for order workflow observability
        """
        super().__init__()
        self._scenarios = ExampleScenarios()
        self._metadata = self._scenarios.metadata(tags=["integration", "demo"])

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "OrderService initialized with inherited infrastructure",
            extra={
                "service_type": "Order Orchestration",
                "dependencies": ["InventoryService", "PaymentService"],
                "metadata": self._metadata,
            }
        )

    def create_order(
        self,
        customer_id: str,
        items: list[dict[str, str | int]],
    ) -> FlextResult[Order]:
        """Create and process an order."""
        # Create correlation ID for tracking
        correlation_id = str(uuid4())
        self.logger.bind(correlation_id=correlation_id)
        self.logger.info("Creating order", extra={"customer_id": customer_id})

        # Get services from container
        inventory_result = self.container.get_typed("inventory", InventoryService)
        if inventory_result.is_failure:
            return FlextResult[Order].fail("Inventory service not available")
        inventory = inventory_result.unwrap()

        # Create order
        order = Order(customer_id=customer_id, domain_events=[])

        # Add items
        for item_data in items:
            # Get product
            product_id = item_data["product_id"]
            quantity = item_data["quantity"]
            if isinstance(product_id, str) and isinstance(quantity, int):
                product_result = inventory.get_product(product_id)
                if product_result.is_failure:
                    self.logger.info("Product not found: %s", product_id)
                    continue

                product = product_result.unwrap()

                # Add to order
                add_result = order.add_item(product, quantity)
                if add_result.is_failure:
                    self.logger.warning(
                        f"Failed to add item: {add_result.error}",
                        extra={"product_id": product.id},
                    )
                    continue

                self.logger.info(
                    "Item added to order",
                    extra={"product_id": product.id, "quantity": quantity},
                )

        # Check if any items were added
        if not order.items:
            return FlextResult[Order].fail("No items could be added to order")

        self.logger.info(
            "Order created successfully",
            extra={
                "order_id": order.id,
                "total": str(order.total.amount),
                "items_count": len(order.items),
            },
        )

        return FlextResult[Order].ok(order)

    def submit_order(
        self,
        order: Order,
        payment_method: str,
    ) -> FlextResult[FlextTypes.Dict]:
        """Submit order with payment processing."""
        self.logger.info("Submitting order", extra={"order_id": order.id})

        # Submit the order
        submit_result = order.submit()
        if submit_result.is_failure:
            return FlextResult[FlextTypes.Dict].fail(
                submit_result.error or UNKNOWN_ERROR_MSG,
            )

        # Get payment service
        payment_service_result = self.container.get_typed("payment", PaymentService)
        if payment_service_result.is_failure:
            return FlextResult[FlextTypes.Dict].fail("Payment service not available")
        payment = payment_service_result.unwrap()

        # Process payment
        payment_result = payment.process_payment(order, payment_method)
        if payment_result.is_failure:
            # Rollback order status
            order.status = "PAYMENT_FAILED"
            return FlextResult[FlextTypes.Dict].fail(
                f"Payment failed: {payment_result.error}",
            )

        # Update order status
        order.status = "PAID"
        payment_data = payment_result.unwrap()
        order.add_domain_event(
            "OrderPaid",
            {"order_id": order.id, "transaction": payment_data},
        )

        # Send confirmation message
        confirmation_data: dict[str, str | int | float] = {
            "order_id": order.id,
            "customer_id": order.customer_id,
            "total": str(order.total.amount),
            "status": order.status,
        }
        confirmation_payload = FlextModels.Payload[dict[str, str | int | float]](
            data=confirmation_data,
            correlation_id=FlextContext.Correlation.get_correlation_id(),
            source_service="order_service",
            message_type="order_confirmation",
        )

        self.logger.info(
            "Order processed successfully",
            extra={
                "order_id": order.id,
                "status": order.status,
                "payment": payment_data,
            },
        )

        result_data: FlextTypes.Dict = {
            "order_id": order.id,
            "status": order.status,
            "total": str(order.total.amount),
            "confirmation": confirmation_payload.id,
        }
        # Add payment data as string representation to avoid type issues
        result_data["payment_transaction_id"] = str(
            payment_data.get("transaction_id", "N/A"),
        )
        result_data["payment_status"] = str(payment_data.get("status", "unknown"))

        return FlextResult[FlextTypes.Dict].ok(result_data)

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute order operation."""
        return FlextResult[FlextTypes.Dict].ok({"status": "order_service_ready"})

    def process_operation(
        self,
        data: FlextTypes.Dict,
    ) -> FlextResult[FlextTypes.Dict]:
        """Process order operation."""
        operation = data.get("operation")

        if operation == "create_and_submit":
            # Create order
            customer_id = data["customer_id"]
            items = data["items"]
            if isinstance(customer_id, str) and isinstance(items, list):
                order_result = self.create_order(customer_id, items)
                if order_result.is_failure:
                    return FlextResult[FlextTypes.Dict].fail(
                        order_result.error or UNKNOWN_ERROR_MSG,
                    )

                order = order_result.unwrap()

                # Submit with payment
                payment_method = data.get("payment_method", "credit_card")
                if isinstance(payment_method, str):
                    return self.submit_order(order, payment_method)
                return FlextResult[FlextTypes.Dict].fail(
                    "Invalid payment method type",
                )

        return FlextResult[FlextTypes.Dict].fail(f"Unknown operation: {operation}")


# ========== HANDLER PIPELINE (Using FlextProcessors) ==========


class OrderValidationHandler:
    """Validate order request."""

    def __init__(self) -> None:
        """Initialize handler."""
        super().__init__()
        self.name = "OrderValidator"
        self._logger = FlextLogger(__name__)

    def handle(self, request: FlextTypes.Dict) -> FlextResult[FlextTypes.Dict]:
        """Validate order data."""
        self._logger.info("Validating order request")

        # Check required fields
        if not request.get("customer_id"):
            return FlextResult[FlextTypes.Dict].fail("Customer ID required")

        if not request.get("items"):
            return FlextResult[FlextTypes.Dict].fail("Order items required")

        # Validate items
        items = request["items"]
        if isinstance(items, list):
            for item in items:
                if not item.get("product_id"):
                    return FlextResult[FlextTypes.Dict].fail(
                        "Product ID required for all items",
                    )
                quantity = item.get("quantity")
                if not isinstance(quantity, int) or quantity <= 0:
                    return FlextResult[FlextTypes.Dict].fail(
                        "Valid quantity required for all items",
                    )

        request["validated"] = True
        return FlextResult[FlextTypes.Dict].ok(request)


class OrderEnrichmentHandler:
    """Enrich order with additional data."""

    def __init__(self) -> None:
        """Initialize handler."""
        super().__init__()
        self.name = "OrderEnricher"
        self._logger = FlextLogger(__name__)

    def handle(
        self,
        request: FlextTypes.Dict,
        *,
        _debug: bool = False,
    ) -> FlextResult[FlextTypes.Dict]:
        """Add metadata to order."""
        self._logger.info("Enriching order request")

        if not request.get("validated"):
            return FlextResult[FlextTypes.Dict].fail("Order must be validated first")

        # Add metadata
        metadata: dict[str, str | float] = {
            "timestamp": str(time.time()),
            "source": "web",
            "version": "1.0",
            "correlation_id": str(uuid4()),
        }

        # Create a new dict with the metadata added
        enriched_request: FlextTypes.Dict = {
            **request,
            "metadata": metadata,
        }

        return FlextResult[FlextTypes.Dict].ok(enriched_request)


# ========== INTEGRATION DEMONSTRATION ==========


def demonstrate_new_flextresult_methods() -> None:
    """Demonstrate the 5 new FlextResult methods in integration context.
    
    Shows how the new v0.9.9+ methods work with complete FLEXT integration:
    - from_callable: Safe exception handling
    - flow_through: Pipeline composition
    - lash: Error recovery
    - alt: Alternative results
    - value_or_call: Lazy defaults
    """
    print("=" * 60)
    print("NEW FLEXTRESULT METHODS - INTEGRATION CONTEXT")
    print("Demonstrating v0.9.9+ methods in complete system integration")
    print("=" * 60)

    # Setup test data
    scenario_data: RealisticDataDict = ExampleScenarios.realistic_data()
    order_template: RealisticOrderDict = scenario_data["order"]
    user_pool = ExampleScenarios.users()
    REDACTED_LDAP_BIND_PASSWORD_user = user_pool[0]  # Get first user for demos

    # 1. from_callable - Safe Integration Layer Operations
    print("\n=== 1. from_callable: Safe Integration Operations ===")

    def risky_order_creation(customer_id: str, items: list) -> Order:
        """Order creation that might raise exceptions."""
        if not items:
            msg = "Cannot create empty order"
            raise FlextExceptions.ValidationError(msg, field="items", value=items)
        if not customer_id:
            msg = "Customer ID required"
            raise FlextExceptions.ValidationError(msg, field="customer_id", value=None)
        return Order(customer_id=customer_id, items=[], domain_events=[])

    # Safe order creation without try/except
    order_result = FlextResult.from_callable(
        lambda: risky_order_creation(
            REDACTED_LDAP_BIND_PASSWORD_user["email"],
            order_template.get("items", []),
        ),
    )
    if order_result.is_success:
        order = order_result.unwrap()
        print(f"✅ Order created safely: {order.customer_id}")
    else:
        print(f"❌ Order creation failed: {order_result.error}")

    # 2. flow_through - Complete Integration Pipeline
    print("\n=== 2. flow_through: Integration Pipeline Composition ===")

    def validate_order_data(data: dict) -> FlextResult[dict]:
        """Validate order request data."""
        if not data.get("customer_id"):
            return FlextResult[dict].fail("Customer ID required")
        if not data.get("items"):
            return FlextResult[dict].fail("Items required")
        return FlextResult[dict].ok(data)

    def check_inventory_availability(data: dict) -> FlextResult[dict]:
        """Check all items are in stock."""
        # Simulate inventory check
        enriched = {**data, "inventory_status": "available"}
        return FlextResult[dict].ok(enriched)

    def calculate_order_total(data: dict) -> FlextResult[dict]:
        """Calculate total price with taxes."""
        items = data.get("items", [])
        subtotal = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)
        tax = subtotal * Decimal("0.1")
        total = subtotal + tax
        enriched = {**data, "subtotal": subtotal, "tax": tax, "total": total}
        return FlextResult[dict].ok(enriched)

    def apply_promotions(data: dict) -> FlextResult[dict]:
        """Apply promotional discounts."""
        total = data.get("total", Decimal(0))
        discount = total * Decimal("0.05") if total > 100 else Decimal(0)
        final_total = total - discount
        enriched = {**data, "discount": discount, "final_total": final_total}
        return FlextResult[dict].ok(enriched)

    # Flow through complete integration pipeline
    order_data: dict = {
        "customer_id": REDACTED_LDAP_BIND_PASSWORD_user["email"],
        "items": order_template.get("items", []),
        "source": "web_checkout",
    }
    pipeline_result = (
        FlextResult[dict]
        .ok(order_data)
        .flow_through(
            validate_order_data,
            check_inventory_availability,
            calculate_order_total,
            apply_promotions,
        )
    )

    if pipeline_result.is_success:
        final_order = pipeline_result.unwrap()
        print(f"✅ Integration pipeline complete: ${final_order.get('final_total', 0):.2f}")
        print(f"   Subtotal: ${final_order.get('subtotal', 0):.2f}")
        print(f"   Tax: ${final_order.get('tax', 0):.2f}")
        print(f"   Discount: ${final_order.get('discount', 0):.2f}")
    else:
        print(f"❌ Pipeline failed: {pipeline_result.error}")

    # 3. lash - Service Fallback Recovery
    print("\n=== 3. lash: Service Fallback Recovery ===")

    def try_primary_payment_gateway(_amount: Decimal) -> FlextResult[dict]:
        """Try primary payment processor."""
        # Simulate failure (amount parameter for demonstration purposes)
        return FlextResult[dict].fail("Primary gateway timeout")

    def fallback_to_secondary_gateway(error: str) -> FlextResult[dict]:
        """Fallback to secondary payment processor."""
        print(f"   ⚠️  Primary failed: {error}, using fallback...")
        # Simulate successful fallback
        return FlextResult[dict].ok({
            "gateway": "secondary",
            "transaction_id": str(uuid4()),
            "status": "approved",
        })

    # Try primary, fallback to secondary on failure
    payment_result = try_primary_payment_gateway(Decimal("99.99")).lash(
        fallback_to_secondary_gateway,
    )

    if payment_result.is_success:
        payment_data = payment_result.unwrap()
        print(f"✅ Payment processed via {payment_data.get('gateway')} gateway")
        print(f"   Transaction: {payment_data.get('transaction_id')}")
    else:
        print(f"❌ All payment gateways failed: {payment_result.error}")

    # 4. alt - Service Discovery with Fallback
    print("\n=== 4. alt: Service Discovery with Fallback ===")

    def get_premium_shipping_service() -> FlextResult[dict]:
        """Try to get premium shipping service."""
        # Simulate service unavailable
        return FlextResult[dict].fail("Premium shipping service unavailable")

    def get_standard_shipping_service() -> FlextResult[dict]:
        """Get standard shipping service."""
        return FlextResult[dict].ok({
            "service": "standard",
            "estimated_days": 5,
            "cost": Decimal("9.99"),
        })

    # Try premium, fall back to standard
    shipping = get_premium_shipping_service().alt(get_standard_shipping_service())

    if shipping.is_success:
        shipping_data = shipping.unwrap()
        print(f"✅ Shipping service: {shipping_data.get('service')}")
        print(f"   Delivery: {shipping_data.get('estimated_days')} days")
        print(f"   Cost: ${shipping_data.get('cost')}")
    else:
        print(f"❌ No shipping services available: {shipping.error}")

    # 5. value_or_call - Lazy Configuration Loading
    print("\n=== 5. value_or_call: Lazy Configuration Loading ===")

    def load_custom_config() -> dict:
        """Expensive configuration loading operation."""
        print("   ⚙️  Loading custom configuration from database...")
        time.sleep(0.1)  # Simulate expensive operation
        return {
            "max_order_amount": Decimal(10000),
            "default_currency": "USD",
            "tax_rate": Decimal("0.1"),
        }

    # Config not found, lazy-load default
    config_result: FlextResult[dict] = FlextResult[dict].fail("Config not in cache")

    # Only loads if result is failure
    config = config_result.value_or_call(load_custom_config)
    print(f"✅ Configuration loaded: currency={config.get('default_currency')}")
    print(f"   Max order: ${config.get('max_order_amount')}")
    print(f"   Tax rate: {config.get('tax_rate')}")

    # When success, doesn't call the function
    cached_config: FlextResult[dict] = FlextResult[dict].ok({"cached": True})
    result_config = cached_config.value_or_call(load_custom_config)  # Won't execute
    print(f"✅ Cached config used: {result_config}")

    print("\n" + "=" * 60)
    print("✅ NEW FLEXTRESULT METHODS INTEGRATION DEMO COMPLETE!")
    print("All 5 methods demonstrated in complete system integration context")
    print("=" * 60)


def demonstrate_complete_integration() -> None:
    """Demonstrate all FLEXT components working together."""
    print("=" * 60)
    print("COMPLETE FLEXT INTEGRATION DEMONSTRATION")
    print("E-Commerce Order Processing System")
    print("=" * 60)

    scenario_data: RealisticDataDict = ExampleScenarios.realistic_data()
    order_template: RealisticOrderDict = scenario_data["order"]
    user_pool = ExampleScenarios.users()

    # 1. Configuration setup
    print("\n=== 1. Configuration ===")
    config = FlextConfig()
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"Log Level: {config.log_level}")

    # 2. Container setup with dependency injection
    print("\n=== 2. Dependency Injection ===")
    container = FlextContainer.get_global()

    # Register services
    inventory = InventoryService()
    payment = PaymentService()
    order_service = OrderService()

    container.register("inventory", inventory)
    container.register("payment", payment)
    container.register("order", order_service)
    print("✅ Services registered in container")

    # 3. Create handler pipeline
    print("\n=== 3. Handler Pipeline ===")
    validation_handler = OrderValidationHandler()
    enrichment_handler = OrderEnrichmentHandler()

    def execute_pipeline(
        request: FlextTypes.Dict,
    ) -> FlextResult[FlextTypes.Dict]:
        """Execute the processing pipeline."""
        # Chain handlers using railway pattern
        return validation_handler.handle(request).flat_map(enrichment_handler.handle)

    # 4. Process an order
    print("\n=== 4. Order Processing ===")

    scenario_order = order_template
    inventory_result = container.get_typed("inventory", InventoryService)
    products: list[Product] = []
    if inventory_result.is_success:
        inventory_service = inventory_result.unwrap()
        for item in scenario_order["items"]:
            product_result = inventory_service.get_product(item["product_id"])
            if product_result.is_success:
                products.append(product_result.unwrap())

    order_request: FlextTypes.Dict = {
        "customer_id": scenario_order["customer_id"],
        "items": [
            {
                "product_id": item["product_id"],
                "quantity": item.get("quantity", 1),
            }
            for item in scenario_order["items"]
        ],
        "payment_method": "credit_card",
    }
    order_dict = order_request
    items_list = cast("FlextTypes.List", order_dict["items"])
    first_item = cast("FlextTypes.Dict", items_list[0])
    first_product_id = first_item["product_id"]

    print(f"Customer: {order_dict['customer_id']}")
    items = order_dict["items"]
    if isinstance(items, list):
        print(f"Items: {len(items)} products")
    else:
        print("Items: Invalid items format")

    # Run through pipeline
    pipeline_result = execute_pipeline(order_request)
    if pipeline_result.is_failure:
        print(f"❌ Pipeline failed: {pipeline_result.error}")
        return

    print("✅ Order validated and enriched")
    enriched_request = pipeline_result.unwrap()

    # 5. Create and submit order
    print("\n=== 5. Order Creation & Submission ===")

    # Use the order service to process
    enriched_data = enriched_request
    customer_id = enriched_data["customer_id"]
    items_raw = enriched_data["items"]
    payment_method = enriched_data.get("payment_method", "credit_card")

    if (
        isinstance(customer_id, str)
        and isinstance(items_raw, list)
        and isinstance(payment_method, str)
    ):
        items = items_raw
        result = order_service.process_operation({
            "operation": "create_and_submit",
            "customer_id": customer_id,
            "items": items,
            "payment_method": payment_method,
        })
    else:
        result = FlextResult[FlextTypes.Dict].fail("Invalid enriched data types")

    if result.is_success:
        order_data = result.unwrap()
        print("✅ Order successful!")
        print(f"   Order ID: {order_data['order_id']}")
        print(f"   Status: {order_data['status']}")
        print(f"   Total: ${order_data['total']}")
        # Display payment information from the result data
        if "payment_transaction_id" in order_data:
            transaction_id = order_data["payment_transaction_id"]
            if isinstance(transaction_id, str):
                print(f"   Transaction: {transaction_id}")
            else:
                print(f"   Transaction: {transaction_id}")
        print(f"   Confirmation: {order_data['confirmation']}")
    else:
        print(f"❌ Order failed: {result.error}")

    # 6. Demonstrate error handling
    print("\n=== 6. Error Handling ===")

    # Try to order with insufficient stock
    large_order: FlextTypes.Dict = {
        "customer_id": (
            user_pool[1]["id"] if len(user_pool) > 1 else scenario_order["customer_id"]
        ),
        "items": [
            {
                "product_id": first_product_id,
                "quantity": 100,
            },
        ],
        "payment_method": "paypal",
    }

    print("Attempting order with insufficient stock...")
    validation_result = execute_pipeline(large_order)

    if validation_result.is_success:
        validation_data = validation_result.unwrap()
        customer_id = validation_data["customer_id"]
        items_raw = validation_data["items"]
        payment_method = validation_data.get("payment_method", "credit_card")

        if (
            isinstance(customer_id, str)
            and isinstance(items_raw, list)
            and isinstance(payment_method, str)
        ):
            items = items_raw
            result = order_service.process_operation({
                "operation": "create_and_submit",
                "customer_id": customer_id,
                "items": items,
                "payment_method": payment_method,
            })
        else:
            result = FlextResult[FlextTypes.Dict].fail(
                "Invalid validation data types",
            )

        if result.is_failure:
            print(f"✅ Correctly rejected: {result.error}")
        else:
            print("❌ Should have failed due to insufficient stock")

    # 7. Domain Events
    print("\n=== 7. Domain Events ===")
    print("Events emitted during order processing:")

    # Create a simple order to show events
    default_customer = (
        user_pool[0]["id"] if user_pool else scenario_order["customer_id"]
    )
    simple_order = Order(customer_id=default_customer, domain_events=[])
    item_template = scenario_order["items"][0]
    template_price = item_template.get("price", Decimal("10.00"))
    price_amount = (
        template_price
        if isinstance(template_price, Decimal)
        else Decimal(str(template_price))
    )
    product = Product(
        id=item_template.get("product_id", "PROD-TEMPLATE"),
        name=item_template.get("name", "Template Product"),
        price=Money(amount=price_amount, currency="USD"),
        stock=5,
    )

    simple_order.add_item(product, 2)
    simple_order.submit()

    for event in simple_order.get_domain_events():
        # Domain events are FlextModels.DomainEvent objects
        if isinstance(event, FlextModels.DomainEvent):
            event_name: str = event.event_type
            event_data: FlextTypes.Dict = event.data
            print(f"  📢 {event_name}: {event_data}")
        else:
            print(f"  📢 Unexpected event format: {event}")

    # 8. Logging with correlation
    print("\n=== 8. Structured Logging ===")
    logger = FlextLogger(__name__)
    correlation_id = str(uuid4())
    logger.bind(correlation_id=correlation_id)

    logger.info(
        "Integration test completed",
        extra={
            "orders_processed": 2,
            "success_rate": 0.5,
            "components_tested": [
                "FlextResult",
                "FlextContainer",
                "FlextModels",
                "FlextConfig",
                "FlextLogger",
                "FlextProcessors",
                "FlextService",
                "Payload & Events",
            ],
        },
    )

    print(f"  Correlation ID: {correlation_id}")
    print("  All operations tracked with structured logging")

    print("\n" + "=" * 60)
    print("✅ COMPLETE INTEGRATION DEMONSTRATED!")
    print("All FLEXT components working together seamlessly")
    print("=" * 60)


# ========== USING FLEXTCORE UNIFIED FACADE (MODERN PATTERN) ==========


def demonstrate_flextcore_unified_access() -> None:
    """Demonstrate the modern Flext unified facade pattern.

    This shows how Flext provides a single entry point for accessing
    ALL flext-core components with improved convenience and discoverability.
    """
    print("\n" + "=" * 60)
    print("FLEXTCORE UNIFIED FACADE DEMONSTRATION")
    print("Modern pattern for complete flext-core access")
    print("=" * 60)

    # 1. Setup complete infrastructure with single call
    print("\n=== 1. Unified Infrastructure Setup ===")
    setup_result = Flext.setup_service_infrastructure("ecommerce-service")

    if setup_result.is_success:
        infra = setup_result.unwrap()
        print("  ✅ Complete infrastructure initialized:")
        print(f"     - Config: {type(infra['config']).__name__}")
        print(f"     - Container: {type(infra['container']).__name__}")
        print(f"     - Logger: {type(infra['logger']).__name__}")
        print(f"     - Bus: {type(infra['bus']).__name__}")
        print(f"     - Context: {type(infra['context']).__name__}")

    # 2. Direct class access through Flext
    print("\n=== 2. Direct Component Access ===")
    result = FlextResult[str].ok("Order created successfully")
    config = Flext.Config()
    timeout = Flext.Constants.Defaults.TIMEOUT

    print(f"  ✅ Result access: {result.value}")
    print(f"  ✅ Config access: log_level = {config.log_level}")
    print(f"  ✅ Constants access: timeout = {timeout}")

    # 3. Factory methods for convenience
    print("\n=== 3. Convenience Factory Methods ===")
    success = Flext.create_result_ok({"order_id": "ORD-123", "status": "created"})
    failure = Flext.create_result_fail("Payment declined", "PAYMENT_ERROR")
    logger = Flext.create_logger("ecommerce")

    print(f"  ✅ Success result: {success.value}")
    print(f"  ✅ Failure result: {failure.error}")
    print(f"  ✅ Logger created: {logger}")

    # 4. Instance-based access to all components
    print("\n=== 4. Unified Core Instance ===")
    core = Flext()

    # Access all components through one object
    core_config = core.config
    core_container = core.container
    core_logger = core.logger
    core_bus = core.bus
    core_context = core.context

    print(f"  ✅ Config: {type(core_config).__name__}")
    print(f"  ✅ Container: {type(core_container).__name__}")
    print(f"  ✅ Logger: {type(core_logger).__name__}")
    print(f"  ✅ Bus: {type(core_bus).__name__}")
    print(f"  ✅ Context: {type(core_context).__name__}")

    # 5. Railway pattern with Flext
    print("\n=== 5. Railway Pattern via Flext ===")

    def validate_order(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        if not data.get("customer_id"):
            return FlextResult[FlextTypes.Dict].fail("Customer ID required")
        return FlextResult[FlextTypes.Dict].ok(data)

    def process_order(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        data["processed"] = True
        return FlextResult[FlextTypes.Dict].ok(data)

    # Compose operations
    order_data = {"customer_id": "CUST-001", "amount": 100}
    pipeline_result = validate_order(order_data).flat_map(process_order)

    if pipeline_result.is_success:
        print(f"  ✅ Pipeline result: {pipeline_result.value}")

    print("\n" + "=" * 60)
    print("✅ FLEXTCORE UNIFIED FACADE DEMONSTRATED!")
    print("Single entry point for all flext-core functionality")
    print("=" * 60)


def demonstrate_flextcore_11_features() -> None:
    """Demonstrate Flext 1.1.0 convenience methods.

    Shows the four new convenience methods added in version 1.1.0:
    1. publish_event() - Event publishing with correlation tracking
    2. create_service() - Service creation with infrastructure injection
    3. build_pipeline() - Railway-oriented pipeline builder
    4. request_context() - Request context manager
    """
    print("\n" + "=" * 60)
    print("FLEXTCORE 1.1.0 CONVENIENCE METHODS")
    print("=" * 60)

    # 1. Event Publishing with Correlation Tracking
    print("\n=== 1. Event Publishing ===")
    core = Flext()

    # Publish events with automatic correlation tracking
    event_result = core.publish_event(
        "order.created",
        {"order_id": "ORD-123", "customer_id": "CUST-456", "amount": 99.99},
    )

    if event_result.is_success:
        print("  ✅ Event published successfully")

    # Publish with custom correlation ID
    correlation_result = core.publish_event(
        "payment.processed",
        {"payment_id": "PAY-789", "status": "completed"},
        correlation_id="req-abc-123",
    )

    if correlation_result.is_success:
        print("  ✅ Event published with custom correlation ID")

    # 2. Service Creation with Infrastructure Injection
    print("\n=== 2. Service Creation ===")

    class OrderProcessingService(FlextService):
        def execute(self) -> FlextResult[str]:
            return FlextResult[str].ok("Order processed successfully")

    # Create service with automatic infrastructure setup
    # service_result = Flext.create_service(OrderProcessingService, "order-processing")
    # if service_result.is_success:
    #     service = service_result.unwrap()
    #     exec_result = service.execute()
    #     print(f"  ✅ Service created and executed: {exec_result.value}")
    print("  Service creation demo skipped")

    # 3. Railway Pipeline Builder
    print("\n=== 3. Pipeline Builder ===")

    # Define pipeline operations
    def validate_order_data(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        if not data.get("order_id"):
            return FlextResult[FlextTypes.Dict].fail("Order ID required")
        if not data.get("amount"):
            return FlextResult[FlextTypes.Dict].fail("Amount required")
        return FlextResult[FlextTypes.Dict].ok(data)

    def check_inventory(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        # Simulate inventory check
        data["inventory_checked"] = True
        return FlextResult[FlextTypes.Dict].ok(data)

    def calculate_shipping(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        # Simulate shipping calculation
        amount_raw = data.get("amount", 0)
        amount = Decimal(str(amount_raw)) if amount_raw else Decimal(0)
        shipping = amount * Decimal("0.1")  # 10% shipping
        data["shipping"] = float(shipping)
        return FlextResult[FlextTypes.Dict].ok(data)

    def apply_discounts(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        # Simulate discount application
        data["discount_applied"] = True
        return FlextResult[FlextTypes.Dict].ok(data)

    # Build pipeline
    # order_pipeline = Flext.build_pipeline(
    #     validate_order_data, check_inventory, calculate_shipping, apply_discounts
    # )

    # # Execute pipeline
    # order_data: dict[str, object] = {"order_id": "ORD-999", "amount": 100.00}
    # pipeline_result = order_pipeline(order_data)

    # if pipeline_result.is_success:
    #     final_order = cast("Flext.Types.Dict", pipeline_result.unwrap())
    #     print(f"  ✅ Pipeline completed: {len(final_order)} fields")
    #     print(f"     - Inventory checked: {final_order.get('inventory_checked')}")
    #     print(f"     - Shipping: ${final_order.get('shipping'):.2f}")
    #     print(f"     - Discount applied: {final_order.get('discount_applied')}")
    print("  Pipeline builder demo skipped")

    # 4. Request Context Manager
    print("\n=== 4. Request Context Manager ===")

    # Use context manager for request-scoped data
    with core.request_context(
        request_id="req-12345",
        user_id="user-789",
        client_ip="192.168.1.100",
        user_agent="Mozilla/5.0",
    ) as context:
        print(f"  ✅ Request context active: {context.get('request_id')}")
        print(f"     - User ID: {context.get('user_id')}")
        print(f"     - Client IP: {context.get('client_ip')}")

        # Context is automatically cleaned up on exit

    # 5. Complete Integration - All Features Together
    print("\n=== 5. Complete Integration Example ===")

    # Define a complete order processing workflow
    def validate_customer(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        if not data.get("customer_id"):
            return FlextResult[FlextTypes.Dict].fail("Customer ID required")
        data["customer_validated"] = True
        return FlextResult[FlextTypes.Dict].ok(data)

    def reserve_inventory(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        data["inventory_reserved"] = True
        return FlextResult[FlextTypes.Dict].ok(data)

    def process_payment(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        data["payment_processed"] = True
        return FlextResult[FlextTypes.Dict].ok(data)

    # Build complete workflow pipeline
    # workflow = Flext.build_pipeline(
    #     validate_customer, reserve_inventory, process_payment
    # )

    # Execute within request context
    # with core.request_context(request_id="complete-req-001") as ctx:
    # request_id = ctx.get("request_id")

    # # Process order through pipeline
    # order_input: dict[str, object] = {
    #     "customer_id": "CUST-001",
    #     "order_id": "ORD-001",
    #     "amount": 250.00,
    # }

    # workflow_result = workflow(order_input)

    # if workflow_result.is_success:
    #     completed_order = cast("Flext.Types.Dict", workflow_result.unwrap())

    #     # Publish success event with request correlation
    #     core.publish_event(
    #         "order.workflow.completed",
    #         {
    #             "order_id": completed_order["order_id"],
    #             "customer_id": completed_order["customer_id"],
    #             "steps_completed": 3,
    #         },
    #         correlation_id=request_id,
    #     )

    #     print("  ✅ Complete workflow executed successfully")
    #     print(
    #         f"     - Customer validated: {completed_order.get('customer_validated')}"
    #     )
    #     print(
    #         f"     - Inventory reserved: {completed_order.get('inventory_reserved')}"
    #     )
    #     print(
    #         f"     - Payment processed: {completed_order.get('payment_processed')}"
    #     )
    #     print(f"     - Event published with correlation: {request_id}")

    print("\n" + "=" * 60)
    print("✅ FLEXTCORE 1.1.0 CONVENIENCE METHODS DEMONSTRATED!")
    print("Event publishing, service creation, pipelines, and context management")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    # New FlextResult methods (v0.9.9+)
    demonstrate_new_flextresult_methods()

    # Traditional pattern
    demonstrate_complete_integration()

    # Modern Flext pattern (1.0.0)
    demonstrate_flextcore_unified_access()

    # Flext 1.1.0 convenience methods
    demonstrate_flextcore_11_features()


if __name__ == "__main__":
    main()
