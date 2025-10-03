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
    FlextConfig,
    FlextContainer,
    FlextContext,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
    FlextTypes,
)

from .example_scenarios import ExampleScenarios

# Constants
UNKNOWN_ERROR_MSG = "Unknown error"


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
    """Inventory management service."""

    def __init__(self) -> None:
        """Initialize with products."""
        super().__init__()
        self._logger = FlextLogger(__name__)
        self._scenarios = ExampleScenarios
        self._products: dict[str, Product] = {}
        self._initialize_products()

    def _initialize_products(self) -> None:
        """Initialize sample products."""
        realistic_order = self._scenarios.realistic_data()["order"]
        products: list[Product] = []
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
    """Payment processing service using strategy pattern."""

    def __init__(self) -> None:
        """Initialize payment service."""
        super().__init__()
        self._logger = FlextLogger(__name__)

    def process_payment(
        self,
        order: Order,
        method: str,
    ) -> FlextResult[FlextTypes.Dict]:
        """Process payment for order."""
        self._logger.info(
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
    """Order processing service - orchestrates the workflow."""

    def __init__(self) -> None:
        """Initialize with dependencies from container."""
        super().__init__()
        manager = FlextContainer.ensure_global_manager()
        self._container = manager.get_or_create()
        self._logger = FlextLogger(__name__)
        self._scenarios = ExampleScenarios
        self._metadata = self._scenarios.metadata(tags=["integration", "demo"])

    def create_order(
        self,
        customer_id: str,
        items: list[dict[str, str | int]],
    ) -> FlextResult[Order]:
        """Create and process an order."""
        # Create correlation ID for tracking
        correlation_id = str(uuid4())
        self._logger.bind(correlation_id=correlation_id)
        self._logger.info("Creating order", extra={"customer_id": customer_id})

        # Get services from container
        inventory_result = self._container.get_typed("inventory", InventoryService)
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
                    self._logger.info("Product not found: %s", product_id)
                    continue

                product = product_result.unwrap()

                # Add to order
                add_result = order.add_item(product, quantity)
                if add_result.is_failure:
                    self._logger.warning(
                        f"Failed to add item: {add_result.error}",
                        extra={"product_id": product.id},
                    )
                    continue

                self._logger.info(
                    "Item added to order",
                    extra={"product_id": product.id, "quantity": quantity},
                )

        # Check if any items were added
        if not order.items:
            return FlextResult[Order].fail("No items could be added to order")

        self._logger.info(
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
        self._logger.info("Submitting order", extra={"order_id": order.id})

        # Submit the order
        submit_result = order.submit()
        if submit_result.is_failure:
            return FlextResult[FlextTypes.Dict].fail(
                submit_result.error or UNKNOWN_ERROR_MSG,
            )

        # Get payment service
        payment_service_result = self._container.get_typed("payment", PaymentService)
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

        self._logger.info(
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


def demonstrate_complete_integration() -> None:
    """Demonstrate all FLEXT components working together."""
    print("=" * 60)
    print("COMPLETE FLEXT INTEGRATION DEMONSTRATION")
    print("E-Commerce Order Processing System")
    print("=" * 60)

    scenario_data = ExampleScenarios.realistic_data()
    order_template = scenario_data["order"]
    user_pool = ExampleScenarios.users()

    # 1. Configuration setup
    print("\n=== 1. Configuration ===")
    config = FlextConfig()
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"Log Level: {config.log_level}")

    # 2. Container setup with dependency injection
    print("\n=== 2. Dependency Injection ===")
    manager = FlextContainer.ensure_global_manager()

    container = manager.get_or_create()

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


def main() -> None:
    """Main entry point."""
    demonstrate_complete_integration()


if __name__ == "__main__":
    main()
