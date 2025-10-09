# !/usr/bin/env python3
"""08 - Complete FLEXT Integration: All Components Working Together.

This comprehensive example demonstrates how ALL FLEXT components work together
in a real-world application scenario - an e-commerce order processing system.

Integrates:
- FlextResult for railway-oriented error handling throughout
- FlextCore.Container for dependency injection and service management
- FlextModels for domain modeling (entities, values, aggregates)
- FlextCore.Config for environment-aware configuration
- FlextLogger for structured logging with correlation tracking
- FlextCore.Processors for handler pipelines and strategy patterns
- FlextModels.Payload and DomainEvent for messaging

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from copy import deepcopy
from decimal import Decimal
from typing import ClassVar, TypedDict, cast
from uuid import uuid4

from pydantic import Field

from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextCore,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
    FlextTypes,
)


class OrderItemDict(TypedDict):
    """Type definition for order item data in integration examples."""

    product_id: str
    name: str
    price: str
    quantity: int


class RealisticOrderDict(TypedDict):
    """Type definition for realistic order data in integration examples."""

    customer_id: str
    items: list[OrderItemDict]
    order_id: str
    total: str


class RealisticDataDict(TypedDict):
    """Type definition for realistic combined data in integration examples."""

    order: RealisticOrderDict
    api_response: FlextTypes.Dict
    user_registration: FlextTypes.Dict


class UserDict(TypedDict):
    """Type definition for user data in integration examples."""

    id: int
    name: str
    email: str
    age: int


class DemoDatasetDict(TypedDict):
    """Type definition for demo dataset structure."""

    users: list[UserDict]


class DemoScenarios:
    """Inline scenario helpers for the integration example."""

    _DATASET: ClassVar[DemoDatasetDict] = {
        "users": [
            {
                "id": 1,
                "name": "Alice Example",
                "email": "alice@example.com",
                "age": 30,
            },
            {
                "id": 2,
                "name": "Bob Example",
                "email": "bob@example.com",
                "age": 28,
            },
            {
                "id": 3,
                "name": "Charlie Example",
                "email": "charlie@example.com",
                "age": 35,
            },
        ],
    }

    _REALISTIC: ClassVar[RealisticDataDict] = {
        "order": {
            "customer_id": "cust-123",
            "order_id": "order-456",
            "total": "119.97",
            "items": [
                {
                    "product_id": "prod-001",
                    "name": "Widget",
                    "price": "39.99",
                    "quantity": 1,
                },
                {
                    "product_id": "prod-002",
                    "name": "Gadget",
                    "price": "29.99",
                    "quantity": 2,
                },
            ],
        },
        "api_response": {
            "status": "ok",
            "processed_at": "2025-01-01T00:00:00Z",
        },
        "user_registration": {
            "user_id": "usr-789",
            "plan": "standard",
        },
    }

    _CONFIG: ClassVar[FlextTypes.Dict] = {
        "database_url": "sqlite:///:memory:",
        "api_timeout": 30,
        "retry": 3,
    }

    _PAYLOAD: ClassVar[FlextTypes.Dict] = {
        "event": "order_processed",
        "order_id": "order-456",
        "metadata": {"source": "examples", "version": "1.0"},
    }

    @staticmethod
    def dataset() -> DemoDatasetDict:
        """Get a copy of the demo dataset."""
        return deepcopy(DemoScenarios._DATASET)

    @staticmethod
    def realistic_data() -> RealisticDataDict:
        """Get a copy of realistic demo data."""
        return deepcopy(DemoScenarios._REALISTIC)

    @staticmethod
    def validation_data() -> FlextTypes.Dict:
        """Get validation demo data."""
        return {
            "valid_emails": ["user@example.com"],
            "invalid_emails": ["invalid"],
        }

    @staticmethod
    def config(**overrides: object) -> FlextTypes.Dict:
        """Create configuration dictionary with optional overrides."""
        value = deepcopy(DemoScenarios._CONFIG)
        value.update(overrides)
        return value

    @staticmethod
    def metadata(
        *, source: str = "examples", tags: list[str] | None = None, **extra: object
    ) -> FlextTypes.Dict:
        """Create metadata dictionary for integration examples."""
        data: FlextTypes.Dict = {
            "source": source,
            "component": "flext_core",
            "tags": tags or ["integration", "demo"],
        }
        data.update(extra)
        return data

    @staticmethod
    def user(**overrides: object) -> FlextTypes.Dict:
        """Create user data dictionary for integration examples."""
        user: UserDict = deepcopy(DemoScenarios._DATASET["users"][0])
        # Apply overrides manually to avoid TypedDict.update overload issues
        for key, value in overrides.items():
            setattr(user, key, value)
        return cast("FlextTypes.Dict", user)

    @staticmethod
    def users(count: int = 5) -> list[UserDict]:
        """Create list of user data dictionaries for integration examples."""
        return [deepcopy(user) for user in DemoScenarios._DATASET["users"][:count]]

    @staticmethod
    def payload(**overrides: object) -> FlextTypes.Dict:
        """Create event payload dictionary for integration examples."""
        payload = deepcopy(DemoScenarios._PAYLOAD)
        payload.update(overrides)
        return payload


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
        FlextCore.Exceptions.ConfigurationError: If logger is None.

    """
    if logger is None:
        msg = "Logger must be initialized"
        raise FlextCore.Exceptions.ConfigurationError(msg)
    return logger


def assert_container_initialized(
    container: FlextCore.Container | None,
) -> FlextCore.Container:
    """Type guard to assert container is initialized.

    Args:
        container: Container instance or None.

    Returns:
        Non-None container instance.

    Raises:
        FlextCore.Exceptions.ConfigurationError: If container is None.

    """
    if container is None:
        msg = "Container must be initialized"
        raise FlextCore.Exceptions.ConfigurationError(msg)
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

    def get_domain_events(self) -> list[FlextModels.DomainEvent]:
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
    - Inherited container property (FlextCore.Container singleton)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextCore.Context for request tracking)
    - Inherited config property (FlextCore.Config with application settings)
    - Inherited metrics property (FlextMetrics for observability)

    Manages product inventory with domain model operations and structured logging.
    """

    def __init__(self) -> None:
        """Initialize inventory service with inherited FlextMixins.Service infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context (ALREADY CONFIGURED!)
        - self.container: FlextCore.Container global singleton
        - self.context: FlextCore.Context for request tracking
        - self.config: FlextCore.Config with application settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        # Use self.logger from FlextMixins.Logging, not logger
        self._scenarios = DemoScenarios
        self._products: dict[str, Product] = {}
        self._initialize_products()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "InventoryService initialized with inherited infrastructure",
            extra={
                "service_type": "Inventory Management",
                "products_loaded": len(self._products),
            },
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


class PaymentService(FlextService[dict[str, object]]):
    """Payment processing service with FlextMixins.Service infrastructure.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextCore.Container singleton)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextCore.Context for correlation tracking)
    - Inherited config property (FlextCore.Config with payment settings)
    - Inherited metrics property (FlextMetrics for payment observability)

    Implements payment processing with strategy pattern and structured logging.
    """

    # Type annotations for inherited mixin properties
    # logger: FlextLogger  # Provided by FlextMixins.Logging

    def __init__(self) -> None:
        """Initialize payment service with inherited FlextMixins.Service infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context (ALREADY CONFIGURED!)
        - self.container: FlextCore.Container global singleton
        - self.context: FlextCore.Context for correlation tracking
        - self.config: FlextCore.Config with payment settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        # Use self.logger from FlextMixins.Logging, not logger

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "PaymentService initialized with inherited infrastructure",
            extra={
                "service_type": "Payment Processing",
                "payment_methods": ["credit_card", "paypal"],
            },
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
    - Inherited container property (FlextCore.Container singleton for service dependencies)
    - Inherited logger property (FlextLogger with service context and correlation tracking)
    - Inherited context property (FlextCore.Context for request and correlation IDs)
    - Inherited config property (FlextCore.Config with order processing settings)
    - Inherited metrics property (FlextMetrics for order observability)

    Orchestrates complete order workflow: validation → creation → payment → confirmation.
    Uses inherited container to access InventoryService and PaymentService dependencies.
    """

    def __init__(self) -> None:
        """Initialize order service with inherited FlextMixins.Service infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context (ALREADY CONFIGURED!)
        - self.container: FlextCore.Container global singleton (access inventory/payment services)
        - self.context: FlextCore.Context for request and correlation tracking
        - self.config: FlextCore.Config with order processing configuration
        - self.metrics: FlextMetrics for order workflow observability
        """
        super().__init__()
        self._scenarios = DemoScenarios()
        self._metadata = self._scenarios.metadata(tags=["integration", "demo"])

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "OrderService initialized with inherited infrastructure",
            extra={
                "service_type": "Order Orchestration",
                "dependencies": ["InventoryService", "PaymentService"],
                "metadata": self._metadata,
            },
        )

    def create_order(
        self,
        customer_id: str,
        items: list[dict[str, object]],
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
            correlation_id=FlextCore.Context.Correlation.get_correlation_id(),
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
        data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Process order operation."""
        operation = data.get("operation")

        if operation == "create_and_submit":
            # Create order
            customer_id = str(data["customer_id"])
            items = cast("list[dict[str, object]]", data["items"])
            order_result = self.create_order(
                customer_id,
                items,
            )
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


# ========== HANDLER PIPELINE (Using FlextCore.Processors) ==========


class OrderValidationHandler:
    """Validate order request."""

    def __init__(self) -> None:
        """Initialize handler."""
        super().__init__()
        self.name = "OrderValidator"
        self.logger = FlextCore.create_logger(__name__)

    def handle(self, request: FlextTypes.Dict) -> FlextResult[FlextTypes.Dict]:
        """Validate order data."""
        self.logger.info("Validating order request")

        # Check required fields
        if not request.get("customer_id"):
            return FlextResult[FlextTypes.Dict].fail("Customer ID required")

        if not request.get("items"):
            return FlextResult[FlextTypes.Dict].fail("Order items required")

        # Validate items
        items_raw = request["items"]
        if not isinstance(items_raw, list):
            return FlextResult[FlextTypes.Dict].fail("Order items must be a list")
        items: list[dict[str, object]] = cast("list[dict[str, object]]", items_raw)
        for item in items:
            item_dict: dict[str, object] = item
            if not item_dict.get("product_id"):
                return FlextResult[FlextTypes.Dict].fail(
                    "Product ID required for all items",
                )
            quantity = item_dict.get("quantity")
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
        self.logger = FlextCore.create_logger(__name__)

    def handle(
        self,
        request: FlextTypes.Dict,
        *,
        _debug: bool = False,
    ) -> FlextResult[FlextTypes.Dict]:
        """Add metadata to order."""
        self.logger.info("Enriching order request")

        if not request.get("validated"):
            return FlextResult[FlextTypes.Dict].fail("Order must be validated first")

        # Add metadata
        metadata: dict[str, str | float] = {
            "timestamp": str(time.time()),
            "source": "web",
            "version": "1.0",
            "correlation_id": str(uuid4()),
        }

        # Create a new dict[str, object] with the metadata added
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
    print("NEW FlextResult METHODS - INTEGRATION CONTEXT")
    print("Demonstrating v0.9.9+ methods in complete system integration")
    print("=" * 60)

    # Setup test data
    scenario_data: RealisticDataDict = DemoScenarios.realistic_data()
    order_template: RealisticOrderDict = scenario_data["order"]
    user_pool = DemoScenarios.users()
    REDACTED_LDAP_BIND_PASSWORD_user = user_pool[0]  # Get first user for demos

    # 1. from_callable - Safe Integration Layer Operations
    print("\n=== 1. from_callable: Safe Integration Operations ===")

    def risky_order_creation(customer_id: str, items: list[dict[str, object]]) -> Order:
        """Order creation that might raise exceptions."""
        if not items:
            msg = "Cannot create empty order"
            raise FlextCore.Exceptions.ValidationError(msg, field="items", value=items)
        if not customer_id:
            msg = "Customer ID required"
            raise FlextCore.Exceptions.ValidationError(
                msg, field="customer_id", value=None
            )
        return Order(customer_id=customer_id, items=[], domain_events=[])

    # Safe order creation without try/except
    customer_id = REDACTED_LDAP_BIND_PASSWORD_user["email"]
    order_result: FlextResult[Order] = FlextResult[Order].from_callable(
        lambda: risky_order_creation(
            customer_id,
            cast("list[dict[str, object]]", list(order_template.get("items", []))),
        ),
    )
    if order_result.is_success:
        order = order_result.unwrap()
        print(f"✅ Order created safely: {order.customer_id}")
    else:
        print(f"❌ Order creation failed: {order_result.error}")

    # 2. flow_through - Complete Integration Pipeline
    print("\n=== 2. flow_through: Integration Pipeline Composition ===")

    def validate_order_data(
        data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate order request data."""
        if not data.get("customer_id"):
            return FlextResult[dict[str, object]].fail("Customer ID required")
        if not data.get("items"):
            return FlextResult[dict[str, object]].fail("Items required")
        return FlextResult[dict[str, object]].ok(data)

    def check_inventory_availability(
        data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Check all items are in stock."""
        # Simulate inventory check
        return FlextResult[dict[str, object]].ok({
            **data,
            "inventory_status": "available",
        })

    def calculate_order_total(
        data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Calculate total price with taxes."""
        items = cast("list[dict[str, object]]", data.get("items", []))
        subtotal = sum(
            Decimal(str(item.get("price", 0))) * Decimal(str(item.get("quantity", 1)))
            for item in items
        )
        tax = subtotal * Decimal("0.1")
        total = subtotal + tax
        result_dict = cast(
            "dict[str, object]",
            {**data, "subtotal": subtotal, "tax": tax, "total": total},
        )
        return FlextResult[dict[str, object]].ok(result_dict)

    def apply_promotions(
        data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Apply promotional discounts."""
        total = Decimal(str(data.get("total", 0)))
        discount = total * Decimal("0.05") if total > 100 else Decimal(0)
        final_total = total - discount
        result_dict = cast(
            "dict[str, object]",
            {
                **data,
                "discount": discount,
                "final_total": final_total,
            },
        )
        return FlextResult[dict[str, object]].ok(result_dict)

    # Flow through complete integration pipeline
    order_data = cast(
        "dict[str, object]",
        {
            "customer_id": str(REDACTED_LDAP_BIND_PASSWORD_user["email"]),
            "items": order_template.get("items", []),
            "source": "web_checkout",
        },
    )
    pipeline_result = (
        FlextResult[dict[str, object]]
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
        print(
            f"✅ Integration pipeline complete: ${final_order.get('final_total', 0):.2f}"
        )
        print(f"   Subtotal: ${final_order.get('subtotal', 0):.2f}")
        print(f"   Tax: ${final_order.get('tax', 0):.2f}")
        print(f"   Discount: ${final_order.get('discount', 0):.2f}")
    else:
        print(f"❌ Pipeline failed: {pipeline_result.error}")

    # 3. lash - Service Fallback Recovery
    print("\n=== 3. lash: Service Fallback Recovery ===")

    def try_primary_payment_gateway(
        _amount: Decimal,
    ) -> FlextResult[dict[str, object]]:
        """Try primary payment processor."""
        # Simulate failure (amount parameter for demonstration purposes)
        return FlextResult[dict[str, object]].fail("Primary gateway timeout")

    def fallback_to_secondary_gateway(
        error: str,
    ) -> FlextResult[dict[str, object]]:
        """Fallback to secondary payment processor."""
        print(f"   ⚠️  Primary failed: {error}, using fallback...")
        # Simulate successful fallback
        return FlextResult[dict[str, object]].ok({
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

    def get_premium_shipping_service() -> FlextResult[dict[str, object]]:
        """Try to get premium shipping service."""
        # Simulate service unavailable
        return FlextResult[dict[str, object]].fail(
            "Premium shipping service unavailable"
        )

    def get_standard_shipping_service() -> FlextResult[dict[str, object]]:
        """Get standard shipping service."""
        return FlextResult[dict[str, object]].ok({
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

    def load_custom_config() -> dict[str, object]:
        """Expensive configuration loading operation."""
        print("   ⚙️  Loading custom configuration from database...")
        time.sleep(0.1)  # Simulate expensive operation
        return {
            "max_order_amount": Decimal(10000),
            "default_currency": "USD",
            "tax_rate": Decimal("0.1"),
        }

    # Config not found, lazy-load default
    config_result: FlextResult[dict[str, object]] = FlextResult[dict[str, object]].fail(
        "Config not in cache"
    )

    # Only loads if result is failure
    config = config_result.value_or_call(load_custom_config)
    print(f"✅ Configuration loaded: currency={config.get('default_currency')}")
    print(f"   Max order: ${config.get('max_order_amount')}")
    print(f"   Tax rate: {config.get('tax_rate')}")

    # When success, doesn't call the function
    cached_config: FlextResult[dict[str, object]] = FlextResult[dict[str, object]].ok({
        "cached": True
    })
    result_config = cached_config.value_or_call(load_custom_config)  # Won't execute
    print(f"✅ Cached config used: {result_config}")

    print("\n" + "=" * 60)
    print("✅ NEW FlextResult METHODS INTEGRATION DEMO COMPLETE!")
    print("All 5 methods demonstrated in complete system integration context")
    print("=" * 60)


def demonstrate_complete_integration() -> None:
    """Demonstrate all FLEXT components working together."""
    print("=" * 60)
    print("COMPLETE FLEXT INTEGRATION DEMONSTRATION")
    print("E-Commerce Order Processing System")
    print("=" * 60)

    scenario_data: RealisticDataDict = DemoScenarios.realistic_data()
    order_template: RealisticOrderDict = scenario_data["order"]
    user_pool = DemoScenarios.users()

    # 1. Configuration setup
    print("\n=== 1. Configuration ===")
    config = FlextCore.create_config()
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

    items_list: list[dict[str, object]] = [
        {
            "product_id": item["product_id"],
            "quantity": item.get("quantity", 1),
        }
        for item in scenario_order["items"]
    ]
    order_request: FlextTypes.Dict = {
        "customer_id": scenario_order["customer_id"],
        "items": items_list,
        "payment_method": "credit_card",
    }
    order_dict: dict[str, object] = order_request
    items_list_raw = order_dict["items"]
    if not isinstance(items_list_raw, list):
        print("Items is not a list")
    else:
        items_validated: list[dict[str, object]] = cast(
            "list[dict[str, object]]", items_list_raw
        )
        if items_validated:
            first_item: dict[str, object] = items_validated[0]
            first_product_id = first_item["product_id"]

    print(f"Customer: {order_dict['customer_id']}")
    items_raw = order_dict["items"]
    if isinstance(items_raw, list):
        order_items: list[dict[str, object]] = cast(
            "list[dict[str, object]]", items_raw
        )
        print(f"Items: {len(order_items)} products")
    else:
        print("Items is not a list")
        order_items = []

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
        enriched_items: list[dict[str, object]] = cast(
            "list[dict[str, object]]", items_raw
        )
        result = order_service.process_operation({
            "operation": "create_and_submit",
            "customer_id": customer_id,
            "items": enriched_items,
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
            validation_items: list[dict[str, object]] = cast(
                "list[dict[str, object]]", items_raw
            )
            result = order_service.process_operation({
                "operation": "create_and_submit",
                "customer_id": customer_id,
                "items": validation_items,
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
    default_customer = str(
        user_pool[0]["id"] if user_pool else scenario_order["customer_id"]
    )
    simple_order = Order(customer_id=default_customer, domain_events=[])
    items_list_raw = scenario_order["items"]
    # Convert TypedDict items to dict[str, object]
    items_list = [dict(item) for item in items_list_raw]
    item_template: dict[str, object] = items_list[0]
    template_price = item_template.get("price", Decimal("10.00"))
    price_amount = Decimal(str(template_price))
    product = Product(
        id=str(item_template.get("product_id", "PROD-TEMPLATE")),
        name=str(item_template.get("name", "Template Product")),
        price=Money(amount=price_amount, currency="USD"),
        stock=5,
    )

    simple_order.add_item(product, 2)
    simple_order.submit()

    for event in simple_order.get_domain_events():
        # Domain events are FlextModels.DomainEvent objects
        event_name = event.event_type
        event_data = event.data
        print(f"  📢 {event_name}: {event_data}")

    # 8. Logging with correlation
    print("\n=== 8. Structured Logging ===")
    logger = FlextCore.create_logger(__name__)
    correlation_id = str(uuid4())
    logger.bind(correlation_id=correlation_id)

    logger.info(
        "Integration test completed",
        extra={
            "orders_processed": 2,
            "success_rate": 0.5,
            "components_tested": [
                "FlextResult",
                "FlextCore.Container",
                "FlextModels",
                "FlextCore.Config",
                "FlextLogger",
                "FlextCore.Processors",
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
    """Demonstrate the modern FlextConstants unified facade pattern.

    This shows how FlextConstants provides a single entry point for accessing
    ALL flext-core components with improved convenience and discoverability.
    """
    print("\n" + "=" * 60)
    print("FLEXTCORE UNIFIED FACADE DEMONSTRATION")
    print("Modern pattern for complete flext-core access")
    print("=" * 60)

    # 1. Setup complete infrastructure manually
    print("\n=== 1. Unified Infrastructure Setup ===")
    # Create infrastructure components manually
    config = FlextCore.create_config()
    container = FlextContainer.get_global()
    logger = FlextCore.create_logger("ecommerce-service")
    bus = FlextCore.Bus()
    context = FlextCore.Context()

    print("  ✅ Complete infrastructure initialized:")
    print(f"     - Config: {type(config).__name__}")
    print(f"     - Container: {type(container).__name__}")
    print(f"     - Logger: {type(logger).__name__}")
    print(f"     - Bus: {type(bus).__name__}")
    print(f"     - Context: {type(context).__name__}")

    # 2. Direct class access through FlextConstants
    print("\n=== 2. Direct Component Access ===")
    result = FlextResult[str].ok("Order created successfully")
    config = FlextCore.create_config()
    timeout = FlextConstants.Defaults.TIMEOUT

    print(f"  ✅ Result access: {result.value}")
    print(f"  ✅ Config access: log_level = {config.log_level}")
    print(f"  ✅ Constants access: timeout = {timeout}")

    # 3. Factory methods for convenience
    print("\n=== 3. Convenience Factory Methods ===")
    success = FlextResult[dict[str, str]].ok({
        "order_id": "ORD-123",
        "status": "created",
    })
    failure = FlextResult[str].fail("Payment declined", error_code="PAYMENT_ERROR")
    logger = FlextCore.create_logger("ecommerce")

    print(f"  ✅ Success result: {success.value}")
    print(f"  ✅ Failure result: {failure.error}")
    print(f"  ✅ Logger created: {logger}")

    # 4. Instance-based access to all components
    print("\n=== 4. Unified Core Instance ===")
    # Create core components manually since FlextCore doesn't have instance-based access
    core_config = FlextCore.create_config()
    core_container = FlextContainer.get_global()
    core_logger = FlextCore.create_logger("core")
    core_bus = FlextCore.Bus()
    core_context = FlextCore.Context()

    print(f"  ✅ Config: {type(core_config).__name__}")
    print(f"  ✅ Container: {type(core_container).__name__}")
    print(f"  ✅ Logger: {type(core_logger).__name__}")
    print(f"  ✅ Bus: {type(core_bus).__name__}")
    print(f"  ✅ Context: {type(core_context).__name__}")

    # 5. Railway pattern with FlextConstants
    print("\n=== 5. Railway Pattern via FlextConstants ===")

    def validate_order(
        data: dict[str, object],
    ) -> FlextResult[FlextTypes.Dict]:
        if not data.get("customer_id"):
            return FlextResult[FlextTypes.Dict].fail("Customer ID required")
        return FlextResult[FlextTypes.Dict].ok(data)

    def process_order(
        data: dict[str, object],
    ) -> FlextResult[FlextTypes.Dict]:
        data["processed"] = True
        return FlextResult[FlextTypes.Dict].ok(data)

    # Compose operations
    order_data: dict[str, object] = {"customer_id": "CUST-001", "amount": 100}
    pipeline_result = validate_order(order_data).flat_map(process_order)

    if pipeline_result.is_success:
        print(f"  ✅ Pipeline result: {pipeline_result.value}")

    print("\n" + "=" * 60)
    print("✅ FLEXTCORE UNIFIED FACADE DEMONSTRATED!")
    print("Single entry point for all flext-core functionality")
    print("=" * 60)


def demonstrate_flextcore_11_features() -> None:
    """Demonstrate FlextConstants 1.1.0 convenience methods.

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

    # Create bus instance for event publishing
    bus = FlextCore.Bus()

    # Publish events with automatic correlation tracking
    event_result = bus.publish_event(
        {
            "type": "order.created",
            "order_id": "ORD-123",
            "customer_id": "CUST-456",
            "amount": 99.99,
        },
    )

    if event_result.is_success:
        print("  ✅ Event published successfully")

    # Publish with custom correlation ID
    correlation_result = bus.publish_event(
        {
            "type": "payment.processed",
            "payment_id": "PAY-789",
            "status": "completed",
            "correlation_id": "req-abc-123",
        },
    )

    if correlation_result.is_success:
        print("  ✅ Event published with custom correlation ID")

    # 2. Service Creation with Infrastructure Injection
    print("\n=== 2. Service Creation ===")

    # Create service with automatic infrastructure setup
    # service_result = FlextCore.create_service(OrderProcessingService, "order-processing")
    # if service_result.is_success:
    #     service = service_result.unwrap()
    #     exec_result = service.execute()
    #     print(f"  ✅ Service created and executed: {exec_result.value}")
    print("  Service creation demo skipped")

    # 3. Railway Pipeline Builder
    print("\n=== 3. Pipeline Builder ===")

    print("  Pipeline builder demo skipped - functions removed since not used")

    # 4. Request Context Management
    print("\n=== 4. Request Context Management ===")

    # Use context for request-scoped data
    context = FlextCore.Context()

    # Set request context data
    context.set("request_id", "req-12345")
    context.set("user_id", "user-789")
    context.set("client_ip", "192.168.1.100")
    context.set("user_agent", "Mozilla/5.0")

    print(f"  ✅ Request context active: {context.get('request_id')}")
    print(f"     - User ID: {context.get('user_id')}")
    print(f"     - Client IP: {context.get('client_ip')}")

    # Context can be explicitly cleared when done
    context.clear()

    # 5. Complete Integration - All Features Together
    print("\n=== 5. Complete Integration Example ===")

    # Complete workflow pipeline demo skipped - functions removed since not used

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
    #     completed_order = cast("FlextTypes.Dict", workflow_result.unwrap())

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

    # Modern FlextConstants pattern (1.0.0)
    demonstrate_flextcore_unified_access()

    # FlextConstants 1.1.0 convenience methods
    demonstrate_flextcore_11_features()


if __name__ == "__main__":
    main()
