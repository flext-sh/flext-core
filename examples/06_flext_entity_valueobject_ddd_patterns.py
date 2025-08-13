#!/usr/bin/env python3
"""Domain-Driven Design patterns with FlextEntity.

Demonstrates entity lifecycle management, value objects,
domain events, and aggregate patterns.
- Version management and optimistic locking
- Entity lifecycle operations with state transitions
- Aggregate patterns and bounded contexts
- Repository patterns for data persistence simulation
- Domain services for complex business logic
- Entity relationships and object composition
- Performance characteristics and optimization
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from flext_core import (
    FlextEntity,
    FlextResult,
    FlextUtilities,
)

from .shared_domain import (
    Address,
    Age,
    EmailAddress as Email,
    Money,
    Order,
    OrderItem,
    User,
    User as SharedUser,
    UserStatus,
)
from .shared_example_helpers import run_example_demonstration

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    class CustomerFactory(Protocol):
        """Factory protocol for creating `Customer` instances."""

        def __call__(self, **kwargs: object) -> FlextResult[Customer]:
            """Create a new `Customer` from keyword arguments."""
            ...

    class ProductFactory(Protocol):
        """Factory protocol for creating `Product` instances."""

        def __call__(self, **kwargs: object) -> FlextResult[Product]:
            """Create a new `Product` from keyword arguments."""
            ...

    class OrderFactory(Protocol):
        """Factory protocol for creating `Order` instances."""

        def __call__(self, **kwargs: object) -> FlextResult[Order]:
            """Create a new `Order` from keyword arguments."""
            ...

# =============================================================================
# DDD VALIDATION CONSTANTS - Domain validation constraints
# =============================================================================

# Currency and address validation constants
CURRENCY_CODE_LENGTH = 3  # Standard ISO 4217 currency code length
MIN_STREET_ADDRESS_LENGTH = 5  # Minimum characters for street address
MIN_CITY_NAME_LENGTH = 2  # Minimum characters for city name
MIN_POSTAL_CODE_LENGTH = 3  # Minimum characters for postal code
MIN_COUNTRY_NAME_LENGTH = 2  # Minimum characters for country name

# Email validation constants
EMAIL_PARTS_COUNT = 2  # Number of parts in email (local@domain)

# Name validation constants
MIN_NAME_LENGTH = 2  # Minimum characters for names
MAX_CUSTOMER_NAME_LENGTH = 100  # Maximum characters for customer name
MAX_COMPANY_NAME_LENGTH = 200  # Maximum characters for company name

# Reason validation constants
MIN_REASON_LENGTH = 5  # Minimum characters for general reasons
MIN_STATUS_CHANGE_REASON_LENGTH = 10  # Minimum characters for status change reasons
MIN_CANCELLATION_REASON_LENGTH = 10  # Minimum characters for cancellation reasons

# Tracking validation constants
MIN_TRACKING_NUMBER_LENGTH = 5  # Minimum characters for tracking number

# Financial validation constants
MAX_CREDIT_LIMIT = 100000  # Maximum credit limit amount
CURRENCY_PRECISION_TOLERANCE = 0.01  # Tolerance for currency amount comparisons

# =============================================================================
# ENHANCED DOMAIN MODELS - Using shared domain with DDD patterns
# =============================================================================


# =============================================================================
# DOMAIN ENTITIES - Rich domain models with business logic
# =============================================================================


class Customer(SharedUser):
    """Enhanced customer entity using shared domain models."""

    registration_date: str
    credit_limit: Money
    total_orders: int = 0

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate customer business rules."""
        # First validate the inherited shared domain rules
        base_validation = super().validate_domain_rules()
        if base_validation.is_failure:
            return base_validation

        # Validate credit limit
        credit_validation = self.credit_limit.validate_business_rules()
        if credit_validation.is_failure:
            return FlextResult.fail(
                f"Credit limit validation failed: {credit_validation.error}",
            )

        # Business rule: Credit limit must be reasonable
        if self.credit_limit.amount > MAX_CREDIT_LIMIT:
            return FlextResult.fail("Credit limit cannot exceed 100,000")

        # Business rule: Total orders cannot be negative
        if self.total_orders < 0:
            return FlextResult.fail("Total orders cannot be negative")

        return FlextResult.ok(None)

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate customer domain rules."""
        return self.validate_business_rules()

    @property
    def is_active(self) -> bool:
        """Check if customer is active based on status."""
        return self.status == UserStatus.ACTIVE

    def activate(self) -> FlextResult[User]:
        """Activate customer account."""
        if self.is_active:
            return FlextResult.fail("Customer is already active")

        # Create new version with activation
        try:
            activated_customer = self.model_copy(update={"status": UserStatus.ACTIVE})
        except Exception as e:
            return FlextResult.fail(f"Failed to create activated customer: {e}")

        # Add domain event
        activated_customer.add_domain_event(
            "CustomerActivated",
            {
                "customer_id": self.id,
                "activation_date": FlextUtilities.generate_iso_timestamp(),
                "previous_status": "inactive",
            },
        )

        return FlextResult.ok(cast("User", activated_customer))

    def deactivate(self, reason: str) -> FlextResult[Customer]:
        """Deactivate customer account using structured validation."""
        # Run all deactivation checks in sequence
        validation_methods: list[Callable[[], FlextResult[None]]] = [
            self._validate_customer_is_active_for_deactivation,
            lambda: self._validate_deactivation_reason(reason),
        ]

        for validation_method in validation_methods:
            result = validation_method()
            if result.is_failure:
                return FlextResult.fail(result.error or "Validation failed")

        # Execute deactivation process
        return self._create_deactivated_customer().flat_map(
            lambda customer: self._add_deactivation_event(customer, reason)
        )

    def _validate_customer_is_active_for_deactivation(self) -> FlextResult[None]:
        """Validate customer is active and can be deactivated."""
        if not self.is_active:
            return FlextResult.fail("Customer is already inactive")
        return FlextResult.ok(None)

    def _validate_deactivation_reason(self, reason: str) -> FlextResult[None]:
        """Validate deactivation reason meets requirements."""
        if not reason or len(reason.strip()) < MIN_STATUS_CHANGE_REASON_LENGTH:
            return FlextResult.fail(
                "Deactivation reason must be at least 10 characters",
            )
        return FlextResult.ok(None)

    def _create_deactivated_customer(self) -> FlextResult[Customer]:
        """Create new customer instance with inactive status."""
        try:
            deactivated_customer = self.model_copy(
                update={"status": UserStatus.INACTIVE}
            )
            return FlextResult.ok(deactivated_customer)
        except Exception as e:
            return FlextResult.fail(f"Failed to create deactivated customer: {e}")

    def _add_deactivation_event(
        self, customer: Customer, reason: str
    ) -> FlextResult[Customer]:
        """Add domain event for customer deactivation."""
        customer.add_domain_event(
            "CustomerDeactivated",
            {
                "customer_id": self.id,
                "deactivation_date": FlextUtilities.generate_iso_timestamp(),
                "reason": reason,
                "previous_status": "active",
            },
        )

        return FlextResult.ok(customer)

    def update_address(self, new_address: Address) -> FlextResult[Customer]:
        """Update customer address using railway-oriented programming."""
        return (
            self._validate_customer_active_for_address_update()
            .flat_map(lambda _: self._validate_new_address(new_address))
            .flat_map(lambda _: self._create_customer_with_updated_address(new_address))
            .flat_map(
                lambda customer: self._add_address_update_event(customer, new_address)
            )
        )

    def _validate_customer_active_for_address_update(self) -> FlextResult[None]:
        """Validate customer is active for address updates."""
        if not self.is_active:
            return FlextResult.fail("Cannot update address for inactive customer")
        return FlextResult.ok(None)

    def _validate_new_address(self, new_address: Address) -> FlextResult[None]:
        """Validate new address meets domain requirements."""
        validation = new_address.validate_business_rules()
        if validation.is_failure:
            return FlextResult.fail(
                f"New address validation failed: {validation.error}",
            )
        return FlextResult.ok(None)

    def _create_customer_with_updated_address(
        self, new_address: Address
    ) -> FlextResult[Customer]:
        """Create new customer instance with updated address."""
        try:
            updated_customer = self.model_copy(update={"address": new_address})
            return FlextResult.ok(updated_customer)
        except Exception as e:
            return FlextResult.fail(f"Failed to create updated customer: {e}")

    def _add_address_update_event(
        self, customer: Customer, new_address: Address
    ) -> FlextResult[Customer]:
        """Add domain event for address update."""
        customer.add_domain_event(
            "CustomerAddressUpdated",
            {
                "customer_id": self.id,
                "old_address": str(self.address),
                "new_address": str(new_address),
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        return FlextResult.ok(customer)

    def increase_credit_limit(self, amount: Money) -> FlextResult[Customer]:
        """Increase customer credit limit using railway-oriented programming.

        REFACTORED: Eliminated 9 returns using FlextResult chain pattern.
        Applied SOLID principles with single responsibility validation methods.
        """
        return (
            self._validate_customer_active()
            .flat_map(lambda _: self._validate_currency_match(amount))
            .flat_map(lambda _: self._calculate_new_credit_limit(amount))
            .flat_map(self._validate_credit_limit_maximum)
            .flat_map(self._create_updated_customer)
            .flat_map(self._add_credit_limit_event)
        )

    def _validate_customer_active(self) -> FlextResult[None]:
        """Validate customer is active for credit limit operations."""
        if not self.is_active:
            return FlextResult.fail(
                "Cannot increase credit limit for inactive customer"
            )
        return FlextResult.ok(None)

    def _validate_currency_match(self, amount: Money) -> FlextResult[None]:
        """Validate currency matches current credit limit."""
        if self.credit_limit.currency != amount.currency:
            return FlextResult.fail(
                f"Currency mismatch: {self.credit_limit.currency} vs {amount.currency}"
            )
        return FlextResult.ok(None)

    def _calculate_new_credit_limit(self, amount: Money) -> FlextResult[Money]:
        """Calculate new credit limit safely."""
        new_limit_result = self.credit_limit.add(amount)
        if new_limit_result.is_failure:
            return FlextResult.fail(
                f"Failed to calculate new credit limit: {new_limit_result.error}"
            )

        new_limit = new_limit_result.data
        if new_limit is None:
            return FlextResult.fail("Failed to calculate new credit limit")

        return FlextResult.ok(new_limit)

    def _validate_credit_limit_maximum(self, new_limit: Money) -> FlextResult[Money]:
        """Validate new credit limit doesn't exceed maximum."""
        if new_limit.amount > MAX_CREDIT_LIMIT:
            return FlextResult.fail(
                "Credit limit increase would exceed maximum of 100,000"
            )
        return FlextResult.ok(new_limit)

    def _create_updated_customer(self, new_limit: Money) -> FlextResult[Customer]:
        """Create updated customer with new credit limit."""
        try:
            updated_customer = self.model_copy(update={"credit_limit": new_limit})
            return FlextResult.ok(updated_customer)
        except Exception as e:
            return FlextResult.fail(f"Failed to create updated customer: {e}")

    def _add_credit_limit_event(
        self, updated_customer: Customer
    ) -> FlextResult[Customer]:
        """Add domain event for credit limit increase."""
        updated_customer.add_domain_event(
            "CustomerCreditLimitIncreased",
            {
                "customer_id": self.id,
                "old_limit": str(self.credit_limit),
                "new_limit": str(updated_customer.credit_limit),
                "increase_amount": str(
                    updated_customer.credit_limit.amount - self.credit_limit.amount
                ),
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        return FlextResult.ok(updated_customer)

    def increment_order_count(self) -> FlextResult[Customer]:
        """Increment total order count."""
        if not self.is_active:
            return FlextResult.fail("Cannot increment orders for inactive customer")

        # Create new version with incremented order count
        try:
            updated_customer = self.model_copy(
                update={"total_orders": self.total_orders + 1}
            )
        except Exception as e:
            return FlextResult.fail(f"Failed to create updated customer: {e}")

        # Add domain event
        updated_customer.add_domain_event(
            "CustomerOrderCountIncremented",
            {
                "customer_id": self.id,
                "old_count": self.total_orders,
                "new_count": self.total_orders + 1,
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        return FlextResult.ok(updated_customer)


class Product(FlextEntity):
    """Product entity with inventory and pricing logic."""

    name: str
    price: Money
    category: str
    stock_quantity: int
    is_available: bool = True
    minimum_stock: int = 10

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate product business rules using structured validation."""
        # Run all validation checks in sequence
        validation_methods = [
            self._validate_product_name,
            self._validate_product_price,
            self._validate_stock_quantities,
            self._validate_product_category,
        ]

        for validation_method in validation_methods:
            result = validation_method()
            if result.is_failure:
                return result

        return FlextResult.ok(None)

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate product domain rules."""
        return self.validate_business_rules()

    def _validate_product_name(self) -> FlextResult[None]:
        """Validate product name requirements."""
        if not self.name or len(self.name.strip()) < MIN_NAME_LENGTH:
            return FlextResult.fail("Product name must be at least 2 characters")

        if len(self.name) > MAX_COMPANY_NAME_LENGTH:
            return FlextResult.fail("Product name cannot exceed 200 characters")

        return FlextResult.ok(None)

    def _validate_product_price(self) -> FlextResult[None]:
        """Validate product price requirements."""
        # Validate price domain rules first
        price_validation = self.price.validate_business_rules()
        if price_validation.is_failure:
            return FlextResult.fail(
                f"Price validation failed: {price_validation.error}",
            )

        # Business rule: Price must be positive
        if self.price.amount <= 0:
            return FlextResult.fail("Product price must be positive")

        return FlextResult.ok(None)

    def _validate_stock_quantities(self) -> FlextResult[None]:
        """Validate stock quantity requirements."""
        if self.stock_quantity < 0:
            return FlextResult.fail("Stock quantity cannot be negative")

        if self.minimum_stock < 0:
            return FlextResult.fail("Minimum stock cannot be negative")

        return FlextResult.ok(None)

    def _validate_product_category(self) -> FlextResult[None]:
        """Validate product category requirements."""
        valid_categories = ["electronics", "clothing", "books", "home", "sports"]
        if self.category not in valid_categories:
            return FlextResult.fail(
                f"Invalid category. Must be one of: {valid_categories}",
            )

        return FlextResult.ok(None)

    def is_low_stock(self) -> bool:
        """Check if product is low on stock."""
        return self.stock_quantity <= self.minimum_stock

    def is_out_of_stock(self) -> bool:
        """Check if product is out of stock."""
        return self.stock_quantity == 0

    def update_price(self, new_price: Money) -> FlextResult[Product]:
        """Update product price using railway-oriented programming pattern."""
        # Railway-oriented programming: Chain all validations and operations
        return (
            self._validate_product_available()
            .flat_map(lambda _: self._validate_new_price(new_price))
            .flat_map(lambda _: self._validate_price_currency_match(new_price))
            .flat_map(lambda _: self._create_updated_product_with_price(new_price))
            .flat_map(lambda product: self._add_price_update_event(product, new_price))
        )

    def _validate_product_available(self) -> FlextResult[None]:
        """Validate that product is available for price updates."""
        if not self.is_available:
            return FlextResult.fail("Cannot update price for unavailable product")
        return FlextResult.ok(None)

    def _validate_new_price(self, new_price: Money) -> FlextResult[None]:
        """Validate the new price meets business rules."""
        validation = new_price.validate_business_rules()
        if validation.is_failure:
            return FlextResult.fail(f"New price validation failed: {validation.error}")

        if new_price.amount <= 0:
            return FlextResult.fail("New price must be positive")

        return FlextResult.ok(None)

    def _validate_price_currency_match(self, new_price: Money) -> FlextResult[None]:
        """Validate that new price currency matches current price currency."""
        if self.price.currency != new_price.currency:
            return FlextResult.fail(
                f"Currency mismatch: {self.price.currency} vs {new_price.currency}",
            )
        return FlextResult.ok(None)

    def _create_updated_product_with_price(
        self, new_price: Money
    ) -> FlextResult[Product]:
        """Create new product instance with updated price."""
        try:
            updated_product = self.model_copy(update={"price": new_price})
            return FlextResult.ok(updated_product)
        except Exception as e:
            return FlextResult.fail(f"Failed to create updated product: {e}")

    def _add_price_update_event(
        self,
        updated_product: Product,
        new_price: Money,
    ) -> FlextResult[Product]:
        """Add domain event for price update."""
        updated_product.add_domain_event(
            "ProductPriceUpdated",
            {
                "product_id": self.id,
                "old_price": str(self.price),
                "new_price": str(new_price),
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        return FlextResult.ok(updated_product)

    def adjust_stock(self, quantity_change: int, reason: str) -> FlextResult[Product]:
        """Adjust stock quantity with reason using railway-oriented programming.

        REFACTORED: Eliminated 6 returns using FlextResult chain pattern.
        Applied SOLID principles with single responsibility validation methods.
        """
        return (
            self._validate_stock_adjustment_reason(reason)
            .flat_map(lambda _: self._validate_stock_availability(quantity_change))
            .flat_map(self._create_updated_product_with_stock)
            .flat_map(
                lambda product: self._add_stock_adjustment_event(
                    product, quantity_change, reason
                )
            )
        )

    def _validate_stock_adjustment_reason(self, reason: str) -> FlextResult[None]:
        """Validate stock adjustment reason requirements."""
        if not reason or len(reason.strip()) < MIN_REASON_LENGTH:
            return FlextResult.fail(
                "Stock adjustment reason must be at least 5 characters"
            )
        return FlextResult.ok(None)

    def _validate_stock_availability(self, quantity_change: int) -> FlextResult[int]:
        """Validate stock availability and return new stock quantity."""
        new_stock = self.stock_quantity + quantity_change
        if new_stock < 0:
            return FlextResult.fail(
                f"Insufficient stock. Current: {self.stock_quantity}, "
                f"Requested: {abs(quantity_change)}"
            )
        return FlextResult.ok(new_stock)

    def _create_updated_product_with_stock(
        self, new_stock: int
    ) -> FlextResult[Product]:
        """Create new product instance with updated stock quantity."""
        try:
            updated_product = self.model_copy(update={"stock_quantity": new_stock})
            return FlextResult.ok(updated_product)
        except Exception as e:
            return FlextResult.fail(f"Failed to create updated product: {e}")

    def _add_stock_adjustment_event(
        self,
        updated_product: Product,
        quantity_change: int,
        reason: str,
    ) -> FlextResult[Product]:
        """Add domain event for stock adjustment."""
        # Determine event type
        event_type = (
            "ProductStockIncreased" if quantity_change > 0 else "ProductStockDecreased"
        )

        # Add domain event
        updated_product.add_domain_event(
            event_type,
            {
                "product_id": self.id,
                "old_stock": self.stock_quantity,
                "new_stock": updated_product.stock_quantity,
                "quantity_change": quantity_change,
                "reason": reason,
                "is_low_stock": updated_product.is_low_stock(),
                "is_out_of_stock": updated_product.is_out_of_stock(),
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        return FlextResult.ok(updated_product)

    def make_unavailable(self, reason: str) -> FlextResult[Product]:
        """Make product unavailable using railway-oriented programming.

        REFACTORED: Eliminated 6 returns using FlextResult chain pattern.
        Applied SOLID principles with single responsibility validation methods.
        """
        return (
            self._validate_product_is_available()
            .flat_map(lambda _: self._validate_unavailability_reason(reason))
            .flat_map(lambda _: self._create_unavailable_product())
            .flat_map(lambda product: self._add_unavailability_event(product, reason))
        )

    def _validate_product_is_available(self) -> FlextResult[None]:
        """Validate product is available and can be made unavailable."""
        if not self.is_available:
            return FlextResult.fail("Product is already unavailable")
        return FlextResult.ok(None)

    def _validate_unavailability_reason(self, reason: str) -> FlextResult[None]:
        """Validate unavailability reason meets requirements."""
        if not reason or len(reason.strip()) < MIN_STATUS_CHANGE_REASON_LENGTH:
            return FlextResult.fail(
                "Unavailability reason must be at least 10 characters"
            )
        return FlextResult.ok(None)

    def _create_unavailable_product(self) -> FlextResult[Product]:
        """Create new product instance with unavailable status."""
        try:
            unavailable_product = self.model_copy(update={"is_available": False})
            return FlextResult.ok(unavailable_product)
        except Exception as e:
            return FlextResult.fail(f"Failed to create unavailable product: {e}")

    def _add_unavailability_event(
        self, unavailable_product: Product, reason: str
    ) -> FlextResult[Product]:
        """Add domain event for product unavailability."""
        unavailable_product.add_domain_event(
            "ProductMadeUnavailable",
            {
                "product_id": self.id,
                "reason": reason,
                "stock_at_time": self.stock_quantity,
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        return FlextResult.ok(unavailable_product)


# =============================================================================
# AGGREGATE ROOT - Order aggregate managing order items
# =============================================================================


# =============================================================================
# REPOSITORY PATTERNS - Data persistence simulation
# =============================================================================


class CustomerRepository:
    """Repository pattern for customer persistence."""

    def __init__(self) -> None:
        """Initialize CustomerRepository with empty storage."""
        self.customers: dict[str, Customer] = {}

    def save(self, customer: Customer) -> FlextResult[str]:
        """Save customer to repository."""
        self.customers[customer.id] = customer
        print(
            f"💾 Customer saved: {customer.name} "
            f"(ID: {customer.id}, Version: {customer.version})",
        )
        return FlextResult.ok(customer.id)

    def find_by_id(self, customer_id: str) -> FlextResult[Customer]:
        """Find customer by ID."""
        if customer_id not in self.customers:
            return FlextResult.fail(f"Customer not found: {customer_id}")

        customer = self.customers[customer_id]
        print(f"🔍 Customer found: {customer.name} (Version: {customer.version})")
        return FlextResult.ok(customer)

    def find_by_email(self, email: str) -> FlextResult[Customer]:
        """Find customer by email."""
        for customer in self.customers.values():
            customer_email = getattr(customer, "email_address", None) or getattr(
                customer, "email", None
            )
            if customer_email and getattr(customer_email, "email", None) == email:
                print(f"🔍 Customer found by email: {customer.name}")
                return FlextResult.ok(customer)

        return FlextResult.fail(f"Customer not found with email: {email}")

    def find_active_customers(self) -> list[Customer]:
        """Find all active customers."""
        active = [c for c in self.customers.values() if c.is_active]
        print(f"📋 Found {len(active)} active customers")
        return active

    def get_all(self) -> list[Customer]:
        """Get all customers."""
        return list(self.customers.values())


class ProductRepository:
    """Repository pattern for product persistence."""

    def __init__(self) -> None:
        """Initialize ProductRepository with empty storage."""
        self.products: dict[str, Product] = {}

    def save(self, product: Product) -> FlextResult[str]:
        """Save product to repository."""
        self.products[product.id] = product
        print(
            f"💾 Product saved: {product.name} "
            f"(ID: {product.id}, Version: {product.version})",
        )
        return FlextResult.ok(product.id)

    def find_by_id(self, product_id: str) -> FlextResult[Product]:
        """Find product by ID."""
        if product_id not in self.products:
            return FlextResult.fail(f"Product not found: {product_id}")

        product = self.products[product_id]
        print(f"🔍 Product found: {product.name} (Version: {product.version})")
        return FlextResult.ok(product)

    def find_available_products(self) -> list[Product]:
        """Find all available products."""
        available = [p for p in self.products.values() if p.is_available]
        print(f"📋 Found {len(available)} available products")
        return available

    def find_low_stock_products(self) -> list[Product]:
        """Find products with low stock."""
        low_stock = [p for p in self.products.values() if p.is_low_stock()]
        print(f"⚠️ Found {len(low_stock)} low stock products")
        return low_stock

    def get_all(self) -> list[Product]:
        """Get all products."""
        return list(self.products.values())


class OrderRepository:
    """Repository pattern for order persistence."""

    def __init__(self) -> None:
        """Initialize OrderRepository with empty storage."""
        self.orders: dict[str, Order] = {}

    def save(self, order: Order) -> FlextResult[str]:
        """Save order to repository."""
        self.orders[order.id] = order
        print(
            f"💾 Order saved: {order.id} "
            f"(Status: {order.status}, Version: {order.version})",
        )
        return FlextResult.ok(order.id)

    def find_by_id(self, order_id: str) -> FlextResult[Order]:
        """Find order by ID."""
        if order_id not in self.orders:
            return FlextResult.fail(f"Order not found: {order_id}")

        order = self.orders[order_id]
        print(f"🔍 Order found: {order.id} (Status: {order.status})")
        return FlextResult.ok(order)

    def find_by_customer(self, customer_id: str) -> list[Order]:
        """Find orders by customer."""
        customer_orders = [
            o for o in self.orders.values() if o.customer_id == customer_id
        ]
        print(f"📋 Found {len(customer_orders)} orders for customer {customer_id}")
        return customer_orders

    def find_by_status(self, status: str) -> list[Order]:
        """Find orders by status."""
        status_orders = [o for o in self.orders.values() if o.status == status]
        print(f"📋 Found {len(status_orders)} orders with status: {status}")
        return status_orders

    def get_all(self) -> list[Order]:
        """Get all orders."""
        return list(self.orders.values())


# =============================================================================
# DOMAIN SERVICES - Complex business logic
# =============================================================================


class OrderDomainService:
    """Domain service for complex order operations."""

    def __init__(
        self,
        customer_repo: CustomerRepository,
        product_repo: ProductRepository,
        order_repo: OrderRepository,
    ) -> None:
        """Initialize OrderDomainService with repository dependencies.

        Args:
            customer_repo: Customer repository for customer operations
            product_repo: Product repository for inventory operations
            order_repo: Order repository for order persistence

        """
        self.customer_repo = customer_repo
        self.product_repo = product_repo
        self.order_repo = order_repo

    def create_order(
        self,
        customer_id: str,
        product_orders: list[tuple[str, int]],  # (product_id, quantity)
        shipping_address: Address,
    ) -> FlextResult[Order]:
        """Create a new order with comprehensive business validation.

        Refactored to reduce complexity while preserving DDD patterns.
        Uses private methods for clear separation of concerns.
        """
        print(f"🛒 Creating order for customer: {customer_id}")

        # Validate prerequisites
        customer_result = self._validate_customer_for_order(customer_id)
        if customer_result.is_failure:
            return FlextResult.fail(
                customer_result.error or "Customer validation failed"
            )

        address_result = self._validate_shipping_address(shipping_address)
        if address_result.is_failure:
            return FlextResult.fail(address_result.error or "Address validation failed")

        # Process order items
        items_result = self._create_and_validate_order_items(product_orders)
        if items_result.is_failure:
            return FlextResult.fail(
                items_result.error or "Order items validation failed"
            )

        if items_result.data is None:
            return FlextResult.fail("Order items creation returned None")

        order_items, total_amount = items_result.data

        # Create and finalize order
        return self._create_and_persist_order(
            customer_id, order_items, shipping_address, total_amount
        )

    def _validate_customer_for_order(self, customer_id: str) -> FlextResult[None]:
        """Validate customer exists and is eligible for orders."""
        customer_result = self.customer_repo.find_by_id(customer_id)
        if customer_result.is_failure:
            return FlextResult.fail(
                f"Customer validation failed: {customer_result.error}"
            )

        customer = customer_result.data
        if customer is None:
            return FlextResult.fail("Customer validation returned None data")

        if not customer.is_active:
            return FlextResult.fail("Cannot create order for inactive customer")

        return FlextResult.ok(None)

    def _validate_shipping_address(self, address: Address) -> FlextResult[None]:
        """Validate shipping address meets domain requirements."""
        address_validation = address.validate_business_rules()
        if address_validation.is_failure:
            return FlextResult.fail(
                f"Shipping address validation failed: {address_validation.error}"
            )
        return FlextResult.ok(None)

    def _create_and_validate_order_items(
        self, product_orders: list[tuple[str, int]]
    ) -> FlextResult[tuple[list[OrderItem], Money]]:
        """Create and validate all order items, calculate total."""
        order_items: list[OrderItem] = []
        total_amount = Money(amount=Decimal("0.0"), currency="USD")

        for product_id, quantity in product_orders:
            item_result = self._create_single_order_item(product_id, quantity)
            if item_result.is_failure:
                return FlextResult.fail(
                    item_result.error or "Failed to create order item"
                )

            order_item = item_result.data
            if order_item is None:
                return FlextResult.fail("Failed to create order item")
            order_items.append(order_item)

            # Update total amount
            total_result = self._add_item_to_total(order_item, total_amount)
            if total_result.is_failure:
                return FlextResult.fail(
                    total_result.error or "Failed to calculate total"
                )

            new_total = total_result.data
            if new_total is None:
                return FlextResult.fail("Failed to calculate total amount")
            total_amount = new_total

        return FlextResult.ok((order_items, total_amount))

    def _create_single_order_item(
        self, product_id: str, quantity: int
    ) -> FlextResult[OrderItem]:
        """Create and validate a single order item."""
        # Get and validate product
        product_result = self._get_and_validate_product(product_id, quantity)
        if product_result.is_failure:
            return FlextResult.fail(product_result.error or "Product validation failed")

        product = product_result.data
        if product is None:
            return FlextResult.fail("Product data is None")

        # Create order item
        order_item = OrderItem(
            product_id=product_id,
            product_name=product.name,
            unit_price=product.price,
            quantity=quantity,
        )

        # Validate order item domain rules
        item_validation = order_item.validate_business_rules()
        if item_validation.is_failure:
            return FlextResult.fail(
                f"Order item validation failed: {item_validation.error}"
            )

        return FlextResult.ok(order_item)

    def _get_and_validate_product(
        self, product_id: str, quantity: int
    ) -> FlextResult[Product]:
        """Get product and validate availability and stock."""
        product_result = self.product_repo.find_by_id(product_id)
        if product_result.is_failure:
            return FlextResult.fail(f"Product not found: {product_id}")

        product = product_result.data
        if product is None:
            return FlextResult.fail("Product lookup returned None data")

        if not product.is_available:
            return FlextResult.fail(f"Product not available: {product.name}")

        if product.stock_quantity < quantity:
            return FlextResult.fail(
                f"Insufficient stock for {product.name}. "
                f"Available: {product.stock_quantity}, Requested: {quantity}"
            )

        return FlextResult.ok(product)

    def _add_item_to_total(
        self, order_item: OrderItem, current_total: Money
    ) -> FlextResult[Money]:
        """Add order item total to running total."""
        if hasattr(order_item, "total_price"):
            item_total_result = order_item.total_price()
        else:
            item_total_result = FlextResult.fail("total_price method not available")

        if item_total_result.is_failure:
            return FlextResult.fail(
                f"Failed to calculate item total: {item_total_result.error}"
            )

        if item_total_result.data is None:
            return FlextResult.fail("Item total calculation returned None")

        total_result = current_total.add(item_total_result.data)
        if total_result.is_failure:
            return FlextResult.fail(
                f"Failed to calculate order total: {total_result.error}"
            )

        total_data = total_result.data
        if total_data is None:
            return FlextResult.fail("Failed to calculate total")
        return FlextResult.ok(total_data)

    def _create_and_persist_order(
        self,
        customer_id: str,
        order_items: list[OrderItem],
        shipping_address: Address,
        total_amount: Money,
    ) -> FlextResult[Order]:
        """Create order entity, add events, and persist."""
        # Create order entity using factory
        order_factory = create_order_factory()
        order_result = order_factory(
            customer_id=customer_id,
            items=order_items,
            shipping_address=shipping_address,
        )

        if order_result.is_failure:
            return FlextResult.fail(f"Failed to create order: {order_result.error}")

        order = order_result.data
        if order is None:
            return FlextResult.fail("Order creation returned None")

        # Validate complete order
        order_validation = order.validate_domain_rules()
        if order_validation.is_failure:
            return FlextResult.fail(
                f"Order validation failed: {order_validation.error}"
            )

        # Add domain event
        event_result = self._add_order_creation_event(order, customer_id)
        if event_result.is_failure:
            return FlextResult.fail(event_result.error or "Failed to add domain event")

        order = event_result.data or order

        # Persist order
        save_result = self.order_repo.save(order)
        if save_result.is_failure:
            return FlextResult.fail(f"Failed to save order: {save_result.error}")

        print(f"✅ Order created successfully: {order.id} (Total: {total_amount})")
        return FlextResult.ok(order)

    def _add_order_creation_event(
        self, order: Order, customer_id: str
    ) -> FlextResult[Order]:
        """Add domain event for order creation."""
        order.add_domain_event(
            "OrderCreated",
            {
                "order_id": order.id,
                "customer_id": customer_id,
                "item_count": len(order.items),
                "total_amount": str(getattr(order, "total_amount", "N/A")),
                "shipping_address": str(order.shipping_address),
                "creation_date": getattr(order, "order_date", None),
            },
        )

        return FlextResult.ok(order)

    def fulfill_order(self, order_id: str, tracking_number: str) -> FlextResult[Order]:
        """Fulfill order by confirming and shipping using railway-oriented programming.

        REFACTORED: Eliminated 6 returns using FlextResult chain pattern.
        Applied SOLID principles with single responsibility methods.
        """
        print(f"📦 Fulfilling order: {order_id}")

        return (
            self._find_order_for_fulfillment(order_id)
            .flat_map(self._confirm_order_safely)
            .flat_map(self._save_confirmed_order)
            .flat_map(
                lambda confirmed_order: self._ship_order_safely(
                    confirmed_order, tracking_number
                )
            )
            .flat_map(self._save_shipped_order)
            .flat_map(
                lambda shipped_order: self._update_product_stock_for_order(
                    shipped_order, order_id
                )
            )
        )

    def _find_order_for_fulfillment(self, order_id: str) -> FlextResult[Order]:
        """Find order by ID for fulfillment."""
        order_result = self.order_repo.find_by_id(order_id)
        if order_result.is_failure:
            return order_result

        order = order_result.data
        if order is None:
            return FlextResult.fail(f"Order not found: {order_id}")

        return FlextResult.ok(order)

    def _confirm_order_safely(self, order: Order) -> FlextResult[Order]:
        """Confirm order safely, handling missing methods."""
        try:
            if hasattr(order, "confirm"):
                confirmed_result = order.confirm()
                if confirmed_result.is_failure:
                    return confirmed_result
                confirmed_order = confirmed_result.data
                if confirmed_order is None:
                    return FlextResult.fail("Order confirmation returned None")
                return FlextResult.ok(confirmed_order)
            # If method doesn't exist, use the order as-is
            return FlextResult.ok(order)
        except AttributeError:
            # If method doesn't exist, use the order as-is
            return FlextResult.ok(order)

    def _save_confirmed_order(self, confirmed_order: Order) -> FlextResult[Order]:
        """Save confirmed order to repository."""
        save_result = self.order_repo.save(confirmed_order)
        if save_result.is_failure:
            return FlextResult.fail(
                f"Failed to save confirmed order: {save_result.error}"
            )
        return FlextResult.ok(confirmed_order)

    def _ship_order_safely(
        self, confirmed_order: Order, tracking_number: str
    ) -> FlextResult[Order]:
        """Ship order safely, handling missing methods."""
        try:
            if hasattr(confirmed_order, "ship_order"):
                shipped_result = confirmed_order.ship_order(tracking_number)
                if hasattr(shipped_result, "is_failure") and shipped_result.is_failure:
                    return FlextResult.fail("Order shipping failed")
                if hasattr(shipped_result, "data"):
                    shipped_order = shipped_result.data
                    if shipped_order is None:
                        return FlextResult.fail("Order shipping returned None")
                    return FlextResult.ok(cast("Order", shipped_order))
                return FlextResult.ok(confirmed_order)
            # If method doesn't exist, use the confirmed order as-is
            return FlextResult.ok(confirmed_order)
        except (AttributeError, TypeError):
            # If method doesn't exist or fails, use the confirmed order as-is
            return FlextResult.ok(confirmed_order)

    def _save_shipped_order(self, shipped_order: Order) -> FlextResult[Order]:
        """Save shipped order to repository."""
        save_result = self.order_repo.save(shipped_order)
        if save_result.is_failure:
            return FlextResult.fail(
                f"Failed to save shipped order: {save_result.error}"
            )
        return FlextResult.ok(shipped_order)

    def _update_product_stock_for_order(
        self, shipped_order: Order, order_id: str
    ) -> FlextResult[Order]:
        """Update product stock for all items in the order."""
        # Update product stock
        for item in shipped_order.items:
            product_result = self.product_repo.find_by_id(item.product_id)
            if product_result.success:
                product = product_result.data
                if product is not None:
                    stock_result = product.adjust_stock(
                        -item.quantity,
                        f"Order fulfillment: {order_id}",
                    )
                    if stock_result.success and stock_result.data is not None:
                        self.product_repo.save(stock_result.data)

        print(f"✅ Order fulfilled successfully: {order_id}")
        return FlextResult.ok(shipped_order)


# =============================================================================
# FACTORY PATTERNS - Entity creation with defaults
# =============================================================================


def create_customer_factory() -> CustomerFactory:
    """Create factory for customers with defaults."""

    def factory(**kwargs: object) -> FlextResult[Customer]:
        try:
            # Set required defaults for User base class
            defaults = {
                "name": "Default Customer",
                "email_address": Email(email="default@example.com"),
                "age": Age(value=30),
                "status": UserStatus.PENDING,
                "registration_date": FlextUtilities.generate_iso_timestamp(),
                "credit_limit": Money(amount=Decimal("5000.0"), currency="USD"),
                "total_orders": 0,
            }
            # Update with provided values
            defaults_dict = dict(defaults)
            defaults_dict.update(kwargs)
            defaults = defaults_dict

            # Create customer instance using model validation
            customer = Customer.model_validate(defaults)

            # Validate the customer
            validation = customer.validate_business_rules()
            if validation.is_failure:
                return FlextResult.fail(
                    f"Customer validation failed: {validation.error}"
                )

            return FlextResult.ok(customer)
        except Exception as e:
            return FlextResult.fail(f"Failed to create customer: {e}")

    return factory


def create_product_factory() -> ProductFactory:
    """Create factory for products with defaults."""

    def factory(**kwargs: object) -> FlextResult[Product]:
        try:
            # Set defaults
            defaults = {
                "id": FlextUtilities.generate_entity_id(),
                "is_available": True,
                "minimum_stock": 10,
                "stock_quantity": 100,
            }
            # Update with provided values
            defaults_dict = dict(defaults)
            defaults_dict.update(kwargs)
            defaults = defaults_dict

            # Create product instance using model validation
            product = Product.model_validate(defaults)

            # Validate the product
            validation = product.validate_business_rules()
            if validation.is_failure:
                return FlextResult.fail(
                    f"Product validation failed: {validation.error}"
                )

            return FlextResult.ok(product)
        except Exception as e:
            return FlextResult.fail(f"Failed to create product: {e}")

    return factory


def create_order_factory() -> OrderFactory:
    """Create factory for orders with defaults."""

    def factory(**kwargs: object) -> FlextResult[Order]:
        try:
            # Set defaults with explicit typing for object compatibility
            defaults: dict[str, object] = {
                "id": FlextUtilities.generate_entity_id(),
                "order_date": FlextUtilities.generate_iso_timestamp(),
                "status": "pending",
            }
            # Update with provided values
            defaults.update(kwargs)

            # Create order instance using model validation
            order = Order.model_validate(defaults)

            # Validate the order
            validation = order.validate_domain_rules()
            if validation.is_failure:
                return FlextResult.fail(f"Order validation failed: {validation.error}")

            return FlextResult.ok(order)
        except Exception as e:
            return FlextResult.fail(f"Failed to create order: {e}")

    return factory


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def demonstrate_value_objects() -> None:
    """Demonstrate value object patterns."""
    print("\n💎 Value Objects Demonstration")
    print("=" * 50)

    # Money value objects
    print("📋 Money Value Objects:")
    usd_10 = Money(amount=Decimal("10.50"), currency="USD")
    usd_20 = Money(amount=Decimal("20.00"), currency="USD")
    eur_15 = Money(amount=Decimal("15.75"), currency="EUR")

    print(f"  💵 USD Amount 1: {usd_10}")
    print(f"  💵 USD Amount 2: {usd_20}")
    print(f"  💶 EUR Amount: {eur_15}")

    # Money operations
    print("\n📋 Money Operations:")

    # Addition (same currency)
    sum_result = usd_10.add(usd_20)
    if sum_result.success:
        print(f"  ✅ {usd_10} + {usd_20} = {sum_result.data}")

    # Addition (different currency - should fail)
    invalid_sum = usd_10.add(eur_15)
    if invalid_sum.is_failure:
        print(f"  ❌ {usd_10} + {eur_15} failed: {invalid_sum.error}")

    # Multiplication
    doubled_result = usd_10.multiply(Decimal("2.0"))
    if doubled_result.success:
        print(f"  ✅ {usd_10} x 2 = {doubled_result.data}")

    # Address value objects
    print("\n📋 Address Value Objects:")
    address1 = Address(
        street="123 Main Street",
        city="Springfield",
        postal_code="12345",
        country="USA",
    )

    address2 = Address(
        street="456 Oak Avenue",
        city="Springfield",
        postal_code="12346",
        country="USA",
    )

    print(f"  🏠 Address 1: {address1}")
    print(f"  🏠 Address 2: {address2}")
    print(f"  🏙️ Same city: {address1.is_same_city(address2)}")

    # Email value objects
    print("\n📋 Email Value Objects:")
    email1 = Email(email="john@example.com")
    email2 = Email(email="invalid-email")

    print(f"  📧 Valid Email: {email1}")

    # Validate invalid email
    validation = email2.validate_business_rules()
    if validation.is_failure:
        print(f"  ❌ Invalid Email '{email2}': {validation.error}")


def demonstrate_entity_lifecycle() -> None:
    """Demonstrate entity lifecycle operations."""
    print("\n🔄 Entity Lifecycle Demonstration")
    print("=" * 50)

    customer = _create_demo_customer()
    if customer is None:
        return

    customer = _demo_update_address(customer)
    customer = _demo_increase_credit(customer)
    customer = _demo_increment_orders(customer)
    _demo_print_events(customer)


def _create_demo_customer() -> Customer | None:
    """Create and print a demo customer for lifecycle operations."""
    factory = create_customer_factory()
    result = factory(
        name="Alice Johnson",
        email_address=Email(email="alice@company.com"),
        address=Address(
            street="123 Business Ave",
            city="Enterprise City",
            postal_code="12345",
            country="USA",
        ),
    )

    if result.is_failure or result.data is None:
        print(f"❌ Customer creation failed: {result.error or 'no data'}")
        return None

    customer = result.data
    print(
        f"✅ Customer created: {customer.name} "
        f"(ID: {customer.id}, Version: {customer.version})",
    )
    print("\n📋 Customer Operations:")
    return customer


def _demo_update_address(customer: Customer) -> Customer:
    """Run and print the address update flow, returning the latest customer."""
    new_address = Address(
        street="456 New Street",
        city="New City",
        postal_code="67890",
        country="USA",
    )
    updated_result = customer.update_address(new_address)
    if updated_result.success and isinstance(updated_result.data, Customer):
        updated_customer = updated_result.data
        print(f"✅ Address updated (Version: {updated_customer.version})")
        return updated_customer
    return customer


def _demo_increase_credit(customer: Customer) -> Customer:
    """Run and print the credit increase flow, returning the latest customer."""
    credit_increase = Money(amount=Decimal("2000.0"), currency="USD")
    credit_result = customer.increase_credit_limit(credit_increase)
    if credit_result.success and isinstance(credit_result.data, Customer):
        updated = credit_result.data
        print(
            f"✅ Credit limit increased to {updated.credit_limit} "
            f"(Version: {updated.version})",
        )
        return updated
    return customer


def _demo_increment_orders(customer: Customer) -> Customer:
    """Increment orders once and print the outcome, returning the latest customer."""
    order_result = customer.increment_order_count()
    if order_result.success and isinstance(order_result.data, Customer):
        updated = order_result.data
        print(
            f"✅ Order count incremented to {updated.total_orders} "
            f"(Version: {updated.version})",
        )
        return updated
    return customer


def _demo_print_events(customer: Customer) -> None:
    """Print and clear domain events for the given customer."""
    print("\n📋 Domain Events:")
    events = customer.clear_events()
    for i, event in enumerate(events, 1):
        event_type = event.get_metadata("event_type") if hasattr(event, "get_metadata") else "Unknown Event Type"
        print(f"  📝 Event {i}: {event_type}")
        data_repr = event.data if hasattr(event, "data") else event
        print(f"     Data: {data_repr}")


def _setup_aggregate_repositories() -> tuple[
    CustomerRepository, ProductRepository, OrderRepository
]:
    """Setup repositories for aggregate demonstration."""
    return CustomerRepository(), ProductRepository(), OrderRepository()


def _create_test_customer(customer_repo: CustomerRepository) -> Customer | None:
    """Create and save test customer for aggregate demonstration."""
    customer_factory = create_customer_factory()
    customer_result = customer_factory(
        name="Bob Smith",
        email_address=Email(email="bob@company.com"),
        address=Address(
            street="789 Customer Lane",
            city="Shopping City",
            postal_code="54321",
            country="USA",
        ),
    )

    if customer_result.is_failure:
        print(f"❌ Failed to create customer: {customer_result.error}")
        return None

    customer = customer_result.data
    if customer is None:
        print("❌ Failed to create customer: None returned")
        return None
    # Customer is already correctly typed
    customer_repo.save(customer)
    return customer


def _create_test_products(product_repo: ProductRepository) -> list[Product]:
    """Create and save test products for aggregate demonstration."""
    product_factory = create_product_factory()
    products_data = [
        {
            "name": "Wireless Headphones",
            "price": Money(amount=Decimal("199.99"), currency="USD"),
            "category": "electronics",
            "stock_quantity": 50,
        },
        {
            "name": "USB Cable",
            "price": Money(amount=Decimal("19.99"), currency="USD"),
            "category": "electronics",
            "stock_quantity": 100,
        },
    ]

    products: list[Product] = []
    for product_data in products_data:
        product_result = product_factory(**product_data)
        if product_result.success:
            product = product_result.data
            if product is not None:
                product_repo.save(product)
                products.append(product)
                print(f"  ✅ Product created: {product.name}")
    return products


def _create_order_via_service(
    order_service: OrderDomainService,
    customer: Customer,
    products: list[Product],
) -> Order | None:
    """Create order using domain service."""
    shipping_address = Address(
        street="123 Delivery Street",
        city="Ship City",
        postal_code="98765",
        country="USA",
    )

    product_orders = [
        (products[0].id, 2),  # 2 headphones
        (products[1].id, 3),  # 3 USB cables
    ]

    order_result = order_service.create_order(
        customer.id,
        product_orders,
        shipping_address,
    )

    if order_result.is_failure:
        print(f"❌ Order creation failed: {order_result.error}")
        return None

    order = order_result.data
    if order is None:
        print("❌ Order creation returned None")
        return None

    total_amount_str = str(getattr(order, "total_amount", "N/A"))
    print(f"✅ Order created: {order.id} (Total: {total_amount_str})")
    return order


def _process_order_lifecycle(
    order_service: OrderDomainService,
    order_repo: OrderRepository,
    order: Order,
) -> None:
    """Process order lifecycle (fulfill and deliver)."""
    print("\n📋 Order Lifecycle:")

    # Fulfill order
    tracking_number = "TRK123456789"
    fulfill_result = order_service.fulfill_order(order.id, tracking_number)
    if not fulfill_result.success:
        return

    fulfilled_order = fulfill_result.data
    if fulfilled_order is None:
        print("❌ Order fulfillment returned None")
        return
    print(f"✅ Order fulfilled: Status {fulfilled_order.status}")

    # Deliver order (method may not exist in Order class)
    if hasattr(fulfilled_order, "deliver_order"):
        deliver_result = fulfilled_order.deliver_order()
        if deliver_result.success:
            delivered_order = deliver_result.data
            if delivered_order is None:
                print("❌ Order delivery returned None")
                return
            order_repo.save(delivered_order)
            print(f"✅ Order delivered: Status {delivered_order.status}")
    else:
        # If method doesn't exist, simulate delivery
        print("✅ Order delivered (simulated): Status delivered")


def _display_updated_stock(
    product_repo: ProductRepository, products: list[Product]
) -> None:
    """Display updated product stock after order processing."""
    print("\n📋 Updated Product Stock:")
    for product in products:
        updated_result = product_repo.find_by_id(product.id)
        if updated_result.success:
            updated_product = updated_result.data
            if updated_product is None:
                print(f"  📦 Product {product.id}: Not found")
                continue
            print(
                f"  📦 {updated_product.name}: Stock {updated_product.stock_quantity}",
            )


def demonstrate_aggregate_patterns() -> None:
    """Demonstrate aggregate patterns with orders."""
    print("\n🏢 Aggregate Patterns Demonstration")
    print("=" * 50)

    # Setup repositories
    customer_repo, product_repo, order_repo = _setup_aggregate_repositories()

    # Setup test data
    print("📋 Setting up test data:")
    customer = _create_test_customer(customer_repo)
    if customer is None:
        return

    products = _create_test_products(product_repo)
    if not products:
        print("❌ No products created")
        return

    # Create order using domain service
    print("\n📋 Order Creation via Domain Service:")
    order_service = OrderDomainService(customer_repo, product_repo, order_repo)
    order = _create_order_via_service(order_service, customer, products)
    if order is None:
        return

    # Process order lifecycle
    _process_order_lifecycle(order_service, order_repo, order)

    # Display updated stock
    _display_updated_stock(product_repo, products)


def _create_test_customers_for_repo(
    customer_repo: CustomerRepository,
) -> list[Customer]:
    """Create test customers for repository demonstration."""
    customer_factory = create_customer_factory()
    customers_data = [
        {
            "name": "Alice Smith",
            "email_address": Email(email="alice@example.com"),
            "address": Address(
                street="123 Main St",
                city="City1",
                postal_code="12345",
                country="USA",
            ),
        },
        {
            "name": "Bob Johnson",
            "email_address": Email(email="bob@example.com"),
            "address": Address(
                street="456 Oak Ave",
                city="City2",
                postal_code="67890",
                country="USA",
            ),
        },
    ]

    customers = []
    for customer_data in customers_data:
        result = customer_factory(**customer_data)
        if result.success:
            customer = result.data
            if customer is not None:
                customer_repo.save(customer)
                customers.append(customer)
    return customers


def _demonstrate_customer_queries(
    customer_repo: CustomerRepository, customers: list[Customer]
) -> None:
    """Demonstrate customer repository queries."""
    print("\n📋 Repository Queries:")

    # Find by email
    email_result = customer_repo.find_by_email("alice@example.com")
    if email_result.success:
        found_customer = email_result.data
        if found_customer is None:
            print("  🔍 Customer not found by email")
        else:
            print(f"  🔍 Found customer by email: {found_customer.name}")

    # Find active customers
    active_customers = customer_repo.find_active_customers()
    print(f"  👥 Active customers: {len(active_customers)}")

    # Deactivate a customer
    if customers:
        deactivate_result = customers[0].deactivate(
            "Customer requested account closure",
        )
        if deactivate_result.success:
            deactivated = deactivate_result.data
            if deactivated is not None:
                customer_repo.save(deactivated)
                print(f"  ❌ Customer deactivated: {deactivated.name}")

    # Check active customers again
    active_customers = customer_repo.find_active_customers()
    print(f"  👥 Active customers after deactivation: {len(active_customers)}")


def _demonstrate_product_operations(product_repo: ProductRepository) -> None:
    """Demonstrate product repository operations."""
    print("\n📋 Product Repository Operations:")
    product_factory = create_product_factory()

    products_data = [
        {
            "name": "Laptop",
            "price": Money(amount=Decimal("999.99"), currency="USD"),
            "category": "electronics",
            "stock_quantity": 5,  # Low stock
        },
        {
            "name": "Book",
            "price": Money(amount=Decimal("29.99"), currency="USD"),
            "category": "books",
            "stock_quantity": 0,  # Out of stock
        },
    ]

    for product_data in products_data:
        product_result = product_factory(**product_data)
        if product_result.success:
            product = product_result.data
            if product is not None:
                product_repo.save(product)

    # Find low stock products
    low_stock = product_repo.find_low_stock_products()
    for product in low_stock:
        print(f"  ⚠️ Low stock: {product.name} ({product.stock_quantity} units)")


def demonstrate_repository_patterns() -> None:
    """Demonstrate repository patterns."""
    print("\n🗄️ Repository Patterns Demonstration")
    print("=" * 50)

    # Create repositories
    customer_repo = CustomerRepository()
    product_repo = ProductRepository()

    # Create and save customers
    print("📋 Customer Repository Operations:")
    customers = _create_test_customers_for_repo(customer_repo)

    # Demonstrate customer queries
    _demonstrate_customer_queries(customer_repo, customers)

    # Demonstrate product operations
    _demonstrate_product_operations(product_repo)


def demonstrate_version_management() -> None:
    """Demonstrate optimistic locking and version management."""
    print("\n🔒 Version Management Demonstration")
    print("=" * 50)

    # Create product
    product_factory = create_product_factory()
    product_result = product_factory(
        name="Test Product",
        price=Money(amount=Decimal("100.0"), currency="USD"),
        category="electronics",
    )

    if product_result.is_failure:
        print(f"❌ Product creation failed: {product_result.error}")
        return

    product = product_result.data
    if product is None:
        print("❌ Product creation returned None")
        return
    print(f"📦 Product created: {product.name} (Version: {product.version})")

    # Simulate concurrent modifications
    print("\n📋 Concurrent Modification Simulation:")

    # First modification: Update price
    new_price = Money(amount=Decimal("120.0"), currency="USD")
    price_update_result = product.update_price(new_price)
    if price_update_result.success:
        updated_product_1 = price_update_result.data
        if updated_product_1 is not None:
            print(
                f"  💰 Price updated: {updated_product_1.price} "
                f"(Version: {updated_product_1.version})",
            )

        # Second modification: Adjust stock
        if updated_product_1 is not None:
            stock_update_result = updated_product_1.adjust_stock(-10, "Sales")
            if stock_update_result.success:
                updated_product_2 = stock_update_result.data
                if updated_product_2 is not None:
                    print(
                        f"  📦 Stock adjusted: {updated_product_2.stock_quantity} "
                        f"(Version: {updated_product_2.version})",
                    )

                    # Show version progression
                    print("\n📊 Version History:")
                    print(f"  📝 Original: Version {product.version}")
                    print(
                        f"  📝 After price update: Version {updated_product_1.version}"
                    )
                    print(
                        f"  📝 After stock update: Version {updated_product_2.version}"
                    )


def demonstrate_performance_characteristics() -> None:
    """Demonstrate performance characteristics of entities."""
    print("\n⚡ Performance Characteristics Demonstration")
    print("=" * 50)

    # Entity creation performance
    print("📋 Entity Creation Performance:")
    customer_factory = create_customer_factory()

    operations = 1000
    start_time = time.time()

    for i in range(operations):
        customer_factory(
            name=f"Customer {i}",
            email_address=Email(email=f"customer{i}@example.com"),
            address=Address(
                street=f"{i} Test Street",
                city="Test City",
                postal_code="12345",
                country="USA",
            ),
        )

    creation_time = time.time() - start_time
    print(
        f"  🔹 {operations} Customer creations: {creation_time:.4f}s "
        f"({operations / creation_time:.0f}/s)",
    )

    # Entity operation performance
    print("\n📋 Entity Operation Performance:")

    # Create a customer for operations
    customer_result = customer_factory(
        name="Performance Test Customer",
        email_address=Email(email="perf@example.com"),
        address=Address(
            street="123 Perf St",
            city="Perf City",
            postal_code="12345",
            country="USA",
        ),
    )

    if customer_result.success:
        customer = customer_result.data
        if customer is None:
            print("❌ Performance test customer creation returned None")
            return

        # Test copy_with performance
        start_time = time.time()
        current_customer = customer

        for _i in range(100):
            result = current_customer.increment_order_count()
            if result.success and result.data is not None:
                current_customer = result.data

        operation_time = time.time() - start_time
        print(
            f"  🔹 100 Entity operations: {operation_time:.4f}s "
            f"({100 / operation_time:.0f}/s)",
        )

        # Final state
        if current_customer is not None:
            print(
                f"  📊 Final state: Version {current_customer.version}, "
                f"Orders {current_customer.total_orders}",
            )

    # Value object operation performance
    print("\n📋 Value Object Performance:")
    money1 = Money(amount=Decimal("100.0"), currency="USD")
    money2 = Money(amount=Decimal("50.0"), currency="USD")

    start_time = time.time()
    for _ in range(operations):
        money1.add(money2)

    value_time = time.time() - start_time
    print(
        f"  🔹 {operations} Money additions: {value_time:.4f}s "
        f"({operations / value_time:.0f}/s)",
    )


def main() -> None:
    """Run comprehensive FlextEntity/ValueObject DDD demonstration."""
    examples = [
        ("Value Object Patterns", demonstrate_value_objects),
        ("Entity Lifecycle Management", demonstrate_entity_lifecycle),
        ("Aggregate Patterns and Domain Services", demonstrate_aggregate_patterns),
        ("Repository Patterns", demonstrate_repository_patterns),
        ("Version Management and Optimistic Locking", demonstrate_version_management),
        ("Performance Characteristics", demonstrate_performance_characteristics),
    ]

    run_example_demonstration(
        "FLEXT ENTITY/VALUE OBJECT - DOMAIN-DRIVEN DESIGN PATTERNS DEMONSTRATION",
        examples,
    )


if __name__ == "__main__":
    main()
