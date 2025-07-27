#!/usr/bin/env python3
"""FLEXT Entity/ValueObject - Domain-Driven Design Patterns Example.

Demonstrates comprehensive Domain-Driven Design (DDD) patterns using FlextEntity
with entity lifecycle management, value objects, domain events, and aggregates.

Features demonstrated:
- Entity creation with domain validation and business rules
- Value object patterns for domain modeling
- Domain event handling and event sourcing
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

from flext_core import FlextResult
from flext_core.entities import FlextEntity, FlextEntityFactory
from flext_core.utilities import FlextUtilities
from flext_core.value_objects import FlextValueObject

# =============================================================================
# VALUE OBJECTS - Immutable domain concepts
# =============================================================================


class Money(FlextValueObject):
    """Value object representing monetary amount with currency."""

    amount: float
    currency: str

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate money domain rules."""
        if self.amount < 0:
            return FlextResult.fail("Money amount cannot be negative")

        if not self.currency or len(self.currency) != 3:
            return FlextResult.fail("Currency must be 3-letter code")

        return FlextResult.ok(None)

    def add(self, other: Money) -> FlextResult[Money]:
        """Add money values (same currency only)."""
        if self.currency != other.currency:
            return FlextResult.fail(
                f"Cannot add different currencies: {self.currency} + {other.currency}",
            )

        result = Money(amount=self.amount + other.amount, currency=self.currency)
        validation = result.validate_domain_rules()
        if validation.is_failure:
            return FlextResult.fail(validation.error or "Invalid money calculation")

        return FlextResult.ok(result)

    def multiply(self, factor: float) -> FlextResult[Money]:
        """Multiply money by factor."""
        if factor < 0:
            return FlextResult.fail("Cannot multiply money by negative factor")

        result = Money(amount=self.amount * factor, currency=self.currency)
        validation = result.validate_domain_rules()
        if validation.is_failure:
            return FlextResult.fail(validation.error or "Invalid money calculation")

        return FlextResult.ok(result)

    def __str__(self) -> str:
        """Strings representation of money."""
        return f"{self.amount:.2f} {self.currency}"


class Address(FlextValueObject):
    """Value object representing a physical address."""

    street: str
    city: str
    postal_code: str
    country: str

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate address domain rules."""
        if not self.street or len(self.street.strip()) < 5:
            return FlextResult.fail("Street address must be at least 5 characters")

        if not self.city or len(self.city.strip()) < 2:
            return FlextResult.fail("City must be at least 2 characters")

        if not self.postal_code or len(self.postal_code.strip()) < 3:
            return FlextResult.fail("Postal code must be at least 3 characters")

        if not self.country or len(self.country.strip()) < 2:
            return FlextResult.fail("Country must be at least 2 characters")

        return FlextResult.ok(None)

    def is_same_city(self, other: Address) -> bool:
        """Check if addresses are in the same city."""
        return (
            self.city.lower() == other.city.lower()
            and self.country.lower() == other.country.lower()
        )

    def __str__(self) -> str:
        """String representation of address."""
        return f"{self.street}, {self.city}, {self.postal_code}, {self.country}"


class Email(FlextValueObject):
    """Value object representing an email address."""

    value: str

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate email domain rules."""
        if not self.value or "@" not in self.value:
            return FlextResult.fail("Invalid email format")

        parts = self.value.split("@")
        if len(parts) != 2:
            return FlextResult.fail("Email must have exactly one @ symbol")

        local, domain = parts
        if not local or not domain:
            return FlextResult.fail("Email local and domain parts cannot be empty")

        if "." not in domain:
            return FlextResult.fail("Email domain must contain at least one dot")

        return FlextResult.ok(None)

    def get_domain(self) -> str:
        """Extract domain from email."""
        return self.value.split("@")[1] if "@" in self.value else ""

    def __str__(self) -> str:
        """Strings representation of email."""
        return self.value


# =============================================================================
# DOMAIN ENTITIES - Rich domain models with business logic
# =============================================================================


class Customer(FlextEntity):
    """Customer entity with comprehensive business rules."""

    name: str
    email: Email
    address: Address
    registration_date: str
    is_active: bool = True
    credit_limit: Money
    total_orders: int = 0

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate customer domain rules."""
        if not self.name or len(self.name.strip()) < 2:
            return FlextResult.fail("Customer name must be at least 2 characters")

        if len(self.name) > 100:
            return FlextResult.fail("Customer name cannot exceed 100 characters")

        # Validate email
        email_validation = self.email.validate_domain_rules()
        if email_validation.is_failure:
            return FlextResult.fail(
                f"Email validation failed: {email_validation.error}",
            )

        # Validate address
        address_validation = self.address.validate_domain_rules()
        if address_validation.is_failure:
            return FlextResult.fail(
                f"Address validation failed: {address_validation.error}",
            )

        # Validate credit limit
        credit_validation = self.credit_limit.validate_domain_rules()
        if credit_validation.is_failure:
            return FlextResult.fail(
                f"Credit limit validation failed: {credit_validation.error}",
            )

        # Business rule: Credit limit must be reasonable
        if self.credit_limit.amount > 100000:
            return FlextResult.fail("Credit limit cannot exceed 100,000")

        # Business rule: Total orders cannot be negative
        if self.total_orders < 0:
            return FlextResult.fail("Total orders cannot be negative")

        return FlextResult.ok(None)

    def activate(self) -> FlextResult[Customer]:
        """Activate customer account."""
        if self.is_active:
            return FlextResult.fail("Customer is already active")

        # Create new version with activation
        result = self.copy_with(is_active=True)
        if result.is_failure:
            return result

        activated_customer = result.data

        # Add domain event
        event_result = activated_customer.add_domain_event(
            "CustomerActivated",
            {
                "customer_id": self.id,
                "activation_date": FlextUtilities.generate_iso_timestamp(),
                "previous_status": "inactive",
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add activation event: {event_result.error}",
            )

        return FlextResult.ok(activated_customer)

    def deactivate(self, reason: str) -> FlextResult[Customer]:
        """Deactivate customer account."""
        if not self.is_active:
            return FlextResult.fail("Customer is already inactive")

        if not reason or len(reason.strip()) < 10:
            return FlextResult.fail(
                "Deactivation reason must be at least 10 characters",
            )

        # Create new version with deactivation
        result = self.copy_with(is_active=False)
        if result.is_failure:
            return result

        deactivated_customer = result.data

        # Add domain event
        event_result = deactivated_customer.add_domain_event(
            "CustomerDeactivated",
            {
                "customer_id": self.id,
                "deactivation_date": FlextUtilities.generate_iso_timestamp(),
                "reason": reason,
                "previous_status": "active",
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add deactivation event: {event_result.error}",
            )

        return FlextResult.ok(deactivated_customer)

    def update_address(self, new_address: Address) -> FlextResult[Customer]:
        """Update customer address."""
        if not self.is_active:
            return FlextResult.fail("Cannot update address for inactive customer")

        # Validate new address
        validation = new_address.validate_domain_rules()
        if validation.is_failure:
            return FlextResult.fail(
                f"New address validation failed: {validation.error}",
            )

        # Create new version with updated address
        result = self.copy_with(address=new_address)
        if result.is_failure:
            return result

        updated_customer = result.data

        # Add domain event
        event_result = updated_customer.add_domain_event(
            "CustomerAddressUpdated",
            {
                "customer_id": self.id,
                "old_address": str(self.address),
                "new_address": str(new_address),
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add address update event: {event_result.error}",
            )

        return FlextResult.ok(updated_customer)

    def increase_credit_limit(self, amount: Money) -> FlextResult[Customer]:
        """Increase customer credit limit."""
        if not self.is_active:
            return FlextResult.fail(
                "Cannot increase credit limit for inactive customer",
            )

        # Validate same currency
        if self.credit_limit.currency != amount.currency:
            return FlextResult.fail(
                f"Currency mismatch: {self.credit_limit.currency} vs {amount.currency}",
            )

        # Calculate new credit limit
        new_limit_result = self.credit_limit.add(amount)
        if new_limit_result.is_failure:
            return FlextResult.fail(
                f"Failed to calculate new credit limit: {new_limit_result.error}",
            )

        new_limit = new_limit_result.data

        # Business rule: Total credit limit cannot exceed 100,000
        if new_limit.amount > 100000:
            return FlextResult.fail(
                "Credit limit increase would exceed maximum of 100,000",
            )

        # Create new version with updated credit limit
        result = self.copy_with(credit_limit=new_limit)
        if result.is_failure:
            return result

        updated_customer = result.data

        # Add domain event
        event_result = updated_customer.add_domain_event(
            "CustomerCreditLimitIncreased",
            {
                "customer_id": self.id,
                "old_limit": str(self.credit_limit),
                "new_limit": str(new_limit),
                "increase_amount": str(amount),
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add credit limit event: {event_result.error}",
            )

        return FlextResult.ok(updated_customer)

    def increment_order_count(self) -> FlextResult[Customer]:
        """Increment total order count."""
        if not self.is_active:
            return FlextResult.fail("Cannot increment orders for inactive customer")

        # Create new version with incremented order count
        result = self.copy_with(total_orders=self.total_orders + 1)
        if result.is_failure:
            return result

        updated_customer = result.data

        # Add domain event
        event_result = updated_customer.add_domain_event(
            "CustomerOrderCountIncremented",
            {
                "customer_id": self.id,
                "old_count": self.total_orders,
                "new_count": self.total_orders + 1,
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add order count event: {event_result.error}",
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

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate product domain rules."""
        if not self.name or len(self.name.strip()) < 2:
            return FlextResult.fail("Product name must be at least 2 characters")

        if len(self.name) > 200:
            return FlextResult.fail("Product name cannot exceed 200 characters")

        # Validate price
        price_validation = self.price.validate_domain_rules()
        if price_validation.is_failure:
            return FlextResult.fail(
                f"Price validation failed: {price_validation.error}",
            )

        # Business rule: Price must be positive
        if self.price.amount <= 0:
            return FlextResult.fail("Product price must be positive")

        # Business rule: Stock quantities must be non-negative
        if self.stock_quantity < 0:
            return FlextResult.fail("Stock quantity cannot be negative")

        if self.minimum_stock < 0:
            return FlextResult.fail("Minimum stock cannot be negative")

        # Business rule: Category must be valid
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
        """Update product price."""
        if not self.is_available:
            return FlextResult.fail("Cannot update price for unavailable product")

        # Validate new price
        validation = new_price.validate_domain_rules()
        if validation.is_failure:
            return FlextResult.fail(f"New price validation failed: {validation.error}")

        if new_price.amount <= 0:
            return FlextResult.fail("New price must be positive")

        # Currency must match
        if self.price.currency != new_price.currency:
            return FlextResult.fail(
                f"Currency mismatch: {self.price.currency} vs {new_price.currency}",
            )

        # Create new version with updated price
        result = self.copy_with(price=new_price)
        if result.is_failure:
            return result

        updated_product = result.data

        # Add domain event
        event_result = updated_product.add_domain_event(
            "ProductPriceUpdated",
            {
                "product_id": self.id,
                "old_price": str(self.price),
                "new_price": str(new_price),
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add price update event: {event_result.error}",
            )

        return FlextResult.ok(updated_product)

    def adjust_stock(self, quantity_change: int, reason: str) -> FlextResult[Product]:
        """Adjust stock quantity with reason."""
        if not reason or len(reason.strip()) < 5:
            return FlextResult.fail(
                "Stock adjustment reason must be at least 5 characters",
            )

        new_stock = self.stock_quantity + quantity_change
        if new_stock < 0:
            return FlextResult.fail(
                f"Insufficient stock. Current: {self.stock_quantity}, Requested: {abs(quantity_change)}",
            )

        # Create new version with adjusted stock
        result = self.copy_with(stock_quantity=new_stock)
        if result.is_failure:
            return result

        updated_product = result.data

        # Determine event type
        event_type = (
            "ProductStockIncreased" if quantity_change > 0 else "ProductStockDecreased"
        )

        # Add domain event
        event_result = updated_product.add_domain_event(
            event_type,
            {
                "product_id": self.id,
                "old_stock": self.stock_quantity,
                "new_stock": new_stock,
                "quantity_change": quantity_change,
                "reason": reason,
                "is_low_stock": updated_product.is_low_stock(),
                "is_out_of_stock": updated_product.is_out_of_stock(),
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add stock adjustment event: {event_result.error}",
            )

        return FlextResult.ok(updated_product)

    def make_unavailable(self, reason: str) -> FlextResult[Product]:
        """Make product unavailable."""
        if not self.is_available:
            return FlextResult.fail("Product is already unavailable")

        if not reason or len(reason.strip()) < 10:
            return FlextResult.fail(
                "Unavailability reason must be at least 10 characters",
            )

        # Create new version as unavailable
        result = self.copy_with(is_available=False)
        if result.is_failure:
            return result

        unavailable_product = result.data

        # Add domain event
        event_result = unavailable_product.add_domain_event(
            "ProductMadeUnavailable",
            {
                "product_id": self.id,
                "reason": reason,
                "stock_at_time": self.stock_quantity,
                "update_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add unavailability event: {event_result.error}",
            )

        return FlextResult.ok(unavailable_product)


# =============================================================================
# AGGREGATE ROOT - Order aggregate managing order items
# =============================================================================


class OrderItem(FlextValueObject):
    """Order item value object."""

    product_id: str
    product_name: str
    unit_price: Money
    quantity: int

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate order item rules."""
        if not self.product_id:
            return FlextResult.fail("Product ID is required")

        if not self.product_name or len(self.product_name.strip()) < 2:
            return FlextResult.fail("Product name must be at least 2 characters")

        # Validate unit price
        price_validation = self.unit_price.validate_domain_rules()
        if price_validation.is_failure:
            return FlextResult.fail(
                f"Unit price validation failed: {price_validation.error}",
            )

        if self.unit_price.amount <= 0:
            return FlextResult.fail("Unit price must be positive")

        if self.quantity <= 0:
            return FlextResult.fail("Quantity must be positive")

        return FlextResult.ok(None)

    def calculate_total(self) -> FlextResult[Money]:
        """Calculate total price for this item."""
        return self.unit_price.multiply(float(self.quantity))

    def __str__(self) -> str:
        """Strings representation of order item."""
        total_result = self.calculate_total()
        total_str = str(total_result.data) if total_result.is_success else "ERROR"
        return f"{self.quantity}x {self.product_name} @ {self.unit_price} = {total_str}"


class Order(FlextEntity):
    """Order aggregate root managing the complete order lifecycle."""

    customer_id: str
    order_date: str
    items: list[OrderItem]
    status: str = "pending"
    shipping_address: Address
    total_amount: Money

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate order domain rules."""
        if not self.customer_id:
            return FlextResult.fail("Customer ID is required")

        if not self.items:
            return FlextResult.fail("Order must have at least one item")

        # Validate all items
        for i, item in enumerate(self.items):
            item_validation = item.validate_domain_rules()
            if item_validation.is_failure:
                return FlextResult.fail(
                    f"Item {i} validation failed: {item_validation.error}",
                )

        # Validate shipping address
        address_validation = self.shipping_address.validate_domain_rules()
        if address_validation.is_failure:
            return FlextResult.fail(
                f"Shipping address validation failed: {address_validation.error}",
            )

        # Validate total amount
        total_validation = self.total_amount.validate_domain_rules()
        if total_validation.is_failure:
            return FlextResult.fail(
                f"Total amount validation failed: {total_validation.error}",
            )

        # Business rule: Valid order statuses
        valid_statuses = ["pending", "confirmed", "shipped", "delivered", "cancelled"]
        if self.status not in valid_statuses:
            return FlextResult.fail(
                f"Invalid order status. Must be one of: {valid_statuses}",
            )

        # Business rule: Verify calculated total matches
        calculated_total = self.calculate_total()
        if calculated_total.is_success:
            if abs(calculated_total.data.amount - self.total_amount.amount) > 0.01:
                return FlextResult.fail(
                    "Order total amount does not match calculated total",
                )

        return FlextResult.ok(None)

    def calculate_total(self) -> FlextResult[Money]:
        """Calculate total order amount."""
        if not self.items:
            return FlextResult.fail("Cannot calculate total for empty order")

        # Start with first item total
        first_item = self.items[0]
        first_total_result = first_item.calculate_total()
        if first_total_result.is_failure:
            return FlextResult.fail(
                f"Failed to calculate first item total: {first_total_result.error}",
            )

        total = first_total_result.data

        # Add remaining items
        for item in self.items[1:]:
            item_total_result = item.calculate_total()
            if item_total_result.is_failure:
                return FlextResult.fail(
                    f"Failed to calculate item total: {item_total_result.error}",
                )

            total_result = total.add(item_total_result.data)
            if total_result.is_failure:
                return FlextResult.fail(
                    f"Failed to add item total: {total_result.error}",
                )

            total = total_result.data

        return FlextResult.ok(total)

    def confirm_order(self) -> FlextResult[Order]:
        """Confirm the order."""
        if self.status != "pending":
            return FlextResult.fail(f"Cannot confirm order with status: {self.status}")

        # Create new version with confirmed status
        result = self.copy_with(status="confirmed")
        if result.is_failure:
            return result

        confirmed_order = result.data

        # Add domain event
        event_result = confirmed_order.add_domain_event(
            "OrderConfirmed",
            {
                "order_id": self.id,
                "customer_id": self.customer_id,
                "total_amount": str(self.total_amount),
                "item_count": len(self.items),
                "confirmation_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add confirmation event: {event_result.error}",
            )

        return FlextResult.ok(confirmed_order)

    def ship_order(self, tracking_number: str) -> FlextResult[Order]:
        """Ship the order."""
        if self.status != "confirmed":
            return FlextResult.fail(f"Cannot ship order with status: {self.status}")

        if not tracking_number or len(tracking_number.strip()) < 5:
            return FlextResult.fail("Tracking number must be at least 5 characters")

        # Create new version with shipped status
        result = self.copy_with(status="shipped")
        if result.is_failure:
            return result

        shipped_order = result.data

        # Add domain event
        event_result = shipped_order.add_domain_event(
            "OrderShipped",
            {
                "order_id": self.id,
                "customer_id": self.customer_id,
                "tracking_number": tracking_number,
                "shipping_address": str(self.shipping_address),
                "ship_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add shipping event: {event_result.error}",
            )

        return FlextResult.ok(shipped_order)

    def deliver_order(self) -> FlextResult[Order]:
        """Mark order as delivered."""
        if self.status != "shipped":
            return FlextResult.fail(f"Cannot deliver order with status: {self.status}")

        # Create new version with delivered status
        result = self.copy_with(status="delivered")
        if result.is_failure:
            return result

        delivered_order = result.data

        # Add domain event
        event_result = delivered_order.add_domain_event(
            "OrderDelivered",
            {
                "order_id": self.id,
                "customer_id": self.customer_id,
                "delivery_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add delivery event: {event_result.error}",
            )

        return FlextResult.ok(delivered_order)

    def cancel_order(self, reason: str) -> FlextResult[Order]:
        """Cancel the order."""
        if self.status in ["delivered", "cancelled"]:
            return FlextResult.fail(f"Cannot cancel order with status: {self.status}")

        if not reason or len(reason.strip()) < 10:
            return FlextResult.fail(
                "Cancellation reason must be at least 10 characters",
            )

        # Create new version with cancelled status
        result = self.copy_with(status="cancelled")
        if result.is_failure:
            return result

        cancelled_order = result.data

        # Add domain event
        event_result = cancelled_order.add_domain_event(
            "OrderCancelled",
            {
                "order_id": self.id,
                "customer_id": self.customer_id,
                "previous_status": self.status,
                "reason": reason,
                "cancellation_date": FlextUtilities.generate_iso_timestamp(),
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add cancellation event: {event_result.error}",
            )

        return FlextResult.ok(cancelled_order)


# =============================================================================
# REPOSITORY PATTERNS - Data persistence simulation
# =============================================================================


class CustomerRepository:
    """Repository pattern for customer persistence."""

    def __init__(self) -> None:
        self.customers: dict[str, Customer] = {}

    def save(self, customer: Customer) -> FlextResult[str]:
        """Save customer to repository."""
        self.customers[customer.id] = customer
        print(
            f"💾 Customer saved: {customer.name} (ID: {customer.id}, Version: {customer.version})",
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
            if customer.email.value == email:
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
        self.products: dict[str, Product] = {}

    def save(self, product: Product) -> FlextResult[str]:
        """Save product to repository."""
        self.products[product.id] = product
        print(
            f"💾 Product saved: {product.name} (ID: {product.id}, Version: {product.version})",
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
        self.orders: dict[str, Order] = {}

    def save(self, order: Order) -> FlextResult[str]:
        """Save order to repository."""
        self.orders[order.id] = order
        print(
            f"💾 Order saved: {order.id} (Status: {order.status}, Version: {order.version})",
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
        self.customer_repo = customer_repo
        self.product_repo = product_repo
        self.order_repo = order_repo

    def create_order(
        self,
        customer_id: str,
        product_orders: list[tuple[str, int]],  # (product_id, quantity)
        shipping_address: Address,
    ) -> FlextResult[Order]:
        """Create a new order with business validation."""
        print(f"🛒 Creating order for customer: {customer_id}")

        # Validate customer exists and is active
        customer_result = self.customer_repo.find_by_id(customer_id)
        if customer_result.is_failure:
            return FlextResult.fail(
                f"Customer validation failed: {customer_result.error}",
            )

        customer = customer_result.data
        if not customer.is_active:
            return FlextResult.fail("Cannot create order for inactive customer")

        # Validate shipping address
        address_validation = shipping_address.validate_domain_rules()
        if address_validation.is_failure:
            return FlextResult.fail(
                f"Shipping address validation failed: {address_validation.error}",
            )

        # Create order items and validate stock
        order_items = []
        total_amount = Money(amount=0.0, currency="USD")

        for product_id, quantity in product_orders:
            # Find product
            product_result = self.product_repo.find_by_id(product_id)
            if product_result.is_failure:
                return FlextResult.fail(f"Product not found: {product_id}")

            product = product_result.data

            # Validate product availability
            if not product.is_available:
                return FlextResult.fail(f"Product not available: {product.name}")

            # Validate stock
            if product.stock_quantity < quantity:
                return FlextResult.fail(
                    f"Insufficient stock for {product.name}. Available: {product.stock_quantity}, Requested: {quantity}",
                )

            # Create order item
            order_item = OrderItem(
                product_id=product_id,
                product_name=product.name,
                unit_price=product.price,
                quantity=quantity,
            )

            # Validate order item
            item_validation = order_item.validate_domain_rules()
            if item_validation.is_failure:
                return FlextResult.fail(
                    f"Order item validation failed: {item_validation.error}",
                )

            order_items.append(order_item)

            # Add to total
            item_total_result = order_item.calculate_total()
            if item_total_result.is_failure:
                return FlextResult.fail(
                    f"Failed to calculate item total: {item_total_result.error}",
                )

            total_result = total_amount.add(item_total_result.data)
            if total_result.is_failure:
                return FlextResult.fail(
                    f"Failed to calculate order total: {total_result.error}",
                )

            total_amount = total_result.data

        # Create order
        order_id = FlextUtilities.generate_entity_id()
        order = Order(
            id=order_id,
            customer_id=customer_id,
            order_date=FlextUtilities.generate_iso_timestamp(),
            items=order_items,
            shipping_address=shipping_address,
            total_amount=total_amount,
        )

        # Validate order
        order_validation = order.validate_domain_rules()
        if order_validation.is_failure:
            return FlextResult.fail(
                f"Order validation failed: {order_validation.error}",
            )

        # Add domain event
        event_result = order.add_domain_event(
            "OrderCreated",
            {
                "order_id": order.id,
                "customer_id": customer_id,
                "item_count": len(order_items),
                "total_amount": str(total_amount),
                "shipping_address": str(shipping_address),
                "creation_date": order.order_date,
            },
        )

        if event_result.is_failure:
            return FlextResult.fail(
                f"Failed to add order creation event: {event_result.error}",
            )

        # Save order
        save_result = self.order_repo.save(order)
        if save_result.is_failure:
            return FlextResult.fail(f"Failed to save order: {save_result.error}")

        print(f"✅ Order created successfully: {order.id} (Total: {total_amount})")
        return FlextResult.ok(order)

    def fulfill_order(self, order_id: str, tracking_number: str) -> FlextResult[Order]:
        """Fulfill order by confirming and shipping."""
        print(f"📦 Fulfilling order: {order_id}")

        # Find order
        order_result = self.order_repo.find_by_id(order_id)
        if order_result.is_failure:
            return order_result

        order = order_result.data

        # Confirm order
        confirmed_result = order.confirm_order()
        if confirmed_result.is_failure:
            return confirmed_result

        confirmed_order = confirmed_result.data

        # Save confirmed order
        save_result = self.order_repo.save(confirmed_order)
        if save_result.is_failure:
            return FlextResult.fail(
                f"Failed to save confirmed order: {save_result.error}",
            )

        # Ship order
        shipped_result = confirmed_order.ship_order(tracking_number)
        if shipped_result.is_failure:
            return shipped_result

        shipped_order = shipped_result.data

        # Save shipped order
        save_result = self.order_repo.save(shipped_order)
        if save_result.is_failure:
            return FlextResult.fail(
                f"Failed to save shipped order: {save_result.error}",
            )

        # Update product stock
        for item in order.items:
            product_result = self.product_repo.find_by_id(item.product_id)
            if product_result.is_success:
                product = product_result.data
                stock_result = product.adjust_stock(
                    -item.quantity,
                    f"Order fulfillment: {order_id}",
                )
                if stock_result.is_success:
                    self.product_repo.save(stock_result.data)

        print(f"✅ Order fulfilled successfully: {order_id}")
        return FlextResult.ok(shipped_order)


# =============================================================================
# FACTORY PATTERNS - Entity creation with defaults
# =============================================================================


def create_customer_factory() -> object:
    """Create factory for customers with defaults."""
    return FlextEntityFactory.create_entity_factory(
        Customer,
        defaults={
            "registration_date": FlextUtilities.generate_iso_timestamp(),
            "is_active": True,
            "credit_limit": Money(amount=5000.0, currency="USD"),
            "total_orders": 0,
        },
    )


def create_product_factory() -> object:
    """Create factory for products with defaults."""
    return FlextEntityFactory.create_entity_factory(
        Product,
        defaults={
            "is_available": True,
            "minimum_stock": 10,
            "stock_quantity": 100,
        },
    )


def create_order_factory() -> object:
    """Create factory for orders with defaults."""
    return FlextEntityFactory.create_entity_factory(
        Order,
        defaults={
            "order_date": FlextUtilities.generate_iso_timestamp(),
            "status": "pending",
        },
    )


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def demonstrate_value_objects() -> None:
    """Demonstrate value object patterns."""
    print("\n💎 Value Objects Demonstration")
    print("=" * 50)

    # Money value objects
    print("📋 Money Value Objects:")
    usd_10 = Money(amount=10.50, currency="USD")
    usd_20 = Money(amount=20.00, currency="USD")
    eur_15 = Money(amount=15.75, currency="EUR")

    print(f"  💵 USD Amount 1: {usd_10}")
    print(f"  💵 USD Amount 2: {usd_20}")
    print(f"  💶 EUR Amount: {eur_15}")

    # Money operations
    print("\n📋 Money Operations:")

    # Addition (same currency)
    sum_result = usd_10.add(usd_20)
    if sum_result.is_success:
        print(f"  ✅ {usd_10} + {usd_20} = {sum_result.data}")

    # Addition (different currency - should fail)
    invalid_sum = usd_10.add(eur_15)
    if invalid_sum.is_failure:
        print(f"  ❌ {usd_10} + {eur_15} failed: {invalid_sum.error}")

    # Multiplication
    doubled_result = usd_10.multiply(2.0)
    if doubled_result.is_success:
        print(f"  ✅ {usd_10} × 2 = {doubled_result.data}")  # noqa: RUF001

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
    email1 = Email(value="john@example.com")
    email2 = Email(value="invalid-email")

    print(f"  📧 Valid Email: {email1} (Domain: {email1.get_domain()})")

    # Validate invalid email
    validation = email2.validate_domain_rules()
    if validation.is_failure:
        print(f"  ❌ Invalid Email '{email2}': {validation.error}")


def demonstrate_entity_lifecycle() -> None:
    """Demonstrate entity lifecycle operations."""
    print("\n🔄 Entity Lifecycle Demonstration")
    print("=" * 50)

    # Create customer factory
    customer_factory = create_customer_factory()

    # Create customer
    print("📋 Customer Creation:")
    customer_data = {
        "name": "Alice Johnson",
        "email": Email(value="alice@company.com"),
        "address": Address(
            street="123 Business Ave",
            city="Enterprise City",
            postal_code="12345",
            country="USA",
        ),
    }

    customer_result = customer_factory(**customer_data)
    if customer_result.is_failure:
        print(f"❌ Customer creation failed: {customer_result.error}")
        return

    customer = customer_result.data
    print(
        f"✅ Customer created: {customer.name} (ID: {customer.id}, Version: {customer.version})",
    )

    # Customer operations
    print("\n📋 Customer Operations:")

    # Update address
    new_address = Address(
        street="456 New Street",
        city="New City",
        postal_code="67890",
        country="USA",
    )

    updated_result = customer.update_address(new_address)
    if updated_result.is_success:
        updated_customer = updated_result.data
        print(f"✅ Address updated (Version: {updated_customer.version})")
        customer = updated_customer

    # Increase credit limit
    credit_increase = Money(amount=2000.0, currency="USD")
    credit_result = customer.increase_credit_limit(credit_increase)
    if credit_result.is_success:
        credit_customer = credit_result.data
        print(
            f"✅ Credit limit increased to {credit_customer.credit_limit} (Version: {credit_customer.version})",
        )
        customer = credit_customer

    # Increment order count
    order_result = customer.increment_order_count()
    if order_result.is_success:
        order_customer = order_result.data
        print(
            f"✅ Order count incremented to {order_customer.total_orders} (Version: {order_customer.version})",
        )
        customer = order_customer

    # View domain events
    print("\n📋 Domain Events:")
    events = customer.clear_events()
    for i, event in enumerate(events, 1):
        event_type = event.get_metadata("event_type")
        print(f"  📝 Event {i}: {event_type}")
        print(f"     Data: {event.data}")


def demonstrate_aggregate_patterns() -> None:
    """Demonstrate aggregate patterns with orders."""
    print("\n🏢 Aggregate Patterns Demonstration")
    print("=" * 50)

    # Create repositories
    customer_repo = CustomerRepository()
    product_repo = ProductRepository()
    order_repo = OrderRepository()

    # Setup test data
    print("📋 Setting up test data:")

    # Create customer
    customer_factory = create_customer_factory()
    customer_result = customer_factory(
        name="Bob Smith",
        email=Email(value="bob@company.com"),
        address=Address(
            street="789 Customer Lane",
            city="Shopping City",
            postal_code="54321",
            country="USA",
        ),
    )

    if customer_result.is_failure:
        print(f"❌ Failed to create customer: {customer_result.error}")
        return

    customer = customer_result.data
    customer_repo.save(customer)

    # Create products
    product_factory = create_product_factory()

    products_data = [
        {
            "name": "Wireless Headphones",
            "price": Money(amount=199.99, currency="USD"),
            "category": "electronics",
            "stock_quantity": 50,
        },
        {
            "name": "USB Cable",
            "price": Money(amount=19.99, currency="USD"),
            "category": "electronics",
            "stock_quantity": 100,
        },
    ]

    products = []
    for product_data in products_data:
        product_result = product_factory(**product_data)
        if product_result.is_success:
            product = product_result.data
            product_repo.save(product)
            products.append(product)
            print(f"  ✅ Product created: {product.name}")

    # Create order using domain service
    print("\n📋 Order Creation via Domain Service:")
    order_service = OrderDomainService(customer_repo, product_repo, order_repo)

    shipping_address = Address(
        street="123 Delivery Street",
        city="Ship City",
        postal_code="98765",
        country="USA",
    )

    # Order 2 headphones and 3 USB cables
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
        return

    order = order_result.data
    print(f"✅ Order created: {order.id} (Total: {order.total_amount})")

    # Order lifecycle
    print("\n📋 Order Lifecycle:")

    # Fulfill order
    tracking_number = "TRK123456789"
    fulfill_result = order_service.fulfill_order(order.id, tracking_number)
    if fulfill_result.is_success:
        fulfilled_order = fulfill_result.data
        print(f"✅ Order fulfilled: Status {fulfilled_order.status}")

        # Deliver order
        deliver_result = fulfilled_order.deliver_order()
        if deliver_result.is_success:
            delivered_order = deliver_result.data
            order_repo.save(delivered_order)
            print(f"✅ Order delivered: Status {delivered_order.status}")

    # Check updated product stock
    print("\n📋 Updated Product Stock:")
    for product in products:
        updated_result = product_repo.find_by_id(product.id)
        if updated_result.is_success:
            updated_product = updated_result.data
            print(
                f"  📦 {updated_product.name}: Stock {updated_product.stock_quantity}",
            )


def demonstrate_repository_patterns() -> None:
    """Demonstrate repository patterns."""
    print("\n🗄️ Repository Patterns Demonstration")
    print("=" * 50)

    # Create repositories
    customer_repo = CustomerRepository()
    product_repo = ProductRepository()

    # Create and save customers
    print("📋 Customer Repository Operations:")
    customer_factory = create_customer_factory()

    customers_data = [
        {
            "name": "Alice Smith",
            "email": Email(value="alice@example.com"),
            "address": Address(
                street="123 Main St",
                city="City1",
                postal_code="12345",
                country="USA",
            ),
        },
        {
            "name": "Bob Johnson",
            "email": Email(value="bob@example.com"),
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
        if result.is_success:
            customer = result.data
            customer_repo.save(customer)
            customers.append(customer)

    # Repository queries
    print("\n📋 Repository Queries:")

    # Find by email
    email_result = customer_repo.find_by_email("alice@example.com")
    if email_result.is_success:
        found_customer = email_result.data
        print(f"  🔍 Found customer by email: {found_customer.name}")

    # Find active customers
    active_customers = customer_repo.find_active_customers()
    print(f"  👥 Active customers: {len(active_customers)}")

    # Deactivate a customer
    if customers:
        deactivate_result = customers[0].deactivate(
            "Customer requested account closure",
        )
        if deactivate_result.is_success:
            deactivated = deactivate_result.data
            customer_repo.save(deactivated)
            print(f"  ❌ Customer deactivated: {deactivated.name}")

    # Check active customers again
    active_customers = customer_repo.find_active_customers()
    print(f"  👥 Active customers after deactivation: {len(active_customers)}")

    # Product repository operations
    print("\n📋 Product Repository Operations:")
    product_factory = create_product_factory()

    products_data = [
        {
            "name": "Laptop",
            "price": Money(amount=999.99, currency="USD"),
            "category": "electronics",
            "stock_quantity": 5,  # Low stock
        },
        {
            "name": "Book",
            "price": Money(amount=29.99, currency="USD"),
            "category": "books",
            "stock_quantity": 0,  # Out of stock
        },
    ]

    for product_data in products_data:
        result = product_factory(**product_data)
        if result.is_success:
            product = result.data
            product_repo.save(product)

    # Find low stock products
    low_stock = product_repo.find_low_stock_products()
    for product in low_stock:
        print(f"  ⚠️ Low stock: {product.name} ({product.stock_quantity} units)")


def demonstrate_version_management() -> None:
    """Demonstrate optimistic locking and version management."""
    print("\n🔒 Version Management Demonstration")
    print("=" * 50)

    # Create product
    product_factory = create_product_factory()
    product_result = product_factory(
        name="Test Product",
        price=Money(amount=100.0, currency="USD"),
        category="electronics",
    )

    if product_result.is_failure:
        print(f"❌ Product creation failed: {product_result.error}")
        return

    product = product_result.data
    print(f"📦 Product created: {product.name} (Version: {product.version})")

    # Simulate concurrent modifications
    print("\n📋 Concurrent Modification Simulation:")

    # First modification: Update price
    price_update_result = product.update_price(Money(amount=120.0, currency="USD"))
    if price_update_result.is_success:
        updated_product_1 = price_update_result.data
        print(
            f"  💰 Price updated: {updated_product_1.price} (Version: {updated_product_1.version})",
        )

        # Second modification: Adjust stock
        stock_update_result = updated_product_1.adjust_stock(-10, "Sales")
        if stock_update_result.is_success:
            updated_product_2 = stock_update_result.data
            print(
                f"  📦 Stock adjusted: {updated_product_2.stock_quantity} (Version: {updated_product_2.version})",
            )

            # Show version progression
            print("\n📊 Version History:")
            print(f"  📝 Original: Version {product.version}")
            print(f"  📝 After price update: Version {updated_product_1.version}")
            print(f"  📝 After stock update: Version {updated_product_2.version}")


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
            email=Email(value=f"customer{i}@example.com"),
            address=Address(
                street=f"{i} Test Street",
                city="Test City",
                postal_code="12345",
                country="USA",
            ),
        )

    creation_time = time.time() - start_time
    print(
        f"  🔹 {operations} Customer creations: {creation_time:.4f}s ({operations / creation_time:.0f}/s)",
    )

    # Entity operation performance
    print("\n📋 Entity Operation Performance:")

    # Create a customer for operations
    customer_result = customer_factory(
        name="Performance Test Customer",
        email=Email(value="perf@example.com"),
        address=Address(
            street="123 Perf St",
            city="Perf City",
            postal_code="12345",
            country="USA",
        ),
    )

    if customer_result.is_success:
        customer = customer_result.data

        # Test copy_with performance
        start_time = time.time()
        current_customer = customer

        for i in range(100):
            result = current_customer.increment_order_count()
            if result.is_success:
                current_customer = result.data

        operation_time = time.time() - start_time
        print(
            f"  🔹 100 Entity operations: {operation_time:.4f}s ({100 / operation_time:.0f}/s)",
        )

        # Final state
        print(
            f"  📊 Final state: Version {current_customer.version}, Orders {current_customer.total_orders}",
        )

    # Value object operation performance
    print("\n📋 Value Object Performance:")
    money1 = Money(amount=100.0, currency="USD")
    money2 = Money(amount=50.0, currency="USD")

    start_time = time.time()
    for _ in range(operations):
        money1.add(money2)

    value_time = time.time() - start_time
    print(
        f"  🔹 {operations} Money additions: {value_time:.4f}s ({operations / value_time:.0f}/s)",
    )


def main() -> None:
    """Run comprehensive FlextEntity/ValueObject DDD demonstration."""
    print("=" * 80)
    print("🏢 FLEXT ENTITY/VALUE OBJECT - DOMAIN-DRIVEN DESIGN PATTERNS DEMONSTRATION")
    print("=" * 80)

    # Example 1: Value Objects
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: Value Object Patterns")
    print("=" * 60)
    demonstrate_value_objects()

    # Example 2: Entity Lifecycle
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: Entity Lifecycle Management")
    print("=" * 60)
    demonstrate_entity_lifecycle()

    # Example 3: Aggregate Patterns
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: Aggregate Patterns and Domain Services")
    print("=" * 60)
    demonstrate_aggregate_patterns()

    # Example 4: Repository Patterns
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: Repository Patterns")
    print("=" * 60)
    demonstrate_repository_patterns()

    # Example 5: Version Management
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Version Management and Optimistic Locking")
    print("=" * 60)
    demonstrate_version_management()

    # Example 6: Performance Characteristics
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 6: Performance Characteristics")
    print("=" * 60)
    demonstrate_performance_characteristics()

    print("\n" + "=" * 80)
    print("🎉 FLEXT ENTITY/VALUE OBJECT DDD DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
