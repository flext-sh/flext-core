# Development Best Practices

Guidelines for developing with FLEXT Core based on Clean Architecture and Domain-Driven Design principles.

## Core Principles

### 1. Railway-Oriented Programming

Always use `FlextResult` for operations that can fail:

```python
from flext_core import FlextResult

# ✅ GOOD - Explicit error handling
def process_payment(amount: Decimal, card: str) -> FlextResult[Transaction]:
    """Process payment with explicit error handling."""
    if amount <= 0:
        return FlextResult[None].fail("Amount must be positive")

    if not validate_card(card):
        return FlextResult[None].fail("Invalid card number")

    transaction = Transaction(amount=amount, card=card)
    return FlextResult[None].ok(transaction)

# ❌ BAD - Using exceptions for business logic
def process_payment_bad(amount: Decimal, card: str) -> Transaction:
    if amount <= 0:
        raise ValueError("Amount must be positive")  # Don't use exceptions

    return Transaction(amount=amount, card=card)
```

### 2. Type Safety First

Use comprehensive type hints with generics:

```python
from typing import TypeVar, Generic
from flext_core import FlextResult

T = TypeVar('T')
E = TypeVar('E')

# ✅ GOOD - Full type annotations
def transform_data[T, U](
    data: T,
    transformer: Callable[[T], U]
) -> FlextResult[U]:
    """Transform data with type safety."""
    try:
        result = transformer(data)
        return FlextResult[None].ok(result)
    except Exception as e:
        return FlextResult[None].fail(f"Transformation failed: {e}")

# ❌ BAD - Missing type hints
def transform_data_bad(data, transformer):
    return transformer(data)
```

### 3. Dependency Injection

Use the container pattern for service management:

```python
from flext_core import get_flext_container

# ✅ GOOD - Dependencies injected
class OrderService:
    def __init__(self, payment_gateway, inventory_service, email_service):
        self.payment = payment_gateway
        self.inventory = inventory_service
        self.email = email_service

    def process_order(self, order: Order) -> FlextResult[None]:
        # Use injected dependencies
        payment_result = self.payment.charge(order.total)
        if payment_result.is_failure:
            return payment_result

        return FlextResult[None].ok(None)

# Setup with container
container = FlextContainer.get_global()
container.register("payment", PaymentGateway())
container.register("inventory", InventoryService())
container.register("email", EmailService())

# ❌ BAD - Hard-coded dependencies
class OrderServiceBad:
    def __init__(self):
        self.payment = PaymentGateway()  # Hard-coded
        self.inventory = InventoryService()  # Hard-coded
```

## Architecture Patterns

### Clean Architecture Layers

Organize code by architectural boundaries:

```python
# Domain Layer - Pure business logic
# domain/entities.py
from flext_core import FlextModels.Entity, FlextResult

class Product(FlextModels.Entity):
    """Domain entity with business rules."""
    name: str
    price: Decimal
    stock: int

    def can_fulfill(self, quantity: int) -> bool:
        """Business rule - pure domain logic."""
        return self.stock >= quantity

    def reserve(self, quantity: int) -> FlextResult[None]:
        """Business operation."""
        if not self.can_fulfill(quantity):
            return FlextResult[None].fail(f"Insufficient stock: {self.stock} < {quantity}")

        self.stock -= quantity
        self.add_domain_event("ProductReserved", {
            "product_id": self.id,
            "quantity": quantity,
            "remaining": self.stock
        })
        return FlextResult[None].ok(None)

# Application Layer - Use case orchestration
# application/use_cases.py
class CreateOrderUseCase:
    """Application service orchestrating domain logic."""

    def __init__(self, order_repo, product_repo, event_bus):
        self.order_repo = order_repo
        self.product_repo = product_repo
        self.event_bus = event_bus

    def execute(self, request: CreateOrderRequest) -> FlextResult[Order]:
        """Execute use case."""
        # Load domain entities
        products = []
        for item in request.items:
            product_result = self.product_repo.find_by_id(item.product_id)
            if product_result.is_failure:
                return product_result
            products.append(product_result.unwrap())

        # Apply domain logic
        order = Order.create(request.customer_id)
        for product, item in zip(products, request.items):
            add_result = order.add_item(product, item.quantity)
            if add_result.is_failure:
                return add_result

        # Persist and publish events
        save_result = self.order_repo.save(order)
        if save_result.success:
            for event in order.get_events():
                self.event_bus.publish(event)

        return save_result

# Infrastructure Layer - External concerns
# infrastructure/repositories.py
class OrderRepository:
    """Infrastructure implementation."""

    def __init__(self, database):
        self.db = database

    def save(self, order: Order) -> FlextResult[Order]:
        """Persist order to database."""
        try:
            self.db.execute(
                "INSERT INTO orders ...",
                order.to_dict()
            )
            return FlextResult[None].ok(order)
        except DatabaseError as e:
            return FlextResult[None].fail(f"Database error: {e}")
```

### Domain-Driven Design

Model your domain with rich entities:

```python
from flext_core import FlextModels.Entity, FlextModels.Value, FlextAggregates

# Value Object - Immutable, no identity
class Address(FlextModels.Value):
    """Address value object."""
    street: str
    city: str
    postal_code: str
    country: str

    def format(self) -> str:
        """Format address for display."""
        return f"{self.street}, {self.city} {self.postal_code}, {self.country}"

# Entity - Has identity, mutable
class OrderItem(FlextModels.Entity):
    """Order item entity."""
    product_id: str
    product_name: str
    quantity: int
    unit_price: Decimal

    @property
    def total(self) -> Decimal:
        """Calculate item total."""
        return self.unit_price * self.quantity

    def adjust_quantity(self, new_quantity: int) -> FlextResult[None]:
        """Adjust item quantity."""
        if new_quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")

        old_quantity = self.quantity
        self.quantity = new_quantity

        self.add_domain_event("QuantityAdjusted", {
            "item_id": self.id,
            "old_quantity": old_quantity,
            "new_quantity": new_quantity
        })

        return FlextResult[None].ok(None)

# Aggregate Root - Consistency boundary
class Order(FlextAggregates):
    """Order aggregate root."""
    customer_id: str
    items: list[OrderItem]
    shipping_address: Address
    status: OrderStatus

    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        """Add item maintaining invariants."""
        if self.status != OrderStatus.DRAFT:
            return FlextResult[None].fail("Cannot modify confirmed order")

        # Check product availability
        if not product.can_fulfill(quantity):
            return FlextResult[None].fail(f"Product {product.name} insufficient stock")

        # Reserve stock
        reserve_result = product.reserve(quantity)
        if reserve_result.is_failure:
            return reserve_result

        # Add item
        item = OrderItem(
            id=generate_id("item"),
            product_id=product.id,
            product_name=product.name,
            quantity=quantity,
            unit_price=product.price
        )
        self.items.append(item)

        self.add_domain_event("ItemAdded", {
            "order_id": self.id,
            "item_id": item.id,
            "product_id": product.id,
            "quantity": quantity
        })

        return FlextResult[None].ok(None)

    @property
    def total(self) -> Decimal:
        """Calculate order total."""
        return sum(item.total for item in self.items)

    def can_ship(self) -> bool:
        """Check if order can be shipped."""
        return (
            self.status == OrderStatus.PAID and
            self.shipping_address is not None and
            len(self.items) > 0
        )
```

## Error Handling Patterns

### Graceful Degradation

Handle failures with fallbacks:

```python
def get_user_preferences(user_id: str) -> FlextResult[dict]:
    """Get user preferences with fallback."""
    # Try primary source
    cache_result = cache.get(f"prefs:{user_id}")
    if cache_result.success:
        return cache_result

    # Try database
    db_result = database.get_preferences(user_id)
    if db_result.success:
        # Update cache for next time
        cache.set(f"prefs:{user_id}", db_result.unwrap())
        return db_result

    # Return defaults
    logger.warning(f"Using default preferences for user {user_id}")
    return FlextResult[None].ok(DEFAULT_PREFERENCES)
```

### Error Aggregation

Collect multiple errors:

```python
def validate_order(order: Order) -> FlextResult[Order]:
    """Validate order with multiple checks."""
    errors = []

    # Validate customer
    if not order.customer_id:
        errors.append("Customer ID required")

    # Validate items
    if not order.items:
        errors.append("Order must have at least one item")

    for item in order.items:
        if item.quantity <= 0:
            errors.append(f"Invalid quantity for item {item.product_name}")
        if item.unit_price < 0:
            errors.append(f"Invalid price for item {item.product_name}")

    # Validate shipping
    if not order.shipping_address:
        errors.append("Shipping address required")

    # Return aggregated result
    if errors:
        return FlextResult[None].fail(" | ".join(errors))

    return FlextResult[None].ok(order)
```

## Testing Strategies

### Test Organization

Structure tests by architectural layer:

```python
# tests/unit/domain/test_order.py
def test_order_add_item_success():
    """Test adding item to order."""
    order = Order(customer_id="customer_123")
    product = Product(id="prod_1", name="Widget", price=10.00, stock=5)

    result = order.add_item(product, 2)

    assert result.success
    assert len(order.items) == 1
    assert order.items[0].quantity == 2
    assert product.stock == 3  # Stock reduced

def test_order_add_item_insufficient_stock():
    """Test adding item with insufficient stock."""
    order = Order(customer_id="customer_123")
    product = Product(id="prod_1", name="Widget", price=10.00, stock=1)

    result = order.add_item(product, 2)

    assert result.is_failure
    assert "insufficient stock" in result.error.lower()
    assert len(order.items) == 0
    assert product.stock == 1  # Stock unchanged

# tests/integration/test_create_order_use_case.py
def test_create_order_complete_flow():
    """Test complete order creation flow."""
    # Setup
    container = setup_test_container()
    use_case = CreateOrderUseCase(
        container.get("order_repo").unwrap(),
        container.get("product_repo").unwrap(),
        container.get("event_bus").unwrap()
    )

    # Execute
    request = CreateOrderRequest(
        customer_id="customer_123",
        items=[
            OrderItemRequest(product_id="prod_1", quantity=2),
            OrderItemRequest(product_id="prod_2", quantity=1)
        ]
    )
    result = use_case.execute(request)

    # Verify
    assert result.success
    order = result.unwrap()
    assert order.customer_id == "customer_123"
    assert len(order.items) == 2

    # Verify events published
    events = container.get("event_bus").unwrap().get_published()
    assert any(e.type == "OrderCreated" for e in events)
```

### Test Fixtures

Use fixtures for common test data:

```python
# tests/conftest.py
import pytest
from flext_core import FlextContainer

@pytest.fixture
def container():
    """Provide clean container for tests."""
    return FlextContainer()

@pytest.fixture
def test_product():
    """Provide test product."""
    return Product(
        id="test_prod_1",
        name="Test Widget",
        price=Decimal("9.99"),
        stock=100
    )

@pytest.fixture
def test_order(test_product):
    """Provide test order with items."""
    order = Order(customer_id="test_customer")
    order.add_item(test_product, 2)
    return order
```

## Performance Optimization

### Lazy Loading

Load data only when needed:

```python
class Order(FlextAggregates):
    """Order with lazy-loaded items."""

    def __init__(self, **data):
        super().__init__(**data)
        self._items = None  # Lazy-loaded
        self._items_loaded = False

    @property
    def items(self) -> list[OrderItem]:
        """Load items on first access."""
        if not self._items_loaded:
            self._items = self._load_items()
            self._items_loaded = True
        return self._items

    def _load_items(self) -> list[OrderItem]:
        """Load items from repository."""
        # Implementation depends on your architecture
        pass
```

### Batch Operations

Process multiple items efficiently:

```python
def process_orders_batch(order_ids: list[str]) -> FlextResult[list[Order]]:
    """Process multiple orders efficiently."""
    # Load all at once
    orders_result = repository.find_by_ids(order_ids)
    if orders_result.is_failure:
        return orders_result

    orders = orders_result.unwrap()
    processed = []
    errors = []

    # Process each order
    for order in orders:
        result = process_single_order(order)
        if result.success:
            processed.append(result.unwrap())
        else:
            errors.append(f"Order {order.id}: {result.error}")

    # Return results
    if errors and not processed:
        return FlextResult[None].fail(" | ".join(errors))

    if errors:
        logger.warning(f"Partial batch failure: {errors}")

    return FlextResult[None].ok(processed)
```

## Code Quality

### Linting and Formatting

Always run quality checks:

```bash
# Before committing
make validate  # Runs all checks

# Individual checks
make lint       # Code style
make type-check # Type safety
make test       # Test suite
make format     # Auto-format
```

### Documentation

Document public APIs:

```python
def process_payment(
    amount: Decimal,
    payment_method: PaymentMethod,
    customer: Customer
) -> FlextResult[Payment]:
    """Process customer payment.

    Args:
        amount: Payment amount in customer's currency.
        payment_method: Customer's selected payment method.
        customer: Customer making the payment.

    Returns:
        FlextResult containing processed Payment on success,
        or error message on failure.

    Example:
        >>> payment = process_payment(
        ...     amount=Decimal("99.99"),
        ...     payment_method=card,
        ...     customer=customer
        ... )
        >>> if payment.success:
        ...     print(f"Payment ID: {payment.unwrap().id}")
    """
```

## Security Considerations

### Input Validation

Always validate external input:

```python
def create_user(request_data: dict) -> FlextResult[User]:
    """Create user with validation."""
    # Sanitize input
    email = request_data.get("email", "").strip().lower()
    name = request_data.get("name", "").strip()

    # Validate
    if not email or "@" not in email:
        return FlextResult[None].fail("Invalid email")

    if not name or len(name) < 2:
        return FlextResult[None].fail("Invalid name")

    if len(email) > 255:  # Prevent DoS
        return FlextResult[None].fail("Email too long")

    # Create user
    user = User(email=email, name=name)
    return FlextResult[None].ok(user)
```

### Sensitive Data

Never log sensitive information:

```python
def authenticate(email: str, password: str) -> FlextResult[User]:
    """Authenticate user."""
    logger.info(f"Authentication attempt for email: {email}")
    # Never log: logger.info(f"Password: {password}")  # NEVER DO THIS

    user_result = find_user_by_email(email)
    if user_result.is_failure:
        logger.warning(f"User not found: {email}")
        return FlextResult[None].fail("Invalid credentials")  # Generic message

    user = user_result.unwrap()
    if not verify_password(password, user.password_hash):
        logger.warning(f"Invalid password for: {email}")
        return FlextResult[None].fail("Invalid credentials")  # Same generic message

    logger.info(f"Successful authentication: {email}")
    return FlextResult[None].ok(user)
```

## Summary

### DO

✅ Use FlextResult for all fallible operations  
✅ Apply comprehensive type hints  
✅ Follow Clean Architecture layers  
✅ Use dependency injection  
✅ Write tests for success and failure paths  
✅ Document public APIs  
✅ Validate all external input  
✅ Handle errors explicitly

### DON'T

❌ Use exceptions for business logic  
❌ Create hard-coded dependencies  
❌ Skip type annotations  
❌ Ignore FlextResult failures  
❌ Mix architectural layers  
❌ Log sensitive data  
❌ Trust external input

---

For more examples, see the [Examples Guide](../examples/overview.md).
