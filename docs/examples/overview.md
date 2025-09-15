# FLEXT-Core Examples

**Practical examples demonstrating FLEXT-Core foundation patterns**

---

## Quick Start Examples

### Core Imports

All FLEXT-Core functionality is available through root-level imports:

```python
from flext_core import (
    FlextResult,          # Railway-oriented programming
    FlextContainer,       # Dependency injection
    FlextModels,          # Domain modeling patterns
    FlextConfig,          # Configuration management
    FlextLogger,          # Structured logging
    FlextDomainService,   # Service architecture
)
```

## Example 1: Railway-Oriented Programming

### Basic Error Handling

```python
from flext_core import FlextResult

def divide_numbers(a: float, b: float) -> FlextResult[float]:
    """Safe division with explicit error handling."""
    if b == 0:
        return FlextResult[float].fail("Cannot divide by zero")
    return FlextResult[float].ok(a / b)

# Usage
result = divide_numbers(10, 2)
if result.is_success:
    print(f"Result: {result.unwrap()}")  # Result: 5.0
else:
    print(f"Error: {result.error}")
```

### Chaining Operations

```python
from flext_core import FlextResult

def validate_number(x: float) -> FlextResult[float]:
    """Validate number is positive."""
    if x <= 0:
        return FlextResult[float].fail("Number must be positive")
    return FlextResult[float].ok(x)

def square_root(x: float) -> FlextResult[float]:
    """Calculate square root."""
    import math
    return FlextResult[float].ok(math.sqrt(x))

# Chain operations with automatic error propagation
result = (
    validate_number(16.0)
    .flat_map(square_root)  # Only runs if validation succeeds
    .map(lambda x: round(x, 2))  # Transform the result
)

if result.is_success:
    print(f"Square root: {result.unwrap()}")  # Square root: 4.0
```

## Example 2: Dependency Injection

### Service Registration

```python
from flext_core import FlextContainer, FlextResult

class DatabaseService:
    """Example database service."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def save(self, data: dict) -> FlextResult[str]:
        # Simulate saving to database
        record_id = f"record_{hash(str(data)) % 10000:04d}"
        return FlextResult[str].ok(record_id)

class UserService:
    """User service with injected dependencies."""

    def __init__(self, database: DatabaseService):
        self.database = database

    def create_user(self, name: str, email: str) -> FlextResult[dict]:
        user_data = {"name": name, "email": email}

        save_result = self.database.save(user_data)
        if save_result.is_failure:
            return FlextResult[dict].fail(f"Save failed: {save_result.error}")

        user_data["id"] = save_result.unwrap()
        return FlextResult[dict].ok(user_data)

# Setup container
container = FlextContainer.get_global()

# Register services
db_service = DatabaseService("sqlite:///app.db")
container.register("database", db_service)

user_service = UserService(db_service)
container.register("user_service", user_service)

# Use services
service_result = container.get("user_service")
if service_result.is_success:
    user_service = service_result.unwrap()
    result = user_service.create_user("John Doe", "john@example.com")
    if result.is_success:
        print(f"User created: {result.unwrap()}")
```

## Example 3: Domain Modeling

### Entity with Business Logic

```python
from flext_core import FlextModels, FlextResult
from datetime import datetime

class BankAccount(FlextModels.Entity):
    """Bank account entity with business rules."""

    account_number: str
    balance: float
    owner_name: str
    is_active: bool = True
    daily_limit: float = 1000.0
    daily_withdrawn: float = 0.0

    def deposit(self, amount: float) -> FlextResult[float]:
        """Deposit money into account."""
        if amount <= 0:
            return FlextResult[float].fail("Deposit amount must be positive")

        if not self.is_active:
            return FlextResult[float].fail("Account is not active")

        self.balance += amount
        self.add_domain_event("MoneyDeposited", {
            "account": self.account_number,
            "amount": amount,
            "new_balance": self.balance
        })

        return FlextResult[float].ok(self.balance)

    def withdraw(self, amount: float) -> FlextResult[float]:
        """Withdraw money from account."""
        if amount <= 0:
            return FlextResult[float].fail("Withdrawal amount must be positive")

        if not self.is_active:
            return FlextResult[float].fail("Account is not active")

        if amount > self.balance:
            return FlextResult[float].fail("Insufficient funds")

        if self.daily_withdrawn + amount > self.daily_limit:
            return FlextResult[float].fail("Daily withdrawal limit exceeded")

        self.balance -= amount
        self.daily_withdrawn += amount

        self.add_domain_event("MoneyWithdrawn", {
            "account": self.account_number,
            "amount": amount,
            "new_balance": self.balance
        })

        return FlextResult[float].ok(self.balance)

# Usage
account = BankAccount(
    id="acc_001",
    account_number="1234567890",
    balance=1000.0,
    owner_name="Jane Smith"
)

# Deposit money
deposit_result = account.deposit(500.0)
if deposit_result.is_success:
    print(f"New balance after deposit: ${deposit_result.unwrap():.2f}")

# Withdraw money
withdraw_result = account.withdraw(200.0)
if withdraw_result.is_success:
    print(f"New balance after withdrawal: ${withdraw_result.unwrap():.2f}")

# Check domain events
events = account.get_domain_events()
for event in events:
    print(f"Event: {event}")
```

### Value Objects

```python
from flext_core import FlextModels, FlextResult
from decimal import Decimal

class Money(FlextModels.Value):
    """Immutable money value object."""

    amount: Decimal
    currency: str = "USD"

    def add(self, other: 'Money') -> FlextResult['Money']:
        """Add two money values."""
        if self.currency != other.currency:
            return FlextResult[Money].fail(
                f"Cannot add {self.currency} and {other.currency}"
            )

        return FlextResult[Money].ok(Money(
            amount=self.amount + other.amount,
            currency=self.currency
        ))

    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"

# Usage
price = Money(amount=Decimal("19.99"), currency="USD")
tax = Money(amount=Decimal("1.80"), currency="USD")

total_result = price.add(tax)
if total_result.is_success:
    print(f"Total: {total_result.unwrap()}")  # USD 21.79
```

## Example 4: Service Architecture

### Domain Service with Dependencies

```python
from flext_core import FlextDomainService, FlextContainer, FlextLogger, FlextResult

class OrderService(FlextDomainService):
    """Order service using FLEXT-Core patterns."""

    def __init__(self) -> None:
        super().__init__()
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)

    def process_order(self, order_data: dict) -> FlextResult[dict]:
        """Process order with logging and error handling."""
        self._logger.info("Processing order", extra={"order_data": order_data})

        # Validate order data
        if not order_data.get("customer_id"):
            return FlextResult[dict].fail("Customer ID is required")

        if not order_data.get("items"):
            return FlextResult[dict].fail("Order must have items")

        # Process order
        order_id = f"order_{hash(str(order_data)) % 10000:04d}"
        processed_order = {
            "id": order_id,
            "status": "processed",
            **order_data
        }

        self._logger.info(
            "Order processed successfully",
            extra={"order_id": order_id}
        )

        return FlextResult[dict].ok(processed_order)

# Usage
order_service = OrderService()

order_data = {
    "customer_id": "cust_123",
    "items": [
        {"product": "Widget", "quantity": 2, "price": 10.00}
    ]
}

result = order_service.process_order(order_data)
if result.is_success:
    print(f"Order processed: {result.unwrap()}")
```

## Example 5: Configuration Management

### Environment-Aware Configuration

```python
from flext_core import FlextConfig
from pydantic import BaseSettings, Field

class AppConfig(BaseSettings):
    """Application configuration with environment support."""

    app_name: str = Field("FLEXT Demo", description="Application name")
    debug: bool = Field(False, description="Debug mode")
    database_url: str = Field("sqlite:///app.db", description="Database URL")
    api_port: int = Field(8000, description="API port")
    log_level: str = Field("INFO", description="Logging level")

    class Config:
        env_file = ".env"
        env_prefix = "APP_"
        case_sensitive = False

# Usage
config = AppConfig()

print(f"App: {config.app_name}")
print(f"Database: {config.database_url}")
print(f"Port: {config.api_port}")
print(f"Debug: {config.debug}")

# Configuration integrates with other FLEXT-Core patterns
if config.debug:
    from flext_core import FlextLogger
    logger = FlextLogger(__name__)
    logger.info("Running in debug mode", extra={"config": config.dict()})
```

## Example 6: Complete Application Pattern

### Putting It All Together

```python
from flext_core import (
    FlextResult, FlextContainer, FlextModels,
    FlextDomainService, FlextLogger
)
from pydantic import BaseSettings

# Configuration
class AppConfig(BaseSettings):
    database_url: str = "sqlite:///shop.db"
    debug: bool = False

    class Config:
        env_prefix = "SHOP_"

# Domain Model
class Product(FlextModels.Entity):
    name: str
    price: float
    stock: int

    def reduce_stock(self, quantity: int) -> FlextResult[None]:
        if quantity > self.stock:
            return FlextResult[None].fail("Insufficient stock")

        self.stock -= quantity
        self.add_domain_event("StockReduced", {
            "product_id": self.id,
            "quantity": quantity,
            "remaining_stock": self.stock
        })

        return FlextResult[None].ok(None)

# Services
class InventoryService(FlextDomainService):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._logger = FlextLogger(__name__)

    def purchase_product(self, product_id: str, quantity: int) -> FlextResult[dict]:
        """Purchase product reducing stock."""
        self._logger.info(
            "Processing purchase",
            extra={"product_id": product_id, "quantity": quantity}
        )

        # In real app, load from database
        product = Product(
            id=product_id,
            name="Sample Product",
            price=29.99,
            stock=10
        )

        # Reduce stock
        stock_result = product.reduce_stock(quantity)
        if stock_result.is_failure:
            return FlextResult[dict].fail(f"Purchase failed: {stock_result.error}")

        # Create purchase record
        purchase = {
            "product_id": product_id,
            "quantity": quantity,
            "total_price": product.price * quantity,
            "remaining_stock": product.stock
        }

        self._logger.info("Purchase completed", extra=purchase)
        return FlextResult[dict].ok(purchase)

# Application Setup
def setup_application() -> FlextContainer:
    """Setup application with dependency injection."""
    container = FlextContainer.get_global()

    # Register configuration
    config = AppConfig()
    container.register("config", config)

    # Register services
    inventory_service = InventoryService(config)
    container.register("inventory", inventory_service)

    return container

# Usage
if __name__ == "__main__":
    container = setup_application()

    # Get service and use it
    service_result = container.get("inventory")
    if service_result.is_success:
        inventory = service_result.unwrap()

        result = inventory.purchase_product("prod_001", 3)
        if result.is_success:
            print(f"Purchase successful: {result.unwrap()}")
        else:
            print(f"Purchase failed: {result.error}")
```

## Running Examples

### Prerequisites

```bash
# Navigate to project directory
cd flext-core

# Install dependencies
make setup

# Verify installation
PYTHONPATH=src python -c "from flext_core import FlextResult; print('Ready!')"
```

### Testing Examples

Save any example to a Python file and run:

```bash
# Save example to file
cat > example_railway.py << 'EOF'
from flext_core import FlextResult

def divide_numbers(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult[float].fail("Cannot divide by zero")
    return FlextResult[float].ok(a / b)

result = divide_numbers(10, 2)
print(f"Result: {result.unwrap()}" if result.is_success else f"Error: {result.error}")
EOF

# Run with correct Python path
PYTHONPATH=src python example_railway.py
```

## Best Practices

### 1. Always Use FlextResult for Error Handling

```python
# ✅ Good - Explicit error handling
def safe_operation(data: dict) -> FlextResult[str]:
    if not data:
        return FlextResult[str].fail("Data required")
    return FlextResult[str].ok("processed")

# ❌ Avoid - Exception-based errors
def unsafe_operation(data: dict) -> str:
    if not data:
        raise ValueError("Data required")
    return "processed"
```

### 2. Chain Operations with Railway Pattern

```python
# ✅ Good - Chained operations
result = (
    validate_input(data)
    .flat_map(process_data)
    .map(format_output)
)

# ❌ Avoid - Nested error checking
validation = validate_input(data)
if validation.is_success:
    processing = process_data(validation.unwrap())
    if processing.is_success:
        result = format_output(processing.unwrap())
```

### 3. Use Dependency Injection

```python
# ✅ Good - Dependencies injected
class Service:
    def __init__(self, database, logger):
        self.database = database
        self.logger = logger

# ❌ Avoid - Hard-coded dependencies
class Service:
    def __init__(self):
        self.database = DatabaseConnection()  # Hard-coded
        self.logger = Logger()  # Hard-coded
```

---

**Example Status**: All examples use FLEXT-Core v0.9.0 actual API and have been tested with the current implementation. For more advanced patterns, see [development/best-practices.md](../development/best-practices.md).