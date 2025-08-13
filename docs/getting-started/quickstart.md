# Quick Start Guide

Get started with FLEXT Core in 5 minutes with practical, working examples.

## Installation

### Development Setup

```bash
# Clone repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Setup environment
make setup

# Verify installation
python -c "from flext_core import FlextResult; print('✅ Ready')"
```

### As a Dependency

```bash
# Using Poetry (recommended)
poetry add flext-core

# Using pip
pip install flext-core
```

## Core Concepts

### 1. Railway-Oriented Programming with FlextResult

Replace exceptions with explicit, type-safe error handling:

```python
from flext_core import FlextResult

def divide(a: float, b: float) -> FlextResult[float]:
    """Safe division without exceptions."""
    if b == 0:
        return FlextResult.fail("Cannot divide by zero")
    return FlextResult.ok(a / b)

# Chain operations safely
result = (
    divide(10, 2)           # Returns FlextResult.ok(5.0)
    .map(lambda x: x * 2)   # Transform: 10.0
    .flat_map(lambda x: divide(x, 4))  # Chain: 2.5
)

if result.success:
    print(f"Result: {result.unwrap()}")  # Result: 2.5
else:
    print(f"Error: {result.error}")
```

### 2. Dependency Injection Container

Manage services with a global, type-safe container:

```python
from flext_core import get_flext_container

# Get singleton container
container = get_flext_container()

# Register services
class DatabaseService:
    def connect(self) -> str:
        return "Connected to database"

class EmailService:
    def send(self, to: str, subject: str) -> str:
        return f"Email sent to {to}: {subject}"

container.register("database", DatabaseService())
container.register("email", EmailService())

# Retrieve services safely
db_result = container.get("database")
if db_result.success:
    db = db_result.unwrap()
    print(db.connect())  # Connected to database

# Use in classes
class OrderService:
    def __init__(self):
        self.db = container.get("database").unwrap()
        self.email = container.get("email").unwrap()

    def create_order(self, customer_email: str) -> FlextResult[str]:
        # Use injected services
        self.db.connect()
        self.email.send(customer_email, "Order Confirmed")
        return FlextResult.ok("Order created successfully")
```

### 3. Configuration Management

Environment-aware settings with validation:

```python
from flext_core import FlextSettings

class AppSettings(FlextSettings):
    """Application configuration."""
    app_name: str = "MyApp"
    debug: bool = False
    database_url: str = "sqlite:///app.db"
    api_key: str = "default-key"
    max_retries: int = 3

    class Config:
        env_prefix = "APP_"  # Reads from APP_DEBUG, APP_DATABASE_URL, etc.

# Automatically loads from environment variables
settings = AppSettings()

# Use in application
if settings.debug:
    print(f"Debug mode enabled for {settings.app_name}")
```

### 4. Domain-Driven Design

Build rich domain models with business logic:

```python
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot
from decimal import Decimal

# Value Object - Immutable, no identity
class Money(FlextValueObject):
    amount: Decimal
    currency: str

    def add(self, other: 'Money') -> FlextResult['Money']:
        if self.currency != other.currency:
            return FlextResult.fail("Currency mismatch")
        return FlextResult.ok(Money(
            amount=self.amount + other.amount,
            currency=self.currency
        ))

# Entity - Has identity, mutable
class Product(FlextEntity):
    name: str
    price: Money
    stock: int

    def purchase(self, quantity: int) -> FlextResult[Money]:
        if quantity > self.stock:
            return FlextResult.fail(f"Insufficient stock: {self.stock} available")

        self.stock -= quantity
        total = Money(
            amount=self.price.amount * quantity,
            currency=self.price.currency
        )

        # Raise domain event
        self.add_domain_event("ProductPurchased", {
            "product_id": self.id,
            "quantity": quantity,
            "total": str(total.amount)
        })

        return FlextResult.ok(total)

# Aggregate Root - Consistency boundary
class ShoppingCart(FlextAggregateRoot):
    customer_id: str
    items: list[tuple[Product, int]] = []

    def add_product(self, product: Product, quantity: int) -> FlextResult[None]:
        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")

        # Check stock availability
        if product.stock < quantity:
            return FlextResult.fail(f"Only {product.stock} items available")

        self.items.append((product, quantity))

        self.add_domain_event("ProductAddedToCart", {
            "cart_id": self.id,
            "product_id": product.id,
            "quantity": quantity
        })

        return FlextResult.ok(None)

    def checkout(self) -> FlextResult[Money]:
        if not self.items:
            return FlextResult.fail("Cart is empty")

        total = Money(amount=Decimal("0"), currency="USD")

        for product, quantity in self.items:
            purchase_result = product.purchase(quantity)
            if purchase_result.is_failure:
                return purchase_result

            add_result = total.add(purchase_result.unwrap())
            if add_result.is_failure:
                return add_result
            total = add_result.unwrap()

        self.add_domain_event("CartCheckedOut", {
            "cart_id": self.id,
            "total": str(total.amount),
            "currency": total.currency
        })

        return FlextResult.ok(total)
```

## Complete Example: User Registration System

Here's a real-world example combining all patterns:

```python
from flext_core import (
    FlextResult,
    get_flext_container,
    FlextEntity,
    FlextSettings
)
import hashlib
from datetime import datetime

# Configuration
class AppConfig(FlextSettings):
    database_url: str = "sqlite:///users.db"
    smtp_host: str = "localhost"
    smtp_port: int = 587
    require_email_verification: bool = True

    class Config:
        env_prefix = "USER_SYSTEM_"

# Domain Model
class User(FlextEntity):
    username: str
    email: str
    password_hash: str
    is_verified: bool = False
    created_at: datetime

    def verify_email(self) -> FlextResult[None]:
        if self.is_verified:
            return FlextResult.fail("Email already verified")

        self.is_verified = True
        self.add_domain_event("UserEmailVerified", {
            "user_id": self.id,
            "email": self.email
        })
        return FlextResult.ok(None)

    def change_password(self, old_hash: str, new_hash: str) -> FlextResult[None]:
        if self.password_hash != old_hash:
            return FlextResult.fail("Invalid current password")

        self.password_hash = new_hash
        self.add_domain_event("UserPasswordChanged", {
            "user_id": self.id,
            "changed_at": datetime.now().isoformat()
        })
        return FlextResult.ok(None)

# Services
class PasswordService:
    def hash_password(self, password: str) -> FlextResult[str]:
        if len(password) < 8:
            return FlextResult.fail("Password must be at least 8 characters")

        hashed = hashlib.sha256(password.encode()).hexdigest()
        return FlextResult.ok(hashed)

    def verify_password(self, password: str, hash: str) -> FlextResult[bool]:
        check_hash = hashlib.sha256(password.encode()).hexdigest()
        return FlextResult.ok(check_hash == hash)

class EmailService:
    def __init__(self, config: AppConfig):
        self.config = config

    def send_verification_email(self, user: User) -> FlextResult[None]:
        if not self.config.require_email_verification:
            return FlextResult.ok(None)

        # Simulate email sending
        print(f"📧 Verification email sent to {user.email}")
        return FlextResult.ok(None)

class UserRepository:
    def __init__(self):
        self.users: dict[str, User] = {}

    def save(self, user: User) -> FlextResult[None]:
        self.users[user.id] = user
        return FlextResult.ok(None)

    def find_by_email(self, email: str) -> FlextResult[User]:
        for user in self.users.values():
            if user.email == email:
                return FlextResult.ok(user)
        return FlextResult.fail(f"User not found with email: {email}")

# Use Case
class UserRegistrationService:
    def __init__(self):
        container = get_flext_container()
        self.config = AppConfig()
        self.password_service = PasswordService()
        self.email_service = EmailService(self.config)
        self.repository = UserRepository()

    def register_user(self, username: str, email: str,
                      password: str) -> FlextResult[User]:
        """Complete user registration flow."""

        # Validate input
        validation = self._validate_registration(username, email, password)
        if validation.is_failure:
            return validation

        # Check if user exists
        existing = self.repository.find_by_email(email)
        if existing.success:
            return FlextResult.fail("Email already registered")

        # Hash password
        hash_result = self.password_service.hash_password(password)
        if hash_result.is_failure:
            return hash_result.map(lambda _: User(...))  # Type conversion

        # Create user
        user = User(
            id=f"user_{hashlib.md5(email.encode()).hexdigest()[:8]}",
            username=username,
            email=email,
            password_hash=hash_result.unwrap(),
            created_at=datetime.now()
        )

        # Save user
        save_result = self.repository.save(user)
        if save_result.is_failure:
            return FlextResult.fail(f"Failed to save user: {save_result.error}")

        # Send verification email
        email_result = self.email_service.send_verification_email(user)
        if email_result.is_failure:
            # Log but don't fail registration
            print(f"Warning: Email failed: {email_result.error}")

        return FlextResult.ok(user)

    def _validate_registration(self, username: str, email: str,
                              password: str) -> FlextResult[None]:
        """Validate registration input."""
        if not username or len(username) < 3:
            return FlextResult.fail("Username must be at least 3 characters")

        if "@" not in email or "." not in email:
            return FlextResult.fail("Invalid email format")

        if len(password) < 8:
            return FlextResult.fail("Password must be at least 8 characters")

        return FlextResult.ok(None)

# Usage
def main():
    print("🚀 User Registration System\n")

    service = UserRegistrationService()

    # Successful registration
    result = service.register_user(
        username="johndoe",
        email="john@example.com",
        password="SecurePass123"
    )

    if result.success:
        user = result.unwrap()
        print(f"✅ User registered successfully!")
        print(f"   ID: {user.id}")
        print(f"   Username: {user.username}")
        print(f"   Email: {user.email}")
        print(f"   Verified: {user.is_verified}")
    else:
        print(f"❌ Registration failed: {result.error}")

    # Failed registration (duplicate email)
    result2 = service.register_user(
        username="janedoe",
        email="john@example.com",  # Same email
        password="AnotherPass456"
    )

    if result2.is_failure:
        print(f"\n❌ Expected error: {result2.error}")

if __name__ == "__main__":
    main()
```

## Testing Your Code

FLEXT Core is designed for testability:

```python
import pytest
from flext_core import FlextResult, FlextContainer

def test_user_registration():
    """Test user registration flow."""
    service = UserRegistrationService()

    # Test successful registration
    result = service.register_user(
        username="testuser",
        email="test@example.com",
        password="TestPass123"
    )

    assert result.success
    user = result.unwrap()
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert not user.is_verified

def test_password_validation():
    """Test password validation."""
    service = PasswordService()

    # Test short password
    result = service.hash_password("short")
    assert result.is_failure
    assert "8 characters" in result.error

    # Test valid password
    result = service.hash_password("ValidPassword123")
    assert result.success
    assert len(result.unwrap()) == 64  # SHA256 hex length

def test_container_isolation():
    """Test container service isolation."""
    container = FlextContainer()  # New isolated container

    # Register test service
    container.register("test", "test_value")

    # Verify registration
    result = container.get("test")
    assert result.success
    assert result.unwrap() == "test_value"
```

## Next Steps

### Learn More

- [Architecture Overview](../architecture/overview.md) - Understand the design
- [API Reference](../api/core.md) - Complete API documentation
- [Examples](../../examples/) - More working examples

### Development Commands

```bash
# Run tests
make test

# Check code quality
make lint
make type-check

# Run all validations
make validate

# Generate coverage report
make coverage-html
```

### Best Practices

1. **Always use FlextResult** for operations that can fail
2. **Chain operations** with map and flat_map
3. **Register services early** in application startup
4. **Keep domain logic** in entities and value objects
5. **Use type hints** for all function signatures

---

**Ready to build?** You now have everything needed to create robust applications with FLEXT Core!
