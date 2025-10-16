# Getting Started with FLEXT-Core

Quick start guide for using FLEXT-Core v0.9.9 - the foundation library providing railway-oriented programming, dependency injection, and domain-driven design patterns with Python 3.13+.

## Prerequisites

- **Python**: 3.13+ (required)
- **Poetry**: Latest version (recommended) or pip
- **Git**: For source checkout

Verify your environment:

```bash
python --version  # Should be 3.13+
poetry --version  # Latest Poetry
```

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Setup development environment (includes pre-commit hooks)
make setup

# Or install dependencies only
make install
```

### Verification

```bash
# Quick verification
python -c "from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities; print('✅ FLEXT-Core ready')"

# Check version
python -c "from flext_core import __version__; print(f'FLEXT-Core {__version__}')"
```

## Core Concepts

### 1. Railway Pattern (FlextResult)

Handle errors without exceptions using the Result monad:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

def divide(a: int, b: int) -> FlextResult[float]:
    """Division with explicit error handling."""
    if b == 0:
        return FlextResult[float].fail("Division by zero")
    return FlextResult[float].ok(a / b)

# Usage
result = divide(10, 2)
if result.is_success:
    value = result.unwrap()  # Safe: 5.0
    print(f"Result: {value}")
else:
    print(f"Error: {result.error}")

# Chaining operations (railway-oriented composition)
result = (
    divide(10, 2)
    .map(lambda x: x * 2)           # Transform success value
    .flat_map(lambda x: divide(x, 5))  # Chain another operation
    .map_error(lambda e: f"Calculation failed: {e}")
)
```

### 2. Dependency Injection (FlextContainer)

Manage dependencies with the global singleton container:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

# Get global container instance
container = FlextContainer.get_global()

# Register services
logger = FlextLogger(__name__)
container.register("logger", logger)

# Retrieve services
logger_result = container.get("logger")
if logger_result.is_success:
    retrieved_logger = logger_result.unwrap()
    retrieved_logger.info("Container working!")
```

### 3. Domain Modeling (FlextModels)

Create domain entities with Pydantic v2 validation:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

# Entity - has identity
class User(FlextModels.Entity):
    """User entity with validation."""
    name: str
    email: str
    age: int

    def model_post_init(self, __context: object) -> None:
        """Post-initialization validation."""
        if self.age < 0:
            raise ValueError("Age cannot be negative")
        if "@" not in self.email:
            raise ValueError("Invalid email format")

# Value Object - compared by value
class Address(FlextModels.Value):
    """Address value object."""
    street: str
    city: str
    zip_code: str

# Create instances
user = User(id="user_123", name="Alice", email="alice@example.com", age=30)
address = Address(street="123 Main St", city="Springfield", zip_code="12345")

print(f"User: {user.name} ({user.email})")
```

### 4. Domain Services (FlextService)

Encapsulate business logic in domain services:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class UserService(FlextService):
    """User domain service."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = FlextLogger(__name__)

    def create_user(self, name: str, email: str, age: int) -> FlextResult[User]:
        """Create user with validation."""
        self.logger.info("Creating user", extra={"name": name})

        # Validate
        if age < 18:
            return FlextResult[User].fail("User must be 18 or older")

        if "@" not in email:
            return FlextResult[User].fail("Invalid email format")

        # Create entity
        try:
            user = User(id=f"user_{name.lower()}", name=name, email=email, age=age)
            self.logger.info("User created", extra={"user_id": user.id})
            return FlextResult[User].ok(user)
        except ValueError as e:
            return FlextResult[User].fail(str(e))

# Usage
service = UserService()
result = service.create_user("Bob", "bob@example.com", 25)

if result.is_success:
    user = result.unwrap()
    print(f"✅ Created: {user.name}")
else:
    print(f"❌ Error: {result.error}")
```

### 5. Configuration (FlextConfig)

Manage application configuration with multiple sources:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

# Define configuration schema
class AppConfig(FlextConfig):
    """Application configuration."""
    app_name: str = "myapp"
    debug: bool = False
    max_connections: int = 100

# Create configuration instance
config = AppConfig()
print(f"App: {config.app_name}")
print(f"Debug: {config.debug}")

# Create for specific environment
config = AppConfig()
```

### 6. Structured Logging (FlextLogger)

Use structured logging with context propagation:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

# Create logger
logger = FlextLogger(__name__)

# Log with structured data
logger.info("Application started")
logger.info("User login", extra={"user_id": "user_123", "ip": "192.168.1.1"})

# Log errors
# Log errors with Result pattern
result = divide(10, 0)
if not result.is_success:
    logger.error("Calculation failed", extra={"error": result.error})
```

## Complete Example

Here's a complete example combining all concepts:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

# 1. Define domain model
class Product(FlextModels.Entity):
    """Product entity."""
    name: str
    price: float
    quantity: int

    def model_post_init(self, __context: object) -> None:
        """Validate product."""
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if self.quantity < 0:
            raise ValueError("Quantity cannot be negative")

# 2. Create domain service
class ProductService(FlextService):
    """Product management service."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = FlextLogger(__name__)
        self._products: dict[str, Product] = {}

    def create_product(self, name: str, price: float, quantity: int) -> FlextResult[Product]:
        """Create a new product."""
        self.logger.info("Creating product", extra={"name": name})

        # Validation
        if price <= 0:
            return FlextResult[Product].fail("Price must be positive")

        # Create entity
        try:
            product = Product(
                id=f"product_{name.lower().replace(' ', '_')}",
                name=name,
                price=price,
                quantity=quantity
            )
            self._products[product.id] = product
            self.logger.info("Product created", extra={"product_id": product.id})
            return FlextResult[Product].ok(product)
        except ValueError as e:
            return FlextResult[Product].fail(str(e))

    def get_product(self, product_id: str) -> FlextResult[Product]:
        """Get product by ID."""
        product = self._products.get(product_id)
        if product is None:
            return FlextResult[Product].fail(f"Product not found: {product_id}")
        return FlextResult[Product].ok(product)

# 3. Setup dependency injection
container = FlextContainer.get_global()
product_service = ProductService()
container.register("product_service", product_service)

# 4. Use the service
service_result = container.get("product_service")
if service_result.is_success:
    service = service_result.unwrap()

    # Create products
    laptop = service.create_product("Laptop", 999.99, 10)
    if laptop.is_success:
        print(f"✅ Created: {laptop.unwrap().name}")

    mouse = service.create_product("Mouse", 29.99, 50)
    if mouse.is_success:
        print(f"✅ Created: {mouse.unwrap().name}")

    # Retrieve product
    product_result = service.get_product("product_laptop")
    if product_result.is_success:
        product = product_result.unwrap()
        print(f"📦 Product: {product.name} - ${product.price}")
```

## Running Tests

### Quick Test

```bash
# Run all tests
make test

# Run with coverage
make test-coverage
```

### Specific Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Specific module
pytest tests/unit/test_result.py -v
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory for more examples
2. **Read Architecture**: See [Architecture Overview](../architecture/overview.md) for architecture details
3. **API Reference**: Check [API Reference](../api-reference/) for complete API documentation
4. **Development Guide**: See [Development Guide](../development/contributing.md) for contributing guidelines

## Common Patterns

### Pattern 1: Validation with Railway

```python
def validate_and_process(data: dict) -> FlextResult[FlextTypes.Dict]:
    """Validate and process data."""
    return (
        validate_schema(data)
        .flat_map(lambda d: validate_business_rules(d))
        .map(lambda d: transform_data(d))
        .map_error(lambda e: f"Processing failed: {e}")
    )
```

### Pattern 2: Service with DI

```python
class MyService(FlextService):
    def __init__(self) -> None:
        super().__init__()
        self._container = FlextContainer.get_global()
        self._logger_result = self._container.get("logger")

    def process(self, data: dict) -> FlextResult[FlextTypes.Dict]:
        if self._logger_result.is_success:
            self._logger_result.unwrap().info("Processing", extra=data)
        # Business logic here
        return FlextResult[FlextTypes.Dict].ok(data)
```

### Pattern 3: Domain Event

```python
class Order(FlextModels.AggregateRoot):
    """Order aggregate root."""
    items: FlextTypes.StringList
    total: float

    def place_order(self) -> FlextResult[None]:
        """Place order and emit event."""
        if self.total <= 0:
            return FlextResult[None].fail("Order total must be positive")

        # Emit domain event
        self.add_domain_event("OrderPlaced", {"order_id": self.id, "total": self.total})

        return FlextResult[None].ok(None)
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're using Python 3.13+:

```bash
python --version
# Should be 3.13 or higher
```

### Type Errors

If you get type errors during development:

```bash
# Run type checker
make type-check

# Or directly
mypy src/
pyright src/
```

### Test Failures

If tests fail:

```bash
# Run tests with verbose output
pytest tests/ -v --tb=short

# Run specific failing test
pytest tests/unit/test_result.py::TestFlextResult::test_ok -v
```

## Getting Help

- **Documentation**: Browse `docs/` directory
- **Issues**: Report issues on GitHub
- **Examples**: Check `examples/` for working examples
- **Tests**: Look at `tests/` for usage patterns

---

**Ready to build with FLEXT-Core!** 🚀
