# Getting Started with FLEXT-Core

**Foundation library installation and first steps**
**Date**: September 17, 2025 | **Version**: 0.9.0

---

## Prerequisites

### System Requirements

- **Python**: 3.13+ (required for type safety features)
- **Poetry**: Latest version for dependency management
- **Git**: For source code access

### Environment Setup

```bash
# Verify Python version
python --version  # Should be 3.13+

# Install Poetry if not available
curl -sSL https://install.python-poetry.org | python3 -

# Verify Poetry installation
poetry --version
```

---

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Set up development environment
make setup

# Verify installation
python -c "from flext_core import FlextResult; print('✓ FLEXT-Core ready')"
```

### Verify Installation

```bash
# Run quality checks
make validate

# Run tests
make test

# Check coverage
pytest tests/ --cov=src --cov-report=term
```

---

## Basic Usage Examples

### Railway Pattern with FlextResult

```python
from flext_core import FlextResult

def safe_divide(a: float, b: float) -> FlextResult[float]:
    """Safe division with error handling."""
    if b == 0:
        return FlextResult[float].fail("Division by zero")
    return FlextResult[float].ok(a / b)

# Usage
result = safe_divide(10, 2)
if result.is_success:
    print(f"Result: {result.unwrap()}")  # 5.0
else:
    print(f"Error: {result.error}")

# Chain operations
result = (
    safe_divide(10, 2)
    .map(lambda x: x * 2)
    .flat_map(lambda x: safe_divide(x, 5))
)
print(result.unwrap())  # 2.0
```

### Dependency Injection with FlextContainer

```python
from flext_core import FlextContainer

# Service interface
class DatabaseService:
    def connect(self) -> str:
        return "Connected to database"

# Register service
container = FlextContainer.get_global()
db_service = DatabaseService()
register_result = container.register("database", db_service)

if register_result.is_success:
    # Retrieve service
    service_result = container.get("database")
    if service_result.is_success:
        db = service_result.unwrap()
        print(db.connect())
```

### Domain Modeling with FlextModels

```python
from flext_core import FlextModels, FlextResult

class User(FlextModels.Entity):
    """User entity with business logic."""
    name: str
    email: str
    is_active: bool = False

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.is_active:
            return FlextResult[None].fail("User already active")

        self.is_active = True
        self.add_domain_event("UserActivated", {"user_id": self.id})
        return FlextResult[None].ok(None)

# Usage
user = User(name="John Doe", email="john@example.com")
activation_result = user.activate()

if activation_result.is_success:
    print(f"User {user.name} activated successfully")
```

### Configuration Management with FlextConfig

```python
from flext_core import FlextConfig

class AppConfig(FlextConfig):
    """Application configuration with environment variables."""
    database_url: str
    api_key: str
    debug: bool = False
    port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False

# Create .env file:
# DATABASE_URL=postgresql://localhost:5432/mydb
# API_KEY=your-secret-key
# DEBUG=true
# PORT=3000

config = AppConfig()
print(f"Database: {config.database_url}")
print(f"Debug mode: {config.debug}")
```

---

## Core Patterns Overview

### 1. Railway-Oriented Programming

FlextResult[T] provides type-safe error handling without exceptions:

```python
# Success path
success = FlextResult[str].ok("Hello")
print(success.unwrap())  # "Hello"

# Failure path
failure = FlextResult[str].fail("Error occurred")
print(failure.error)  # "Error occurred"

# Pattern matching
result = some_operation()
match result:
    case result if result.is_success:
        process_success(result.unwrap())
    case result if result.is_failure:
        handle_error(result.error)
```

### 2. Dependency Injection Pattern

Singleton container for service lifecycle management:

```python
# Register services at application startup
container = FlextContainer.get_global()
container.register("logger", LoggingService())
container.register("cache", CacheService())
container.register("metrics", MetricsService())

# Use throughout application
def process_request(data: dict):
    logger = container.get("logger").unwrap()
    cache = container.get("cache").unwrap()

    logger.info("Processing request")
    # ... business logic
```

### 3. Domain-Driven Design

Structured domain modeling with clear boundaries:

```python
# Value Object (immutable)
class Money(FlextModels.Value):
    amount: Decimal
    currency: str

    def add(self, other: "Money") -> FlextResult["Money"]:
        if self.currency != other.currency:
            return FlextResult["Money"].fail("Currency mismatch")
        return FlextResult["Money"].ok(
            Money(amount=self.amount + other.amount, currency=self.currency)
        )

# Entity (with identity)
class Account(FlextModels.Entity):
    balance: Money
    owner_id: str

    def withdraw(self, amount: Money) -> FlextResult[None]:
        if amount.amount > self.balance.amount:
            return FlextResult[None].fail("Insufficient funds")

        # Business logic here
        return FlextResult[None].ok(None)

# Aggregate Root (consistency boundary)
class BankingSystem(FlextModels.AggregateRoot):
    accounts: list[Account]

    def transfer_money(self, from_id: str, to_id: str, amount: Money) -> FlextResult[None]:
        # Ensure consistency across multiple entities
        pass
```

---

## Development Workflow

### Code Quality Standards

```bash
# Before committing
make validate      # Run all quality checks
make format        # Auto-format code
make test          # Run test suite

# Type checking
make type-check    # MyPy strict mode

# Linting
make lint          # Ruff linting
```

### Testing Your Code

```bash
# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests
pytest tests/ -m "not slow"          # Skip performance tests

# Coverage analysis
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html for detailed coverage report
```

---

## Next Steps

1. **Read the Architecture Guide**: [docs/architecture.md](architecture.md) - Understanding design patterns
2. **Explore API Reference**: [docs/api-reference.md](api-reference.md) - Complete API documentation
3. **Configuration Guide**: [docs/configuration.md](configuration.md) - Advanced configuration patterns
4. **Integration Patterns**: [docs/integration.md](integration.md) - Ecosystem integration
5. **Working Examples**: [examples/](../examples/) - Real-world usage patterns

---

## Common Issues

### Import Errors

```python
# ✓ Correct imports
from flext_core import FlextResult, FlextContainer, FlextModels

# ❌ Avoid internal imports
from flext_core.result import FlextResult  # Don't do this
```

### Type Safety

```python
# ✓ Correct type annotations
def process_data(data: str) -> FlextResult[str]:
    return FlextResult[str].ok(data.upper())

# ❌ Missing type annotations
def process_data(data):  # Missing types
    return FlextResult.ok(data.upper())  # Missing generic type
```

### Container Usage

```python
# ✓ Correct container pattern
container = FlextContainer.get_global()
service_result = container.get("service")
if service_result.is_success:
    service = service_result.unwrap()

# ❌ Direct access without error handling
service = container.get("service").unwrap()  # Can fail
```

For more troubleshooting information, see [docs/troubleshooting.md](troubleshooting.md).

---

**Getting Started Complete** - Ready to build with FLEXT-Core foundation patterns!