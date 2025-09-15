# Getting Started with FLEXT-Core

**Foundation library for the FLEXT ecosystem** providing railway-oriented programming, dependency injection, and domain modeling patterns with type safety.

---

## Installation

### Requirements

- Python 3.13+
- Poetry (for development)
- Dependencies managed via pyproject.toml

### Development Installation

```bash
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup

# Verify installation
PYTHONPATH=src python -c "from flext_core import FlextResult; print('FLEXT-Core ready')"
```

### Production Installation

*Note: Package not yet published to PyPI. Use development installation for now.*

---

## Quick Start

### Basic Error Handling with Railway Pattern

```python
from flext_core import FlextResult

def divide_numbers(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult[float].fail("Division by zero")
    return FlextResult[float].ok(a / b)

# Use the result
result = divide_numbers(10, 2)
if result.is_success:
    print(f"Result: {result.unwrap()}")  # Result: 5.0
else:
    print(f"Error: {result.error}")

# Chain operations with railway pattern
def process_numbers(a: float, b: float, c: float) -> FlextResult[float]:
    return (
        divide_numbers(a, b)
        .map(lambda x: x * c)  # Transform success value
        .filter(lambda x: x > 0, "Result must be positive")  # Conditional check
    )
```

### Dependency Injection Container

```python
from flext_core import FlextContainer

# Get global container singleton
container = FlextContainer.get_global()

# Register services (returns FlextResult for error handling)
register_result = container.register("database", DatabaseConnection())
if register_result.is_failure:
    print(f"Registration failed: {register_result.error}")

# Retrieve services with type safety
db_result = container.get("database")
if db_result.is_success:
    database = db_result.unwrap()
```

### Domain Modeling Patterns

```python
from flext_core import FlextModels, FlextResult

# Entity with identity and lifecycle
class User(FlextModels.Entity):
    name: str
    email: str
    is_active: bool = False

    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult[None].fail("User already active")

        self.is_active = True
        self.add_domain_event("UserActivated", {"user_id": self.id})
        return FlextResult[None].ok(None)

# Value Object (immutable)
class Email(FlextModels.Value):
    address: str

    def validate(self) -> FlextResult[None]:
        if "@" not in self.address:
            return FlextResult[None].fail("Invalid email format")
        return FlextResult[None].ok(None)

# Usage
user = User(name="John", email="john@example.com")
activation_result = user.activate()
```

### Service Architecture

```python
from flext_core import FlextDomainService, FlextLogger, FlextResult

class UserService(FlextDomainService):
    def __init__(self) -> None:
        super().__init__()
        self._logger = FlextLogger(__name__)

    def create_user(self, user_data: dict) -> FlextResult[User]:
        self._logger.info("Creating user", extra={"data": user_data})

        if not user_data.get("email"):
            return FlextResult[User].fail("Email is required")

        user = User(**user_data)
        self._logger.info("User created successfully", extra={"user_id": user.id})
        return FlextResult[User].ok(user)
```

---

## Core Patterns

### Railway-Oriented Programming

Chain operations with automatic error propagation:

```python
from flext_core import FlextResult

def process_user_data(raw_data: dict) -> FlextResult[User]:
    """Complete user processing pipeline with error handling."""
    return (
        validate_input(raw_data)
        .flat_map(lambda d: parse_user_data(d))
        .flat_map(lambda d: enrich_user_data(d))
        .map(lambda d: create_user(d))
        .filter(lambda u: u.email is not None, "User must have email")
    )

# If any step fails, the entire pipeline fails gracefully
result = process_user_data(user_input)
```

### Configuration Management

```python
from flext_core import FlextConfig

class AppConfig(FlextConfig):
    """Application configuration with environment integration."""
    database_url: str
    api_key: str
    debug: bool = False
    timeout: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False

# Automatically loads from environment variables or .env file
config = AppConfig()
```

### Logging with Structure

```python
from flext_core import FlextLogger

logger = FlextLogger(__name__)

# Structured logging with context
logger.info(
    "User operation completed",
    extra={
        "user_id": "123",
        "operation": "activate",
        "duration_ms": 150
    }
)
```

---

## Testing Your Installation

### Quick Verification

```bash
# Run basic tests to verify installation
PYTHONPATH=src python -c "
from flext_core import FlextResult, FlextContainer, FlextModels
print('✅ Core imports successful')

# Test FlextResult
result = FlextResult[str].ok('test')
assert result.is_success
print('✅ FlextResult working')

# Test Container
container = FlextContainer.get_global()
print('✅ FlextContainer working')

print('FLEXT-Core installation verified!')
"
```

### Run Test Suite

```bash
# Run quality checks
make check              # Quick validation (lint + type-check)
make validate           # Full validation (lint + type + security + test)

# Run tests with coverage
make test               # Full test suite with 84% coverage
make test-fast          # Tests without coverage reporting
```

---

## Development Commands

Essential commands for working with FLEXT-Core:

```bash
# Setup and installation
make setup              # Complete development environment setup
make install            # Install dependencies only

# Quality gates (mandatory before commits)
make validate           # All quality gates (lint + type + security + test)
make check              # Quick validation (lint + type-check)
make lint               # Code linting with Ruff
make type-check         # MyPy type checking (strict mode)
make test               # Test suite with coverage (84% current)

# Development utilities
make format             # Auto-format code
make fix                # Auto-fix linting issues
make shell              # Python REPL with project loaded
make clean              # Clean build artifacts
```

---

## Integration with FLEXT Ecosystem

FLEXT-Core serves as the foundation for the FLEXT ecosystem with:

- **Error Handling**: `FlextResult[T]` provides railway-oriented programming across all projects
- **Dependency Injection**: `FlextContainer.get_global()` manages services throughout the ecosystem
- **Domain Modeling**: `FlextModels` provides Entity, Value Object, and Aggregate Root patterns
- **Configuration**: `FlextConfig` offers environment-aware settings management
- **Logging**: `FlextLogger` provides structured logging with context
- **Type Safety**: Complete Python 3.13+ type annotations set the ecosystem standard

---

## Next Steps

- **[Architecture](architecture.md)** - Design patterns and architectural principles
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](examples/)** - Working code examples and use cases
- **[Development](development.md)** - Contributing guidelines and development workflow