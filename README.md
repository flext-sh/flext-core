# flext-core

**Enterprise foundation library providing railway-oriented programming, dependency injection, and domain-driven design patterns for the FLEXT ecosystem.**

## Overview

FLEXT Core is the architectural foundation for 32+ projects in the FLEXT data integration ecosystem. It provides type-safe error handling, enterprise patterns, and clean architecture principles that eliminate common boilerplate and ensure consistency across all ecosystem projects.

### Key Features

- 🚂 **Railway-Oriented Programming** - Type-safe error handling with `FlextResult[T]` pattern
- 💉 **Dependency Injection** - Enterprise DI container with singleton management
- 🏛️ **Domain-Driven Design** - Rich entities, value objects, and aggregates
- 🎯 **Clean Architecture** - Clear separation between layers and concerns
- 🔒 **Type Safety** - MyPy strict mode with comprehensive type hints
- 📊 **Observability** - Built-in structured logging and correlation IDs
- 🧩 **Extensible** - Plugin-ready architecture for ecosystem growth

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Setup development environment
make setup

# Verify installation
python -c "from flext_core import FlextResult; print('✅ Working')"
```

### Basic Usage

#### Railway-Oriented Error Handling

```python
from flext_core import FlextResult

def process_user(user_id: str) -> FlextResult[User]:
    """All operations return FlextResult for composability."""
    if not user_id:
        return FlextResult[None].fail("Invalid user ID")

    user = User(id=user_id, name="John Doe")
    return FlextResult[None].ok(user)

# Chain operations safely
result = (
    process_user("123")
    .flat_map(lambda u: validate_user(u))
    .map(lambda u: enrich_user_data(u))
    .map_error(lambda e: log_error(e))
)

if result.success:
    user = result.unwrap()
    print(f"Processed user: {user.name}")
else:
    print(f"Error: {result.error}")
```

#### Dependency Injection Container

```python
from flext_core import get_flext_container

# Get global container instance
container = get_flext_container()

# Register services
container.register("database", DatabaseService())
container.register_factory("logger", lambda: create_logger())

# Retrieve services with type safety
db_result = container.get("database")
if db_result.success:
    db = db_result.unwrap()
    db.connect()
```

#### Domain-Driven Design

```python
from flext_core import FlextEntity, FlextValue, FlextAggregateRoot

class Email(FlextValue):
    """Immutable value object with built-in validation."""
    address: str

    def validate(self) -> FlextResult[None]:
        if "@" not in self.address:
            return FlextResult[None].fail("Invalid email format")
        return FlextResult[None].ok(None)

class User(FlextEntity):
    """Entity with identity and business logic."""
    name: str
    email: Email
    is_active: bool = False

    def activate(self) -> FlextResult[None]:
        """Business operations return FlextResult."""
        if self.is_active:
            return FlextResult[None].fail("User already active")

        self.is_active = True
        self.add_domain_event("UserActivated", {"user_id": self.id})
        return FlextResult[None].ok(None)

class Account(FlextAggregateRoot):
    """Aggregate root managing consistency boundaries."""
    owner: User
    balance: Decimal

    def withdraw(self, amount: Decimal) -> FlextResult[None]:
        if amount > self.balance:
            return FlextResult[None].fail("Insufficient funds")

        self.balance -= amount
        self.add_domain_event("MoneyWithdrawn", {
            "account_id": self.id,
            "amount": str(amount)
        })
        return FlextResult[None].ok(None)
```

## Architecture

### Layer Organization

```
flext-core/
├── Foundation Layer              # Core patterns and primitives
│   ├── result.py                # FlextResult railway pattern
│   ├── container.py             # Dependency injection
│   ├── exceptions.py            # Exception hierarchy
│   └── constants.py             # Enums and constants
│
├── Domain Layer                  # Business logic patterns
│   ├── entities.py              # DDD entities
│   ├── value_objects.py         # DDD value objects
│   ├── aggregate_root.py        # DDD aggregates
│   └── domain_services.py       # Domain services
│
├── Application Layer             # Use case orchestration
│   ├── commands.py              # CQRS commands
│   ├── handlers.py              # Command/query handlers
│   ├── validation.py            # Business validation
│   └── interfaces.py            # Port interfaces
│
└── Infrastructure Layer          # External concerns
    ├── config.py                # Configuration management
    ├── loggings.py              # Structured logging
    ├── observability.py         # Monitoring/metrics
    └── payload.py               # Event/message patterns
```

### Pattern Flow

```mermaid
graph LR
    A[Input] --> B[Validation]
    B --> C{FlextResult}
    C -->|Success| D[Business Logic]
    C -->|Failure| E[Error Handler]
    D --> F[Domain Event]
    F --> G[Response]
    E --> G
```

## Development

### Quality Gates

All code must pass these checks before commit:

```bash
# Run all quality checks (MANDATORY)
make validate

# Individual checks
make lint        # Code style (ruff)
make type-check  # Type safety (mypy strict)
make test        # Tests with 75% coverage
make security    # Security scanning
```

### Testing

```bash
# Run full test suite
make test

# Run specific test categories
poetry run pytest -m unit         # Unit tests only
poetry run pytest -m integration  # Integration tests
poetry run pytest -m "not slow"   # Fast tests only

# Run specific test file
poetry run pytest tests/unit/core/test_result.py -v

# Generate coverage report
make coverage-html
```

### Code Style

- **Line Length**: 79 characters (PEP8 strict)
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all public functions
- **Naming**: `FlextXxx` prefix for all exports

## Ecosystem Integration

FLEXT Core is the foundation for the entire FLEXT ecosystem:

```
flext-core (Foundation Library)
    ├── Infrastructure Libraries (6 projects)
    │   ├── flext-db-oracle       # Oracle database patterns
    │   ├── flext-ldap            # LDAP integration
    │   ├── flext-grpc            # gRPC communication
    │   └── flext-meltano         # Data orchestration
    │
    ├── Application Services (5 projects)
    │   ├── flext-api             # REST API (FastAPI)
    │   ├── flext-auth            # Authentication
    │   └── flext-web             # Web interface
    │
    ├── Singer Ecosystem (15 projects)
    │   ├── Taps (5)              # Data extraction
    │   ├── Targets (5)           # Data loading
    │   └── DBT (4)               # Data transformation
    │
    └── Runtime Services (Go)
        ├── FlexCore              # Distributed runtime
        └── FLEXT Service         # Control panel
```

### Breaking Changes Policy

As a foundation library for 32+ projects:

1. **Semantic Versioning**: Strict adherence to semver
2. **Deprecation Warnings**: 2 version cycles before removal
3. **Migration Guides**: Provided for all breaking changes
4. **Compatibility Testing**: Against all dependent projects

## Documentation

- [Getting Started](docs/getting-started/quickstart.md) - Quick introduction
- [Architecture Guide](docs/architecture/overview.md) - System design
- [API Reference](docs/api/core.md) - Complete API documentation
- [Examples](examples/) - Working code examples
- [Contributing](CONTRIBUTING.md) - Development guidelines

## Requirements

- Python 3.13+
- Poetry 1.8+
- Make (for development commands)

## Dependencies

Minimal runtime dependencies for maximum portability:

- `pydantic>=2.11.7` - Data validation and settings
- `pydantic-settings>=2.10.1` - Configuration management
- `structlog>=25.4.0` - Structured logging

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup

# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
make validate

# Submit pull request
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/flext-sh/flext-core/issues)
- **Documentation**: [Full Documentation](https://flext-sh.github.io/flext-core/)
- **Examples**: [Working Examples](examples/)

---

**FLEXT Core** - Foundation for enterprise data integration
