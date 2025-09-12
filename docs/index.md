# FLEXT Core Documentation

**Enterprise foundation library for railway-oriented programming and clean architecture**

## Overview

FLEXT Core is the architectural foundation for 32+ projects in the FLEXT data integration ecosystem. It provides type-safe error handling, enterprise patterns, and clean architecture principles that eliminate common boilerplate and ensure consistency across all ecosystem projects.

## Quick Start

### Installation

```bash
# Clone and setup development environment
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup

# Verify installation
python -c "from flext_core import FlextResult; print('✅ Working')"
```

### Core Patterns

#### Railway-Oriented Programming

```python
from flext_core import FlextResult

def validate_age(age: int) -> FlextResult[int]:
    if age < 0:
        return FlextResult[None].fail("Age cannot be negative")
    if age > 150:
        return FlextResult[None].fail("Age seems unrealistic")
    return FlextResult[None].ok(age)

# Chain operations safely
result = (
    validate_age(25)
    .map(lambda x: x * 365)  # Convert to days
    .flat_map(lambda days: calculate_experience(days))
    .map_error(lambda e: f"Validation failed: {e}")
)

if result.success:
    print(f"Experience: {result.unwrap()}")
else:
    print(f"Error: {result.error}")
```

#### Dependency Injection

```python
from flext_core import get_flext_container

# Global singleton container
container = FlextContainer.get_global()

# Register services
container.register("database", DatabaseService())
container.register_factory("cache", lambda: CacheService())

# Retrieve with type safety
db = container.get("database").unwrap()
cache = container.get("cache").unwrap()
```

#### Domain-Driven Design

```python
from flext_core import FlextModels
from decimal import Decimal

class Money(FlextModels.Value):
    """Immutable value object."""
    amount: Decimal
    currency: str

    def add(self, other: Money) -> FlextResult[Money]:
        if self.currency != other.currency:
            return FlextResult[None].fail("Currency mismatch")
        return FlextResult[None].ok(Money(
            amount=self.amount + other.amount,
            currency=self.currency
        ))

class Account(FlextModels.Entity):
    """Entity with identity and lifecycle."""
    owner_name: str
    balance: Money
    is_active: bool = True

    def deposit(self, amount: Money) -> FlextResult[None]:
        if not self.is_active:
            return FlextResult[None].fail("Account is inactive")

        result = self.balance.add(amount)
        if result.success:
            self.balance = result.unwrap()
            self.add_domain_event("MoneyDeposited", {
                "account_id": self.id,
                "amount": str(amount.amount),
                "currency": amount.currency
            })
        return result.map(lambda _: None)
```

## Documentation Structure

### 📚 Core Documentation

- [**Architecture Overview**](architecture/overview.md) - System design and patterns
- [**Getting Started**](getting-started/quickstart.md) - Quick introduction
- [**API Reference**](api/core.md) - Complete API documentation
- [**Examples**](examples/overview.md) - Working code examples

### 🏗️ Architecture Guides

- [**Clean Architecture**](architecture/overview.md) - Layer separation and dependencies
- [**Component Hierarchy**](architecture/component-hierarchy.md) - Module organization
- [**Design Patterns**](architecture/patterns.md) - Implemented patterns

### 🛠️ Development

- [**Best Practices**](development/best-practices.md) - Coding standards
- [**Testing Guide**](development/testing.md) - Writing and running tests
- [**Contributing**](../CONTRIBUTING.md) - How to contribute

### ⚙️ Configuration

- [**Configuration Management**](configuration/overview.md) - Settings and environment
- [**Secrets Management**](configuration/secrets.md) - Handling sensitive data

## Architecture

### Layer Organization

```
┌─────────────────────────────────────────────┐
│           Presentation Layer                 │
│         (External Interfaces)               │
├─────────────────────────────────────────────┤
│           Application Layer                  │
│    (Commands, Handlers, Use Cases)         │
├─────────────────────────────────────────────┤
│            Domain Layer                     │
│  (Entities, Value Objects, Aggregates)     │
├─────────────────────────────────────────────┤
│         Infrastructure Layer                │
│   (Config, Logging, External Services)     │
├═════════════════════════════════════════════┤
│          Foundation Layer                   │
│    (FlextResult, FlextContainer, Types)    │
└─────────────────────────────────────────────┘
```

### Core Components

| Component                     | Purpose                         | Status     |
| ----------------------------- | ------------------------------- | ---------- |
| **FlextResult[T]**            | Railway-oriented error handling | ✅ Stable  |
| **FlextContainer**            | Dependency injection container  | ✅ Stable  |
| **FlextModels.Entity**        | DDD entities with identity      | ✅ Stable  |
| **FlextModels.Value**         | Immutable value objects         | ✅ Stable  |
| **FlextModels.AggregateRoot** | Aggregate consistency boundary  | ✅ Stable  |
| **FlextCommand**              | CQRS command pattern            | 🔄 Active  |
| **FlextMessageHandler**       | Command/query handlers          | 🔄 Active  |
| **FlextEvent**                | Domain events                   | 📋 Planned |

## Ecosystem Integration

FLEXT Core serves as the foundation for:

### Python Libraries (29 projects)

- **Infrastructure**: flext-db-oracle, flext-ldap, flext-grpc
- **Applications**: flext-api, flext-auth, flext-web
- **Singer Taps**: flext-tap-oracle, flext-tap-ldap
- **Singer Targets**: flext-target-oracle, flext-target-ldap
- **DBT Projects**: flext-dbt-oracle, flext-dbt-ldap

### Go Services

- **FlexCore**: Distributed runtime engine
- **FLEXT Service**: Control panel and orchestration

### Integration Pattern

```python
# All ecosystem projects use the same patterns
from flext_core import FlextResult, get_flext_container

class OracleService:
    def query(self, sql: str) -> FlextResult[list]:
        """All operations return FlextResult."""
        try:
            results = self.connection.execute(sql)
            return FlextResult[None].ok(results)
        except Exception as e:
            return FlextResult[None].fail(str(e))

# Register in global container
container = FlextContainer.get_global()
container.register("oracle", OracleService())
```

## Quality Standards

### Requirements

- **Python**: 3.13+ only (no backward compatibility)
- **Coverage**: 75% minimum test coverage
- **Type Safety**: MyPy strict mode with zero errors
- **Code Style**: PEP8 with 79 character line limit
- **Documentation**: All public APIs documented

### Quality Gates

```bash
# Must pass before any commit
make validate  # Runs all checks

# Individual checks
make lint       # Code style
make type-check
make test       # Test suite
make security   # Security scan
```

## Current Status

### Version 0.9.0

#### ✅ Stable Features

- FlextResult railway-oriented programming
- FlextContainer dependency injection
- Domain modeling (Entity, ValueObject, AggregateRoot)
- Configuration management
- Structured logging

#### 🔄 In Development

- CQRS command/query bus
- Event sourcing patterns
- Advanced validation
- Plugin architecture

#### 📋 Planned

- Cross-language bridge (Python-Go)
- Distributed patterns
- Performance optimizations

## Getting Help

- **Quick Start**: [Getting Started Guide](getting-started/quickstart.md)
- **API Docs**: [Complete API Reference](api/core.md)
- **Examples**: [Working Examples](../examples/)
- **Issues**: [GitHub Issues](https://github.com/flext-sh/flext-core/issues)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup

# Run quality checks
make validate

# Run specific tests
poetry run pytest tests/unit/core/test_result.py -v
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

---

**FLEXT Core** - Foundation for enterprise data integration
**Version**: 0.9.0 | **Python**: 3.13+ | **License**: MIT
