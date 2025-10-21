# FLEXT-Core Documentation

Professional Documentation · Status: Production Ready · Version: 0.9.9
Last Updated: 2025-10-21

This comprehensive documentation covers FLEXT-Core, the foundation library for the
FLEXT ecosystem. It provides railway-oriented programming, dependency injection,
domain-driven design patterns, and comprehensive type safety with Python 3.13+.

> **✨ New in v0.9.9**: Enhanced 5-layer architecture (Layers 0, 0.5, 1, 2, 3, 4) with zero-dependency constants layer and runtime bridge. See [CLAUDE.md](../CLAUDE.md) for AI-assisted development workflow with Serena MCP integration.
>
> **📚 New Comprehensive Guides** (October 2025):
> - **[Railway-Oriented Programming](./guides/railway-oriented-programming.md)** - FlextResult[T] patterns with real examples
> - **[Advanced Dependency Injection](./guides/dependency-injection-advanced.md)** - FlextContainer type-safe patterns
> - **[Domain-Driven Design](./guides/domain-driven-design.md)** - FlextModels with practical examples
> - **[Anti-Patterns & Best Practices](./guides/anti-patterns-best-practices.md)** - Common mistakes and solutions
> - **[Pydantic v2 Patterns](./guides/pydantic-v2-patterns.md)** - Production patterns for ecosystem projects

## Documentation Structure

```text
docs/
├── README.md                 # This file - documentation overview
├── INDEX.md                  # Navigation guide to all documentation
│
├── api-reference/           # ✅ Complete API reference (ALL FILES)
│   ├── foundation.md        # Core foundation classes (Result, Container, etc.)
│   ├── domain.md           # Domain layer (Models, Services, etc.)
│   ├── application.md      # Application layer (Bus, Handlers, etc.)
│   └── infrastructure.md   # Infrastructure layer (Config, Logging, etc.)
│
├── guides/                  # ✅ Core guides (6/10 complete)
│   ├── getting-started.md           # ✅ Installation and quick start
│   ├── railway-oriented-programming.md   # ✅ FlextResult[T] comprehensive guide
│   ├── dependency-injection-advanced.md  # ✅ FlextContainer advanced patterns
│   ├── domain-driven-design.md      # ✅ FlextModels and DDD patterns
│   ├── anti-patterns-best-practices.md   # ✅ Common mistakes and solutions
│   ├── pydantic-v2-patterns.md      # ✅ Pydantic v2 ecosystem patterns
│   ├── configuration.md             # 🔄 Planned
│   ├── error-handling.md            # 🔄 Planned
│   ├── testing.md                   # 🔄 Planned
│   └── troubleshooting.md           # 🔄 Planned
│
├── architecture/            # ⚠️ Partial (1/4 complete)
│   ├── overview.md         # ✅ High-level architecture
│   ├── clean-architecture.md # 🔄 Planned
│   ├── patterns.md         # 🔄 Planned
│   └── decisions.md        # 🔄 Planned (Architecture Decision Records)
│
├── development/            # ⚠️ Partial (1/1 complete)
│   └── contributing.md     # ✅ How to contribute
│
├── standards/              # ⚠️ Partial (1/3 complete)
│   ├── development.md      # ✅ Coding standards and conventions
│   ├── python.md           # 🔄 Planned
│   └── documentation.md    # 🔄 Planned
│
└── improvements/           # Documentation audit reports
    └── PHASE1_COMPLETION_SUMMARY.md  # Quality audit results
```

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup

# Verify installation
python -c "from flext_core import __version__; print(f'✅ FLEXT-Core v{__version__} ready')"
```

```python
from flext_core import FlextContainer
from flext_core import FlextResult

# Railway-oriented error handling
result = FlextResult[str].ok("Success!")
if result.is_success:
    value = result.unwrap()

# Dependency injection
container = FlextContainer.get_global()
container.register("logger", FlextLogger(**name**))


# Domain modeling with DDD patterns
class User(FlextModels.Entity):
    name: str
    email: str
```

## Core Concepts

### 1. Railway-Oriented Programming

FLEXT-Core uses the `FlextResult[T]` monad for error handling without exceptions:

```python
def divide(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult[float].fail("Division by zero")
    return FlextResult[float].ok(a / b)

result = divide(10, 2)
if result.is_success:
    print(f"Result: {result.unwrap()}")
```

### 2. Dependency Injection

Global container with type-safe service registration:

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()
container.register("database", DatabaseService())
db = container.get("database")
```

### 3. Domain-Driven Design

Entity, Value Object, and Aggregate Root patterns:

```python
from typing import List
from decimal import Decimal
from flext_core import FlextModels
from flext_core import FlextResult

class Order(FlextModels.Entity):
    customer_id: str
    items: List[OrderItem]
    total: Decimal

    def calculate_total(self) -> FlextResult[Decimal]:
        # Business logic here
        pass
```

- **Zero MyPy Errors**: Type safety guaranteed
- **75%+ Test Coverage**: Comprehensive testing
- **Python 3.13+**: Modern Python features
- **Pydantic v2**: Latest validation framework

## Getting Help

- **[API Reference](./api-reference/)**:
  Complete API documentation
- **[GitHub Issues](https://github.com/flext-sh/flext-core/issues)**:
  Report bugs or request features
- **[GitHub Discussions](https://github.com/flext-sh/flext-core/discussions)**:
  Ask questions and share ideas

## Contributing

See [Contributing Guide](./development/contributing.md) for development guidelines
and workflow.

---

**FLEXT-Core v0.9.9** - Production-ready foundation for enterprise Python applications
with railway-oriented programming, dependency injection, and domain-driven design
patterns.
