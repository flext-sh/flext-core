# FLEXT-Core Documentation

Professional Documentation · Status: Production Ready · Version: 0.9.9
Last Updated: 2025-10-16

This comprehensive documentation covers FLEXT-Core, the foundation library for the
FLEXT ecosystem. It provides railway-oriented programming, dependency injection,
domain-driven design patterns, and comprehensive type safety with Python 3.13+.

> **✨ New in v0.9.9**: Enhanced 5-layer architecture (Layers 0, 0.5, 1, 2, 3, 4) with zero-dependency constants layer and runtime bridge. See [CLAUDE.md](../CLAUDE.md) for AI-assisted development workflow with Serena MCP integration.

## Documentation Structure

```text
docs/
├── README.md                 # This file - documentation overview
├── api-reference/           # Complete API reference
│   ├── foundation.md        # Core foundation classes (Result, Container, etc.)
│   ├── domain.md           # Domain layer (Models, Services, etc.)
│   ├── application.md      # Application layer (Bus, Handlers, etc.)
│   └── infrastructure.md   # Infrastructure layer (Config, Logging, etc.)
├── guides/                 # User and developer guides
│   ├── getting-started.md  # Installation and quick start
│   ├── configuration.md    # Configuration management
│   ├── error-handling.md   # Railway pattern and error handling
│   ├── dependency-injection.md # Container and DI patterns
│   ├── domain-modeling.md  # DDD patterns and best practices
│   ├── testing.md          # Testing strategies and patterns
│   └── troubleshooting.md  # Common issues and solutions
├── architecture/           # Architecture and design
│   ├── overview.md         # High-level architecture
│   ├── clean-architecture.md # Clean architecture principles
│   ├── patterns.md         # Design patterns used
│   └── decisions.md        # Architecture decision records
├── development/            # Development workflow
│   ├── contributing.md     # How to contribute
│   ├── standards.md        # Coding standards and conventions
│   ├── workflow.md         # Development workflow
│   └── quality.md          # Quality assurance processes
└── standards/              # Standards and guidelines
    ├── python.md           # Python coding standards
    ├── documentation.md    # Documentation standards
    └── templates.md        # Document templates
```

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup

# Verify installation
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
from flext_core import FlextUtilities; print('✅ FLEXT-Core v0.9.9 ready')"
```

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

container = FlextContainer.get_global()
container.register("database", DatabaseService())
db = container.get("database")
```

### 3. Domain-Driven Design

Entity, Value Object, and Aggregate Root patterns:

```python
from typing import List
from decimal import Decimal
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
