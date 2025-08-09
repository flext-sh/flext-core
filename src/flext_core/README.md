# FLEXT Core Source Code

**The Architectural Foundation Implementation for Enterprise Data Integration**

This directory contains the complete source code implementation of FLEXT Core, providing foundational patterns used across all 32 projects in the FLEXT ecosystem.

## 🏗️ Source Code Architecture

FLEXT Core is organized into **6 architectural layers**, each providing specific patterns and capabilities:

### **Foundation Layer** - Type System & Core Contracts

Core foundational contracts that all other modules depend on.

| Module                             | Purpose             | Key Exports                                          |
| ---------------------------------- | ------------------- | ---------------------------------------------------- |
| [`__init__.py`](__init__.py)       | Public API gateway  | `FlextResult`, `FlextContainer`, `FlextEntity`, etc. |
| [`typings.py`](typings.py)         | Centralized types   | `FlextTypes`, `TAnyDict`, `TLogMessage`, all types   |
| [`constants.py`](constants.py)     | Ecosystem constants | `FlextLogLevel`, `Platform`, port definitions        |
| [`__version__.py`](__version__.py) | Version management  | `get_version_info()`, `is_feature_available()`       |

### **Core Pattern Layer** - Railway-Oriented Programming

Railway-oriented programming foundation and dependency injection.

| Module                           | Purpose                | Key Exports                                   |
| -------------------------------- | ---------------------- | --------------------------------------------- |
| [`result.py`](result.py)         | FlextResult[T] pattern | `FlextResult`, railway-oriented methods       |
| [`container.py`](container.py)   | Enterprise DI system   | `FlextContainer`, `get_flext_container()`     |
| [`exceptions.py`](exceptions.py) | Exception hierarchy    | `FlextException`, business context exceptions |
| [`utilities.py`](utilities.py)   | Pure utility functions | Performance tracking, ID generation           |

### **Configuration Layer** - System Integration & Logging

Configuration, logging, and external integration contracts.

| Module                                             | Purpose              | Key Exports                                          |
| -------------------------------------------------- | -------------------- | ---------------------------------------------------- |
| [`config.py`](config.py)                           | Base configuration   | `FlextSettings`, environment integration         |
| [`loggings.py`](loggings.py)                       | Structured logging   | `FlextLogger`, `get_logger()`, correlation IDs       |
| [`payload.py`](payload.py)                         | Message patterns     | `FlextPayload`, `FlextEvent`, `FlextMessage`         |
| [`interfaces.py`](interfaces.py)                   | Protocol definitions | `FlextValidator`, `FlextService`, Clean Architecture |
| [`config_models.py`](config_models.py)             | Configuration models | Database, Redis, Oracle, LDAP configs                |
| [`config_hierarchical.py`](config_hierarchical.py) | Hierarchical config  | Configuration composition patterns                   |

### **Domain Layer** - Domain-Driven Design Patterns

Rich domain modeling patterns following DDD principles.

| Module                                     | Purpose          | Key Exports                                  |
| ------------------------------------------ | ---------------- | -------------------------------------------- |
| [`entities.py`](entities.py)               | Domain entities  | `FlextEntity`, rich business logic           |
| [`value_objects.py`](value_objects.py)     | Immutable values | `FlextValueObject`, attribute-based equality |
| [`aggregate_root.py`](aggregate_root.py)   | DDD aggregates   | `FlextAggregateRoot`, invariants & events    |
| [`domain_services.py`](domain_services.py) | Domain services  | `FlextDomainService`, business operations    |
| [`models.py`](models.py)                   | General models   | Universal data structures, semantic types    |

### **CQRS Layer** - Command Query Responsibility Segregation

CQRS patterns for enterprise scalability.

| Module                           | Purpose          | Key Exports                               |
| -------------------------------- | ---------------- | ----------------------------------------- |
| [`commands.py`](commands.py)     | Command patterns | `FlextCommand`, `FlextCommandBus`         |
| [`handlers.py`](handlers.py)     | Handler patterns | `FlextCommandHandler`, message processing |
| [`validation.py`](validation.py) | Input validation | Validation system, business rules         |

### **Extension Layer** - Reusable Behaviors & Cross-Cutting Concerns

Reusable patterns and cross-cutting concerns.

| Module                           | Purpose               | Key Exports                             |
| -------------------------------- | --------------------- | --------------------------------------- |
| [`mixins.py`](mixins.py)         | Reusable behaviors    | `TimestampMixin`, `SoftDeleteMixin`     |
| [`decorators.py`](decorators.py) | Enterprise decorators | `@with_correlation_id`, `@with_metrics` |
| [`fields.py`](fields.py)         | Field metadata        | Enhanced data modeling support          |
| [`guards.py`](guards.py)         | Validation guards     | Type safety, validation builders        |
| [`core.py`](core.py)             | FlextCore main class  | Integration of all patterns             |

### **Specialized Modules** - Extension Points & Legacy Support

Specialized processing components and legacy compatibility.

| Module                                         | Purpose               | Key Exports                               |
| ---------------------------------------------- | --------------------- | ----------------------------------------- |
| [`schema_processing.py`](schema_processing.py) | Processing components | LDIF/ACL processing patterns              |
| [`singer_base.py`](singer_base.py)             | Legacy compatibility  | Singer pattern compatibility (deprecated) |
| [`testing_utilities.py`](testing_utilities.py) | Testing support       | Standardized test configurations          |

## 🎯 Module Usage Patterns

### **Foundation Usage**

```python
# Essential imports for all FLEXT projects
from flext_core import FlextResult, FlextContainer, FlextEntity
from flext_core.typings import FlextTypes, TAnyDict, TLogMessage
from flext_core.constants import FlextLogLevel, Platform
```

### **Railway-Oriented Programming**

```python
# Type-safe error handling throughout ecosystem
def process_data(data: dict) -> FlextResult[ProcessedData]:
    return (
        validate_input(data)
        .map(transform_data)
        .flat_map(save_to_database)
        .map(format_response)
    )
```

### **Domain Modeling**

```python
# Rich domain entities with business logic
class User(FlextEntity):
    name: str
    email: str

    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult.fail("User already active")

        self.is_active = True
        self.add_domain_event({"type": "UserActivated"})
        return FlextResult.ok(None)
```

### **Enterprise Configuration**

```python
# Environment-aware configuration management
class AppSettings(FlextSettings):
    database_url: str = "postgresql://localhost/app"
    log_level: str = "INFO"

    class Config:
        env_prefix = "APP_"
```

### **Structured Logging**

```python
# Correlation ID support and enterprise observability
logger = get_logger(__name__)
with create_log_context(logger, request_id="123", user_id="456"):
    logger.info("Processing request", operation="create_user")
```

## 📊 Ecosystem Integration

### **Project Dependencies**

- **32 Projects** directly depend on FLEXT Core patterns
- **15,000+ Function Signatures** use FlextResult[T] across ecosystem
- **Zero Downtime** requirement for changes affecting dependent projects

### **Cross-Language Integration**

- **Python-Go Bridge**: FlextCore (Go) integrates via Python bridge
- **Type Safety**: Comprehensive type annotations ensure cross-service reliability
- **Serialization**: Cross-language data contract compatibility

### **Development Status (v0.9.0 → 1.0.0)**

- ✅ **Production Ready**: Core patterns, dependency injection, domain modeling
- 🚧 **In Development**: Event Sourcing, advanced CQRS, plugin architecture
- 📋 **Planned**: Python-Go bridge, enterprise observability, distributed tracing

## 🔧 Development Standards

### **Quality Requirements**

- **Test Coverage**: 95% minimum (currently 95%+)
- **Type Safety**: Strict MyPy with zero tolerance for type errors
- **Code Quality**: Ruff linting with comprehensive rule set
- **Security**: Bandit + pip-audit scanning
- **Line Length**: 79 characters (strict PEP8 compliance)

### **Documentation Standards**

- **Comprehensive Docstrings**: Every module, class, and method documented
- **Module Role Documentation**: Clear architectural layer positioning
- **Usage Patterns**: Real-world examples and ecosystem integration
- **Development Status**: Current capabilities and roadmap integration

### **Architectural Constraints**

- **Dependency Direction**: Dependencies flow inward following Clean Architecture
- **Layer Separation**: Clear boundaries with no reverse dependencies
- **Type Safety**: FlextResult[T] for all error-prone operations
- **Immutability**: Value objects frozen, entities mutable with controlled state

## 📚 Related Documentation

- [**CLAUDE.md**](../../CLAUDE.md) - Development guidance and quality gates
- [**README.md**](../../README.md) - Project overview and getting started
- [**docs/architecture/**](../../docs/architecture/) - Detailed architectural documentation
- [**docs/TODO.md**](../../docs/TODO.md) - Development roadmap and current priorities
- [**examples/**](../../examples/) - 17 comprehensive working examples

## 🎯 Navigation Tips

1. **Start with Foundation**: Begin with `__init__.py`, `result.py`, and `container.py`
2. **Follow Layers**: Progress through layers in architectural order
3. **Check Examples**: Reference working examples for usage patterns
4. **Review Tests**: Examine test files for comprehensive usage scenarios
5. **Understand Context**: Each module's docstring explains its ecosystem role

---

**FLEXT Core v0.9.0** - The architectural foundation enabling enterprise-grade data integration across 32 interconnected projects. Every module serves a specific role in building reliable, scalable, and maintainable data integration solutions.
