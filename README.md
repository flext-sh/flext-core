# FLEXT-Core

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Foundation Library](https://img.shields.io/badge/role-foundation-brightgreen.svg)](#)
[![v0.9.9 RC](https://img.shields.io/badge/version-0.9.9--rc-orange.svg)](#)
[![Documentation](https://img.shields.io/badge/docs-organized-blue.svg)](./docs/)
[![GitHub](https://img.shields.io/badge/github-flext--core-black.svg)](https://github.com/flext/flext-core)

Foundation library for the FLEXT ecosystem providing railway-oriented programming patterns, dependency injection container, domain-driven design patterns, and type safety with Python 3.13+.

> **✅ Phase 1-2 COMPLETE - ALL PYDANTIC V2 DUPLICATIONS REMOVED** · v0.9.9 RC · **1793 passing tests** · **0 failures** · **0 linting violations** · **0 src/ type errors** · **81.02% coverage** · **4 Pydantic v2 duplications removed total** (3 validators + 1 base model class) · **11 architectural validators retained** (FlextResult railway patterns) · **Foundation for 32+ FLEXT projects**

## 📚 Documentation

**Documentation available in [./docs/](./docs/)** - Implementation guides, API reference, and working examples

- **[🚀 Getting Started](./docs/guides/getting-started.md)** - Installation and basic usage
- **[🏗️ Architecture](./docs/architecture/overview.md)** - System design and patterns
- **[🔌 API Reference](./docs/api-reference/)** - Complete API documentation
- **[👥 Contributing](./docs/development/contributing.md)** - Development guidelines

---

## 🎯 Mission & Role in FLEXT Ecosystem

### **Foundation Library for Data Integration**

FLEXT-Core provides patterns and infrastructure used by 32+ specialized projects across the FLEXT ecosystem for data integration and system operations.

### **Core Responsibilities**

1. **🏗️ Railway-Oriented Programming** - `FlextResult[T]` for error handling without exceptions
2. **💉 Dependency Injection** - `FlextContainer` singleton for service management
3. **🎯 Domain-Driven Design** - Entity and value object patterns with domain services
4. **🔒 Type Safety** - Python 3.13+ with strict type annotations and Pydantic v2
5. **📊 Configuration Management** - Environment-aware settings with validation
6. **🔍 Logging Infrastructure** - Structured logging with multiple output formats
7. **🚌 Event-Driven Architecture** - Message bus and event dispatching
8. **⚡ Processing Pipeline** - Handler chains and processor patterns

### **Ecosystem Integration**

FLEXT-Core provides the **architectural patterns** that all FLEXT projects inherit:

| Project Type                | Projects                           | Integration Pattern                               |
| --------------------------- | ---------------------------------- | ------------------------------------------------- |
| **Core Libraries**          | flext-api, flext-auth, flext-grpc  | Direct inheritance of foundation patterns         |
| **Infrastructure**          | flext-ldap, flext-db-oracle        | Railway patterns for LDAP and database operations |
| **Data Integration**        | flext-meltano, Singer taps/targets | Configuration and processing patterns             |
| **Quality & Observability** | flext-quality, flext-observability | Logging and monitoring integration                |

### **Foundation Patterns Used Across Ecosystem**

```python
# Railway-oriented error handling (784+ usages across ecosystem)
result = operation()
if result.is_success():
    data = result.unwrap()
else:
    error = result.unwrap_failure()

# Dependency injection (120+ services across ecosystem)
container = FlextContainer()
container.register_singleton(DatabaseConnection, create_db_connection)

# Domain-driven design (95+ entities across ecosystem)
@dataclass(frozen=True)
class UserId(ValueObject):
    value: str
```

---

## 🏗️ Current Implementation

### **Source Architecture**

```
src/flext_core/
├── api.py              # Main API interface (750+ lines)
├── bus.py              # Event bus and messaging (856+ lines)
├── config.py           # Configuration management (423+ lines)
├── constants.py        # Configuration constants
├── container.py        # Dependency injection container (612+ lines)
├── context.py          # Execution context management (387+ lines)
├── dispatcher.py       # Event dispatching (298+ lines)
├── exceptions.py       # Custom exception hierarchy
├── handlers.py         # Handler pattern implementations (445+ lines)
├── loggings.py         # Logging infrastructure (534+ lines)
├── mixins.py           # Reusable mixin classes
├── models.py           # Pydantic models and entities (389+ lines)
├── processors.py       # Processing pipeline (267+ lines)
├── protocols.py        # Protocol definitions and typing
├── py.typed            # Type checking marker
├── registry.py         # Component registry (198+ lines)
├── result.py           # Railway-oriented programming (445+ lines)
├── service.py          # Service layer base classes (323+ lines)
├── typings.py          # Type definitions and protocols
└── utilities.py        # Utility functions and helpers (456+ lines)
```

### **Key Architectural Components**

#### **FlextResult[T] - Railway Pattern**

```python
# Success path
success_result = FlextResult.success(User(id="123", name="John"))

# Failure path
error_result = FlextResult.failure(ValidationError("Invalid input"))

# Railway operations
final_result = (
    validate_input(input_data)
    .and_then(process_data)
    .and_then(save_to_database)
    .map(transform_result)
)
```

#### **FlextContainer - Dependency Injection**

```python
container = FlextContainer()

# Register services
container.register_singleton(DatabaseConnection, create_db_connection)
container.register_transient(UserService, UserService)

# Resolve dependencies
db_connection = container.resolve(DatabaseConnection)
user_service = container.resolve(UserService)
```

#### **FlextBus - Event-Driven Architecture**

```python
bus = FlextBus()

# Register handlers
@bus.handler("user.created")
def handle_user_created(event: UserCreatedEvent):
    send_welcome_email(event.user_id)

# Publish events
bus.publish(UserCreatedEvent(user_id="123"))
```

---

## 🏗️ Recent Architectural Improvements (v0.9.9)

### Layer 0 Foundation Architecture

**Status**: ✅ Completed | **Impact**: Eliminates circular dependencies across 32+ ecosystem projects

The new Layer 0 architecture provides a solid foundation with zero internal dependencies:

```python
# Layer 0: Foundation layer - constants, types, protocols (no external dependencies)
from flext_core import FlextConstants, FlextTypes, FlextProtocols

# Access 50+ error codes for exception handling
error_code = FlextConstants.Errors.VALIDATION_FAILED
timeout = FlextConstants.Configuration.DEFAULT_TIMEOUT
email_pattern = FlextConstants.Validation.EMAIL_PATTERN

# Type system with 50+ TypeVars for generic programming
from flext_core.typings import T, T_co, T_contra
```

**Key Features**:

- ✅ **50+ Error Codes**: Categorized exception handling across ecosystem
- ✅ **Validation Patterns**: Email, URL, UUID, phone number patterns
- ✅ **Configuration Defaults**: Timeouts, network settings, logging levels
- ✅ **Platform Constants**: HTTP status codes, encodings, file paths
- ✅ **Complete Immutability**: All constants marked with `typing.Final`

### Layer 0.5 Runtime Bridge

**Status**: ✅ Completed | **Impact**: External library integration without circular dependencies

The runtime bridge exposes external libraries while maintaining proper dependency hierarchy:

```python
# Layer 0.5: Runtime bridge - external library integration
from flext_core import FlextRuntime

# Type guards and validation using Layer 0 patterns
if FlextRuntime.is_valid_email(email):
    process_email(email)

# Serialization utilities
json_data = FlextRuntime.serialize_to_json(data)
structured_logs = FlextRuntime.get_structured_logger(__name__)
```

**Key Features**:

- ✅ **Type Guards**: Email, URL, UUID validation using Layer 0 patterns
- ✅ **Serialization**: JSON conversion with FLEXT defaults
- ✅ **External Libraries**: Direct access to structlog, dependency_injector
- ✅ **Structured Logging**: Pre-configured with FLEXT patterns
- ✅ **Sequence Utilities**: Type checking for collections

### Quality Achievements

**Test Coverage**: 74% (1502 tests passing, 2 failures) - **Target**: 79% - 2 handler-related test failures under investigation

**Module Coverage Breakdown**:

- **Foundation Layer**: 95%+ (result.py 91%, container.py 81%, typings.py 100%, constants.py 96%, protocols.py 100%)
- **Domain Layer**: 68% (models.py 57%, service.py 87%, mixins.py 84%, utilities.py 73%)
- **Application Layer**: 77% (bus.py 83%, handlers.py 77%, dispatcher.py 72%, processors.py 63%, registry.py 92%)
- **Infrastructure Layer**: 82% (config.py 72%, loggings.py 80%, context.py 94%, decorators.py 84%, exceptions.py 82%)
- **Runtime Bridge**: 98% (runtime.py 98%)

**Quality Gates Status**:

- ⚠️ **Ruff Linting**: 1 violation (PGH003 in test file) - needs fixes
- ⚠️ **Type Checking**: 14 errors (Pyrefly strict mode) - needs investigation and fixes
- ⚠️ **Test Suite**: 1502 tests passing, 2 failures (handler dict message tests under investigation)
- ⚠️ **Coverage**: 74% (target 79% for 1.0.0)

---

## ✨ Phase 1 Architectural Enhancements (v0.9.9)

**Status**: ✅ Completed | **Impact**: Foundation for ecosystem-wide code reduction

### **New Features**

#### **1. Context Enrichment in FlextMixins**

Automatic context management for structured logging and distributed tracing:

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

class PaymentService(FlextService[FlextTypes.Dict]):
    """Service with automatic context enrichment."""

    def process_payment(self, payment_id: str, amount: float, user_id: str) -> FlextResult[dict]:
        # Generate correlation ID for distributed tracing
        correlation_id = self._with_correlation_id()

        # Set user context for audit trail
        self._with_user_context(user_id, payment_id=payment_id)

        # Set operation context for tracking
        self._with_operation_context("process_payment", amount=amount)

        # All logs now include full context automatically
        self.logger.info("Processing payment", payment_id=payment_id, amount=amount)

        # Business logic here...

        # Clean up context
        self._clear_operation_context()

        return FlextResult[dict].ok({"status": "completed", "correlation_id": correlation_id})
```

**Available Methods**:

- `_enrich_context(**context_data)` - Add service metadata to logs
- `_with_correlation_id(correlation_id?)` - Set/generate correlation IDs for tracing
- `_with_user_context(user_id, **user_data)` - Set user audit context
- `_with_operation_context(operation_name, **data)` - Set operation tracking
- `_clear_operation_context()` - Clean up context after operations

#### **2. Automatic Context in FlextService & FlextHandlers**

Both `FlextService` and `FlextHandlers` now automatically enrich context in `__init__`:

```python
class UserService(FlextService[User]):
    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        # Context automatically includes:
        # - service_type: "UserService"
        # - service_module: "my_app.services.user"

class CreateOrderHandler(FlextHandlers.CommandHandler[CreateOrderCommand, Order]):
    def __init__(self, **data: object) -> None:
        config = FlextModels.Cqrs.Handler(
            handler_name="CreateOrderHandler",
            handler_type="command",
        )
        super().__init__(config=config)
        # Context automatically includes:
        # - handler_name: "CreateOrderHandler"
        # - handler_type: "command"
        # - handler_class: "CreateOrderHandler"
```

#### **3. Helper Method for Context Enrichment**

`FlextService.execute_with_context_enrichment()` provides structured context management with automatic setup and cleanup:

```python
class OrderService(FlextService[Order]):
    def process_order(
        self,
        order_id: str,
        customer_id: str,
        correlation_id: str | None = None,
    ) -> FlextResult[Order]:
        """Process order with complete automation."""
        return self.execute_with_context_enrichment(
            operation_name="process_order",
            correlation_id=correlation_id,
            user_id=customer_id,
            order_id=order_id,
        )
        # Automatically handles:
        # - Correlation ID generation/setting
        # - User context enrichment
        # - Operation context tracking
        # - Performance tracking
        # - Operation logging (start/complete/error)
        # - Context cleanup
```

### **Benefits**

- ✅ **Zero Boilerplate** - No manual context setup required
- ✅ **Distributed Tracing** - Automatic correlation ID generation
- ✅ **Audit Trail** - User context automatically captured
- ✅ **Operation Tracking** - Performance and lifecycle tracking
- ✅ **Structured Logging** - All logs include rich context
- ✅ **Ecosystem Ready** - Available to all 32+ dependent projects

### **Examples**

See `examples/15_automation_showcase.py` for complete working examples demonstrating:

- Basic service with automatic context enrichment
- Payment service with correlation ID tracking
- Order service using context enrichment helper method

Additional examples available in `examples/` directory (00-14: basic patterns through advanced integration)

---

## 🚀 1.0.0 Release Roadmap

**Target Date**: October 2025 | **Current**: v0.9.9 Release Candidate

### Why 1.0.0 Matters

FLEXT-Core serves as the **foundation for 32+ dependent packages** in the FLEXT ecosystem. The 1.0.0 release targets:

- **🔒 API Stability**: No breaking changes throughout the 1.x series (📋 Planned for 1.0.0)
- **⚡ ABI Compatibility**: Locked dependency versions to prevent ecosystem breakage (📋 Planned for 1.0.0)
- **🏭 Quality Standards**: Zero linting violations, zero type checking errors, 79%+ test coverage (📋 Target for 1.0.0)
- **🛠️ Deprecation Strategy**: Minimum 2 minor version cycle before removing deprecated features (📋 Planned for 1.0.0)

### Release Timeline (5 Weeks)

#### Phase 1: API Stabilization & Documentation (Weeks 1-2) ✅

- ✅ **ABI Finalization**: Dependency versions locked, semantic versioning defined
- ✅ **API Documentation**: Stability guarantees documented (VERSIONING.md, API_STABILITY.md)
- 🔄 **Documentation**: README.md roadmap, CLAUDE.md guidelines, migration guide (in progress)

#### Phase 2: Quality Assurance & Ecosystem Testing (Weeks 2-3)

- Test coverage enhancement (74% → 79%+ target)
- Security audit with pip-audit and vulnerability scanning
- Top 5 dependent project validation (flext-api, flext-cli, flext-ldap, flext-auth, flext-web)
- Backward compatibility verification

#### Phase 3: Performance & Optimization (Weeks 3-4)

- Performance baseline establishment
- Critical path optimization (FlextResult, FlextContainer)
- Memory usage profiling and optimization
- Benchmark suite implementation

#### Phase 4: Release Preparation (Week 4)

- Release artifact creation (CHANGELOG.md, migration documentation)
- CI/CD pipeline for automated releases
- Documentation review and finalization
- Release candidate testing

---

#### Phase 5: 1.0.0 Launch & Ecosystem Migration (Week 5)

- Official 1.0.0 release on PyPI
- Ecosystem-wide migration coordination
- Community announcement and documentation
- Post-release monitoring and hotfix readiness

### Stability Guarantees

**What's Guaranteed in 1.x**:

- ✅ **FlextResult[T]** - Railway pattern with dual `.value`/`.data` access
- ✅ **FlextContainer** - Dependency injection singleton API
- ✅ **FlextModels** - DDD patterns (Entity, Value, AggregateRoot)
- ✅ **FlextLogger** - Structured logging interface
- ✅ **FlextConfig** - Configuration management API
- ✅ **FlextBus** - Messaging patterns
- ✅ **HTTP Primitives** - Constants, request/response models (new in 0.9.9)

**Semantic Versioning Promise**:

- **MAJOR** (1.x → 2.0): Breaking changes only, minimum 6 months notice
- **MINOR** (1.0 → 1.1): New features, backward compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, security patches

See [VERSIONING.md](VERSIONING.md) and [API_STABILITY.md](API_STABILITY.md) for complete details.

---

## Core Features

**Core Modules** (v0.9.9):

- ✅ **FlextResult[T]** - Railway-oriented programming with dual `.value`/`.data` access for ABI stability
- ✅ **FlextContainer** - Singleton dependency injection with typed service keys and lifecycle management
- ✅ **FlextModels** - Domain-driven design with Entity/Value/AggregateRoot patterns (Pydantic v2)
- ✅ **FlextLogger** - Structured logging with context propagation and correlation tracking
- ✅ **FlextConfig** - Layered configuration with .env, TOML, and YAML support
- ✅ **FlextBus** - Command/Query/Event bus with middleware pipeline and caching
- ✅ **FlextContext** - Request/operation context with correlation IDs and metadata
- ✅ **FlextDispatcher** - Unified command/query dispatcher with registry support
- ✅ **FlextTypes** - Type system with 50+ TypeVars, protocols, and domain-specific types

**Quality Metrics** (v0.9.9):

- **Ruff**: 1 violation (PGH003) - needs fixes
- **Pyrefly/MyPy**: 14 errors (strict mode) - under investigation
- **Coverage**: 74% (target 79% for 1.0.0)
- **Tests**: 1502 passing (unit + integration + patterns), 2 failures

---

## Architecture Overview

**Foundation Layer**:

- `FlextResult[T]` - Monadic error handling with railway-oriented composition
- `FlextContainer` - Dependency injection singleton with typed service resolution
- `FlextExceptions` - Exception hierarchy with error codes and error message formatting
- `FlextConstants` - Centralized constants, validation patterns, and enumerations
- `FlextTypes` - Type system with 50+ TypeVars, runtime protocols, and type aliases

**Domain Layer**:

- `FlextModels` - DDD patterns (Entity, Value, AggregateRoot)
- `FlextService` - Domain service base with Pydantic Generic[T]
- `FlextMixins` - Reusable behaviors (timestamps, serialization, validation)
- `FlextUtilities` - Domain utilities (validation, conversion, type guards)

**Application Layer**:

- `FlextHandlers` - Handler registry and execution
- `FlextBus` - Message bus with middleware pipeline
- `FlextDispatcher` - Unified dispatcher façade
- `FlextRegistry` - Handler registry management

**Infrastructure Layer**:

- `FlextConfig` - Configuration management with multiple sources
- `FlextLogger` - Structured logging with context propagation
- `FlextContext` - Request/operation context tracking
- `FlextProcessors` - Message processing orchestration
- `FlextProtocols` - Runtime-checkable interfaces

---

## Installation

```bash
# Clone and setup
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup

# Verify installation
python -c "from flext_core import FlextResult, FlextContainer, FlextModels; print('✅ FLEXT-Core v0.9.9 ready')"
```

## Quick Start Example

```python
from flext_core import FlextResult, FlextContainer, FlextModels, FlextService

# 1. Railway Pattern - Error handling without exceptions
def validate_email(email: str) -> FlextResult[str]:
    if "@" not in email:
        return FlextResult[str].fail("Invalid email format")
    return FlextResult[str].ok(email)

result = validate_email("user@example.com")
if result.is_success:
    email = result.unwrap()  # Safe extraction after success check
    print(f"✅ Valid email: {email}")

# 2. Dependency Injection - Global container
container = FlextContainer.get_global()
container.register("logger", FlextLogger(__name__))

logger_result = container.get("logger")
if logger_result.is_success:
    logger = logger_result.unwrap()
    logger.info("Application started")

# 3. Domain Modeling - DDD patterns with Pydantic v2
class User(FlextModels.Entity):
    """User entity with validation."""
    name: str
    email: str
    age: int

    def model_post_init(self, __context: object) -> None:
        """Validate after initialization."""
        if self.age < 0:
            raise ValueError("Age cannot be negative")

user = User(id="user_123", name="Alice", email="alice@example.com", age=30)
print(f"Created user: {user.name} with ID: {user.entity_id}")

# 4. Domain Service - Business logic encapsulation
class UserService(FlextService):
    """User domain service."""

    def create_user(self, name: str, email: str, age: int) -> FlextResult[User]:
        """Create a new user with validation."""
        email_result = validate_email(email)
        if email_result.is_failure:
            return FlextResult[User].fail(f"Email validation failed: {email_result.error}")

        try:
            user = User(id=f"user_{name.lower()}", name=name, email=email, age=age)
            return FlextResult[User].ok(user)
        except ValueError as e:
            return FlextResult[User].fail(str(e))

service = UserService()
user_result = service.create_user("Bob", "bob@example.com", 25)
if user_result.is_success:
    print(f"✅ User created: {user_result.unwrap().name}")
```

---

## Module Reference

| Category           | Module          | Coverage | Description                              |
| ------------------ | --------------- | -------- | ---------------------------------------- |
| **Foundation**     | `result.py`     | 92%      | Railway pattern with monadic composition |
|                    | `container.py`  | 81%      | Dependency injection singleton           |
|                    | `typings.py`    | 100%     | Type system (50+ TypeVars)               |
|                    | `constants.py`  | 98%      | Centralized constants                    |
|                    | `exceptions.py` | 59%      | Exception hierarchy                      |
| **Domain**         | `models.py`     | 55%      | DDD patterns (Entity/Value/Aggregate)    |
|                    | `service.py`    | 67%      | Domain service base class                |
|                    | `mixins.py`     | 84%      | Reusable behaviors                       |
|                    | `utilities.py`  | 66%      | Domain utilities                         |
| **Application**    | `bus.py`        | 91%      | Message bus with middleware              |
|                    | `handlers.py`   | 78%      | Handler registry                         |
|                    | `dispatcher.py` | 54%      | Unified dispatcher                       |
|                    | `registry.py`   | 91%      | Handler registry management              |
|                    | `processors.py` | 64%      | Message processing                       |
| **Infrastructure** | `config.py`     | 68%      | Configuration management                 |
|                    | `loggings.py`   | 66%      | Structured logging                       |
|                    | `context.py`    | 72%      | Context tracking                         |
|                    | `protocols.py`  | 100%     | Runtime protocols                        |
|                    | `decorators.py` | 84%      | Cross-cutting concerns                   |

---

## Development Workflow

### Setup

```bash
# Initial setup with pre-commit hooks
make setup

# Install dependencies
make install
```

### Quality Assurance

```bash
# Complete validation pipeline
make validate     # lint + type-check + security + test

# Individual checks
make lint         # Ruff linting (ZERO tolerance)
make type-check   # MyPy strict + PyRight
make test         # Full test suite with coverage
make security     # Bandit + pip-audit

# Quick checks
make check        # lint + type-check only
make format       # Auto-format code
make fix          # Auto-fix linting issues
```

### Testing

```bash
# Run all tests
make test                    # All tests with coverage

# Specific test types
make test-unit              # Unit tests only
make test-integration       # Integration tests only
make test-fast              # Tests without coverage

# Coverage reports
make coverage-html          # Generate HTML coverage report
pytest --cov=src --cov-report=term-missing
```

### Quality Standards

- **Python**: 3.13+ (required)
- **Linting**: Ruff (1 violation: PGH003 in test file)
- **Type Checking**: MyPy strict mode + PyRight (14 errors in strict mode)
- **Line Length**: 79 characters (PEP 8 strict)
- **Coverage**: Current 74%, target 79% for 1.0.0
- **Tests**: 1502 passing, 2 failures (unit + integration + patterns)

---

## Test Organization

```
tests/
├── unit/           # Unit tests (core functionality)
│   ├── test_result.py
│   ├── test_container.py
│   ├── test_models.py
│   └── ... (20+ test modules)
├── integration/    # Integration tests (component interaction)
│   ├── test_config_singleton_integration.py
│   ├── test_service.py
│   └── test_wildcard_exports.py
├── patterns/       # Pattern tests (CQRS, DDD, architectural)
│   ├── test_patterns.py
│   ├── test_patterns_commands.py
│   └── test_advanced_patterns.py
└── conftest.py     # Shared fixtures and configuration
```

### Running Specific Tests

```bash
# By module
pytest tests/unit/test_result.py -v
pytest tests/unit/test_container.py::TestFlextContainer::test_singleton -v

# By marker
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Exclude slow tests

# With coverage
pytest tests/unit/test_result.py --cov=src/flext_core/result.py --cov-report=term-missing
```

---

## Contributing

### Before Submitting PR

```bash
# Run complete validation
make validate

# Ensure zero violations
make lint        # Must pass with ZERO violations
make type-check  # Must pass with ZERO errors
make test        # All tests must pass
```

### Guidelines

- Use `FlextResult[T]` for all operations that can fail
- Register services with `FlextContainer.get_global()`
- Follow DDD patterns with `FlextModels.Entity/Value/AggregateRoot`
- Use `FlextLogger` with context propagation
- Write tests using `flext_tests` infrastructure (no mocks)
- Keep line length to 79 characters (PEP 8)
- Use Python 3.13+ syntax and features

---

## Documentation

- **[📚 Documentation](./docs/)**: Getting started guide, architecture documentation, and API reference
- **Getting Started**: [`docs/guides/getting-started.md`](./docs/guides/getting-started.md)
- **Architecture**: [`docs/architecture/overview.md`](./docs/architecture/overview.md)
- **API Reference**: [`docs/api-reference/`](./docs/api-reference/)
- **Development**: [`docs/development/contributing.md`](./docs/development/contributing.md)
- **Standards**: [`docs/standards/`](./docs/standards/)
- **Pydantic v2 Modernization**: [`docs/pydantic-v2-modernization/README.md`](./docs/pydantic-v2-modernization/README.md)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/flext-sh/flext-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/flext-sh/flext-core/discussions)
- **Security**: Report vulnerabilities privately to FLEXT maintainers

---

## 🔗 Cross-Reference Navigation

### By Use Case

| Need                     | Quick Link                                                        | Related                                                   |
| ------------------------ | ----------------------------------------------------------------- | --------------------------------------------------------- |
| **Get Started**          | [Getting Started Guide](./docs/guides/getting-started.md)         | [Architecture](./docs/architecture/overview.md)           |
| **Error Handling**       | [Railway Patterns](./docs/guides/railway-oriented-programming.md) | [FlextResult API](./docs/api-reference/foundation.md)     |
| **Dependency Injection** | [DI Advanced](./docs/guides/dependency-injection-advanced.md)     | [FlextContainer API](./docs/api-reference/foundation.md)  |
| **Data Models**          | [DDD Guide](./docs/guides/domain-driven-design.md)                | [FlextModels API](./docs/api-reference/domain.md)         |
| **Configuration**        | [Getting Started](./docs/guides/getting-started.md)               | [FlextConfig API](./docs/api-reference/infrastructure.md) |
| **Development**          | [Standards](./docs/standards/development.md)                      | [Contributing](./docs/development/contributing.md)        |

### By Layer

- **Layer 0 (Constants)**: [Foundation API](./docs/api-reference/foundation.md) → constants.py, typings.py, protocols.py
- **Layer 0.5 (Runtime)**: [Foundation API](./docs/api-reference/foundation.md) → runtime.py
- **Layer 1 (Foundation)**: [Foundation API](./docs/api-reference/foundation.md) → FlextResult, FlextContainer, FlextExceptions
- **Layer 2 (Domain)**: [Domain API](./docs/api-reference/domain.md) → FlextModels, FlextService, FlextMixins, FlextUtilities
- **Layer 3 (Application)**: [Application API](./docs/api-reference/application.md) → FlextHandlers, FlextBus, FlextDispatcher, FlextProcessors
- **Layer 4 (Infrastructure)**: [Infrastructure API](./docs/api-reference/infrastructure.md) → FlextConfig, FlextLogger, FlextContext

### By Feature Status

- **✅ Implemented (v0.9.9)**: All core modules, FlextResult, FlextContainer, FlextModels, FlextService context enrichment
- **🔄 In Progress**: Ecosystem compliance testing, coverage improvements
- **📋 Planned (v1.0.0)**: Additional integration patterns, extended example suite

### Documentation Index

- **[docs/INDEX.md](./docs/INDEX.md)** - Master navigation with 4-level learning paths
- **[docs/pydantic-v2-modernization/README.md](./docs/pydantic-v2-modernization/README.md)** - Migration patterns and best practices

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

**FLEXT-Core v0.9.9** - Foundation library for the FLEXT ecosystem serving 32+ dependent packages with railway-oriented programming patterns, dependency injection, and domain-driven design.

**Version 1.0.0 target** (October 2025) with guaranteed API stability, locked dependencies, and ecosystem testing across all dependent projects. See [VERSIONING.md](VERSIONING.md) and [API_STABILITY.md](API_STABILITY.md) for stability guarantees.
