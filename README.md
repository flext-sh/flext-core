# FLEXT-Core

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Foundation Library](https://img.shields.io/badge/role-foundation-brightgreen.svg)](#)
[![v0.9.9 RC](https://img.shields.io/badge/version-0.9.9--rc-orange.svg)](#)
[![Documentation](https://img.shields.io/badge/docs-organized-blue.svg)](./docs/)
[![GitHub](https://img.shields.io/badge/github-flext--core-black.svg)](https://github.com/flext/flext-core)

**Foundation library** for the FLEXT ecosystem providing railway-oriented programming, dependency injection, domain-driven design patterns, and comprehensive type safety with Python 3.13+.

> **✅ Status**: v0.9.9 Release Candidate · 75% test coverage · 1,163 passing tests · Zero QA violations · **Foundation for 32+ FLEXT projects**

## 📚 Documentation

**Complete documentation available in [./docs/](./docs/)** - Comprehensive guides, API reference, and examples

- **[🚀 Getting Started](./docs/guides/getting-started.md)** - Installation and basic usage
- **[🏗️ Architecture](./docs/architecture/overview.md)** - System design and patterns
- **[🔌 API Reference](./docs/api-reference/)** - Complete API documentation
- **[👥 Contributing](./docs/development/contributing.md)** - Development guidelines

---

## 🎯 Mission & Role in FLEXT Ecosystem

### **Foundation for Enterprise Data Integration**

FLEXT-Core serves as the **architectural foundation** for the entire FLEXT enterprise data integration platform, providing essential patterns and infrastructure that power 32+ specialized projects across the ecosystem.

### **Core Responsibilities**

1. **🏗️ Railway-Oriented Programming** - `FlextResult[T]` for comprehensive error handling
2. **💉 Dependency Injection** - `FlextContainer` for clean, testable architectures
3. **🎯 Domain-Driven Design** - Rich entities, value objects, and domain services
4. **🔒 Type Safety** - Python 3.13+ with comprehensive typing and Pydantic v2
5. **📊 Configuration Management** - Environment-aware settings with validation
6. **🔍 Logging Infrastructure** - Structured logging with multiple output formats
7. **🚌 Event-Driven Architecture** - Message bus and event dispatching
8. **⚡ Processing Pipeline** - Handler chains and processor patterns

### **Ecosystem Integration**

FLEXT-Core provides the **architectural patterns** that all FLEXT projects inherit:

| Project Type         | Projects                           | Integration Pattern                        |
| -------------------- | ---------------------------------- | ------------------------------------------ |
| **Core Libraries**   | flext-api, flext-auth, flext-grpc  | Direct inheritance of foundation patterns  |
| **Infrastructure**   | flext-ldap, flext-db-oracle        | Railway patterns for enterprise operations |
| **Data Integration** | flext-meltano, Singer taps/targets | Configuration and processing patterns      |
| **Enterprise Tools** | flext-quality, flext-observability | Logging and monitoring integration         |

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

## ✨ Phase 1 Architectural Enhancements (v0.9.9)

**Status**: ✅ Completed | **Impact**: Foundation for ecosystem-wide code reduction

### **New Features**

#### **1. Context Enrichment in FlextMixins.Service**

Automatic context management for structured logging and distributed tracing:

```python
from flext_core import FlextService, FlextResult

class PaymentService(FlextService[dict[str, object]]):
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
        config = FlextModels.CqrsConfig.Handler(
            handler_name="CreateOrderHandler",
            handler_type="command",
        )
        super().__init__(config=config)
        # Context automatically includes:
        # - handler_name: "CreateOrderHandler"
        # - handler_type: "command"
        # - handler_class: "CreateOrderHandler"
```

#### **3. Helper Method for Complete Automation**

`FlextService.execute_with_context_enrichment()` provides complete automation:

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

See `examples/automation_showcase.py` for complete working examples demonstrating:
- Basic service with automatic context enrichment
- Payment service with correlation ID tracking
- Order service using context enrichment helper method

---

## 🚀 1.0.0 Release Roadmap

**Target Date**: October 2025 | **Current**: v0.9.9 Release Candidate

### Why 1.0.0 Matters

FLEXT-Core serves as the **foundation for 32+ dependent packages** in the FLEXT ecosystem. The 1.0.0 release represents our commitment to:

- **🔒 API Stability**: Zero breaking changes throughout the 1.x series
- **⚡ ABI Compatibility**: Locked dependency versions prevent ecosystem breakage
- **🏭 Production Readiness**: Enterprise-grade quality with comprehensive testing
- **🛠️ Long-term Support**: Minimum 2 minor version deprecation cycle

### Release Timeline (5 Weeks)

#### Phase 1: API Stabilization & Documentation (Weeks 1-2) ✅

- ✅ **ABI Finalization**: Dependency versions locked, semantic versioning strategy defined
- ✅ **API Guarantees**: Comprehensive stability documentation (VERSIONING.md, API_STABILITY.md)
- 🔄 **Documentation**: README.md roadmap, CLAUDE.md guidelines, migration guide (in progress)

#### Phase 2: Quality Assurance & Ecosystem Testing (Weeks 2-3)

- Test coverage enhancement (75% → 79%+ target)
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

## 🚀 1.0.0 Release Roadmap

**Target Date**: October 2025 | **Current**: v0.9.9 Release Candidate

### Why 1.0.0 Matters

FLEXT-Core serves as the **foundation for 32+ dependent packages** in the FLEXT ecosystem. The 1.0.0 release represents our commitment to:

- **API Stability**: Zero breaking changes throughout the 1.x series
- **ABI Compatibility**: Locked dependency versions prevent ecosystem breakage
- **Production Readiness**: Enterprise-grade quality with comprehensive testing
- **Long-term Support**: Minimum 2 minor version deprecation cycle

### Release Timeline (5 Weeks)

#### Phase 1: API Stabilization & Documentation (Weeks 1-2) ✅

- ✅ **ABI Finalization**: Dependency versions locked, semantic versioning strategy defined
- ✅ **API Guarantees**: Comprehensive stability documentation (VERSIONING.md, API_STABILITY.md)
- 🔄 **Documentation**: README.md roadmap, CLAUDE.md guidelines, migration guide (in progress)

#### Phase 2: Quality Assurance & Ecosystem Testing (Weeks 2-3)

- Test coverage enhancement (75% → 79%+ target)
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

**Production-Ready Foundation**:

- ✅ **FlextResult[T]** - Railway-oriented programming with dual `.value`/`.data` access for ABI stability
- ✅ **FlextContainer** - Singleton dependency injection with typed service keys and lifecycle management
- ✅ **FlextModels** - Domain-driven design with Entity/Value/AggregateRoot patterns (Pydantic v2)
- ✅ **FlextLogger** - Structured logging with context propagation and correlation tracking
- ✅ **FlextConfig** - Layered configuration with .env, TOML, and YAML support
- ✅ **FlextBus** - Command/Query/Event bus with middleware pipeline and caching
- ✅ **FlextContext** - Request/operation context with correlation IDs and metadata
- ✅ **FlextDispatcher** - Unified command/query dispatcher with registry support
- ✅ **FlextTypes** - Comprehensive type system with 50+ TypeVars and type aliases

**Quality Metrics**:

- **Ruff**: Zero violations
- **PyRight/MyPy**: Zero errors (strict mode)
- **Coverage**: 75% (proven stable), targeting 79% for 1.0.0
- **Tests**: 1,163 passing (unit + integration + patterns)

---

## Architecture Overview

**Foundation Layer**:

- `FlextResult[T]` - Monadic error handling with railway-oriented composition
- `FlextContainer` - Dependency injection singleton with typed service resolution
- `FlextExceptions` - Comprehensive exception hierarchy with error codes
- `FlextConstants` - Centralized constants and enumerations
- `FlextTypes` - Complete type system (TypeVars, Protocols, Aliases)

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
python -c "from flext_core import FlextResult; print('✅ FLEXT-Core v0.9.9 ready')"
```

## Quick Start Example

```python
from flext_core import (
    FlextResult,
    FlextContainer,
    FlextLogger,
    FlextModels,
    FlextService,
)

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
print(f"Created user: {user.name} with ID: {user.id}")

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
| **Foundation**     | `result.py`     | 95%      | Railway pattern with monadic composition |
|                    | `container.py`  | 99%      | Dependency injection singleton           |
|                    | `typings.py`    | 100%     | Type system (50+ TypeVars)               |
|                    | `constants.py`  | 100%     | Centralized constants                    |
|                    | `exceptions.py` | 62%      | Exception hierarchy                      |
| **Domain**         | `models.py`     | 65%      | DDD patterns (Entity/Value/Aggregate)    |
|                    | `service.py`    | 92%      | Domain service base class                |
|                    | `mixins.py`     | 57%      | Reusable behaviors                       |
|                    | `utilities.py`  | 66%      | Domain utilities                         |
| **Application**    | `bus.py`        | 94%      | Message bus with middleware              |
|                    | `cqrs.py`       | 100%     | CQRS patterns                            |
|                    | `handlers.py`   | 66%      | Handler registry                         |
|                    | `dispatcher.py` | 45%      | Unified dispatcher                       |
|                    | `registry.py`   | 91%      | Handler registry management              |
|                    | `processors.py` | 56%      | Message processing                       |
| **Infrastructure** | `config.py`     | 90%      | Configuration management                 |
|                    | `loggings.py`   | 72%      | Structured logging                       |
|                    | `context.py`    | 66%      | Context tracking                         |
|                    | `protocols.py`  | 99%      | Runtime protocols                        |
|                    | `version.py`    | 100%     | Version management                       |

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
- **Linting**: Ruff (ZERO violations)
- **Type Checking**: MyPy strict mode + PyRight (ZERO errors in src/)
- **Line Length**: 79 characters (PEP 8 strict)
- **Coverage**: Current 75%, baseline achieved, target 79% for 1.0.0
- **Tests**: 1,163 passing (unit + integration + patterns)

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

## Roadmap to 1.0.0

### Current Status (v0.9.9)

- ✅ Core API stable and production-ready
- ✅ Zero QA violations (Ruff + MyPy + PyRight)
- ✅ 1,163 tests passing
- ✅ Coverage at 75% (baseline achieved, targeting 79% for 1.0.0)

### 1.0.0 Requirements

1. **Coverage**: Reach 79% minimum (currently 75%)
   - Priority: dispatcher (45%), processors (56%), mixins (57%)
   - Already achieved 75% baseline - only 4% more to target
   - Add functional tests for error paths and edge cases
2. **API Stability**: Maintain backward compatibility
   - Keep dual `.value`/`.data` access on FlextResult
   - Preserve container singleton pattern
   - No breaking changes to public API
3. **Documentation**: Complete API reference
   - Document all public classes and methods
   - Add usage examples for each module
   - Update architecture documentation

### Timeline

- **Target**: October 2025
- **Focus**: Quality over features
- **Commitment**: Zero breaking changes in 1.x series

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

- **[📚 Complete Documentation](./docs-new/)**: Comprehensive guides and API reference
- **Getting Started**: [`docs-new/guides/getting-started.md`](./docs-new/guides/getting-started.md)
- **Architecture**: [`docs-new/architecture/overview.md`](./docs-new/architecture/overview.md)
- **API Reference**: [`docs-new/api-reference/`](./docs-new/api-reference/)
- **Development**: [`docs-new/development/contributing.md`](./docs-new/development/contributing.md)
- **Standards**: [`docs-new/standards/`](./docs-new/standards/)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/flext-sh/flext-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/flext-sh/flext-core/discussions)
- **Security**: Report vulnerabilities privately to FLEXT maintainers

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

**FLEXT-Core v0.9.9** - Production-ready foundation for the FLEXT ecosystem powering 32+ dependent packages with railway-oriented programming, dependency injection, and domain-driven design patterns.

**On the road to 1.0.0** (October 2025) with guaranteed API stability, locked dependencies, and comprehensive ecosystem testing. See [VERSIONING.md](VERSIONING.md) and [API_STABILITY.md](API_STABILITY.md) for our stability commitment.
