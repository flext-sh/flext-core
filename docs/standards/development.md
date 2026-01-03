# Development Standards

**FLEXT-Core Development Standards | Version: 0.9.9 | Status: Production Ready**

This document outlines the development standards, patterns, and quality requirements for FLEXT-Core - the foundation library for the entire FLEXT ecosystem.

## 🎯 Mission & Authority

**CRITICAL ROLE**: FLEXT-Core is the FOUNDATION library for the entire FLEXT ecosystem. Every change here impacts 32+ dependent projects. This requires the highest quality standards and zero tolerance for breaking changes.

**CORE RESPONSIBILITIES**:

- ✅ **Railway Pattern Foundation**: FlextResult[T] with .data/.value compatibility
- ✅ **Dependency Injection**: FlextContainer.get_global() with type safety
- ✅ **Domain Models**: FlextModels.Entity/Value/AggregateRoot for DDD patterns
- ✅ **Service Architecture**: FlextService with Pydantic Generic[T] base
- ✅ **Type Safety**: Complete type annotations for ecosystem-wide consistency
- ✅ **Zero Breaking Changes**: Maintain API compatibility across versions
- ✅ **Evidence-Based Quality**: 79% coverage proven stable, targeting 85% for 1.0.0

## 📋 Quality Imperatives

### Zero Tolerance Standards

**MANDATORY - NO EXCEPTIONS:**

1. **Ruff Violations**: ZERO allowed (checked via `make lint`)
2. **MyPy Errors**: ZERO allowed (checked via `make type-check`)
3. **PyRight Errors**: ZERO allowed (enhanced type checking)
4. **Test Failures**: ZERO allowed (comprehensive test suite)
5. **Breaking Changes**: ZERO allowed without migration guide

### Quality Gates

**PRE-COMMIT (Required):**

```bash
# All must pass with ZERO violations
make lint          # Ruff linting
make type-check    # MyPy + PyRight
make test         # Full test suite
make security     # Bandit + pip-audit
```

**PRE-PUBLISH (Required):**

```bash
# Ecosystem validation
make validate-all    # All projects in ecosystem
make coverage-html   # Coverage visualization
make benchmark      # Performance regression tests
```

## 🏗️ Architecture Standards

### Clean Architecture Compliance

**Dependency Rule (STRICT):**

```
Infrastructure → Application → Domain → Foundation
     (outer)         ↓          ↓         (inner)

Inner layers know NOTHING about outer layers.
```

**Layer Responsibilities:**

1. **Foundation Layer** (No Dependencies):
   - FlextResult[T] - Railway pattern
   - FlextContainer - Dependency injection
   - t - Type system
   - FlextConstants - Centralized constants

2. **Domain Layer** (Foundation Only):
   - FlextModels - DDD patterns
   - FlextService - Domain services
   - x - Reusable behaviors

3. **Application Layer** (Foundation + Domain):
   - FlextDispatcher - Message routing
   - h - Handler registry
   - FlextRegistry - Component management

4. **Infrastructure Layer** (All Layers):
   - FlextSettings - Configuration management
   - FlextLogger - Structured logging
   - FlextContext - Context propagation

### Module Organization

**Required Structure:**

```
src/flext_core/
├── __init__.py          # Public API exports ONLY
├── result.py           # Railway pattern (FlextResult)
├── container.py        # DI singleton (FlextContainer)
├── typings.py          # Type system (t)
├── constants.py        # Constants (FlextConstants)
├── models.py           # DDD base classes (FlextModels)
├── service.py          # Domain service base (FlextService)
├── dispatcher.py      # Command/query dispatcher
├── config.py          # Configuration (FlextSettings)
├── loggings.py        # Logging (FlextLogger)
└── ... (other modules)
```

**Import Standards:**

```python
# ✅ CORRECT - Direct imports (import only what you need)
from flext_core import FlextResult, FlextService, FlextModels

# ❌ WRONG - Star imports in production
from flext_core import *

# ❌ WRONG - Relative imports in public APIs
from .result import FlextResult
```

## 🔧 Development Workflow

### Environment Setup

**Required Tools:**

- Python 3.13+ (MANDATORY)
- Poetry 1.8+ (recommended)
- Git (for version control)
- Make (for build automation)

**Setup Process:**

```bash
# 1. Clone repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# 2. Setup development environment
make setup

# 3. Verify installation
python -c "import flext_core; print('✅ Ready')"
```

### Development Commands

**Quality Assurance (Required for all changes):**

```bash
# Complete validation pipeline
make validate      # lint + type-check + test + security

# Individual checks
make lint         # Ruff linting (ZERO tolerance)
make type-check   # MyPy strict + PyRight (ZERO errors)
make test         # Full test suite (1,163+ tests)
make security     # Security audit

# Development helpers
make format       # Auto-format code
make fix          # Auto-fix linting issues
make check        # Quick validation (lint + type-check)
```

**Testing Strategy:**

```bash
# Test categories
make test-unit         # Unit tests (fast, isolated)
make test-integration  # Integration tests (component interaction)
make test-patterns     # Pattern tests (CQRS, DDD, architectural)

# Coverage analysis
make coverage-html     # Generate HTML coverage report
pytest --cov=src --cov-report=term-missing:skip-covered

# Specific modules
pytest tests/unit/test_result.py -v
pytest tests/unit/test_container.py::TestFlextContainer::test_singleton -v
```

## 📝 Code Standards

### Python Standards

**Language Requirements:**

- Python 3.13+ ONLY (MANDATORY)
- PEP 8 compliance (79 character limit)
- Type hints on ALL functions/classes
- Modern syntax (structural pattern matching, etc.)

**Code Style:**

```python
# ✅ CORRECT - Type hints, proper naming
def process_user(user_id: str) -> FlextResult[User]:
    """Process user with validation."""
    if not user_id:
        return FlextResult[User].fail("User ID required")

    user = self.user_repository.get(user_id)
    return FlextResult[User].ok(user)

# ❌ WRONG - Missing types, poor naming
def do_stuff(x):
    if not x:
        return None
    return x
```

### Pattern Standards

**1. Railway Pattern (MANDATORY for all operations):**

```python
# ✅ CORRECT - Always return FlextResult
def create_user(name: str, email: str) -> FlextResult[User]:
    if not name or not email:
        return FlextResult[User].fail("Name and email required")

    user = User(id=f"user_{name}", name=name, email=email)
    return FlextResult[User].ok(user)

# ❌ WRONG - Using exceptions for control flow
def create_user(name: str, email: str) -> User:
    if not name or not email:
        raise ValueError("Invalid input")
    return User(name, email)
```

**2. Dependency Injection (MANDATORY for all services):**

```python
# ✅ CORRECT - Use global container
class UserService(FlextService):
    def __init__(self) -> None:
        super().__init__()
        self.container = FlextContainer.get_global()
        self.logger = self._get_logger()

    def _get_logger(self) -> FlextLogger:
        result = self.container.get("logger")
        return result.value if result.is_success else FlextLogger(__name__)

# ❌ WRONG - Manual DI or no DI
class UserService:
    def __init__(self, logger: Logger):
        self.logger = logger
```

**3. Domain-Driven Design (MANDATORY for business logic):**

```python
# ✅ CORRECT - Proper DDD patterns
class Order(FlextModels.AggregateRoot):
    customer_id: str
    items: list[OrderItem]
    total: Decimal

    def add_item(self, item: OrderItem) -> FlextResult[None]:
        if self.status != OrderStatus.PENDING:
            return FlextResult[None].fail("Can only modify pending orders")

        self.items.append(item)
        self.add_domain_event("ItemAdded", {"item_id": item.entity_id})
        return FlextResult[None].ok(None)

# ❌ WRONG - Anemic model
class Order:
    def __init__(self, items: list):
        self.items = items
```

## 🔒 API Stability Standards

### Versioning Strategy

**Current State**: v0.9.9 Release Candidate → 1.0.0 Stable (October 2025)

**Stability Guarantees:**

- **FlextResult[T]**: Dual `.value`/`.data` access maintained
- **FlextContainer**: Singleton pattern preserved
- **FlextModels**: DDD patterns locked
- **Public APIs**: Zero breaking changes in 1.x series

**Adding New Features:**

1. Add to existing modules (preferred)
2. New modules with clear deprecation path
3. Feature flags for experimental features

**Deprecation Process:**

1. Mark as deprecated in docstring
2. Add warning in implementation
3. Remove in next major version (2.0.0)
4. Update migration guide

## 🧪 Testing Standards

### Test Organization

**Required Structure:**

```
tests/
├── unit/           # Unit tests (fast, isolated)
│   ├── test_result.py
│   ├── test_container.py
│   └── ... (20+ modules)
├── integration/    # Integration tests (component interaction)
│   ├── test_config_singleton_integration.py
│   └── test_service_integration.py
├── patterns/       # Pattern tests (CQRS, DDD, architectural)
│   ├── test_patterns.py
│   └── test_advanced_patterns.py
└── conftest.py     # Shared fixtures
```

**Test Requirements:**

- **Unit Tests**: 100% coverage target for foundation layer
- **Integration Tests**: Component interaction validation
- **Pattern Tests**: Architectural pattern verification
- **No Mocks**: Use flext_tests infrastructure for real dependencies

### Test Patterns

**Unit Test Example:**

```python
def test_flext_result_ok():
    """Test successful result creation."""
    result = FlextResult[str].ok("success")

    assert result.is_success
    assert not result.is_failure
    assert result.value == "success"
    assert result.data == "success"  # Dual access
```

**Integration Test Example:**

```python
def test_container_singleton():
    """Test container singleton behavior."""
    container1 = FlextContainer.get_global()
    container2 = FlextContainer.get_global()

    assert container1 is container2  # Same instance
```

## 📚 Documentation Standards

### Documentation Requirements

**For Every Public API:**

- Complete docstring with examples
- Type annotations and descriptions
- Usage examples and edge cases
- Performance characteristics
- Related APIs and alternatives

**For Architecture Changes:**

- Update architecture documentation
- Add migration guide for breaking changes
- Update examples and tutorials
- Notify dependent projects

### Documentation Structure

**Required Documentation:**

- README.md (project overview)
- docs/getting-started.md (quick start)
- docs/architecture.md (architecture overview)
- docs/api-reference/ (complete API docs)
- docs/development/ (contributing guide)
- CHANGELOG.md (version history)

## 🔄 Refactoring Standards

### Refactoring Guidelines

**Allowed Refactoring:**

- Performance optimizations (with benchmarks)
- Code clarity improvements (maintain functionality)
- Test coverage improvements
- Documentation enhancements

**Prohibited Changes:**

- Breaking API changes without migration
- Removing public APIs without deprecation
- Changing behavior without tests
- Reducing type safety

### Refactoring Process

1. **Create Issue**: Document proposed changes
2. **Write Tests**: Ensure existing functionality preserved
3. **Implement**: Make changes with full validation
4. **Update Docs**: Update all affected documentation
5. **PR Review**: Comprehensive review by maintainers

## 🚨 Emergency Procedures

### Breaking Issues

**If Breaking Issue Discovered:**

1. **Immediate Action**: Create hotfix branch
2. **Root Cause**: Identify and document cause
3. **Fix Implementation**: Minimal, targeted fix
4. **Test Coverage**: Comprehensive tests for issue
5. **Documentation**: Update with issue details

### Ecosystem Impact

**For Ecosystem-Wide Issues:**

1. **Notify Maintainers**: Immediate notification
2. **Coordinate Response**: Cross-project coordination
3. **Staged Rollout**: Gradual fix deployment
4. **Monitoring**: Enhanced monitoring during fix

## 📊 Quality Metrics

### Target Metrics (1.0.0 Release)

| Metric          | Current (0.9.9) | Target (1.0.0) | Status         |
| --------------- | --------------- | -------------- | -------------- |
| Test Coverage   | 75%             | 79%+           | 🔄 In Progress |
| Total Tests     | 1,163           | 1,500+         | 🔄 In Progress |
| Ruff Violations | 0               | 0              | ✅ Complete    |
| MyPy Errors     | 0               | 0              | ✅ Complete    |
| Public APIs     | 60+             | 60+ (stable)   | ✅ Complete    |

### Coverage by Layer

| Layer          | Current | Target | Status       |
| -------------- | ------- | ------ | ------------ |
| Foundation     | 95%+    | 100%   | ✅ Excellent |
| Domain         | 60-70%  | 80%+   | 🔄 Improving |
| Application    | 50-95%  | 85%+   | 🔄 Mixed     |
| Infrastructure | 70-90%  | 90%+   | ✅ Good      |

## 🔗 Related Documentation

- **[README.md](../../README.md)**: Project overview and quick start
- **[Getting Started](../guides/getting-started.md)**: Installation and basic usage
- **[Architecture Overview](../architecture/overview.md)**: Complete architecture guide
- **[API Reference](../api-reference/)**: Complete API documentation
- **[Contributing Guide](../development/contributing.md)**: How to contribute

---

**FLEXT-Core Development Standards** - Ensuring the highest quality foundation for the entire FLEXT ecosystem.
