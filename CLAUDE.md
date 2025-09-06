# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**References**: See [../CLAUDE.md](../CLAUDE.md) for FLEXT ecosystem standards and [README.md](README.md) for project overview.

## Development Commands

### Essential Development Workflow

```bash
# Initial setup
make setup                 # Complete dev environment setup with pre-commit hooks

# Before any commit (MANDATORY)
make validate              # Run ALL quality gates (lint + type + security + test)
make check                 # Quick validation (lint + type-check only)

# Individual quality checks
make lint                  # Ruff linting with comprehensive rules
make type-check            # MyPy strict mode checking (zero tolerance in src/)
make test                  # Full test suite (75% coverage minimum required)
make security              # Bandit + pip-audit security scanning
make format                # Auto-format code (79 char line limit - PEP8 strict)
make fix                   # Auto-fix linting issues

# Testing commands
make test-unit             # Unit tests only (fast feedback)
make test-integration      # Integration tests only
make test-fast             # Tests without coverage (quick iteration)
make coverage-html         # Generate HTML coverage report

# Development utilities
make shell                 # Python REPL with project loaded
make deps-show             # Show dependency tree
make deps-update           # Update all dependencies
make deps-audit            # Security audit of dependencies
make doctor                # Complete health check with diagnostics
make diagnose              # Project diagnostics (versions, environment)
make clean                 # Clean build artifacts
make clean-all             # Deep clean including venv
make reset                 # Full reset (clean + setup)
make pre-commit            # Run pre-commit hooks manually

# Build and documentation
make build                 # Build the package
make build-clean           # Clean and build
make docs                  # Build documentation with mkdocs
make docs-serve            # Serve documentation locally

# Single letter aliases for speed
make t                     # Alias for test
make l                     # Alias for lint
make f                     # Alias for format
make tc                    # Alias for type-check
make c                     # Alias for clean
make i                     # Alias for install
make v                     # Alias for validate
```

### Running Specific Tests

```bash
# Run specific test file
PYTHONPATH=src poetry run pytest tests/unit/test_result.py -v
PYTHONPATH=src poetry run pytest tests/unit/test_container.py::TestFlextContainer::test_basic_registration -v

# Test with markers
poetry run pytest -m unit              # Unit tests only
poetry run pytest -m integration       # Integration tests only
poetry run pytest -m "not slow"        # Exclude slow tests
poetry run pytest -m core              # Core framework tests
poetry run pytest -m ddd               # Domain-driven design tests

# Advanced test execution
poetry run pytest tests/unit/test_result.py::TestFlextResult::test_map -xvs --tb=long
poetry run pytest -m "unit and not slow" --tb=short -q
poetry run pytest tests/unit/ --cov=src/flext_core --cov-report=term-missing
poetry run pytest --lf --ff -x  # Run last failed tests first with fail-fast
poetry run pytest -n auto tests/unit/  # Parallel execution
poetry run pytest tests/unit/ -k "test_result" -v  # Tests matching pattern

# Coverage analysis
PYTHONPATH=src pytest tests/ --cov=src --cov-report=term-missing
PYTHONPATH=src pytest tests/unit/test_result.py --cov=src/flext_core/result.py --cov-report=term-missing
```

## High-Level Architecture

### Core Pattern: FlextResult Railway

The foundation of error handling across the entire ecosystem - eliminates exceptions in business logic through railway-oriented programming:

```python
from flext_core import FlextResult

def validate_user(data: dict) -> FlextResult[User]:
    """All operations return FlextResult for composability."""
    if not data.get("email"):
        return FlextResult[None].fail("Email required", error_code="VALIDATION_ERROR")
    return FlextResult[None].ok(User(**data))

# Railway-oriented composition - the key pattern
result = (
    validate_user(data)
    .flat_map(lambda u: save_user(u))      # Chain operations (monadic bind)
    .map(lambda u: format_response(u))      # Transform success value
    .map_error(lambda e: log_error(e))      # Handle errors in pipeline
    .filter(lambda u: u.is_active, "User not active")  # Conditional filtering
)

# Safe value extraction
if result.success:
    user = result.unwrap()  # Extract value after success check
else:
    logger.error(f"Operation failed: {result.error}")
    
# Alternative patterns
value = result.unwrap_or(default_user)  # With default
value = result.expect("User validation must succeed")  # With custom error
```

### Core Pattern: Dependency Injection

Global container pattern used across all FLEXT services with type-safe service management:

```python
from flext_core import FlextContainer

# Get global singleton container
container = FlextContainer.get_global()

# Register services (returns FlextResult for error handling)
container.register("database", DatabaseService())
container.register_factory("logger", lambda: create_logger())
container.register_singleton("cache", CacheService())

# Type-safe retrieval with FlextResult
db_result = container.get("database")
if db_result.success:
    db = db_result.unwrap()
    # Use the service
```

### Module Organization

The library follows Clean Architecture with strict layering to prevent circular dependencies:

```
src/flext_core/
├── Foundation Layer (Core Patterns - No Dependencies)
│   ├── result.py           # FlextResult[T] railway pattern (monadic operations)
│   ├── container.py        # Dependency injection with type-safe ServiceKey[T]
│   ├── exceptions.py       # Exception hierarchy with error codes
│   ├── constants.py        # FlextConstants, enums, error messages
│   └── typings.py          # Type variables (T, U, V) and type aliases
│
├── Domain Layer (Business Logic - Depends on Foundation)
│   ├── models.py           # FlextModels.Entity/Value/AggregateRoot (DDD patterns)
│   ├── domain_services.py  # Domain service patterns and operations
│   └── validations.py      # FlextValidations with predicate-based rules
│
├── Application Layer (Use Cases - Depends on Domain)
│   ├── commands.py         # FlextCommands CQRS pattern implementation
│   ├── handlers.py         # FlextHandlers registry and execution
│   └── guards.py           # FlextGuards type guards and validation decorators
│
├── Infrastructure Layer (External Concerns - Depends on Application)
│   ├── config.py           # FlextConfig with Pydantic Settings
│   ├── loggings.py         # Structured logging with structlog
│   ├── protocols.py        # FlextProtocols interface definitions
│   └── context.py          # Request/operation context management
│
└── Support Modules (Cross-cutting Utilities)
    ├── mixins.py           # Reusable behaviors (timestamps, serialization)
    ├── decorators.py       # @safe_result, @validate, @cache decorators
    ├── utilities.py        # FlextUtilities helper functions
    └── core.py             # FlextCore main orchestrator class
```

### Domain Modeling (DDD Patterns)

All ecosystem projects inherit these domain-driven design patterns:

```python
from flext_core import FlextModels, FlextResult

# Value Object - Immutable, compared by value
class Email(FlextModels.Value):
    address: str
    
    def validate(self) -> FlextResult[None]:
        if "@" not in self.address:
            return FlextResult[None].fail("Invalid email")
        return FlextResult[None].ok(None)

# Entity - Has identity and lifecycle
class User(FlextModels.Entity):
    name: str
    email: Email  # Composition with value object
    
    def activate(self) -> FlextResult[None]:
        """Business operations return FlextResult."""
        if self.is_active:
            return FlextResult[None].fail("Already active")
        self.is_active = True
        self.add_domain_event("UserActivated", {"user_id": self.id})
        return FlextResult[None].ok(None)

# Aggregate Root - Consistency boundary
class Account(FlextModels.AggregateRoot):
    owner: User
    balance: Decimal
    
    def withdraw(self, amount: Decimal) -> FlextResult[None]:
        """Enforces business invariants."""
        if amount > self.balance:
            return FlextResult[None].fail("Insufficient funds")
        self.balance -= amount
        self.add_domain_event("MoneyWithdrawn", {"amount": str(amount)})
        return FlextResult[None].ok(None)
```

## Key Development Patterns

### Error Handling Pattern (MANDATORY)

Never raise exceptions in business logic - always return FlextResult:

```python
from flext_core import FlextResult

def business_operation(data: dict) -> FlextResult[ProcessedData]:
    """MANDATORY: All business operations must return FlextResult."""
    if not data:
        return FlextResult[None].fail("Data required", error_code="VALIDATION_ERROR")
    
    # Railway-oriented composition (PREFERRED)
    return (
        validate_data(data)
        .flat_map(lambda d: process_data(d))      # Chain operations
        .map(lambda d: enrich_data(d))            # Transform success
        .recover_with(lambda e: fetch_fallback())  # Error recovery
    )
```

### Import Strategy (CRITICAL)

Always import from root module, never from internals:

```python
# ✅ CORRECT - Import from root
from flext_core import FlextResult, FlextContainer, FlextModels

# ❌ WRONG - Never import from internal modules
from flext_core.result import FlextResult  # FORBIDDEN
from flext_core.models import FlextModels  # FORBIDDEN
```

### Testing with Shared Infrastructure

Tests use consolidated support infrastructure:

```python
# Import from flext_tests support module
from flext_tests import (
    UserFactory,           # Test data factories
    FlextResultFactory,    # Result creation helpers
    FlextMatchers,         # Custom pytest matchers
    TestBuilders,          # Builder pattern for test data
)

# Use provided fixtures from conftest.py
def test_with_container(clean_container):
    """Use clean_container fixture for isolated DI testing."""
    clean_container.register("service", MyService())
    # Test continues...
```

## Quality Standards

### Code Quality Requirements

- **MyPy Strict Mode**: Zero tolerance for type errors in `src/` directory
- **Line Length**: 79 characters maximum (PEP8 strict enforcement)
- **Coverage Minimum**: 75% test coverage (enforced by Makefile)
- **Type Hints**: Required for ALL function signatures and class attributes
- **Naming Convention**: `FlextXxx` prefix for all public exports
- **Docstrings**: Required for all public APIs (Google style)

### Pre-Commit Checks

Before committing, these must pass:
1. `make lint` - No linting errors (ruff)
2. `make type-check` - No type errors (mypy strict)
3. `make test` - All tests pass with 75%+ coverage
4. `make security` - No security vulnerabilities

## Common Troubleshooting

### Import Resolution Issues

```bash
# Verify actual exports (ALWAYS do this first)
grep "^from.*import\|^import" src/flext_core/__init__.py

# Check if class exists and where
grep -r "class ClassName" src/

# Test import directly with PYTHONPATH
PYTHONPATH=src python -c "from flext_core import ClassName"
```

### Type Checking Errors

```bash
# Get detailed type errors with context
mypy src/ --show-error-codes --show-error-context

# Check specific file
PYTHONPATH=src mypy src/flext_core/result.py --strict

# Use pyright for additional checking
PYTHONPATH=src pyright src/ --outputformat text
```

### Test Failures

```bash
# Run single test with verbose output
PYTHONPATH=src pytest tests/unit/test_result.py::TestFlextResult::test_map -xvs

# Debug with Python debugger
PYTHONPATH=src python -m pdb -m pytest tests/unit/test_result.py

# Check test coverage for specific module
PYTHONPATH=src pytest tests/ --cov=src/flext_core/result --cov-report=term-missing
```

## Ecosystem Impact

FLEXT Core is the foundation for 32+ projects. Changes here affect:

- **Infrastructure Libraries** (6): flext-db-oracle, flext-ldap, flext-grpc, etc.
- **Application Services** (5): flext-api, flext-auth, flext-web, etc.
- **Singer Ecosystem** (15): Taps, Targets, DBT transformations
- **Go Services** (2): FlexCore runtime, FLEXT Service control panel

### Breaking Change Protocol

1. Check impact: `grep -r "from flext_core" ../` in dependent projects
2. Document migration path in CHANGELOG.md
3. Use deprecation warnings for 2 version cycles
4. Follow semantic versioning strictly

## Verification-First Development

**CRITICAL**: Always verify before making claims:

```bash
# Before claiming something works
mcp__filesystem__read_text_file path=/path/to/file.py  # Read actual content
PYTHONPATH=src python file.py                          # Test execution

# Before assuming imports exist
mcp__filesystem__read_text_file path=src/flext_core/__init__.py  # Check exports
python -c "from flext_core import Class"                         # Verify import
```

**NEVER** assume based on file names or logical patterns. **ALWAYS** verify with tools.

---

**FLEXT-CORE AUTHORITY**: These guidelines are specific to flext-core development.
**ECOSYSTEM STANDARDS**: See [../CLAUDE.md](../CLAUDE.md) for workspace-wide patterns.
**EVIDENCE-BASED**: All patterns documented here are verified through implementation.