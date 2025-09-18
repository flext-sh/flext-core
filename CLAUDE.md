# FLEXT-CORE CLAUDE.md

**The Foundation Library Development Guide for FLEXT Ecosystem**
**Version**: 2.2.0 | **Authority**: CORE FOUNDATION | **Updated**: 2025-09-18
**Status**: 79% test coverage (proven stable), v0.9.9 Release Candidate preparing for 1.0.0 stable release · 1.0.0 Release Preparation

**References**: See [../CLAUDE.md](../CLAUDE.md) for FLEXT ecosystem standards and [README.md](README.md) for project overview.

**Hierarchy**: This document provides project-specific standards based on workspace-level patterns defined in [../CLAUDE.md](../CLAUDE.md). For architectural principles, quality gates, and MCP server usage, reference the main workspace standards.

---

## 🎯 FLEXT-CORE MISSION (FOUNDATION AUTHORITY)

**CRITICAL ROLE**: flext-core is the FOUNDATION library for the entire FLEXT ecosystem. Every change here impacts 32+ dependent projects. This requires the highest quality standards and zero tolerance for breaking changes.

**CORE RESPONSIBILITIES**:
- ✅ **Railway Pattern Foundation**: FlextResult[T] with .data/.value compatibility
- ✅ **Dependency Injection**: FlextContainer.get_global() with type safety
- ✅ **Domain Models**: FlextModels.Entity/Value/AggregateRoot for DDD patterns
- ✅ **Service Architecture**: FlextDomainService with Pydantic Generic[T] base
- ✅ **Type Safety**: Complete type annotations for ecosystem-wide consistency
- ✅ **Zero Breaking Changes**: Maintain API compatibility across versions
- ✅ **Evidence-Based Quality**: 79% coverage proven stable, targeting 85% for 1.0.0

**ECOSYSTEM IMPACT** (32+ Projects Depend on This):
- **Infrastructure**: flext-db-oracle, flext-ldap, flext-grpc, flext-auth, etc.
- **Applications**: flext-api, flext-cli, flext-web, flext-observability
- **Singer Platform**: 15+ taps, targets, and DBT transformations
- **Data Integration**: Oracle OIC/WMS, client-a OUD migration, client-b

**QUALITY IMPERATIVES**:
- 🔴 **ZERO tolerance** for API breaking changes without deprecation cycle
- 🟢 **85%+ test coverage** with REAL functional tests (current: 79%)
- 🟢 **Zero errors** in MyPy strict mode, PyRight, and Ruff for ALL src/ code
- 🟢 **Complete type annotations** - this sets the standard for entire ecosystem
- 🟢 **Professional documentation** - all public APIs must be perfectly documented

## FLEXT-CORE DEVELOPMENT WORKFLOW (FOUNDATION QUALITY)

### Essential Development Workflow (MANDATORY FOR CORE)

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
│   └── utilities.py        # FlextUtilities.Validation with validation utilities
│
├── Application Layer (Use Cases - Depends on Domain)
│   ├── commands.py         # FlextCommands CQRS pattern implementation
│   ├── handlers.py         # FlextHandlers registry and execution
│
├── Infrastructure Layer (External Concerns - Depends on Application)
│   ├── config.py           # FlextConfig with Pydantic Settings
│   ├── loggings.py         # Structured logging with structlog
│   ├── protocols.py        # FlextProtocols interface definitions
│   └── context.py          # Request/operation context management
│
└── Support Modules (Cross-cutting Utilities)
    ├── mixins.py           # Reusable behaviors (timestamps, serialization)
    ├── utilities.py        # FlextUtilities helper functions
    └── processing.py       # FlextProcessing for orchestration functionality
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

### FlextResult Railway Pattern (ECOSYSTEM FOUNDATION)

**CRITICAL**: This is the foundation error handling pattern for ALL 32+ ecosystem projects. Changes here must maintain backward compatibility.

```python
from flext_core import FlextResult

# VERIFIED API - FlextResult has BOTH .data and .value for compatibility
def foundation_operation(data: dict) -> FlextResult[ProcessedData]:
    """Foundation library operation demonstrating ecosystem-wide patterns."""
    if not data:
        return FlextResult[ProcessedData].fail("Data required", error_code="VALIDATION_ERROR")

    # Railway-oriented composition (ECOSYSTEM STANDARD)
    return (
        validate_data(data)
        .flat_map(lambda d: process_data(d))      # Monadic bind - chain operations
        .map(lambda d: enrich_data(d))            # Transform success value
        .map_error(lambda e: log_error(e))        # Handle errors in pipeline
        .filter(lambda d: d.is_valid, "Invalid data")  # Conditional filtering
    )

# CRITICAL: Backward compatibility - maintain both .data and .value
def access_result_data(result: FlextResult[User]) -> User | None:
    \"\"\"Demonstrate both access patterns for ecosystem compatibility.\"\"\"
    if result.is_success:
        # Both patterns must work for ecosystem compatibility
        user_via_value = result.value      # New preferred API
        user_via_data = result.data        # Legacy compatibility (MUST MAINTAIN)
        user_via_unwrap = result.unwrap()  # Explicit unwrap after success check

        # All three should return the same value
        assert user_via_value == user_via_data == user_via_unwrap
        return user_via_value
    return None
```

### Foundation Export Strategy (ECOSYSTEM API AUTHORITY)

**CRITICAL**: flext-core __init__.py defines the API for entire ecosystem. Changes here impact all dependent projects.

```python
# ✅ CORRECT - Root module imports (ECOSYSTEM STANDARD)
from flext_core import (
    FlextResult,           # Railway pattern - ecosystem foundation
    FlextContainer,        # Dependency injection - use .get_global()
    FlextModels,           # Domain models - Entity/Value/AggregateRoot
    FlextDomainService,    # Service base class - Pydantic Generic[T]
    FlextLogger,           # Structured logging - direct instantiation
    FlextConfig,           # Configuration management
    FlextConstants,        # System constants and enums
    FlextTypes,            # Type definitions and aliases
    FlextUtilities,        # Utilities including validation patterns
)

# ❌ ABSOLUTELY FORBIDDEN - Internal module imports
from flext_core.result import FlextResult     # BREAKS ECOSYSTEM COMPATIBILITY
from flext_core.models import FlextModels     # VIOLATES EXPORT STRATEGY
from flext_core.container import FlextContainer  # BYPASSES API LAYER

# DEVELOPMENT NOTE: If you need to import internally during core development,
# it means the __init__.py export is incomplete - FIX the export, don't bypass it
```

### Core Library Development Patterns (FOUNDATION AUTHORITY)

```python
# VERIFIED API PATTERNS FOR CORE DEVELOPMENT

# 1. FlextResult with backward compatibility (CRITICAL for ecosystem)
class FlextResultExample:
    \"\"\"Demonstrate proper FlextResult usage in core library.\"\"\"

    @staticmethod
    def create_with_compatibility() -> FlextResult[str]:
        \"\"\"Show how core library maintains API compatibility.\"\"\"
        result = FlextResult[str].ok("success")

        # CRITICAL: These must ALL work for ecosystem compatibility
        assert result.value == "success"     # New API
        assert result.data == "success"      # Legacy API (MUST MAINTAIN)
        assert result.unwrap() == "success"  # Explicit API

        return result

# 2. FlextContainer with global singleton pattern
class ContainerExample:
    \"\"\"Demonstrate proper container usage in core library.\"\"\"

    def __init__(self) -> None:
        # VERIFIED PATTERN: Direct class access, no wrapper functions
        self._container = FlextContainer.get_global()

    def register_core_services(self) -> FlextResult[None]:
        \"\"\"Register services using verified container API.\"\"\"
        # Container operations return FlextResult for error handling
        logger_result = self._container.register("logger", FlextLogger(__name__))
        if logger_result.is_failure:
            return FlextResult[None].fail(f"Logger registration failed: {logger_result.error}")

        return FlextResult[None].ok(None)

# 3. FlextDomainService as foundation service pattern
class CoreService(FlextDomainService):
    \"\"\"Core service demonstrating foundation service patterns.\"\"\"

    def __init__(self) -> None:
        super().__init__()  # Initialize Pydantic base
        self._logger = FlextLogger(__name__)

    def perform_core_operation(self, data: dict) -> FlextResult[dict]:
        \"\"\"Core operation with proper error handling and logging.\"\"\"
        self._logger.info("Performing core operation", extra={"data_keys": list(data.keys())})

        # Input validation with early return
        if not data:
            return FlextResult[dict].fail("Input data cannot be empty")

        # Business logic with explicit error handling
        processed_data = {"processed": True, "original": data}

        self._logger.info("Core operation completed successfully")
        return FlextResult[dict].ok(processed_data)
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

## FLEXT-CORE QUALITY STANDARDS (FOUNDATION REQUIREMENTS)

### Code Quality Requirements (ECOSYSTEM FOUNDATION LEVEL)

**CRITICAL**: As the foundation library, flext-core must achieve the highest quality standards. All 32+ dependent projects rely on this quality.

- **MyPy Strict Mode**: ZERO tolerance for type errors in `src/` directory - this sets ecosystem standard
- **PyRight Validation**: ZERO errors - secondary type checking for completeness
- **Line Length**: 79 characters maximum (PEP8 strict enforcement)
- **Coverage Target**: 85%+ real functional test coverage (current: 79%, targeting improvement)
- **Type Hints**: Required for ALL function signatures, class attributes, and public APIs
- **Naming Convention**: `FlextXxx` prefix for ALL public exports (ecosystem consistency)
- **Docstrings**: Required for ALL public APIs with complete examples (Google style)
- **API Compatibility**: ZERO breaking changes without proper deprecation cycle

### Foundation Quality Gates (MANDATORY FOR ALL COMMITS)

**CRITICAL**: These quality gates protect the entire ecosystem from regressions:

```bash
# PHASE 1: Foundation Code Quality (ZERO TOLERANCE)
make lint                    # Ruff: ZERO violations allowed in src/
make type-check             # MyPy strict: ZERO errors in src/
make security               # Bandit: ZERO critical vulnerabilities

# PHASE 2: API Compatibility Validation
python -c "from flext_core import *; print('API imports successful')"
python -c "
from flext_core import FlextResult
result = FlextResult[str].ok('test')
assert hasattr(result, 'data'), 'Legacy .data API missing'
assert hasattr(result, 'value'), 'New .value API missing'
print('API compatibility validated')
"

# PHASE 3: Foundation Test Coverage (EVIDENCE-BASED)
make test                   # 79%+ coverage with REAL functional tests
pytest tests/ --cov=src/flext_core --cov-fail-under=79

# PHASE 4: Ecosystem Impact Validation
# Before any API change, validate dependent projects
echo "Testing ecosystem compatibility..."
for project in ../flext-api ../flext-cli ../flext-auth; do
    if [ -d "$project" ]; then
        echo "Testing $project compatibility..."
        cd "$project" && python -c "from flext_core import FlextResult; print('OK')" && cd - || echo "FAILED: $project"
    fi
done
```

### Foundation Development Standards (ZERO FALLBACK POLICY)

**ABSOLUTELY FORBIDDEN IN FLEXT-CORE**:
- ❌ **Try/except fallback patterns** - core library must handle errors explicitly
- ❌ **Multiple classes per module** - single responsibility, unified classes only
- ❌ **Helper functions outside classes** - everything must be properly organized
- ❌ **ANY type usage** - complete type annotations required
- ❌ **Generic type ignore** - use specific error codes if absolutely necessary
- ❌ **API breaking changes** - maintain compatibility or deprecate properly

**MANDATORY IN FLEXT-CORE**:
- ✅ **Explicit FlextResult error handling** - demonstrate ecosystem patterns
- ✅ **Complete type annotations** - set standard for dependent projects
- ✅ **Backward API compatibility** - maintain .data/.value dual access
- ✅ **Professional documentation** - every public API fully documented
- ✅ **Real functional tests** - minimal mocks, test actual functionality
- ✅ **Zero tolerance quality** - foundation library cannot compromise

## FLEXT-CORE TROUBLESHOOTING (FOUNDATION DIAGNOSTICS)

### API Export Validation (ECOSYSTEM CRITICAL)

```bash
# CRITICAL: Verify all foundation exports are available
echo "=== FOUNDATION API VALIDATION ==="

# 1. Check __init__.py exports completeness
echo "Checking foundation exports..."
python -c "
import sys
sys.path.insert(0, 'src')
from flext_core import (
    FlextResult, FlextContainer, FlextModels, FlextDomainService,
    FlextLogger, FlextConfig, FlextConstants, FlextTypes, FlextUtilities
)
print('✅ All foundation exports available')
"

# 2. Verify FlextResult API compatibility (CRITICAL for ecosystem)
echo "Checking FlextResult API compatibility..."
python -c "
import sys
sys.path.insert(0, 'src')
from flext_core import FlextResult
result = FlextResult[str].ok('test')
assert hasattr(result, 'data'), 'Legacy .data API MISSING - ECOSYSTEM BREAKING'
assert hasattr(result, 'value'), 'New .value API MISSING'
assert hasattr(result, 'unwrap'), 'Unwrap method MISSING'
assert result.data == result.value == 'test', 'API consistency BROKEN'
print('✅ FlextResult API compatibility confirmed')
"

# 3. Check Container API availability
echo "Checking Container API..."
python -c "
import sys
sys.path.insert(0, 'src')
from flext_core import FlextContainer
container = FlextContainer.get_global()
print('✅ Container global access working')
"
```

### Foundation Type Checking (ECOSYSTEM STANDARD)

```bash
# CRITICAL: Foundation library must have ZERO type errors
echo "=== FOUNDATION TYPE VALIDATION ==="

# MyPy strict mode (ZERO tolerance)
echo "MyPy strict validation..."
mypy src/ --strict --show-error-codes --no-error-summary || echo "❌ CRITICAL: Type errors in foundation"

# PyRight additional validation
echo "PyRight validation..."
pyright src/ --outputformat text --level error || echo "❌ CRITICAL: PyRight errors in foundation"

# Type coverage check
echo "Type coverage analysis..."
mypy src/ --html-report type-coverage-report --show-error-codes
```

### Foundation Test Diagnostics (QUALITY ASSURANCE)

```bash
# CRITICAL: Foundation tests must be rock solid
echo "=== FOUNDATION TEST DIAGNOSTICS ==="

# 1. Test coverage analysis with module breakdown
echo "Analyzing test coverage by module..."
pytest tests/ --cov=src/flext_core --cov-report=term-missing --cov-report=html:coverage-report

# 2. Run specific foundation component tests
echo "Testing core components individually..."

# FlextResult tests (CRITICAL - entire ecosystem depends on this)
pytest tests/unit/test_result.py -v --tb=short

# FlextContainer tests (DI foundation)
pytest tests/unit/test_container.py -v --tb=short

# FlextModels tests (Domain foundation)
pytest tests/unit/test_models.py -v --tb=short 2>/dev/null || echo "Models tests not found"

# 3. Integration test validation
pytest tests/integration/ -v --tb=short 2>/dev/null || echo "Integration tests not found"

# 4. Performance regression tests
echo "Checking performance baselines..."
pytest tests/performance/ -v 2>/dev/null || echo "Performance tests not implemented"
```

### Ecosystem Impact Diagnostics (DEPENDENCY VALIDATION)

```bash
# CRITICAL: Validate impact on dependent projects
echo "=== ECOSYSTEM IMPACT VALIDATION ==="

# 1. Check dependent projects can import flext-core
echo "Testing dependent project compatibility..."
for project in ../flext-api ../flext-cli ../flext-auth ../flext-ldap; do
    if [ -d "$project" ]; then
        echo "Testing $project..."
        cd "$project"
        python -c "
import sys
sys.path.insert(0, '../flext-core/src')
try:
    from flext_core import FlextResult, FlextContainer, FlextModels
    print('✅ $project: Core imports successful')
except Exception as e:
    print('❌ $project: Import failed -', e)
            exit(1)
        " || echo "❌ ECOSYSTEM BREAK: $project failed"
        cd - > /dev/null
    fi
done

# 2. API compatibility validation across ecosystem
echo "Validating API consistency across ecosystem..."
python -c "
import sys, os
sys.path.insert(0, 'src')
from flext_core import FlextResult

# Test patterns used across ecosystem
result = FlextResult[dict].ok({'test': True})

# Verify all access patterns work (ecosystem compatibility)
assert result.value == {'test': True}, 'value access broken'
assert result.data == {'test': True}, 'data access broken (ecosystem depends on this)'
assert result.unwrap() == {'test': True}, 'unwrap access broken'
assert result.is_success == True, 'success check broken'

print('✅ API patterns consistent across ecosystem')
"
```

## 🚀 1.0.0 RELEASE PREPARATION GUIDELINES

**CRITICAL: FLEXT-Core v0.9.9 → v1.0.0 Stable Release**

### **Foundation Library 1.0.0 Readiness Assessment**

**Current Status (v0.9.9)**:
- ✅ **79% Test Coverage** - Proven stable across 32+ dependent projects
- ✅ **API Surface Mature** - 20+ stable exports serving entire ecosystem
- ✅ **Zero Breaking Changes** - Railway pattern, DI container, DDD models stable
- ✅ **Type Safety Complete** - Python 3.13 + MyPy strict mode compliant
- ✅ **Quality Gates Perfect** - Zero Ruff issues, complete type coverage

**1.0.0 Release Target**: October 2025 (5-week development cycle)

### **MANDATORY 1.0.0 PREPARATION CHECKLIST**

#### **Phase 1: API Stabilization (Weeks 1-2)**
- [x] **API Surface Audit**: Verified 20+ stable exports
- [x] **Version Update**: v0.9.9 preparation release completed
- [ ] **ABI Finalization**: Lock dependency versions for interface stability
- [ ] **Semantic Versioning**: Finalize breaking change policy
- [ ] **Migration Documentation**: Complete 0.x → 1.0 upgrade guide

#### **Phase 2: Quality Assurance (Weeks 2-3)**
- [ ] **Test Coverage Enhancement**: Target 85% from proven 79% baseline
- [ ] **Ecosystem Integration Testing**: Validate all 32+ dependent projects
- [ ] **Security Audit**: Complete pip-audit and dependency updates
- [ ] **Performance Baselines**: Establish regression test suite

#### **Phase 3: Ecosystem Validation (Weeks 3-4)**
- [ ] **Dependent Project Testing**: flext-api, flext-cli, flext-auth compatibility
- [ ] **Singer Platform Validation**: All taps and targets working
- [ ] **Migration Path Testing**: Upgrade scenarios verified
- [ ] **Documentation Completion**: API reference with examples

#### **Phase 4: Release Preparation (Week 4)**
- [ ] **CI/CD Pipeline**: Automated 1.0.0 release process
- [ ] **CHANGELOG.md**: Complete with breaking changes documentation
- [ ] **Release Artifacts**: Migration tools and compatibility guide
- [ ] **Final Integration Testing**: Cross-ecosystem validation

#### **Phase 5: 1.0.0 Launch (Week 5)**
- [ ] **Tagged Release**: Semantic versioning with stability guarantee
- [ ] **PyPI Publication**: Development Status :: 5 - Production/Stable
- [ ] **Ecosystem Migration**: Support for dependent projects
- [ ] **Community Communication**: Release announcement and support

### **1.0.0 STABILITY GUARANTEES**

**API Compatibility Promise**:
- ✅ **FlextResult[T]** - `.data`/`.value` dual access permanently supported
- ✅ **FlextContainer.get_global()** - Singleton pattern guaranteed stable
- ✅ **FlextModels.Entity/Value/AggregateRoot** - DDD patterns locked
- ✅ **FlextDomainService** - Service base class interface stable
- ✅ **FlextLogger(__name__)** - Logging interface guaranteed

**Breaking Change Policy (1.x series)**:
- **GUARANTEED**: No breaking changes to core APIs in 1.x releases
- **GUARANTEED**: Deprecation cycle minimum 2 minor versions for any changes
- **GUARANTEED**: Migration tools provided for necessary upgrades
- **GUARANTEED**: Backward compatibility maintained through entire 1.x lifecycle

### **POST-1.0.0 DEVELOPMENT ROADMAP**

**1.1.0 - Enhanced Features** (Q4 2025):
- Advanced plugin architecture patterns
- Enhanced performance monitoring integration
- Extended ecosystem validation framework

**1.2.0 - Ecosystem Expansion** (Q1 2026):
- Event sourcing pattern implementations
- Distributed tracing support integration
- Advanced configuration management features

**2.0.0 - Next Generation** (2026):
- Python 3.14+ support with advanced type features
- Breaking changes with comprehensive migration tools
- Advanced architectural pattern evolution

## ECOSYSTEM FOUNDATION IMPACT (CRITICAL AWARENESS)

**FLEXT-CORE DEPENDENCY TREE** - 32+ Projects Depend on This Foundation:

### Direct Dependencies (IMMEDIATE IMPACT)
- **Infrastructure Libraries** (6): flext-db-oracle, flext-ldap, flext-grpc, flext-auth, flext-cli, flext-api
- **Application Services** (5): flext-web, flext-observability, flext-meltano, flext-quality, client-a-oud-mig
- **Data Integration** (8): Oracle OIC/WMS adapters, DBT transformations, ETL pipelines
- **Singer Platform** (15): Taps, Targets, and transformation utilities
- **Runtime Systems** (2): FlexCore Go runtime, FLEXT Service control panel

### Breaking Change Protocol (ZERO TOLERANCE FOR ECOSYSTEM BREAKS)

**MANDATORY STEPS** before ANY API change:

```bash
# 1. ECOSYSTEM IMPACT ANALYSIS (BEFORE any changes)
echo "=== ECOSYSTEM IMPACT ANALYSIS ==="

# Find ALL dependencies across the workspace
echo "Analyzing ecosystem dependencies..."
find ../ -name "*.py" -exec grep -l "from flext_core import\|import flext_core" {} \; | wc -l

# Check specific API usage patterns
grep -r "FlextResult\[.*\]\.data" ../flext-* | wc -l   # Legacy API usage
grep -r "FlextResult\[.*\]\.value" ../flext-* | wc -l  # New API usage
grep -r "FlextContainer\.get_global" ../flext-* | wc -l # Container usage

# 2. BACKWARD COMPATIBILITY VALIDATION
python -c "
# Validate that both old and new APIs continue to work
import sys
sys.path.insert(0, 'src')
from flext_core import FlextResult

result = FlextResult[str].ok('test')

# CRITICAL: Both APIs must work simultaneously
old_api_works = hasattr(result, 'data') and result.data == 'test'
new_api_works = hasattr(result, 'value') and result.value == 'test'

if not (old_api_works and new_api_works):
    print('❌ ECOSYSTEM BREAKING: API compatibility lost')
    exit(1)

print('✅ Backward compatibility maintained')
"

# 3. DEPRECATION PROTOCOL (if breaking changes are necessary)
# - Add deprecation warnings in current version
# - Document migration path in CHANGELOG.md
# - Maintain old API for 2 version cycles minimum
# - Provide automated migration tools where possible
```

### Foundation Quality Metrics (EVIDENCE-BASED TARGETS)

**CURRENT FOUNDATION STATUS** (proven achievable):
- ✅ **79% Test Coverage** - real functional tests, proven stable
- ✅ **Zero MyPy Errors** - strict mode compliance in src/
- ✅ **API Compatibility** - .data/.value dual access working
- ✅ **32+ Projects** - successfully depending on this foundation

**TARGET IMPROVEMENTS** (realistic based on current state):
- 🎯 **85% Test Coverage** - incremental improvement from proven 79%
- 🎯 **Zero PyRight Errors** - secondary type checking compliance
- 🎯 **Complete API Documentation** - all public APIs with examples
- 🎯 **Performance Baselines** - establish performance regression tests

## FOUNDATION VERIFICATION PROTOCOL (ECOSYSTEM PROTECTION)

**CRITICAL FOR FLEXT-CORE**: All changes must be verified against the entire ecosystem impact.

### Pre-Change Verification (MANDATORY)

```bash
# FOUNDATION VERIFICATION PROTOCOL - Run BEFORE any core changes
echo "=== FLEXT-CORE FOUNDATION VERIFICATION ==="

# 1. VERIFY CURRENT API STATE (baseline)
echo "Establishing current API baseline..."
python -c "
import sys
sys.path.insert(0, 'src')

# Test ALL foundation exports
from flext_core import (
    FlextResult, FlextContainer, FlextModels, FlextDomainService,
    FlextLogger, FlextConfig, FlextConstants, FlextTypes, FlextUtilities
)

# Test FlextResult API completeness
result = FlextResult[str].ok('baseline_test')
api_state = {
    'has_data': hasattr(result, 'data'),
    'has_value': hasattr(result, 'value'),
    'has_unwrap': hasattr(result, 'unwrap'),
    'data_value': result.data if hasattr(result, 'data') else None,
    'value_value': result.value if hasattr(result, 'value') else None
}
print('API Baseline:', api_state)
"

# 2. VERIFY ECOSYSTEM COMPATIBILITY (before changes)
echo "Testing ecosystem compatibility baseline..."
for project in ../flext-api ../flext-cli ../flext-auth; do
    if [ -d \"$project\" ]; then
        cd \"$project\"
        python -c \"
import sys
sys.path.insert(0, '../flext-core/src')
from flext_core import FlextResult, FlextContainer
result = FlextResult[str].ok('test')
container = FlextContainer.get_global()
print('✅ $project baseline: OK')
        \" 2>/dev/null || echo \"❌ $project baseline: FAILED\"
        cd - > /dev/null
    fi
done

# 3. TEST COVERAGE BASELINE
echo \"Establishing test coverage baseline...\"
pytest tests/ --cov=src/flext_core --cov-report=term | grep \"TOTAL\" | tail -1
```

### Post-Change Verification (MANDATORY)

```bash
# FOUNDATION POST-CHANGE VERIFICATION - Run AFTER any core changes
echo \"=== POST-CHANGE ECOSYSTEM VALIDATION ===\"

# 1. API COMPATIBILITY CHECK (critical)
python -c \"
import sys
sys.path.insert(0, 'src')
from flext_core import FlextResult

result = FlextResult[str].ok('post_change_test')

# CRITICAL: Ensure no API regressions
assert hasattr(result, 'data'), 'REGRESSION: .data API removed - ECOSYSTEM BREAKING'
assert hasattr(result, 'value'), 'REGRESSION: .value API removed'
assert hasattr(result, 'unwrap'), 'REGRESSION: .unwrap API removed'
assert result.data == result.value == 'post_change_test', 'REGRESSION: API inconsistency'

print('✅ Post-change API compatibility maintained')
\"

# 2. ECOSYSTEM RE-VALIDATION (ensure no breaks)
echo \"Re-validating ecosystem after changes...\"
./validate_ecosystem_compatibility.sh || echo \"❌ ECOSYSTEM BREAK DETECTED\"

# 3. FOUNDATION QUALITY RE-CHECK
make validate || echo \"❌ QUALITY REGRESSION DETECTED\"
```

### Anti-Hallucination Protocol for Core (FOUNDATION TRUTH)

**ABSOLUTE REQUIREMENTS** for core development:

- 🔍 **READ actual __init__.py** before claiming exports exist
- 🔍 **TEST actual imports** before documenting API patterns
- 🔍 **RUN actual code** before claiming functionality works
- 🔍 **VERIFY ecosystem impact** before making any API changes
- 🔍 **MEASURE actual coverage** before claiming test improvements
- 🔍 **VALIDATE dependent projects** before claiming compatibility

```bash
# FOUNDATION TRUTH VERIFICATION
echo \"=== FOUNDATION TRUTH CHECK ===\"

# Read actual exports - NEVER assume
cat src/flext_core/__init__.py | grep \"^from\\|^import\" | head -10

# Test actual imports - NEVER assume
python -c \"
import sys
sys.path.insert(0, 'src')
from flext_core import FlextResult
print('Import successful, methods:', [m for m in dir(FlextResult) if not m.startswith('_')][:10])
\"

# Measure actual metrics - NEVER guess
pytest tests/ --cov=src/flext_core --tb=no -q | tail -3
```

---

## 🔗 MCP SERVER INTEGRATION

### Mandatory MCP Server Usage (FOUNDATION COMPLIANCE)
As defined in [../CLAUDE.md](../CLAUDE.md), all FLEXT development MUST use:
- **serena**: All semantic code operations, symbol analysis, and refactoring
- **sequential-thinking**: Complex problem decomposition and planning
- **context7**: Third-party library documentation and API references
- **github**: Repository operations and pull request management
- **puppeteer**: Web automation and testing interfaces

Foundation library development must demonstrate proper MCP usage patterns for the ecosystem.

---

## FLEXT-CORE DEVELOPMENT SUMMARY

**FOUNDATION AUTHORITY**: flext-core is the bedrock of the FLEXT ecosystem
**ZERO TOLERANCE**: No breaking changes, no quality regressions, no ecosystem impacts
**EVIDENCE-BASED**: All patterns verified against 79% coverage baseline and 32+ dependent projects
**ECOSYSTEM PROTECTION**: Every change validated against entire dependent project tree
**QUALITY LEADERSHIP**: Sets the standard for all ecosystem projects with zero-compromise quality
