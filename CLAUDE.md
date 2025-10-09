# FLEXT-CORE CLAUDE.md

**The Foundation Library Development Guide for FLEXT Ecosystem**
**Version**: 2.4.0 | **Authority**: CORE FOUNDATION | **Updated**: 2025-10-09
**Status**: 79% test coverage (proven stable), v0.9.9 Release Candidate preparing for 1.0.0 stable release · PHASE 2-5 Complete: Zero lazy imports, FlextRuntime.Integration, ecosystem patterns documented

**References**: See [../CLAUDE.md](../CLAUDE.md) for FLEXT ecosystem standards and [README.md](README.md) for project overview.

## 📋 DOCUMENT STRUCTURE & REFERENCES

**Quick Links**:

- **[~/.claude/commands/flext.md](~/.claude/commands/flext.md)**: Optimization command (USE with `/flext` for refactoring)
- **[../CLAUDE.md](../CLAUDE.md)**: Workspace-level domain library standards
- **[README.md](README.md)**: Project overview and quick start

**Document Purpose**:

- **This file**: flext-core specific development patterns, API details, and foundation responsibilities
- **[~/.claude/commands/flext.md](~/.claude/commands/flext.md)**: Practical MCP workflows and refactoring commands
- **[../CLAUDE.md](../CLAUDE.md)**: Domain library architecture and ecosystem standards

**Hierarchy**: This document provides foundation-specific standards. For refactoring workflows and MCP tool usage, use `/flext` command referencing the flext.md guide.

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
- ✅ **Evidence-Based Quality**: 75% coverage proven stable, targeting 85% for 1.0.0

**ECOSYSTEM IMPACT** (32+ Projects Depend on This):

- **Domain Libraries (13)**: flext-cli, flext-ldap, flext-ldif, flext-api, flext-web, flext-db-oracle, flext-meltano, flext-oracle-wms, flext-oracle-oic, flext-auth, flext-observability, flext-grpc
- **Singer Platform (15+)**: Taps, targets, and DBT transformations
- **Enterprise Tools**: algar-oud-mig, gruponos-meltano-native
- **Data Integration**: Oracle OIC/WMS integrations

**NOTE**: All domain libraries build on flext-core patterns. Changes here ripple through entire ecosystem. See [~/.claude/commands/flext.md](~/.claude/commands/flext.md) for complete domain library list.

**QUALITY IMPERATIVES**:

- 🔴 **ZERO tolerance** for API breaking changes without deprecation cycle
- 🟢 **85%+ test coverage** with REAL functional tests (current: 75%)
- 🟢 **Zero errors** in MyPy strict mode, PyRight, and Ruff for ALL src/ code
- 🟢 **Complete type annotations** - this sets the standard for entire ecosystem
- 🟢 **Professional documentation** - all public APIs must be perfectly documented

## FLEXT-CORE DEVELOPMENT WORKFLOW (FOUNDATION QUALITY)

**IMPORTANT**: For refactoring workflows and MCP tool usage, use `/flext` command which references [~/.claude/commands/flext.md](~/.claude/commands/flext.md).

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

### MCP-Based Development Workflow

**RECOMMENDED**: Use serena-flext MCP for code analysis and refactoring:

```bash
# Activate flext-core project in serena
mcp__serena-flext__activate_project project="flext-core"

# List available memories
mcp__serena-flext__list_memories

# Analyze module structure
mcp__serena-flext__list_dir relative_path="src/flext_core" recursive=false

# Get symbol overview
mcp__serena-flext__get_symbols_overview relative_path="src/flext_core/result.py"

# Find symbol references
mcp__serena-flext__find_symbol name_path="FlextResult" relative_path="src/flext_core"
```

**See [~/.claude/commands/flext.md](~/.claude/commands/flext.md) for complete MCP workflow patterns.**

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

**NOTE**: flext-core provides foundation patterns for ALL domain libraries. See [../CLAUDE.md](../CLAUDE.md) for complete domain library architecture and [~/.claude/commands/flext.md](~/.claude/commands/flext.md) for practical usage patterns.

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

### FlextRuntime.Integration Pattern (Layer 0.5 Architecture)

**STATUS**: ✅ PHASE 2 & PHASE 3 Complete - Context integration without circular imports

**CRITICAL ACHIEVEMENT**: Resolved circular import problem between Foundation (Layer 1) and Infrastructure (Layer 4) by introducing Layer 0.5 integration pattern.

**Implementation**:
- **Layer 0.5**: `runtime.py` positioned between constants (Layer 0) and foundation classes (Layer 1+)
- **No Lazy Imports**: All imports at module level - no workarounds or hints
- **Direct structlog**: Integration methods use `structlog.get_logger()` and `structlog.contextvars` directly
- **Single Source of Truth**: structlog.contextvars is the ONLY storage for context data
- **Opt-in Pattern**: Foundation classes explicitly call `FlextRuntime.Integration` methods

**Architectural Layering** (Solves Circular Imports):
```
Layer 4: Infrastructure (context.py, loggings.py) → Uses structlog
    ↑
Layer 1-3: Foundation/Domain/Application (container.py, config.py, models.py)
    ↓ (calls FlextRuntime.Integration methods)
Layer 0.5: runtime.py → Uses structlog directly (NO imports from Layer 1+)
    ↓
Layer 0: constants.py, typings.py (pure Python, no dependencies)
```

**Integrated Foundation Classes** (v0.9.9+):

1. **FlextContainer** - Service resolution tracking:
```python
def _resolve_service(self, name: str) -> FlextResult[object]:
    \"\"\"Resolve service with integration tracking.\"\"\"
    # ... resolution logic ...

    # Integration: Track service resolution
    FlextRuntime.Integration.track_service_resolution(
        name, resolved=True
    )
    return FlextResult[object].ok(service)
```

2. **FlextConfig** - Configuration access tracking:
```python
def get_component_config(self, component: str) -> FlextResult[FlextTypes.Dict]:
    \"\"\"Get component config with access tracking.\"\"\"
    # ... config retrieval ...

    # Integration: Track config access with masking
    sensitive = component in {"database", "security"}
    FlextRuntime.Integration.log_config_access(
        key=f"component.{component}",
        value=config_value if not sensitive else "***MASKED***",
        masked=sensitive
    )
    return FlextResult[FlextTypes.Dict].ok(config_value)
```

3. **FlextModels.AggregateRoot** - Domain event tracking:
```python
def add_domain_event(self, event_name: str, data: FlextTypes.Dict) -> None:
    \"\"\"Add domain event with integration tracking.\"\"\"
    domain_event = FlextModels.DomainEvent(...)
    self.domain_events.append(domain_event)

    # Integration: Track domain event
    FlextRuntime.Integration.track_domain_event(
        event_name=event_name,
        aggregate_id=self.id,
        event_data=data
    )
    # ... existing logging ...
```

**Test Results**: 206/207 tests passing (99.5% success rate, one pre-existing test bug)
- Container tests: 75/75 ✅
- Config tests: 37/38 ✅ (one pre-existing bug)
- Models tests: 19/19 ✅
- Runtime tests: 18/18 ✅
- No circular imports detected ✅

**Quality Validation**:
- ✅ Zero Ruff linting errors
- ✅ MyPy strict mode compliant
- ✅ No lazy imports or workarounds
- ✅ Proper Layer 0.5 architecture maintained

---

## 🏗️ ECOSYSTEM INTEGRATION PATTERNS (v0.9.9+)

**STATUS**: ✅ PHASE 2, 3, 4, and 5 Complete - Comprehensive integration pattern documentation

**CRITICAL ACHIEVEMENT**: Established foundation-wide integration patterns that solve circular imports, eliminate lazy imports (ZERO TOLERANCE), and provide ecosystem-wide context management.

### Layer Hierarchy Rules (ARCHITECTURAL LAW)

**ABSOLUTE RULE**: Higher layers import from lower layers ONLY. Violations cause circular dependencies.

**Layer Architecture** (Strict Enforcement):
```
Layer 5: Unified Facade (base.py)
    ↓ imports from Layer 0-4
Layer 4: Infrastructure (context.py, loggings.py, config.py)
    ↓ imports from Layer 0-3
Layer 3: Application (decorators.py, handlers.py, dispatcher.py, registry.py, bus.py)
    ↓ imports from Layer 0-2
Layer 2: Domain (models.py, service.py, domain_services.py)
    ↓ imports from Layer 0-1
Layer 1: Foundation (result.py, container.py, exceptions.py)
    ↓ imports from Layer 0
Layer 0.5: Integration Bridge (runtime.py)
    ↓ imports structlog directly (NO Layer 1+ imports)
Layer 0: Pure Constants (constants.py, typings.py, protocols.py)
    ↓ no internal dependencies
```

**Layer 0.5 Integration Bridge Pattern**:
- **Purpose**: Enable Layer 1-4 classes to track operations WITHOUT importing Layer 4 (avoids circular imports)
- **Implementation**: `FlextRuntime.Integration` uses `structlog` directly, no Layer 1+ imports
- **Usage**: Foundation classes call `FlextRuntime.Integration.track_*()` methods for observability
- **Benefit**: Zero circular dependencies while maintaining full tracking capabilities

**Example Valid Imports**:
```python
# ✅ CORRECT - Layer 3 imports from Layer 4 (higher → lower)
# decorators.py (Layer 3)
from flext_core.context import FlextContext          # Layer 4
from flext_core.loggings import FlextLogger         # Layer 4

# ✅ CORRECT - Layer 2 imports from Layer 1 (higher → lower)
# models.py (Layer 2)
from flext_core.result import FlextResult           # Layer 1
from flext_core.container import FlextContainer     # Layer 1

# ✅ CORRECT - Layer 1 calls Layer 0.5 (integration bridge)
# container.py (Layer 1)
from flext_core.runtime import FlextRuntime         # Layer 0.5
FlextRuntime.Integration.track_service_resolution(name, resolved=True)
```

**Example FORBIDDEN Imports** (Cause Circular Dependencies):
```python
# ❌ FORBIDDEN - Layer 1 imports from Layer 4 (lower → higher)
# result.py (Layer 1)
from flext_core.loggings import FlextLogger         # CIRCULAR DEPENDENCY!

# ❌ FORBIDDEN - Layer 0.5 imports from Layer 1+ (integration → foundation)
# runtime.py (Layer 0.5)
from flext_core.result import FlextResult           # BREAKS INTEGRATION BRIDGE!

# ❌ FORBIDDEN - Any cross-layer violation
# config.py (Layer 4) importing from decorators.py (Layer 3)
# This is backwards - infrastructure shouldn't import from application
```

### Lazy Import Elimination (ZERO TOLERANCE ENFORCEMENT)

**ABSOLUTE RULE**: ALL imports MUST be at module-level. ZERO lazy imports allowed in flext-core.

**Definition of Lazy Import**:
```python
# ❌ FORBIDDEN - Lazy import (import inside function/method)
def some_function():
    from flext_core.config import FlextConfig  # LAZY IMPORT - FORBIDDEN!
    config = FlextConfig()
    return config

# ❌ FORBIDDEN - Lazy import with noqa workaround
def another_function():
    import structlog  # noqa: PLC0415 - FORBIDDEN WORKAROUND!
    return structlog.get_logger()

# ✅ CORRECT - Module-level import
from flext_core.config import FlextConfig

def some_function():
    config = FlextConfig()  # Use imported class
    return config
```

**Elimination Strategy** (Proven in v0.9.9):

1. **Identify All Lazy Imports**:
   ```bash
   # Use AST parsing to find ALL lazy imports
   python3 -c "
   import ast
   import sys
   from pathlib import Path

   def find_lazy_imports(file_path):
       with open(file_path) as f:
           tree = ast.parse(f.read())

       lazy_imports = []
       for node in ast.walk(tree):
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
               for stmt in node.body:
                   if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                       lazy_imports.append((file_path, node.lineno, node.name))

       return lazy_imports

   # Scan all source files
   for py_file in Path('src/flext_core').rglob('*.py'):
       imports = find_lazy_imports(py_file)
       if imports:
           print(f'Lazy imports in {py_file}: {len(imports)}')
   "
   ```

2. **Categorize by Type**:
   - **External libraries**: Move to module-level (e.g., `import structlog.contextvars`)
   - **Valid layer imports**: Move to module-level if direction is higher → lower
   - **Redundant imports**: Remove entirely (already imported at module level)
   - **Invalid direction imports**: Refactor using Layer 0.5 Integration pattern

3. **Remove and Validate**:
   ```bash
   # After each removal, validate:
   make lint        # Zero PLC0415 violations
   make type-check  # Zero type errors
   make test        # All tests passing
   ```

**Proven Results** (v0.9.9 Achievement):
- **11 lazy imports eliminated** (100% removed)
- **0 remaining lazy imports** (ZERO TOLERANCE achieved)
- **0 circular dependencies** (architectural integrity maintained)
- **1,260/1,260 tests passing** (100% success rate)
- **Zero quality violations** (Ruff, MyPy, PyRight all clean)

### Context Management Integration Pattern (PHASE 4)

**CRITICAL PATTERN**: FlextDecorators integrated with FlextContext for automatic context lifecycle management.

**Implementation Pattern**:

```python
# FlextDecorators with FlextContext integration
from flext_core.context import FlextContext
from flext_core.loggings import FlextLogger

def with_context(**context_vars: object) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to bind context variables for operation duration.

    Automatically manages context lifecycle:
    - Binds context on entry
    - Maintains context during execution
    - Unbinds context on exit (even if exception occurs)
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                # Bind context variables to global logging context
                FlextLogger.bind_global_context(**context_vars)

                return func(*args, **kwargs)

            finally:
                # Always unbind, even on exception
                for key in context_vars:
                    FlextLogger.unbind_global_context(key)

        return wrapper

    return decorator


def track_operation(
    operation_name: str | None = None,
    *,
    track_correlation: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to track operation with context management.

    Features:
    - Automatic correlation ID generation
    - Operation name binding to context
    - Performance tracking via FlextRuntime.Integration (automatic)
    - Proper context cleanup
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            op_name = operation_name or func.__name__

            # Ensure correlation ID if requested
            if track_correlation:
                FlextContext.Utilities.ensure_correlation_id()

            # Bind operation name to context
            FlextLogger.bind_global_context(operation=op_name)

            try:
                # Call the actual function
                # Performance tracking via FlextRuntime.Integration happens automatically
                return func(*args, **kwargs)

            finally:
                # Always unbind operation context
                FlextLogger.unbind_global_context("operation")

        return wrapper

    return decorator
```

**Usage Examples** (Ecosystem Standard):

```python
from flext_core import FlextDecorators

# Example 1: Simple context binding
@FlextDecorators.with_context(service="user_service", version="1.0")
def process_user(user_id: str) -> FlextResult[User]:
    """Context automatically includes service and version."""
    # All logs within this function include service="user_service", version="1.0"
    logger.info("Processing user", user_id=user_id)
    return FlextResult[User].ok(user)

# Example 2: Operation tracking with correlation
@FlextDecorators.track_operation(track_correlation=True)
def handle_request(request: dict) -> FlextResult[dict]:
    """Automatic correlation ID and performance tracking."""
    # Correlation ID automatically generated and propagated
    # Performance metrics tracked automatically via FlextRuntime.Integration
    # Operation name automatically bound to context
    return FlextResult[dict].ok({"status": "success"})

# Example 3: Combined usage
@FlextDecorators.with_correlation()
@FlextDecorators.with_context(component="payment_processor")
@FlextDecorators.track_operation()
def process_payment(payment: dict) -> FlextResult[dict]:
    """Full context management stack."""
    # Correlation ID + component context + operation tracking
    # Performance tracking automatic via FlextRuntime.Integration
    return FlextResult[dict].ok({"processed": True})
```

**Integration Architecture**:
```
┌─────────────────────────────────────────────────┐
│ FlextDecorators (Layer 3 - Application)        │
│ - with_context()                                │
│ - with_correlation()                            │
│ - track_operation()                             │
└────────────────┬────────────────────────────────┘
                 │ imports from ↓
┌────────────────▼────────────────────────────────┐
│ FlextContext (Layer 4 - Infrastructure)        │
│ - Context.Utilities.ensure_correlation_id()    │
│ - Context variables management                  │
└────────────────┬────────────────────────────────┘
                 │ used by ↓
┌────────────────▼────────────────────────────────┐
│ FlextLogger (Layer 4 - Infrastructure)         │
│ - bind_global_context(**kwargs)                │
│ - unbind_global_context(key)                   │
│ - structlog integration (single source)        │
└─────────────────────────────────────────────────┘
```

**Validation Results** (PHASE 4 Complete):
- ✅ **41/41 decorator tests passing** (100%)
- ✅ **No circular imports** (Layer 3 → Layer 4 valid direction)
- ✅ **Context lifecycle managed correctly** (try/finally pattern)
- ✅ **API correctness validated** (proper FlextLogger usage)
- ✅ **Zero quality violations** (Ruff, MyPy clean)

### Integration Quality Gates (MANDATORY ENFORCEMENT)

**CRITICAL**: These quality gates MUST pass before ANY integration work is considered complete.

```bash
# ========================================
# INTEGRATION QUALITY GATE PROTOCOL
# ========================================

echo "=== PHASE 1: LAZY IMPORT DETECTION (ZERO TOLERANCE) ==="

# 1.1. Detect lazy imports with AST parsing
python3 -c "
import ast
from pathlib import Path

def find_lazy_imports(file_path):
    with open(file_path) as f:
        tree = ast.parse(f.read())

    lazy = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in node.body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    lazy.append((node.name, node.lineno))
    return lazy

total_lazy = 0
for py_file in Path('src/flext_core').rglob('*.py'):
    lazy = find_lazy_imports(py_file)
    if lazy:
        print(f'❌ {py_file}: {len(lazy)} lazy imports')
        total_lazy += len(lazy)

if total_lazy == 0:
    print('✅ ZERO LAZY IMPORTS (PERFECT)')
else:
    print(f'❌ FAILED: {total_lazy} lazy imports remaining')
    exit(1)
"

# 1.2. Detect PLC0415 violations (Ruff check)
echo "Checking for import-outside-toplevel violations..."
env PYTHONPATH=src poetry run ruff check src/flext_core/ --select PLC0415 --output-format=concise || exit 1

echo "=== PHASE 2: CIRCULAR IMPORT DETECTION ==="

# 2.1. Validate all modules import successfully
for module in src/flext_core/*.py; do
    module_name=$(basename "$module" .py)
    if [ "$module_name" != "__init__" ]; then
        echo "Testing import: flext_core.$module_name"
        env PYTHONPATH=src python -c "import flext_core.$module_name" || {
            echo "❌ CIRCULAR DEPENDENCY in $module_name"
            exit 1
        }
    fi
done

echo "✅ ALL MODULES IMPORT SUCCESSFULLY"

echo "=== PHASE 3: LAYER HIERARCHY VALIDATION ==="

# 3.1. Verify Layer 0.5 doesn't import from Layer 1+
echo "Validating runtime.py (Layer 0.5) imports..."
grep -E "^from flext_core\.(result|container|models|config|context|loggings)" src/flext_core/runtime.py && {
    echo "❌ LAYER VIOLATION: runtime.py importing from Layer 1+"
    exit 1
} || echo "✅ Layer 0.5 architecture maintained"

# 3.2. Verify no backward imports (lower → higher layer)
echo "Checking for backward layer imports..."
# Add specific checks based on your layer rules

echo "=== PHASE 4: CODE QUALITY VALIDATION ==="

# 4.1. Ruff linting (ZERO violations)
echo "Running Ruff linting..."
env PYTHONPATH=src poetry run ruff check src/flext_core/ --quiet || {
    echo "❌ RUFF VIOLATIONS DETECTED"
    exit 1
}

# 4.2. MyPy strict mode (ZERO errors)
echo "Running MyPy strict mode..."
env PYTHONPATH=src poetry run mypy src/flext_core/ --strict --no-error-summary || {
    echo "❌ TYPE ERRORS DETECTED"
    exit 1
}

echo "=== PHASE 5: TEST VALIDATION ==="

# 5.1. Full test suite (100% success required)
echo "Running full test suite..."
env PYTHONPATH=src poetry run pytest tests/ -q --tb=no || {
    echo "❌ TEST FAILURES DETECTED"
    exit 1
}

# 5.2. Specific integration tests
echo "Running integration tests..."
env PYTHONPATH=src poetry run pytest tests/unit/test_decorators.py -v --tb=short || exit 1
env PYTHONPATH=src poetry run pytest tests/unit/test_context.py -v --tb=short || exit 1
env PYTHONPATH=src poetry run pytest tests/unit/test_container.py -v --tb=short || exit 1
env PYTHONPATH=src poetry run pytest tests/unit/test_config.py -v --tb=short || exit 1
env PYTHONPATH=src poetry run pytest tests/unit/test_models.py -v --tb=short || exit 1

echo "=== PHASE 6: API COMPATIBILITY VALIDATION ==="

# 6.1. Verify FlextResult API compatibility
env PYTHONPATH=src python -c "
from flext_core import FlextResult
result = FlextResult[str].ok('test')
assert hasattr(result, 'data'), 'Legacy .data API missing'
assert hasattr(result, 'value'), 'New .value API missing'
assert result.data == result.value == 'test', 'API inconsistency'
print('✅ FlextResult API compatibility maintained')
"

# 6.2. Verify Container API
env PYTHONPATH=src python -c "
from flext_core import FlextContainer
container = FlextContainer.get_global()
print('✅ Container global access working')
"

echo "=== INTEGRATION QUALITY GATES: ALL PASSED ✅ ==="
```

**Gate Failure Protocol**:
- ❌ **Lazy imports detected**: STOP - remove ALL lazy imports before proceeding
- ❌ **Circular imports detected**: STOP - refactor using Layer 0.5 Integration pattern
- ❌ **Layer violations detected**: STOP - fix import directions
- ❌ **Quality violations**: STOP - fix Ruff/MyPy errors
- ❌ **Test failures**: STOP - fix failing tests
- ❌ **API breakage**: STOP - restore backward compatibility

**Success Criteria** (PHASE 2, 3, 4, 5 Complete):
- ✅ **0 lazy imports** (ZERO TOLERANCE achieved)
- ✅ **0 circular dependencies** (architectural integrity)
- ✅ **0 layer violations** (proper hierarchy maintained)
- ✅ **0 quality violations** (Ruff, MyPy clean)
- ✅ **100% test success** (1,260/1,260 passing)
- ✅ **100% API compatibility** (backward compatibility maintained)

---

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

#### Returns Library Integration (v0.9.9+)

**STATUS**: ✅ Phase 1 Complete - Internal backend integration with full ABI compatibility

**Implementation Details**:
- **Backend**: `dry-python/returns` library (v0.26.0+) integrated as internal storage
- **API Compatibility**: 100% backward compatible - both `.data` and `.value` work simultaneously
- **Internal Storage**: `returns.Result[T_co, str]` powers all FlextResult operations
- **Delegation**: `.map()` internally delegates to `returns.Result.map()` for correctness
- **Validation**: All 254 FlextResult tests passing with returns backend
- **Quality**: Zero ruff linting errors, MyPy strict mode compliant

**Why Returns Library**:
1. **Correctness**: Battle-tested monadic operations from functional programming community
2. **Type Safety**: Proper covariant generic types (`Result[T_co, E_co]`)
3. **Railway Pattern**: Native support for Success/Failure composition
4. **Zero Breaking Changes**: Internal implementation detail, external API unchanged

**Developer Impact**: NONE - FlextResult API remains identical, ecosystem compatibility maintained.

```python
from flext_core import FlextResult

# VERIFIED API - FlextResult has BOTH .data and .value for compatibility
# INTERNAL: Now powered by returns.Result backend for correctness
def foundation_operation(data: dict) -> FlextResult[ProcessedData]:
    """Foundation library operation demonstrating ecosystem-wide patterns."""
    if not data:
        return FlextResult[ProcessedData].fail("Data required", error_code="VALIDATION_ERROR")

    # Railway-oriented composition (ECOSYSTEM STANDARD)
    # INTERNAL: map() delegates to returns.Result.map() for correctness
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

**CRITICAL**: flext-core **init**.py defines the API for entire ecosystem. Changes here impact all dependent projects.

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

**Key Documentation**:

- 📋 [VERSIONING.md](VERSIONING.md) - Semantic versioning strategy and release process
- 📋 [API_STABILITY.md](API_STABILITY.md) - Comprehensive API stability guarantees
- 📋 [README.md](README.md) - Updated with 1.0.0 roadmap and timeline

### **Foundation Library 1.0.0 Readiness Assessment**

**Current Status (v0.9.9)**:

- ✅ **75% Test Coverage** - Proven stable across 32+ dependent projects
- ✅ **API Surface Mature** - 20+ stable exports serving entire ecosystem
- ✅ **Zero Breaking Changes** - Railway pattern, DI container, DDD models stable
- ✅ **Type Safety Complete** - Python 3.13 + MyPy strict mode compliant
- ✅ **Quality Gates Perfect** - Zero Ruff issues, complete type coverage

**1.0.0 Release Target**: October 2025 (5-week development cycle)

### **MANDATORY 1.0.0 PREPARATION CHECKLIST**

#### **Phase 1: API Stabilization (Weeks 1-2)** ✅ COMPLETED

- [x] **API Surface Audit**: Verified 20+ stable exports
- [x] **Version Update**: v0.9.9 preparation release completed
- [x] **ABI Finalization**: Dependency versions locked in pyproject.toml (see VERSIONING.md)
- [x] **Semantic Versioning**: Comprehensive strategy documented (VERSIONING.md)
- [x] **API Stability Guarantees**: Complete documentation (API_STABILITY.md)
- [x] **README.md Roadmap**: 1.0.0 release timeline and stability guarantees added
- [ ] **Migration Documentation**: Create complete 0.x → 1.0 upgrade guide

#### **Phase 2: Quality Assurance (Weeks 2-3)**

- [ ] **Test Coverage Enhancement**: Target 85% from proven 75% baseline
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

> **Complete details in [API_STABILITY.md](API_STABILITY.md) and [VERSIONING.md](VERSIONING.md)**

**API Compatibility Promise**:

- ✅ **FlextResult[T]** - `.data`/`.value` dual access permanently supported (Level 1: 100% stable)
- ✅ **FlextContainer.get_global()** - Singleton pattern guaranteed stable (Level 1: 100% stable)
- ✅ **FlextModels.Entity/Value/AggregateRoot** - DDD patterns locked (Level 1: 100% stable)
- ✅ **FlextService** - Service base class interface stable (Level 1: 100% stable)
- ✅ **FlextLogger(**name**)** - Logging interface guaranteed (Level 1: 100% stable)
- ✅ **HTTP Primitives** - FlextConstants.Http, HttpRequest/HttpResponse models (Level 1: 100% stable, new in 0.9.9)

**Breaking Change Policy (1.x series)**:

- **GUARANTEED**: No breaking changes to core APIs in 1.x releases
- **GUARANTEED**: Deprecation cycle minimum 2 minor versions (6+ months notice)
- **GUARANTEED**: Migration tools and automated utilities provided
- **GUARANTEED**: Backward compatibility maintained through entire 1.x lifecycle
- **GUARANTEED**: Security patches within 48 hours for stability issues

**Dependency Version Locks** (ABI Stability):

- pydantic: `>=2.11.7,<3.0.0` (Pydantic 2.x API stable)
- pydantic-settings: `>=2.10.1,<3.0.0` (aligned with pydantic)
- pyyaml: `>=6.0.2,<7.0.0` (YAML 6.x stable)
- structlog: `>=25.4.0,<26.0.0` (CalVer YY.MINOR)
- typing-extensions: `>=4.12.0,<5.0.0` (type system stability)
- colorlog: `>=6.9.0,<7.0.0` (logging stability)

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
- **Application Services** (5): flext-web, flext-observability, flext-meltano, flext-quality, algar-oud-mig
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

- ✅ **75% Test Coverage** - real functional tests, proven stable
- ✅ **Zero MyPy Errors** - strict mode compliance in src/
- ✅ **API Compatibility** - .data/.value dual access working
- ✅ **32+ Projects** - successfully depending on this foundation

**TARGET IMPROVEMENTS** (realistic based on current state):

- 🎯 **85% Test Coverage** - incremental improvement from proven 75%
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

- 🔍 **READ actual **init**.py** before claiming exports exist
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
