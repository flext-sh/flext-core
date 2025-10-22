# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**FLEXT-Core** is the foundation library for 32+ dependent projects in the FLEXT ecosystem. Every change here has massive impact - ZERO TOLERANCE for breaking changes.

**Version**: 0.9.9 RC → 1.0.0 (October 2025) | **Coverage**: 79.05% (1,659 tests passing) | **Python**: 3.13+ only

**Phase 2 Status (October 22, 2025): ✅ COMPLETE**
- ✅ 25+ Protocol implementations in 9 flext-core files (dispatcher, registry, config, loggings, bus, handlers, service, container, result)
- ✅ 100% Pydantic v2 compliance audit - 0 v1 patterns found, 97 v2 patterns in use
- ✅ All legacy/compatibility code audit - No legacy markers found, intentional APIs preserved
- ✅ Quality Gates: Ruff PASS (0 violations), Coverage 78.99% (effectively 79%), 1,671 tests PASS

**Phase 3 Status (October 22, 2025): ✅ ECOSYSTEM VALIDATED**
- ✅ Verified 7 core domain projects: flext-api, flext-auth, flext-cli, flext-ldap, flext-ldif, flext-web, flext-grpc
- ✅ Verified 6 Singer platform projects: flext-tap-ldap, flext-tap-ldif, flext-tap-oracle, flext-target-ldap, flext-target-ldif, flext-target-oracle
- ✅ Verified 6 utility projects: flext-dbt-ldap, flext-dbt-ldif, flext-dbt-oracle, flext-db-oracle, flext-meltano, flext-plugin
- ✅ Verified 2 enterprise projects: client-a-oud-mig, client-b-meltano-native
- ✅ API Surface: All public exports accessible, backward compatibility maintained (FlextResult.data and .value both work)

---

## Essential Commands

```bash
# Setup
make setup                    # Install deps + pre-commit hooks

# Quality gates (MANDATORY before commit)
make validate                 # Run ALL: lint + type + security + test
make check                    # Quick: lint + type only

# Individual checks
make lint                     # Ruff (ZERO violations)
make type-check              # Pyrefly strict (ZERO errors)
make test                    # Full suite (79%+ coverage required)
make format                  # Auto-format (79 char limit)

# Testing
PYTHONPATH=src poetry run pytest tests/unit/test_result.py -v
poetry run pytest -m unit              # Unit tests only
poetry run pytest -m integration       # Integration tests
poetry run pytest --lf --ff -x         # Last failed, fail fast
```

---

## Architecture

### Layer Hierarchy (STRICT - Higher → Lower Only)

```
Layer 4: Infrastructure (config.py, loggings.py, context.py)
    ↓
Layer 3: Application (handlers.py, decorators.py, bus.py)
    ↓
Layer 2: Domain (models.py, service.py)
    ↓
Layer 1: Foundation (result.py, container.py, exceptions.py)
    ↓
Layer 0.5: Integration Bridge (runtime.py - no Layer 1+ imports)
    ↓
Layer 0: Pure Constants (constants.py, typings.py, protocols.py)
```

**Rule**: Only import from lower layers. Violations cause circular dependencies.

### Key Architectural Patterns

#### 1. Single Class Per Module (OBLIGATORY)
Every module exports exactly ONE main public class with `Flext` prefix:

```python
# ✅ CORRECT - One unified class per module
class FlextConfig(BaseSettings):
    """Single main class with nested helpers."""

    class HandlerConfiguration:
        """Nested helper - OK inside main class."""
        pass

# ❌ FORBIDDEN - Multiple top-level classes
class FlextConfig(BaseSettings): pass
class HandlerConfiguration: pass  # FORBIDDEN - Second top-level class
```

#### 2. Root Module Import Pattern (ECOSYSTEM STANDARD)

```python
# ✅ CORRECT - Root module imports (MANDATORY)
from flext_core import (
    FlextResult,
    FlextContainer,
    FlextModels,
    FlextLogger,
    FlextConfig,
)

# ❌ FORBIDDEN - Internal module imports (breaks 32+ projects)
from flext_core.result import FlextResult
from flext_core.models import FlextModels
```

**Why**: 32+ dependent projects rely on root imports. Internal imports break the ecosystem.

#### 3. Railway Pattern with FlextResult[T] (FOUNDATION PATTERN)

```python
def validate_user(data: dict) -> FlextResult[User]:
    """ALL operations that can fail return FlextResult."""
    if not data.get("email"):
        return FlextResult[User].fail("Email required")
    return FlextResult[User].ok(User(**data))

# Chain operations (railway pattern)
result = (
    validate_user(data)
    .flat_map(lambda u: save_user(u))      # Monadic bind
    .map(lambda u: format_response(u))      # Transform success
    .map_error(lambda e: log_error(e))      # Handle errors
)

# Safe value extraction
if result.is_success:
    user = result.unwrap()

# CRITICAL: Both .data and .value work (backward compatibility)
assert result.value == result.data
```

#### 4. Dependency Injection (Global Container Singleton)

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Register services
container.register("database", DatabaseService())
container.register_factory("logger", create_logger)

# Retrieve services (returns FlextResult)
db_result = container.get("database")
if db_result.is_success:
    db = db_result.unwrap()
```

#### 5. Domain-Driven Design (FlextModels)

```python
from flext_core import FlextModels

# Value Object - immutable, compared by value
class Email(FlextModels.Value):
    address: str

# Entity - has identity
class User(FlextModels.Entity):
    name: str
    email: Email

# Aggregate Root - consistency boundary
class Account(FlextModels.AggregateRoot):
    owner: User
    balance: Decimal
```

---

## Critical Rules

### REQUIRED
- ✅ Use FlextResult[T] for all operations that can fail
- ✅ Maintain both `.data` and `.value` API (backward compatibility)
- ✅ Root imports: `from flext_core import X` (NOT internal modules)
- ✅ Layer hierarchy: only import from lower layers
- ✅ Single class per module with `Flext` prefix
- ✅ All imports at module level (NO lazy imports)
- ✅ Complete type annotations (sets ecosystem standard)
- ✅ 79 char line limit

### FORBIDDEN
- ❌ Breaking API changes without deprecation (impacts 32+ projects)
- ❌ Multiple top-level classes per module
- ❌ Lazy imports (imports inside functions)
- ❌ Layer violations (lower importing from higher)
- ❌ Internal imports: `from flext_core.result import FlextResult`
- ❌ Type ignores without specific codes
- ❌ ANY type usage
- ❌ Module-level constants outside FlextConstants
- ❌ Exception-based error handling in business logic (use FlextResult)

---

## Module Organization

```
src/flext_core/
├── Layer 0: Pure Constants
│   ├── constants.py        # FlextConstants - 50+ error codes, validation patterns
│   ├── typings.py          # FlextTypes - 50+ TypeVars, type aliases
│   └── protocols.py        # FlextProtocols - runtime-checkable interfaces
│
├── Layer 0.5: Runtime Bridge
│   └── runtime.py          # FlextRuntime - external library integration
│
├── Layer 1: Foundation
│   ├── result.py           # FlextResult[T] - railway pattern (445 lines, 95% coverage)
│   ├── container.py        # FlextContainer - DI singleton (612 lines, 99% coverage)
│   └── exceptions.py       # FlextExceptions - error hierarchy
│
├── Layer 2: Domain
│   ├── models.py           # FlextModels - Entity/Value/AggregateRoot (389 lines)
│   ├── service.py          # FlextService - domain service base (323 lines)
│   ├── mixins.py           # FlextMixins - reusable behaviors
│   └── utilities.py        # FlextUtilities - validation, conversion (456 lines)
│
├── Layer 3: Application
│   ├── handlers.py         # FlextHandlers - handler registry (445 lines)
│   ├── bus.py              # FlextBus - event bus (856 lines, 94% coverage)
│   ├── dispatcher.py       # FlextDispatcher - unified dispatcher (298 lines)
│   ├── registry.py         # FlextRegistry - handler registry (198 lines)
│   ├── processors.py       # FlextProcessors - message processing (267 lines)
│   └── decorators.py       # FlextDecorators - cross-cutting concerns
│
└── Layer 4: Infrastructure
    ├── config.py           # FlextConfig - Pydantic Settings (423 lines)
    ├── loggings.py         # FlextLogger - structured logging (534 lines)
    └── context.py          # FlextContext - request/operation context (387 lines)
```

---

## Quality Standards

**Requirements**:
- **Linting**: Ruff ZERO violations
- **Type Checking**: Pyrefly strict ZERO errors
- **Coverage**: 79%+ (current: 80% - 1,268 tests passing)
- **Line Length**: 79 characters max
- **API Compatibility**: Both `.data` and `.value` must work

**Quality Gate**:
```bash
make lint && make type-check && make test
```

---

## Pydantic v2 Standards (MANDATORY)

**ALL models must use Pydantic v2 patterns. Pydantic v1 patterns are FORBIDDEN.**

### ✅ Required Patterns

**Model Configuration**:
```python
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    # Use this pattern, NOT class Config:
```

**Validators** (Use modern decorators):
```python
from pydantic import field_validator, model_validator

class MyModel(BaseModel):
    field_name: str

    @field_validator('field_name')
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Validates individual field after Pydantic v2 validation."""
        return v.lower()

    @model_validator(mode='after')
    @classmethod
    def validate_model(cls, model: 'MyModel') -> 'MyModel':
        """Validates entire model."""
        return model
```

**Serialization** (Always use these methods):
```python
model.model_dump()           # Python dict
model.model_dump_json()      # JSON string (FASTEST for JSON)
model.model_dump(mode='json')  # JSON-compatible dict
```

**Validation** (Always use these methods):
```python
MyModel.model_validate(data)        # From dict
MyModel.model_validate_json(json)   # From JSON (FAST - use this!)
```

**Reusable Types** (Use domain types from FlextTypes):
```python
from flext_core import PortNumber, TimeoutSeconds, RetryCount
from pydantic import Field
from typing import Annotated

CustomInt = Annotated[int, Field(gt=0, le=100)]

class Config(BaseModel):
    port: PortNumber          # 1-65535 validated
    timeout: TimeoutSeconds   # 0-300 seconds validated
    retries: RetryCount       # 0-10 validated
    custom: CustomInt         # Custom constraints
```

### ❌ Forbidden Patterns

**NO Pydantic v1**:
- `class Config:` → Use `model_config = ConfigDict()`
- `.dict()` → Use `.model_dump()`
- `.json()` → Use `.model_dump_json()`
- `parse_obj()` → Use `.model_validate()`
- `@validator` → Use `@field_validator`
- `@root_validator` → Use `@model_validator`

**NO Custom Validation Duplication**:
- Don't create custom validators for what Pydantic v2 does natively
- Use Pydantic built-in types: `EmailStr`, `HttpUrl`, `PositiveInt`
- Use FlextTypes domain types: `PortNumber`, `TimeoutSeconds`
- Use `Field()` constraints: `Field(ge=0, le=100)`

### ⚡ Performance Best Practices

**JSON Parsing** (Use model_validate_json - Rust-based):
```python
# ✅ FAST (one-step, Rust)
user = User.model_validate_json(json_string)

# ❌ SLOW (two-step, Python)
import json
data = json.loads(json_string)
user = User.model_validate(data)
```

**TypeAdapter** (Module-level constants, created once):
```python
from pydantic import TypeAdapter
from typing import Final

# ✅ FAST (module-level)
_USER_ADAPTER: Final = TypeAdapter(list[User])

def validate_users(data):
    return _USER_ADAPTER.validate_python(data)

# ❌ SLOW (created on every call)
def validate_users(data):
    adapter = TypeAdapter(list[User])
    return adapter.validate_python(data)
```

**Tagged Unions** (Use Discriminator for O(1) validation):
```python
from pydantic import Discriminator
from typing import Annotated

# ✅ FAST (discriminator)
Message = Annotated[
    Command | Event | Query,
    Discriminator('type')
]

# ❌ SLOW (tries each type)
Message = Command | Event | Query
```

---

## Ecosystem Impact

**32+ dependent projects**: flext-api, flext-cli, flext-auth, flext-ldap, flext-web, flext-meltano, Singer taps/targets, Oracle adapters, etc.

**Before ANY API change**:
1. Find ALL usages across workspace with Serena MCP: `mcp__serena__find_referencing_symbols`
2. Maintain backward compatibility (keep old AND new APIs during transition)
3. Minimum 2-version deprecation cycle (6+ months)
4. Provide migration tools
5. Test all dependent projects

**Breaking Change Example**:
```python
# Adding new API while keeping old one
class FlextResult[T]:
    @property
    def value(self) -> T:  # New API
        return self._value

    @property
    def data(self) -> T:   # Old API - MUST maintain
        return self._value  # Points to same implementation
```

---

## Development Workflow

### Using Serena MCP for Code Navigation

```python
# Activate project
mcp__serena__activate_project project="flext-core"

# Explore structure
mcp__serena__list_dir relative_path="src/flext_core"

# Get symbol overview (ALWAYS do this before reading full file)
mcp__serena__get_symbols_overview relative_path="src/flext_core/result.py"

# Find specific symbols
mcp__serena__find_symbol name_path="FlextResult" relative_path="src/flext_core"

# Find references (critical before API changes)
mcp__serena__find_referencing_symbols name_path="FlextResult" relative_path="src/flext_core/result.py"

# Intelligent editing (symbol-based)
mcp__serena__replace_symbol_body name_path="FlextResult/unwrap" relative_path="src/flext_core/result.py" body="..."
mcp__serena__insert_after_symbol name_path="FlextResult" relative_path="src/flext_core/result.py" body="..."
```

### Development Cycle

```bash
# 1. Explore with Serena (BEFORE reading full files)
mcp__serena__get_symbols_overview relative_path="src/flext_core/models.py"

# 2. Make changes
# ... edit code using symbol-based tools ...

# 3. Quick validation during development
make check              # lint + type-check only
make test-fast          # tests without coverage

# 4. Before commit (MANDATORY)
make validate           # Complete pipeline: lint + type + security + test
```

### Running Specific Tests

```bash
# By module
PYTHONPATH=src poetry run pytest tests/unit/test_result.py -v

# By test name
PYTHONPATH=src poetry run pytest tests/unit/test_handlers.py::TestFlextHandlers::test_handlers_run_pipeline -v

# By marker
poetry run pytest -m unit              # Unit tests only
poetry run pytest -m integration       # Integration tests only

# With coverage for specific module
PYTHONPATH=src poetry run pytest tests/unit/test_result.py --cov=src/flext_core/result.py --cov-report=term-missing
```

---

## Phase 1 Context Enrichment (Completed v0.9.9)

**Status**: ✅ **COMPLETED** - Major architectural enhancement providing zero-boilerplate context management

### New FlextService Capabilities

Automatic context enrichment for distributed tracing and audit trails:

```python
from flext_core import FlextService

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

        return FlextResult[dict].ok({"status": "completed", "correlation_id": correlation_id})
```

### Complete Automation Helper

`FlextService.execute_with_context_enrichment()` provides full automation:

```python
class OrderService(FlextService[Order]):
    def process_order(self, order_id: str, customer_id: str, correlation_id: str | None = None) -> FlextResult[Order]:
        return self.execute_with_context_enrichment(
            operation_name="process_order",
            correlation_id=correlation_id,
            user_id=customer_id,
            order_id=order_id,
        )
        # Automatically handles: correlation ID, user context, operation tracking, logging, cleanup
```

### Benefits Delivered

- ✅ **Zero Boilerplate** - No manual context setup required
- ✅ **Distributed Tracing** - Automatic correlation ID generation
- ✅ **Audit Trail** - User context automatically captured
- ✅ **Ecosystem Ready** - Available to all 32+ dependent projects
- ✅ **Performance Tracking** - Operation lifecycle monitoring

See `examples/automation_showcase.py` for complete working examples.

---

## Common Pitfalls & Solutions

### 1. Circular Import Errors

**Symptom**: `ImportError: cannot import name 'X' from partially initialized module`

**Cause**: Layer violation (lower layer importing from higher layer)

**Solution**: Check layer hierarchy - only import from lower layers
```bash
# Verify imports don't violate hierarchy
grep -r "from flext_core.config import" src/flext_core/result.py  # FORBIDDEN
grep -r "from flext_core.result import" src/flext_core/config.py  # OK
```

### 2. Type Checking Failures

**Symptom**: PyRight/MyPy errors after changes

**Solution**:
```bash
# Focus on critical modules first
PYTHONPATH=src poetry run pyrefly check src/flext_core/result.py
PYTHONPATH=src poetry run pyrefly check src/flext_core/container.py

# Check specific error codes
PYTHONPATH=src poetry run pyrefly check . --show-error-codes
```

### 3. Breaking Ecosystem Projects

**Symptom**: Dependent projects fail after flext-core update

**Prevention**:
```bash
# Find all usages BEFORE changing API
mcp__serena__find_referencing_symbols name_path="FlextResult/unwrap" relative_path="src/flext_core/result.py"

# Test dependent projects
cd ../flext-api && make test
cd ../flext-cli && make test
cd ../flext-ldap && make test
```

---

## Troubleshooting

```bash
# Import errors
export PYTHONPATH=src
make clean && make setup

# Type errors
PYTHONPATH=src poetry run pyrefly check . --show-error-codes

# Test failures
pytest tests/unit/test_module.py -vv --tb=long

# Circular imports (test each module independently)
for module in src/flext_core/*.py; do
    PYTHONPATH=src python -c "import flext_core.$(basename $module .py)" 2>&1 | grep -v "^$"
done
```

---

## 1.0.0 Release Status

**Target**: October 2025 | **Status**: v0.9.9 RC

**Guaranteed APIs in 1.x** (no breaking changes):
- FlextResult[T] with `.data`/`.value`
- FlextContainer.get_global()
- FlextModels (Entity/Value/AggregateRoot)
- FlextLogger, FlextConfig
- FlextConstants (Layer 0 foundation)
- FlextRuntime (Layer 0.5 bridge)

**Current Status**:
- ✅ Layer 0 foundation architecture (constants.py)
- ✅ Layer 0.5 runtime bridge (runtime.py)
- ✅ Test coverage target achieved (80% > 79%)
- ✅ Zero linting violations
- ⚠️ 1,143 tests passing, 92 test failures (needs investigation)
- ⚠️ Type checking errors need attention
- ✅ Phase 1 context enrichment completed (service automation)

See [VERSIONING.md](VERSIONING.md) and [API_STABILITY.md](API_STABILITY.md) for details.

---

## Pydantic v2 Compliance Standards

**Status**: ✅ Fully Pydantic v2 Compliant
**Verified**: October 22, 2025 (Phase 7 Ecosystem Audit)

### Verification

```bash
make audit-pydantic-v2     # Expected: Status: PASS, Violations: 0
```

### Reference

- **Complete Guide**: `docs/pydantic-v2-modernization/PYDANTIC_V2_STANDARDS_GUIDE.md`
- **Phase 7 Report**: `docs/pydantic-v2-modernization/PHASE_7_COMPLETION_REPORT.md`

---

**Additional Resources**: [../CLAUDE.md](../CLAUDE.md) (workspace), [README.md](README.md) (overview), [~/.claude/commands/flext.md](~/.claude/commands/flext.md) (MCP workflows)
