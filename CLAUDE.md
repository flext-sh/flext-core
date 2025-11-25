# FLEXT-Core Project Guidelines

**Reference**: See [../CLAUDE.md](../CLAUDE.md) for FLEXT ecosystem standards and general rules.

---

## Project Overview

**FLEXT-Core** is the foundation library for 32+ dependent projects in the FLEXT ecosystem. Every change here has massive impact - ZERO TOLERANCE for breaking changes.

**Version**: 0.9.9 RC → 1.0.0 (January 2025)  
**Coverage**: 38.30% (target: 79%+)  
**Python**: 3.13+ only

---

## Architecture

### Module Categories (Dependency Tiers)

**IMPORTANT**: This is a utility library (not a layered application). Dependencies flow from foundational types upward, not strictly hierarchical.

```
Tier 0 (Pure Foundation - ZERO imports from flext_core):
  ├── constants.py     # FlextConstants - error codes, defaults (0 imports)
  ├── typings.py       # FlextTypes - type aliases (0 imports)
  └── protocols.py     # FlextProtocols - interfaces (0 imports)

Tier 0.1 (Configuration - CONTROLS ALL BEHAVIOR):
  └── config.py        # FlextConfig → constants ✅

Tier 0.5 (Runtime Bridge):
  └── runtime.py       # FlextRuntime → constants, typings ✅

Tier 1 (Core Abstractions - Error Handling):
  ├── exceptions.py    # FlextExceptions → config, constants ✅
  └── result.py        # FlextResult → constants, exceptions ✅

Tier 1.5 (Structured Logging - uses Core):
  └── loggings.py      # FlextLogger → result, runtime, typings ✅

Tier 2 (Domain Foundation):
  ├── models.py        # FlextModels → _models/* (Pydantic base classes)
  ├── utilities.py     # FlextUtilities → result ✅
  └── mixins.py        # FlextMixins (reusable behaviors)

Tier 2.5 (Domain + DI):
  ├── container.py     # FlextContainer → config, constants, models, result, runtime, utilities ✅
  ├── service.py       # FlextService → config, container, exceptions, mixins, models, result ✅
  └── context.py       # FlextContext → constants, container, loggings, models, result ✅

Tier 3 (Application Layer):
  ├── Tier 3.1 (Handlers):
  │   └── handlers.py  # FlextHandlers → constants, exceptions, loggings, mixins, models ✅
  │
  ├── Tier 3.2 (Orchestration):
  │   ├── dispatcher.py # FlextDispatcher → constants, context, handlers, mixins, models, result, utilities ✅
  │   └── registry.py   # FlextRegistry → constants, dispatcher, handlers, mixins, models, result ✅
  │
  └── Tier 3.3 (Cross-Cutting):
      └── decorators.py # FlextDecorators → constants, container, context, exceptions, loggings, result ✅
```

**CRITICAL ARCHITECTURAL RULES**:

1. **FlextConfig MUST be Tier 0.1** (just above constants/types) because it:
   - Reads environment variables and provides runtime overrides
   - Controls FlextConstants default values
   - Sets FlextExceptions failure levels and auto-logging behavior
   - Configures FlextLogger output formats, levels, and destinations
   - Defines FlextRuntime correlation ID patterns and context tracking
   - Modifies ALL other modules' behavior via configuration

2. **Why Config can't be higher**:
   - If Tier 1: Circular import with exceptions.py
   - If Tier 4: Circular import with ALL lower tiers
   - Current position (0.1): ✅ NO circular imports detected

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

## Quality Standards

**Requirements**:
- **Linting**: Ruff ZERO violations ✅
- **Type Checking**: Pyrefly strict ZERO errors (needs validation on all modules)
- **Coverage**: 79%+ (current: 38.30% - needs improvement)
- **Line Length**: 79 characters max
- **API Compatibility**: Both `.data` and `.value` must work ✅
- **Circular Dependencies**: ZERO (verified by import tests) ✅

**Quality Gate**:
```bash
make validate  # Runs: lint + type-check + security + test
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

**Additional Resources**: [../CLAUDE.md](../CLAUDE.md) (workspace), [README.md](README.md) (overview), [~/.claude/commands/flext.md](~/.claude/commands/flext.md) (MCP workflows)
