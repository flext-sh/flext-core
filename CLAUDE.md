# FLEXT-Core Project Guidelines

**Reference**: See [../CLAUDE.md](../CLAUDE.md) for FLEXT ecosystem standards and general rules.

---

## Project Overview

**FLEXT-Core** is the foundation library for 32+ dependent projects in the FLEXT ecosystem. Every change here has massive impact - ZERO TOLERANCE for breaking changes.

**Version**: 0.10.0 (December 2025)
**Coverage**: 81.41% (above 73% minimum)
**Python**: 3.13+ only
**Tests**: 2820 tests passing

### Current Status (December 2025)

**Centralized Type System** ✅ **COMPLETED**:
- ✅ **Status**: Migration completed across all `src/` modules
- ✅ **Pattern**: All complex types use `t.Types.*` aliases from `flext_core.typings`
- ✅ **Import**: `from flext_core.typings import t, T, U` (NOT `from flext_core import typings as t`)
- ✅ **TypeVars**: All generic TypeVars imported from `flext_core.typings` (T, U, T_co, T_contra, E, R, P, etc.)
- ✅ **Coverage**: 66 Python files in `src/`, 46+ modules using centralized types
- ✅ **Quality**: Zero Ruff/MyPy errors across entire codebase
- ✅ **Modules Updated**: All modules in `src/flext_core/` and `src/flext_tests/`

**Centralized Constants** ✅ **COMPLETED**:
- ✅ **Status**: `FlextConstants` fully organized with 20+ namespaces
- ✅ **Pattern**: All constants use `c.Namespace.CONSTANT` pattern
- ✅ **Coverage**: All modules using centralized constants
- ✅ **Quality**: Zero duplication, fully typed with `Final`

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

#### 1. Protocol-Based Architecture (SOLID Principles - EXTENSIVELY APPLIED)

**CRITICAL**: Protocols are used EXTENSIVELY throughout `src/` to avoid circular imports, Pydantic forward reference issues, and to follow SOLID principles.

**Protocol Organization** (hierarchical namespaces + ROOT ALIASES):
```python
from flext_core.protocols import p  # FlextProtocols

# =====================================================
# ROOT-LEVEL ALIASES (PREFERRED - CONCISE)
# =====================================================
# Foundation
p.Result[T]              # → p.Foundation.Result[T]
p.ResultLike[T_co]       # → p.Foundation.ResultLike[T_co]
p.Model                  # → p.Foundation.Model
p.HasModelDump           # → p.Foundation.HasModelDump

# Configuration
p.Config                 # → p.Configuration.Config
p.Configurable           # → p.Configuration.Configurable

# Context
p.Ctx                    # → p.Context.Ctx

# Container
p.DI                     # → p.Container.DI

# Domain
p.Service[T]             # → p.Domain.Service[T]
p.Repository[T]          # → p.Domain.Repository[T]
p.HasInvariants          # → p.Domain.Validation.HasInvariants

# Application
p.Handler                # → p.Application.Handler
p.CommandBus             # → p.Application.CommandBus
p.Processor              # → p.Application.Processor
p.Middleware             # → p.Application.Middleware

# Infrastructure
p.Log                    # → p.Infrastructure.Logger.Log
p.StructlogLogger        # → p.Infrastructure.Logger.StructlogLogger
p.Connection             # → p.Infrastructure.Connection
p.Metadata               # → p.Infrastructure.Metadata
p.MetadataProtocol       # → p.Infrastructure.Metadata (backward compat)

# =====================================================
# FULL HIERARCHICAL ACCESS (STILL SUPPORTED)
# =====================================================
# All nested namespaces remain available for backward compatibility:
# p.Foundation.Result[T], p.Domain.Service[T], etc.
```

**Usage Pattern** (MANDATORY for interfaces):
```python
# ✅ PREFERRED - Use root-level aliases (concise)
def execute_service(service: p.Service[str]) -> p.Result[str]:
    """Accept any service implementation via protocol."""
    return service.execute()

# ✅ ALSO CORRECT - Full hierarchical path (backward compat)
def execute_service(service: p.Domain.Service[str]) -> p.Foundation.Result[str]:
    """Accept any service implementation via protocol."""
    return service.execute()

# ✅ CORRECT - Use protocols in return types for abstractions
def get_container() -> p.DI:
    """Return container via protocol interface."""
    container = FlextContainer()
    return cast("p.DI", container)

# ❌ FORBIDDEN - Direct class references in interface signatures
def execute_service(service: FlextService[str]) -> FlextResult[str]:  # FORBIDDEN
    pass
```

**Protocol Benefits**:
- ✅ Eliminates circular import issues
- ✅ Avoids Pydantic forward reference problems
- ✅ Follows Dependency Inversion Principle (SOLID)
- ✅ Enables structural typing (duck typing)
- ✅ Interfaces well-defined and testable

**NO TYPE_CHECKING or Lazy Imports**:
- ❌ **FORBIDDEN**: `TYPE_CHECKING` blocks for protocol imports
- ❌ **FORBIDDEN**: Lazy imports (imports inside functions)
- ✅ **REQUIRED**: All imports at top of file, use protocols directly

#### 2. Single Class Per Module (OBLIGATORY)
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

#### 3. Root Module Import Pattern (ECOSYSTEM STANDARD)

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

#### 4. Centralized Type System (FlextTypes) ✅ **COMPLETED**

**CRITICAL**: All complex types MUST use centralized type aliases from `t.Types` namespace.

**Import Pattern** (MANDATORY):
```python
# ✅ CORRECT - Direct import from typings module
from flext_core.typings import t, T, U, T_co, T_contra, E, R, P

# ❌ FORBIDDEN - Module import then access
from flext_core import typings as t  # MyPy won't resolve nested types correctly

# ❌ FORBIDDEN - Local TypeVar definitions
TResult = TypeVar("TResult")  # FORBIDDEN - Use T from flext_core.typings
TValue = TypeVar("TValue")     # FORBIDDEN - Use T or U from flext_core.typings
```

**Type Alias Usage** (MANDATORY):
```python
from flext_core.typings import t, T, U

# =====================================================
# ROOT-LEVEL ALIASES (PREFERRED - CONCISE)
# =====================================================
# t.PortNumber          → t.Validation.PortNumber
# t.StringDict          → t.Types.StringDict
# t.ConfigurationDict   → t.Types.ConfigurationDict
# t.GeneralValueType    → t.Types.GeneralValueType
# t.ScalarValue         → t.Types.ScalarValue

# ✅ PREFERRED - Use root-level aliases
def process_config(config: t.ConfigurationDict) -> t.StringDict:
    """Use root-level type aliases."""
    pass

# ✅ ALSO CORRECT - Full hierarchical path
def process_config(config: t.Types.ConfigurationDict) -> t.Types.StringDict:
    """Use full path type aliases."""
    pass

# ✅ CORRECT - Use centralized TypeVars
def process_value[T](value: T) -> r[T]:
    """Use T from flext_core.typings."""
    return r[T].ok(value)

# ✅ CORRECT - Multiple TypeVars
def map_dict[T, U](data: dict[T, U]) -> dict[U, T]:
    """Use T, U from flext_core.typings."""
    return {v: k for k, v in data.items()}

# ✅ CORRECT - Enum instance mappings
members_dict: t.StringStrEnumInstanceDict = getattr(
    enum_cls, "__members__", {}
)

# ✅ CORRECT - Exception type mappings
error_classes: t.StringFlextExceptionTypeDict = {
    "ValidationError": e.ValidationError,
    # ...
}

# ❌ FORBIDDEN - Direct type definitions
def process_config(config: dict[str, t.GeneralValueType]) -> dict[str, str]:  # FORBIDDEN
    pass

# ❌ FORBIDDEN - Local TypeVar definitions
TResult = TypeVar("TResult")  # FORBIDDEN
def process[TResult](value: TResult) -> r[TResult]:  # Use T instead
    pass
```

**Available Type Aliases** (in `t.Types` namespace):
- **Configuration**: `ConfigurationDict`, `ConfigurationMapping`, `StringConfigurationDictDict`
- **String mappings**: `StringDict`, `StringIntDict`, `StringFloatDict`, `StringBoolDict`, `StringNumericDict`
- **Enum types**: `StringStrEnumTypeDict`, `StringStrEnumInstanceDict`
- **Exception types**: `StringFlextExceptionTypeDict`
- **Handler types**: `StringHandlerCallableListDict`, `HandlerTypeDict`, `HandlerCallableDict`
- **Settings types**: `StringBaseSettingsTypeDict`
- **Path types**: `StringPathDict`
- **Generic types**: `StringTypeDict`, `StringListDict`, `StringSequenceDict`
- **Metadata types**: `MetadataAttributeDict`
- **And many more...** (see `typings.py` for complete list)

**Available TypeVars** (from `flext_core.typings`):
- **Core generics**: `T`, `T_co` (covariant), `T_contra` (contravariant)
- **Utilities**: `U`, `R`, `E`
- **ParamSpec**: `P` (for decorators)
- **Handlers**: `MessageT_contra`, `ResultT`
- **Config/Models**: `T_Model`, `T_Namespace`, `T_Settings`

**Status**: ✅ **COMPLETED** (January 2025)
- ✅ All modules in `src/flext_core/` using centralized types
- ✅ All modules in `src/flext_tests/` using centralized types
- ✅ All TypeVars imported from `flext_core.typings`
- ✅ Zero local TypeVar definitions
- ✅ Zero Ruff/MyPy errors

**Rules**:
- ✅ All `dict[str, ...]` patterns MUST use `t.Types.*` aliases
- ✅ All `Mapping[str, ...]` patterns SHOULD use `t.Types.*` aliases when available
- ✅ Generic types like `dict[str, T]` where `T` is a type parameter are OK (no replacement needed)
- ✅ All TypeVars MUST be imported from `flext_core.typings` (no local definitions)
- ✅ Zero tolerance for duplicate type definitions

#### 5. Short Aliases Pattern (ECOSYSTEM STANDARD)

**CRITICAL**: Short aliases are the standard pattern for FLEXT modules, providing concise syntax for frequently used types.

**Import Pattern** (MANDATORY):
```python
# ✅ CORRECT - Import short aliases from their modules
from flext_core.result import r       # FlextResult alias
from flext_core.typings import t      # FlextTypes alias
from flext_core.constants import c    # FlextConstants alias
from flext_core.models import m       # FlextModels alias
from flext_core.protocols import p    # FlextProtocols alias
from flext_core.utilities import u    # FlextUtilities alias
from flext_core.exceptions import e   # FlextExceptions alias
from flext_core.decorators import d   # FlextDecorators alias
from flext_core.context import x      # FlextContext alias

# =====================================================
# ALL MODULES NOW HAVE ROOT-LEVEL ALIASES
# =====================================================
# protocols.py: p.Result, p.Service, p.Handler, p.Middleware, etc.
# models.py: m.Command, m.Query, m.Value, m.Entity, etc.
# constants.py: c.VALIDATION_ERROR, c.HandlerType, c.CommonStatus, etc.
# typings.py: t.StringDict, t.PortNumber, t.GeneralValueType, etc.

# ❌ FORBIDDEN - Full class names in type hints
def process(result: FlextResult[str]) -> FlextResult[bool]:  # FORBIDDEN
    pass

# ✅ CORRECT - Short aliases in type hints
def process(result: r[str]) -> r[bool]:
    return r[bool].ok(True) if result.is_success else r[bool].fail("Error")
```

**Lint Configuration**: The `PYI042` rule is ignored in `ruff-shared.toml` to allow short aliases without type annotations.

#### 6. Railway Pattern with FlextResult[T] (FOUNDATION PATTERN)

```python
from flext_core.result import r
from flext_core.typings import t

def validate_user(data: dict) -> r[User]:
    """ALL operations that can fail return FlextResult via r[T] alias."""
    if not data.get("email"):
        return r[User].fail("Email required")
    return r[User].ok(User(**data))

# Chain operations (railway pattern)
result = (
    validate_user(data)
    .flat_map(lambda u: save_user(u))      # Monadic bind
    .map(lambda u: format_response(u))      # Transform success
    .map_error(lambda e: log_error(e))      # Handle errors
)

# Safe value extraction
if result.is_success:
    user = result.value  # Use .value property directly (unwrap() is deprecated)

# CRITICAL: Both .data and .value work (backward compatibility)
assert result.value == result.data
```

**FlextResult Creation Pattern** (MANDATORY):
```python
# ✅ CORRECT - Use r[T].ok() and r[T].fail() directly
def validate_input(data: str) -> r[bool]:
    if not data:
        return r[bool].fail("Input cannot be empty")
    return r[bool].ok(True)

def process_model(model: t.GeneralValueType) -> r[t.GeneralValueType]:
    # Type parameter matches return type
    return r[t.GeneralValueType].ok(model)

# ❌ FORBIDDEN - FlextRuntime.result_* (causes pyright errors)
# These methods return protocol types, not concrete r[T]
def validate_input(data: str) -> r[bool]:
    return FlextRuntime.result_fail("Error")  # FORBIDDEN - pyright error

# ✅ CORRECT - Protocol return types for container/service methods
def get_service() -> p.Foundation.Result[t.GeneralValueType]:
    """When returning from container.get(), use protocol type."""
    return container.get("service_name")
```

**Unified Result Type System** ✅ **COMPLETED** (January 2025):

The Result type architecture has been unified across all modules to ensure seamless interoperability:

| Component | Type | Location | Usage |
|-----------|------|----------|-------|
| `FlextResult[T]` | Concrete class | `result.py` | Main implementation |
| `r = FlextResult` | Short alias | `result.py` | `r[T].ok()`, `r[T].fail()` |
| `p.Foundation.Result[T]` | Protocol | `protocols.py` | Interface definitions |
| `RuntimeResult[T]` | Tier 0.5 class | `runtime.py` | Bootstrap operations |

**Key Rules**:
- ✅ **Return types**: Use `r[T]` in methods that return FlextResult
- ✅ **Parameter types**: Use `r[T]` for input parameters expecting FlextResult
- ✅ **Interface types**: Use `p.Foundation.Result[T]` only in protocol definitions
- ✅ **Tier 0.5**: Only `runtime.py` uses `RuntimeResult` (bootstrap before `result.py` loads)
- ✅ **Backward compatibility**: Both `.data` and `.value` properties work identically

**Migration Pattern** (from protocol to concrete):
```python
# BEFORE (causes mypy errors)
def process() -> p.Foundation.Result[str]:
    return r[str].ok("value")  # Error: Result[Never] vs FlextResult[str]

# AFTER (correct)
def process() -> r[str]:
    return r[str].ok("value")  # OK: Types match
```

#### 7. Dependency Injection (Layered Architecture - Clear Architecture)

**CRITICAL**: FLEXT-Core implements a layered dependency injection pattern following Clear Architecture principles. This ensures services (`FlextConfig`, `FlextLogger`, `FlextContext`, etc.) are easily accessible via DI for downstream projects, reducing complexity and promoting consistent patterns.

**Architecture Layers**:
- **L0.5 (Runtime Bridge)**: `FlextRuntime` is the single surface to access providers/containers/wiring (`Provide`, `inject`) and configuration helpers. Expand capabilities only by exposing new helpers here.
- **L1 (DI Integration)**: `FlextRuntime.DependencyIntegration` owns declarative containers, typed providers (Singleton/Factory/Resource), and `providers.Configuration` for defaults/overrides
- **L1.5 (Service Runtime Bootstrap)**: `FlextRuntime.create_service_runtime` (inherited by `FlextService`) materializes config/context/container in one call with optional overrides, registrations, and wiring
- **L2 (Container)**: `FlextContainer` uses bridge providers to register services, factories, and resources with generics, cloning them for scopes without manual dictionaries
- **L3 (Handlers/Dispatcher)**: Handlers are wired via `wire_modules`, and `@inject`/`Provide` decorators are re-exported by the runtime. Do NOT import `dependency-injector` directly in L3.

**Implementation Rules**:
- Prefer `providers.Resource` for external clients (DB/HTTP/queues), guaranteeing teardown/close and removing manual lifecycle boilerplate
- Use `providers.Configuration` to synchronize defaults/overrides validated by `FlextConfig`; avoid manual merges and preserve override precedence
- Prefer the parameterized `DependencyIntegration.create_container` helper when instantiating DI containers directly
- Service classes should reuse runtime automation by overriding `_runtime_bootstrap_options` on `FlextService` to feed parameters into `create_service_runtime`
- Registrations must use typed providers (Generic[T]) to keep type-safety; avoid extra casts
- Any new DI surface must be exposed via the bridge/runtime or existing facades, never by direct `dependency-injector` imports in upper layers

**Key Services Accessible via DI** (Auto-registered):
- `FlextConfig`: Available as `"config"` - Configuration management (via `container.get("config")` or `container.config` property)
- `FlextLogger`: Available as `"logger"` (factory) - Structured logging (via `container.get("logger")` or `FlextRuntime.structlog()`)
- `FlextContext`: Available as `"context"` - Request/operation context (via `container.get("context")` or `container.context` property)
- `FlextContainer`: Dependency injection container (global singleton via `FlextContainer()` or scoped via `container.scoped()`)
- All domain services via `FlextService` with `_runtime_bootstrap_options` override

**Core Services Auto-Registration**:
When a `FlextContainer` is created, core services are automatically registered with standard names:
- `"config"` → `FlextConfig` instance (singleton)
- `"logger"` → `FlextLogger` factory (creates module logger)
- `"context"` → `FlextContext` instance (singleton)

This enables easy dependency injection in downstream projects without manual registration:

```python
from flext_core import FlextContainer, Provide, inject

container = FlextContainer()

# Core services are automatically available
@inject
def my_handler(
    config=Provide["config"],
    logger=Provide["logger"],
    context=Provide["context"]
):
    # All services injected automatically
    logger.info("Handler executed", app_name=config.app_name)
    return context.get("correlation_id")
```

**Usage in Downstream Projects**:
- Inject services/resources via facades (`FlextContainer`, `wire_modules`, re-exported decorators)
- Do NOT create alternative containers or access `dependency-injector` directly
- Honor inherited configuration contracts: defaults via `providers.Configuration` and user overrides applied by `configure(...)`
- If you need new capabilities (e.g., a new provider), add it to the flext-core bridge and re-export it before using it in the dependent project

See [`docs/guides/dependency_injector_prompt.md`](../docs/guides/dependency_injector_prompt.md) for complete pattern checklist.

#### 8. Dependency Injection (Global Container Singleton - Legacy Pattern)

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Register services
container.register("database", DatabaseService())
container.register_factory("logger", create_logger)

# Retrieve services (returns FlextResult)
db_result = container.get("database")
if db_result.is_success:
    db = db_result.value  # Use .value property directly (unwrap() is deprecated)
```

#### 8. Domain-Driven Design (FlextModels)

```python
from flext_core import FlextModels
from flext_core.models import m  # Short alias

# =====================================================
# ROOT-LEVEL ALIASES (PREFERRED - CONCISE)
# =====================================================
# m.Command     → m.Cqrs.Command
# m.Query       → m.Cqrs.Query
# m.Pagination  → m.Cqrs.Pagination
# m.Value       → m.Entity.Value
# m.Entity      → m.Entity.Entity
# m.AggregateRoot → m.Entity.AggregateRoot
# m.Metadata    → m.Base.Metadata
# m.ProcessingRequest → m.Config.ProcessingRequest
# m.RetryConfiguration → m.Config.RetryConfiguration

# Value Object - immutable, compared by value
class Email(m.Value):  # ✅ PREFERRED: root alias
    address: str

class EmailOld(FlextModels.Entity.Value):  # ✅ ALSO CORRECT: full path
    address: str

# Entity - has identity
class User(m.Entity):
    name: str
    email: Email

# Aggregate Root - consistency boundary
class Account(m.AggregateRoot):
    owner: User
    balance: Decimal

# CQRS Command
class CreateUserCommand(m.Command):
    username: str
    email: str
```

---

## flext_tests Package Architecture

### Overview

The `flext_tests` package provides shared test infrastructure for the FLEXT ecosystem. It follows strict patterns:

- **Single class per module** - Each module exports exactly one main class prefixed with `FlextTests`
- **Extends flext_core** - Test classes extend their `flext_core` counterparts
- **Short aliases** - Essential classes provide single-letter aliases for convenience
- **Maximum reuse** - Uses `FlextUtilities` and other `flext_core` components directly

### Module Structure

```
src/flext_tests/
├── __init__.py          # Public API exports
├── base.py              # FlextTestsServiceBase (extends FlextService) → alias: s
├── builders.py          # FlextTestsBuilders
├── constants.py          # FlextTestsConstants (extends FlextConstants) → alias: c
├── docker.py            # FlextTestDocker
├── domains.py            # FlextTestsDomains
├── factories.py          # FlextTestsFactories
├── files.py              # FlextTestsFileManager
├── matchers.py           # FlextTestsMatchers
├── models.py             # FlextTestsModels (extends FlextModels) → alias: m
├── protocols.py          # FlextTestsProtocols (extends FlextProtocols) → alias: p
├── typings.py            # FlextTestsTypes (extends FlextTypes) → alias: t
└── utilities.py          # FlextTestsUtilities (extends FlextUtilities) → alias: u
```

### Key Classes and Aliases

**Foundation Classes** (extend flext_core, provide short aliases):
- `FlextTestsTypes` → `t` (extends `FlextTypes`)
- `FlextTestsConstants` → `c` (extends `FlextConstants`)
- `FlextTestsProtocols` → `p` (extends `FlextProtocols`)
- `FlextTestsModels` → `m` (extends `FlextModels`)
- `FlextTestsUtilities` → `u` (extends `FlextUtilities`)
- `FlextTestsServiceBase` → `s` (extends `FlextService`)

**Direct Imports from flext_core**:
- `e` - FlextExceptions
- `r` - FlextResult
- `d` - FlextDispatcher
- `x` - FlextMixins

### Usage Pattern

```python
from flext_tests import t, c, p, m, s, u
from flext_core import e, r, d, x

# Use short aliases for test utilities
result = u.Factory.create_result(value="test")
u.Result.assert_success(result)

# Use extended types
config = m.Docker.ContainerConfig(name="test")

# Use extended constants
timeout = c.Docker.DEFAULT_TIMEOUT

# Use extended protocols
logger: p.Infrastructure.Logger.StructlogLogger = ...
```

### FlextTestsUtilities Simplification

**Status**: ✅ **Simplified from 4094 to 333 lines (92% reduction)**

The `FlextTestsUtilities` class was significantly simplified to:
- **Extend `FlextUtilities`** - All `FlextUtilities` functionality available via inheritance
- **Essential methods only** - Removed overengineered nested classes
- **Compatibility maintained** - Old API still works via compatibility aliases

**Structure**:
```python
class FlextTestsUtilities(FlextUtilities):
    """Extends FlextUtilities with test-specific helpers."""

    class Result:
        """Result assertion helpers."""
        assert_success[TResult](result: r[TResult]) -> TResult
        assert_failure[TResult](result: r[TResult]) -> str
        assert_success_with_value[T](result: r[T], expected: T) -> None
        assert_failure_with_error[T](result: r[T], expected: str) -> None

    class TestContext:
        """Context managers for tests."""
        temporary_attribute(target, attribute, value) -> Generator

    class Factory:
        """Test data creation helpers."""
        create_result[T](value: T | None, error: str | None) -> r[T]
        create_test_data(**kwargs) -> dict

    # Compatibility aliases
    class TestUtilities: ...      # Old API compatibility
    class ResultHelpers: ...      # Old API compatibility
    class ModelTestHelpers: ...   # Model validation helpers
    class RegistryHelpers: ...   # Registry creation helpers
    class ConfigHelpers: ...      # Config creation/validation helpers
```

**Key Principle**: Use `FlextUtilities` directly when possible. Only add test-specific helpers that don't exist in `FlextUtilities`.

---

## Essential Commands

```bash
# Setup
make setup                    # Install deps + pre-commit hooks

# Quality gates (MANDATORY before commit)
make validate                 # Run ALL: lint + format-check + type-check + complexity + docstring-check + security + test
make check                    # Quick: lint + type-check only

# Individual checks
make lint                     # Ruff linting (ZERO violations)
make format                   # Auto-format code
make format-check             # Check formatting without modifying
make type-check               # Pyrefly type checking (ZERO errors)
make complexity               # Radon CC + MI analysis
make docstring-check          # Docstring coverage (80%+)
make security                 # Bandit + pip-audit security scan
make test                     # Full suite (80%+ coverage required)

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

## Quality Standards (Updated January 2025)

### Quality Gate Requirements

| Category | Tool | Threshold | Status |
|----------|------|-----------|--------|
| **Coverage** | pytest-cov | 80% minimum (strict) | ✅ |
| **Type Checking** | Pyrefly (Pyright-based) | ZERO errors in src/ | ✅ |
| **Type Checking** | Pyright | ZERO errors in core modules | ✅ |
| **Linting** | Ruff | ZERO violations | ✅ |
| **Security** | Bandit + detect-secrets | ZERO high/medium issues | ✅ |
| **Complexity** | Radon CC + MI | CC ≤ 10, MI ≥ A | ✅ |
| **Docstrings** | interrogate | 80% coverage | ✅ |

### Recent Type Safety Improvements (January 2025)

**Pyright Type Corrections** ✅ **COMPLETED**:
- ✅ **Status**: Core modules corrected with proper type hints
- ✅ **Modules Fixed**: `_utilities/configuration.py`, `_utilities/cache.py`, `_utilities/domain.py`, `_models/context.py`, `_models/config.py`
- ✅ **Pattern**: Used `getattr` with `cast()` for dynamic attribute access
- ✅ **Pattern**: Used explicit type annotations for generic types
- ✅ **Pattern**: Fixed `isinstance` unnecessary checks
- ✅ **Quality**: Zero pyright errors in core modules
- ✅ **Tests**: All tests passing with real execution (no mocks)

**Pyrefly Configuration** ✅ **COMPLETED**:
- ✅ **Status**: All errors corrected in `src/` and `tests/`
- ✅ **Configuration**: Search path configured to exclude backup directories
- ✅ **Tests**: Fixed test class inheritance issues (IOSuccess final class)
- ✅ **Tests**: Fixed method override signatures for error testing
- ✅ **Quality**: Zero pyrefly errors in `src/` and `tests/`

### Quality Gate Command

```bash
make validate  # Runs: lint + format-check + type-check + complexity + docstring-check + security + test
```

### Detailed Requirements

**Code Quality**:
- **Linting**: Ruff ZERO violations ✅
- **Type Checking**: Pyrefly ZERO errors (uses Pyright internally) ✅
- **Coverage**: 80% minimum (strict enforcement via pytest-cov + CI) ✅
- **Tests**: All tests passing ✅
- **Line Length**: 88 characters max (ruff-shared.toml)

**Security (Local + CI)**:
- **Bandit**: ZERO high/medium security issues ✅
- **detect-secrets**: Baseline file required (`.secrets.baseline`) ✅
- **pip-audit**: Dependency vulnerability scanning ✅

**Code Complexity**:
- **Radon CC**: Cyclomatic Complexity ≤ 10 per function ✅
- **Radon MI**: Maintainability Index ≥ A rating ✅

**Documentation**:
- **Docstrings**: 80% coverage via interrogate ✅
- **API Compatibility**: Both `.data` and `.value` must work ✅

**Architecture**:
- **Circular Dependencies**: ZERO (verified by import tests) ✅
- **Type Ignore Comments**: ZERO `# type: ignore` ✅
- **Any Types**: ZERO `Any` type annotations ✅
- **Centralized Types**: All complex types use `t.Types.*` aliases ✅
- **Centralized TypeVars**: All TypeVars imported from `flext_core.typings` ✅
- **Centralized Constants**: All constants use `c.Namespace.CONSTANT` pattern ✅
- **Short Aliases**: All work without lint complaints (`r, t, c, p, m, u, e, s, x, d, h`) ✅

### Lint Configuration Patterns

**BLE001 (Blind Exception Catching)**: Railway-oriented programming pattern requires catching all exceptions from user-provided code. This is configured globally in `ruff-shared.toml`:

```toml
# In ruff-shared.toml - Global ignore
[lint]
    ignore = [
        "BLE001",  # blind-except - Railway-oriented programming: framework code executes user handlers/callables
        # ... other ignores
    ]
```

**Rationale**: Framework code (dispatchers, handlers, containers, utilities) executes user-provided callbacks and must wrap any exception into `FlextResult.fail()` to maintain the railway pattern.

**Pyrefly Configuration**: Handles known limitations with recursive types and complex generics:

```toml
# In pyproject.toml
[tool.pyrefly]
    search-path = [
        ".",
        "src",
        "tests",
        "examples",
        "scripts",
    ]

[tool.pyrefly.errors]
    unknown-name = false         # Recursive type aliases (PEP 695)
    bad-return = false           # Complex generic type inference
    bad-assignment = false       # Union type inference
    no-matching-overload = false # Generic overload resolution
    bad-argument-type = false    # Complex argument typing
    not-iterable = false         # StrEnum iteration (valid Python 3.11+)
    unsupported-operation = false  # pyrefly doesn't understand 'not in' operator
    not-a-type = false           # Union return types with generics
    bad-specialization = false   # LRUCache generic parameterization
    missing-attribute = false    # Docker/files model attribute access
    index-error = false          # Docker container.image.tags indexing
    read-only = false            # Pydantic frozen model validators (use model_validator)
    redundant-cast = false       # Intentional casts for type narrowing
```

**Pyright Type Corrections** (January 2025):
- ✅ Fixed `get_global_instance` type inference using `getattr` with `cast()`
- ✅ Fixed generic type annotations in `collection.py` methods
- ✅ Fixed `isinstance` unnecessary checks
- ✅ Fixed test class inheritance (IOSuccess final class)
- ✅ Fixed test method override signatures

**PYI042**: Ignored globally to allow short alias names (`r`, `t`, `c`, `m`, `p`, `u`) without type annotations.

### Type Cast Patterns

**Status**: Zero `# type: ignore` comments ✅ | ~156 intentional `cast()` usages ✅

**Principle**: Eliminate all `cast()` where possible, but some are intentional by design.

**Eliminated Cast Patterns**:
```python
# ❌ BEFORE (eliminated) - FlextConfig from protocol-typed property
config = cast("FlextConfig", self.config)
config.enable_caching  # Access concrete attribute

# ✅ AFTER - mixins.config now returns FlextConfig directly
config = self.config  # No cast needed
config.enable_caching  # Direct access
```

**Intentional Cast Patterns** (protocol-to-concrete for nested class access):
```python
# ✅ INTENTIONAL - ServiceRuntime.context is p.Context.Ctx protocol
# but FlextContext has nested classes like .Service.service_context()
from flext_core.service import FlextService

class MyService(FlextService):
    def initialize(self) -> None:
        runtime = self._create_initial_runtime()
        # Cast is intentional - protocol can't expose nested classes
        context = cast("FlextContext", runtime.context)
        with context.Service.service_context(...):
            ...
```

**TypeGuard Functions** (preferred alternative to cast when applicable):
```python
from flext_core._utilities.guards import FlextUtilitiesGuards

# Use TypeGuard functions for type narrowing
if FlextUtilitiesGuards.is_config(obj):
    # obj is now typed as p.Configuration.Config
    name = obj.app_name

if FlextUtilitiesGuards.is_handler(obj):
    # obj is now typed as p.Application.Handler
    obj.handle(message)
```

**Available TypeGuard Functions** (in `u.Guards`):
- `is_config()` → `p.Configuration.Config`
- `is_context()` → `p.Context.Ctx`
- `is_container()` → `p.Container.DI`
- `is_command_bus()` → `p.Application.CommandBus`
- `is_handler()` → `p.Application.Handler`
- `is_logger()` → `p.Infrastructure.Logger.StructlogLogger`
- `is_result()` → `p.Foundation.Result`
- `is_service()` → `p.Domain.Service`
- `is_middleware()` → `p.Application.Middleware`

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
PYTHONPATH=src poetry run mypy src/ --strict --show-error-codes
PYTHONPATH=src poetry run pyright src/ --show-error-codes

# Test failures
pytest tests/unit/test_module.py -vv --tb=long

# Circular imports (test each module independently)
for module in src/flext_core/*.py; do
    PYTHONPATH=src python -c "import flext_core.$(basename $module .py)" 2>&1 | grep -v "^$"
done
```

---

**Additional Resources**: [../CLAUDE.md](../CLAUDE.md) (workspace), [README.md](README.md) (overview), [~/.claude/commands/flext.md](~/.claude/commands/flext.md) (MCP workflows)
