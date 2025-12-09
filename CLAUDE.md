# flext-core - FLEXT Core Foundation

**Hierarchy**: PROJECT
**Parent**: [../CLAUDE.md](../CLAUDE.md) - Workspace standards
**Last Update**: 2025-12-07

---

## Project Overview

**FLEXT-Core** is the foundation library for 32+ dependent projects in the FLEXT ecosystem. Every change here has massive impact - ZERO TOLERANCE for breaking changes.

**Version**: 0.10.0 (December 2025)
**Coverage**: 81.41% (above 73% minimum)
**Python**: 3.13+ only
**Tests**: 2820 tests passing

---

## Rule 0 — Cross-Project Alignment
- This file mirrors the root `../CLAUDE.md` standards. Any rule change must be written in the root first and then propagated to this file and to `flext-cli/`, `flext-ldap/`, `flext-ldif/`, and `algar-oud-mig/` `CLAUDE.md` files.
- All agents accept cross-project changes and resolve conflicts in the root `CLAUDE.md` before coding.

## ⚠️ CRITICAL: Architecture Layering (Zero Tolerance)

### Module Import Hierarchy (MANDATORY)

**ABSOLUTELY FORBIDDEN IMPORT PATTERNS**:

```
NEVER IMPORT (regardless of method - direct, lazy, function-local, proxy):

Foundation Modules (models.py, protocols.py, utilities.py, typings.py, constants.py):
  ❌ NEVER import services/*.py
  ❌ NEVER import servers/*.py
  ❌ NEVER import api.py

Infrastructure Modules (servers/*.py):
  ❌ NEVER import services/*.py
  ❌ NEVER import api.py
```

**CORRECT ARCHITECTURE LAYERING**:

```
Tier 0 - Foundation (ZERO internal dependencies):
  ├── constants.py    # Only StrEnum, Final, Literal - NO functions
  ├── typings.py      # Type aliases, TypeVars
  └── protocols.py    # Interface definitions (Protocol classes)

Tier 1 - Domain Foundation:
  ├── models.py       # Pydantic models → constants, typings, protocols
  └── utilities.py    # Helper functions → constants, typings, protocols, models

Tier 2 - Infrastructure:
  └── servers/*.py    # Server implementations → Tier 0, Tier 1 only
                      # NEVER import services/, api.py

Tier 3 - Application (Top Layer):
  ├── services/*.py   # Business logic → All lower tiers
  └── api.py          # Facade/API → All lower tiers
```

**WHY THIS MATTERS**:
- Circular imports cause runtime failures
- Lazy imports are a band-aid, not a solution
- Proper layering ensures testability and maintainability
- Each tier only depends on lower tiers, NEVER on higher tiers

---

## Unified Ecosystem Standards

**CRITICAL**: These standards apply to ALL projects in the FLEXT ecosystem (flext-core, flext-cli, flext-ldif, flext-ldap, etc.). They ensure consistency, maintainability, and type safety across the entire ecosystem.

### Critical Rules (Zero Tolerance)

1. **TYPE_CHECKING**: ❌ PROHIBITED completely — refactor architecture instead.
2. **# type: ignore**: ❌ PROHIBITED completely — fix types properly.
3. **cast()**: ❌ PROHIBITED completely — replace with Models/Protocols/TypeGuards and correct typing.
4. **Any**: ❌ PROHIBITED completely — including docstrings/comments.
5. **Metaclasses/__getattr__/dynamic assignments**: ❌ PROHIBITED — use explicit methods and full namespaces.
6. **Constants**: ❌ No functions/logic in constants.py — only StrEnum/Final/Literal.
7. **Namespace**: ✅ Full namespace always; no root aliases or lazy imports/ImportError fallbacks.
8. **Architecture layering**: ✅ Enforced; lower tiers never import higher tiers (see hierarchy above).
9. **Testing**: ✅ Real implementations only (no mocks/monkeypatch), real data/fixtures, 100% coverage expectation, no functionality loss.

### Architecture Violation Quick Check

**Run before committing:**
```bash
# Quick check for this project
grep -rEn "(from flext_.*\.(services|api) import)" \
  src/*/models.py src/*/protocols.py src/*/utilities.py \
  src/*/constants.py src/*/typings.py src/*/servers/*.py 2>/dev/null

# Expected: ZERO results
# If violations found: Do NOT commit, fix architecture first
```

**See [Ecosystem Standards](../CLAUDE.md) for complete prohibited patterns and remediation examples.**

### Reference Documentation

- **Plans**: See `~/.cursor/plans/` for detailed pattern correction plans
- **Standards**: See `docs/standards/` for coding standards and templates
- **Guides**: See `docs/guides/` for pattern guides and best practices
- **Anti-patterns**: See `docs/guides/anti-patterns-best-practices.md` for common mistakes and solutions

---

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

**Protocol Organization** (hierarchical namespaces - FULL NAMESPACE REQUIRED):
```python
from flext_core.protocols import p  # FlextProtocols

# =====================================================
# FULL NAMESPACE ACCESS (MANDATORY)
# =====================================================
# Foundation
p.Foundation.Result[T]
p.Foundation.ResultLike[T_co]
p.Foundation.Model
p.Foundation.HasModelDump

# Configuration
p.Configuration.Config
p.Configuration.Configurable

# Context
p.Context.Ctx

# Container
p.Container.DI

# Domain
p.Domain.Service[T]
p.Domain.Repository[T]
p.Domain.Validation.HasInvariants

# Application
p.Application.Handler
p.Application.CommandBus
p.Application.Processor
p.Application.Middleware

# Infrastructure
p.Infrastructure.Logger.Log
p.Infrastructure.Logger.StructlogLogger
p.Infrastructure.Connection
p.Infrastructure.Metadata
```

**Usage Pattern** (MANDATORY for interfaces):
```python
# ✅ CORRECT - Use full namespace (MANDATORY)
def execute_service(service: p.Domain.Service[str]) -> p.Foundation.Result[str]:
    """Accept any service implementation via protocol."""
    return service.execute()

# ✅ CORRECT - Use protocols in return types for abstractions
def get_container() -> p.Container.DI:
    """Return container via protocol interface."""
    container = FlextContainer()
    return cast("p.Container.DI", container)

# ❌ FORBIDDEN - Root aliases (p.Result, p.Service, etc.)
def execute_service(service: p.Service[str]) -> p.Result[str]:  # FORBIDDEN
    pass

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
- ❌ **FORBIDDEN**: `TYPE_CHECKING` blocks for protocol imports (ZERO TOLERANCE)
- ❌ **FORBIDDEN**: Lazy imports (imports inside functions)
- ✅ **REQUIRED**: All imports at top of file, use protocols directly
- ✅ **REQUIRED**: Use forward references with `from __future__ import annotations` to avoid circular dependencies

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

**Type Alias Usage** (MANDATORY - FULL NAMESPACE REQUIRED):
```python
from flext_core.typings import t, T, U

# =====================================================
# FULL NAMESPACE ACCESS (MANDATORY)
# =====================================================
# t.Validation.PortNumber
# t.Types.StringDict
# t.Types.ConfigurationDict
# t.Types.GeneralValueType
# t.Types.ScalarValue

# ✅ CORRECT - Use full namespace (MANDATORY)
def process_config(config: t.Types.ConfigurationDict) -> t.Types.StringDict:
    """Use full namespace type aliases."""
    pass

# ❌ FORBIDDEN - Root aliases (t.StringDict, t.ConfigurationDict, etc.)
def process_config(config: t.ConfigurationDict) -> t.StringDict:  # FORBIDDEN
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
# FULL NAMESPACE REQUIRED (NO ROOT ALIASES)
# =====================================================
# protocols.py: p.Foundation.Result, p.Domain.Service, p.Application.Handler, etc.
# models.py: m.Cqrs.Command, m.Cqrs.Query, m.Entity.Value, m.Entity.Entity, etc.
# constants.py: c.Core.VALIDATION_ERROR, c.Core.HandlerType, c.Core.CommonStatus, etc.
# typings.py: t.Types.StringDict, t.Validation.PortNumber, t.Types.GeneralValueType, etc.

# ❌ FORBIDDEN - Full class names in type hints
def process(result: FlextResult[str]) -> FlextResult[bool]:  # FORBIDDEN
    pass

# ✅ CORRECT - Short aliases (r, t, c, p, m, u, e) for concrete classes
def process(result: r[str]) -> r[bool]:
    return r[bool].ok(True) if result.is_success else r[bool].fail("Error")

# ✅ CORRECT - Full namespace for Protocols, Types, Constants, Models
def process_service(service: p.Domain.Service[str]) -> p.Foundation.Result[str]:
    """Use full namespace for protocols."""
    pass

def process_config(config: t.Types.ConfigurationDict) -> t.Types.StringDict:
    """Use full namespace for types."""
    pass
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
# FULL NAMESPACE REQUIRED (MANDATORY)
# =====================================================
# m.Cqrs.Command
# m.Cqrs.Query
# m.Cqrs.Pagination
# m.Entity.Value
# m.Entity.Entity
# m.Entity.AggregateRoot
# m.Base.Metadata
# m.Config.ProcessingRequest
# m.Config.RetryConfiguration

# Value Object - immutable, compared by value
class Email(m.Entity.Value):  # ✅ CORRECT: full namespace
    address: str

# Entity - has identity
class User(m.Entity.Entity):
    name: str
    email: Email

# Aggregate Root - consistency boundary
class Account(m.Entity.AggregateRoot):
    owner: User
    balance: Decimal

# CQRS Command
class CreateUserCommand(m.Cqrs.Command):
    username: str
    email: str
```

---

## 📦 Import and Namespace Guidelines (Critical Architecture)

This section defines **mandatory patterns** for imports, namespaces, and module aggregation across the FLEXT ecosystem. These rules prevent circular imports and ensure maintainability.

### 1. Runtime Import Access (Short Aliases)

**MANDATORY**: Use short aliases at runtime for type annotations and class instantiation:

```python
# ✅ CORRECT - Runtime short aliases (src/ and tests/)
from flext_core.result import r       # FlextResult
from flext_core.typings import t      # FlextTypes
from flext_core.constants import c    # FlextConstants
from flext_core.models import m       # FlextModels
from flext_core.protocols import p    # FlextProtocols
from flext_core.utilities import u    # FlextUtilities
from flext_core.exceptions import e   # FlextExceptions
from flext_core.decorators import d   # FlextDecorators
from flext_core.context import x      # FlextContext
from flext_core.mixins import mx      # FlextMixins

# Usage with full namespace (MANDATORY)
result: r[str] = r[str].ok("value")
config: t.Types.ConfigurationDict = {}
status: c.Core.CommonStatus = c.Core.CommonStatus.OK
entry: m.Ldif.Entry = m.Ldif.Entry(dn="cn=test")
service: p.Domain.Service[str] = my_service

# ❌ FORBIDDEN - Root aliases
status: c.CommonStatus  # WRONG - must use c.Core.CommonStatus
entry: m.Entry          # WRONG - must use m.Ldif.Entry
```

### 2. Module Aggregation Rules (Facades)

**Facade modules** (models.py, utilities.py, protocols.py) aggregate internal submodules:

```python
# =========================================================
# models.py (Facade) - Aggregates _models/*.py
# =========================================================
from flext_core._models.base import FlextBaseModel
from flext_core._models.entity import FlextEntity
from flext_core._models.cqrs import FlextCommand, FlextQuery

class FlextModels:
    """Facade aggregating all model classes."""

    class Base:
        Model = FlextBaseModel

    class Entity:
        Entity = FlextEntity

    class Cqrs:
        Command = FlextCommand
        Query = FlextQuery

# Short alias for runtime access
m = FlextModels

# =========================================================
# IMPORT RULES FOR AGGREGATION
# =========================================================

# ✅ CORRECT - Internal modules (_models/) can import from:
#   - Other _models/* modules
#   - Tier 0 modules (constants, typings, protocols)
#   - NOT from services/, servers/, api.py

# ✅ CORRECT - Facade (models.py) imports from:
#   - All internal _models/* modules
#   - Tier 0 modules

# ❌ FORBIDDEN - Internal modules importing from higher tiers
# _models/base.py importing services/api.py = ARCHITECTURE VIOLATION
```

### 3. Circular Import Avoidance Strategies

**Strategy 1: Forward References with `from __future__ import annotations`**
```python
from __future__ import annotations
from typing import Self

class FlextService:
    def clone(self) -> Self:
        """Self reference works with forward annotations."""
        return type(self)()

    def create_entry(self) -> FlextEntry:  # String not needed - forward ref
        """Forward reference to FlextEntry class defined later."""
        return FlextEntry()

class FlextEntry:
    """Defined after FlextService but can be referenced above."""
    pass
```

**Strategy 2: Protocol-Based Decoupling**
```python
# protocols.py (Tier 0 - no internal imports)
from typing import Protocol

class ServiceProtocol(Protocol):
    def execute(self) -> bool: ...

# services/my_service.py (Tier 3 - can import protocols)
from flext_core.protocols import p

class MyService:
    def process(self, service: p.Domain.Service[str]) -> p.Foundation.Result[str]:
        """Use protocol types to avoid importing concrete classes."""
        return service.execute()
```

**Strategy 3: Dependency Injection**
```python
# Instead of importing services directly, inject them
from flext_core import FlextContainer

class MyHandler:
    def __init__(self, container: FlextContainer) -> None:
        self._container = container

    def process(self) -> None:
        # Get service at runtime instead of importing
        service_result = self._container.get("my_service")
        if service_result.is_success:
            service_result.value.execute()
```

### 4. When Modules Can Import Submodules Directly

**ALLOWED**: Base modules importing from internal submodules to avoid circulars:

```python
# =========================================================
# EXCEPTION: _utilities/builders.py importing from _models/
# =========================================================

# _utilities/builders.py
from flext_ldif._models.config import ProcessConfig  # ✅ ALLOWED

# WHY: _utilities and _models are both Tier 1
# Both are below services/ and api.py
# No circular dependency created

# =========================================================
# EXCEPTION: Base classes importing helpers
# =========================================================

# _models/base.py can import from _models/helpers.py
from flext_core._models.helpers import validate_field  # ✅ ALLOWED

# WHY: Same tier, both internal modules
```

**FORBIDDEN**: Higher tier importing lower tier that imports back:

```python
# ❌ FORBIDDEN PATTERN - Creates circular import
# api.py
from flext_ldif.services.parser import ParserService

# services/parser.py
from flext_ldif.api import FlextLdif  # CIRCULAR!

# ✅ CORRECT - Use protocols or injection
# services/parser.py
from flext_ldif.protocols import p

def parse_with_facade(facade: p.Application.Facade) -> None:
    """Accept protocol, not concrete class."""
    pass
```

### 5. Test Import Patterns

**Tests have special privileges but must follow patterns:**

```python
# tests/unit/test_my_module.py

# ✅ CORRECT - Import from package root
from flext_ldif import FlextLdif
from flext_ldif.models import m
from flext_ldif.constants import c

# ✅ CORRECT - Import test helpers
from tests import tm, tf  # TestsFlextLdifMatchers, TestsFlextLdifFixtures

# ✅ ALLOWED - Tests can import internal modules for testing
from flext_ldif._utilities.builders import ProcessConfigBuilder

# ✅ CORRECT - Use pytest fixtures
@pytest.fixture
def ldif_client() -> FlextLdif:
    return FlextLdif()

# ❌ FORBIDDEN - Don't use TYPE_CHECKING in tests
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # FORBIDDEN even in tests
    from flext_ldif import FlextLdif
```

### 6. Complete Import Hierarchy Reference

```
Tier 0 - Foundation (ZERO internal imports):
├── constants.py    → imports: NOTHING from flext_*
├── typings.py      → imports: NOTHING from flext_*
└── protocols.py    → imports: constants, typings (Tier 0 only)

Tier 1 - Domain Foundation:
├── _models/*.py    → imports: constants, typings, protocols
├── models.py       → imports: _models/*, constants, typings, protocols
├── _utilities/*.py → imports: _models/*, constants, typings, protocols
└── utilities.py    → imports: _utilities/*, models, constants, typings, protocols

Tier 2 - Infrastructure:
└── servers/*.py    → imports: Tier 0, Tier 1
                    → NEVER: services/, api.py

Tier 3 - Application:
├── services/*.py   → imports: ALL lower tiers
└── api.py          → imports: ALL lower tiers (Facade for external use)
```

### 7. Module-Specific Import Rules

| Source Module | Can Import From | Cannot Import From |
|---------------|-----------------|-------------------|
| constants.py | nothing | everything |
| typings.py | nothing | everything |
| protocols.py | constants, typings | everything else |
| _models/*.py | Tier 0, other _models/* | _utilities/*, services/, servers/, api.py |
| models.py | _models/*, Tier 0 | services/, servers/, api.py |
| _utilities/*.py | _models/*, Tier 0, models | services/, servers/, api.py |
| utilities.py | _utilities/*, models, Tier 0 | services/, servers/, api.py |
| servers/*.py | Tier 0, Tier 1 | services/, api.py |
| services/*.py | ALL lower tiers | NOTHING prohibited |
| api.py | ALL lower tiers | NOTHING prohibited |

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

## Automated Fix Scripts

For batch corrections (missing imports, undefined names), use `/tmp/fix_*.sh` scripts with 4 modes: `dry-run`, `backup`, `exec`, `rollback`. **See [../CLAUDE.md](../CLAUDE.md#automated-fix-scripts-batch-corrections)** for template and rules.

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
- **TYPE_CHECKING Blocks**: ZERO (prohibited, use forward references) ✅
- **Type Ignore Comments**: ZERO `# type: ignore` (prohibited, refactor code) ✅
- **__getattr__ Methods**: ZERO (prohibited, use explicit methods) ✅
- **Any Types**: ZERO `Any` type annotations (prohibited, use specific types) ✅
- **Root Aliases**: ZERO (prohibited, use full namespace) ✅
- **Centralized Types**: All complex types use `t.Types.*` aliases ✅
- **Centralized TypeVars**: All TypeVars imported from `flext_core.typings` ✅
- **Centralized Constants**: All constants use `c.Namespace.CONSTANT` pattern ✅
- **Short Aliases**: Concrete classes use short aliases (`r, t, c, p, m, u, e, s, x, d, h`) ✅
- **Full Namespace**: Protocols, Types, Constants, Models use full namespace ✅

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

**Pyrefly Configuration**: Search path configured for proper module resolution:

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
```

**Pyright Type Corrections** (January 2025):
- ✅ Fixed `get_global_instance` type inference using `getattr` with `cast()`
- ✅ Fixed generic type annotations in `collection.py` methods
- ✅ Fixed `isinstance` unnecessary checks
- ✅ Fixed test class inheritance (IOSuccess final class)
- ✅ Fixed test method override signatures

**PYI042**: Ignored globally to allow short alias names (`r`, `t`, `c`, `m`, `p`, `u`) without type annotations.

### Type Cast Patterns

**Status**: Zero `# type: ignore` comments ✅ | Minimizing `cast()` usage aggressively ✅

**Principle**: Minimize `cast()` aggressively - replace with Models/Protocols/TypeGuards where possible. Document intentional ones that remain.

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

**See Also**:
- [Workspace Standards](../CLAUDE.md)
- [flext-ldif Patterns](../flext-ldif/CLAUDE.md)
- [flext-cli Patterns](../flext-cli/CLAUDE.md)
- [Additional Resources](~/.claude/commands/flext.md) (MCP workflows)
