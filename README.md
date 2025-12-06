# FLEXT-Core

FLEXT-Core is the dispatcher-centric foundation library for the FLEXT ecosystem. It provides railway-oriented programming primitives, a layered dependency-injection bridge (runtime + container + handlers), CQRS dispatching, and domain-driven design helpers for Python 3.13+.

## Key Capabilities

- **Railway-oriented results:** `FlextResult` models success and failure without raising exceptions in business code.
- **Protocol-based architecture:** Extensive use of protocols following SOLID principles (Interface Segregation, Dependency Inversion) to eliminate circular imports and Pydantic forward reference issues.
- **Dependency injection:** `FlextContainer` manages shared services with clear scopes (global or custom containers) via the dependency-injector bridge (`Provide`/`inject`).
- **CQRS dispatcher:** `FlextDispatcher` routes commands, queries, and domain events through registered handlers and middleware.
- **Domain primitives:** `FlextModels`, `FlextService`, and mixins simplify entity/value modeling and domain service composition.
- **Infrastructure helpers:** configuration loading, structured logging, and execution context propagation are bundled for consistent cross-cutting concerns.

## Architecture at a Glance

```
src/flext_core/
├── config.py        # Environment-aware configuration helpers
├── constants.py     # Shared defaults and immutables
├── protocols.py     # Protocol definitions (SOLID, hierarchical namespaces)
├── typings.py       # Shared typing aliases and type definitions
├── container.py     # Dependency injection container
├── context.py       # Request/operation context propagation
├── decorators.py    # Cross-cutting decorators and middleware helpers
├── dispatcher.py    # CQRS dispatcher for commands, queries, and events
├── exceptions.py    # Exception hierarchy aligned with result codes
├── handlers.py      # Handler interfaces and base implementations
├── loggings.py      # Structured logging helpers
├── mixins.py        # Reusable mixins for models and services
├── models.py        # DDD entities, values, and aggregates
├── registry.py      # Handler registry utilities
├── result.py        # Railway-oriented `FlextResult`
├── runtime.py       # Integration bridge to external libraries
├── service.py       # Domain service base classes
└── utilities.py     # General-purpose helpers
```

### Protocol-Based Architecture (SOLID)

The codebase extensively uses protocols organized in hierarchical namespaces to follow SOLID principles:

```python
from flext_core.protocols import p  # FlextProtocols

# Use protocols in interfaces (Dependency Inversion Principle)
def process_command(
    dispatcher: p.Application.CommandBus,
    config: p.Configuration.Config,
    logger: p.Infrastructure.Logger.StructlogLogger,
) -> p.Foundation.Result[str]:
    """All interfaces use protocols for flexibility and testability."""
    pass
```

**Benefits**:
- Eliminates circular import issues
- Avoids Pydantic forward reference problems
- Enables structural typing (duck typing)
- Follows Interface Segregation Principle
- Makes code more testable and maintainable

## Quick Start

Install and verify the core imports:

```bash
pip install flext-core
python - <<'PY'
from flext_core import FlextContainer, FlextDispatcher, FlextResult
container = FlextContainer.get_global()
print('flext-core ready', FlextDispatcher.__name__, bool(container))
PY
```

Register and dispatch a simple command:

```python
from dataclasses import dataclass
from flext_core import FlextDispatcher, FlextResult

@dataclass
class CreateUser:
    email: str

dispatcher = FlextDispatcher()

def handle_create_user(message: CreateUser) -> FlextResult[str]:
    if "@" not in message.email:
        return FlextResult[str].fail("invalid email")
    return FlextResult[str].ok(f"created {message.email}")

dispatcher.register_handler(CreateUser, handle_create_user)
result = dispatcher.dispatch(CreateUser(email="user@example.com"))
assert result.is_success
```

### Dependency Injection Architecture (Clear Architecture + DI)

FLEXT-Core implements a layered dependency injection pattern following Clear Architecture principles, ensuring services are easily accessible via DI for downstream projects:

**Architecture Layers**:
- **L0.5 (Runtime Bridge)**: `FlextRuntime` is the single surface to access providers/containers/wiring (`Provide`, `inject`) and configuration helpers
- **L1 (DI Integration)**: `FlextRuntime.DependencyIntegration` owns declarative containers, typed providers (Singleton/Factory/Resource), and `providers.Configuration`
- **L1.5 (Service Runtime Bootstrap)**: `FlextRuntime.create_service_runtime` materializes config/context/container in one call with optional overrides, registrations, and wiring
- **L2 (Container)**: `FlextContainer` uses bridge providers to register services, factories, and resources with generics, cloning them for scopes
- **L3 (Handlers/Dispatcher)**: Handlers are wired via `wire_modules`, and `@inject`/`Provide` decorators are re-exported by the runtime

**Key Services Accessible via DI** (Auto-registered):
- `FlextConfig`: Available as `"config"` - Configuration management with environment variables and validation
- `FlextLogger`: Available as `"logger"` (factory) - Structured logging with context propagation
- `FlextContext`: Available as `"context"` - Request/operation context and correlation IDs
- `FlextContainer`: Dependency injection container with scoped contexts
- All domain services via `FlextService` with `_runtime_bootstrap_options` override

**Core Services Auto-Registration**:
Core services (`config`, `logger`, `context`) are automatically registered when creating a `FlextContainer` instance, making them immediately available for dependency injection:

```python
from flext_core import FlextContainer, FlextConfig, FlextLogger, FlextContext, r

container = FlextContainer()

# Core services are automatically available
config_result: r[FlextConfig] = container.get("config")
logger_result: r[FlextLogger] = container.get("logger")
context_result: r[FlextContext] = container.get("context")

# All return success with the respective service instances
assert config_result.is_success
assert logger_result.is_success
assert context_result.is_success
```

**Usage Example**:

```python
from flext_core import FlextContainer, Provide, inject

container = FlextContainer()
container.register_resource("db", lambda: create_db_client())
container.register_factory("token_factory", lambda: {"token": "abc123"})

# Wire modules for automatic dependency injection
container.wire_modules(modules=[my_module])

@inject
def do_work(
    db=Provide["db"],
    token_factory=Provide["token_factory"]
):
    # Dependencies are resolved automatically by the bridge
    return db, token_factory()

db_client, tokens = do_work()
```

**Service Bootstrap with DI**:

```python
from flext_core import FlextService, r, t

class MyService(FlextService[str]):
    @classmethod
    def _runtime_bootstrap_options(cls) -> t.Types.RuntimeBootstrapOptions:
        return {
            "config_overrides": {"app_name": "my_app"},
            "services": {"feature_flag": True},
            "factories": {"token_factory": lambda: {"token": "svc"}},
            "resources": {"api_client": lambda: connect_client()},
            "wire_modules": [my_module],
        }

    def execute(self) -> r[str]:
        # Access services via container
        token = self.container.get("token_factory").value["token"]
        return r[str].ok(f"token: {token}")
```

The layered bridge keeps dependency-injector usage isolated to the runtime (L0.5) and container (L2) while handlers (L3) use only the public `Provide`/`inject` API. **Projects consuming flext-core should NEVER import `dependency-injector` directly** - always use the facades exposed by `FlextRuntime` and `FlextContainer`.

See [`docs/dependency_injector_prompt.md`](docs/dependency_injector_prompt.md) for the complete pattern checklist and architectural rules.

When you need a standalone DI container (for tests or auxiliary modules) without manual follow-up registration, use the parameterized factory helper:

```python
from flext_core import FlextRuntime

di_container = FlextRuntime.DependencyIntegration.create_container(
    config={"feature_flags": {"beta": True}},
    services={"static_value": 123},
    factories={"token_factory": lambda: {"token": "abc123"}},
    resources={"api_client": lambda: connect_client()},
    wire_modules=[my_module],
    factory_cache=False,  # opt-in per-call factories instead of singletons
)

# Service runtimes can use the same automation
# FlextService inherits FlextMixins (which extends FlextRuntime) and bootstraps
# a scoped runtime with optional overrides, registrations, and wiring hooks.

class ReportingService(FlextService[r[str]]):
    @classmethod
    def _runtime_bootstrap_options(cls) -> t.Types.RuntimeBootstrapOptions:
        return {
            "config_overrides": {"app_name": "reports"},
            "services": {"feature_flag": True},
            "factories": {"token_factory": lambda: {"token": "svc"}},
            "resources": {"api_client": lambda: connect_client()},
            "wire_modules": [reporting_module],
        }

    def execute(self) -> r[str]:
        token = self.container.get[dict]("token_factory").value["token"]
        return r[str].ok(f"token: {token}")
```

## Type System Guidelines

### Centralized Type System (FlextTypes) ✅ **COMPLETED**

**MANDATORY**: All complex types MUST use centralized type aliases from `t.Types` namespace.

```python
from flext_core.typings import t, T, U  # ✅ CORRECT - Direct import

# ✅ CORRECT - Use centralized types
def process_config(config: t.Types.ConfigurationDict) -> t.Types.StringDict:
    """Use centralized type aliases."""
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
members_dict: t.Types.StringStrEnumInstanceDict = getattr(
    enum_cls, "__members__", {}
)

# ❌ FORBIDDEN - Direct type definitions
def process_config(config: dict[str, t.GeneralValueType]) -> dict[str, str]:  # FORBIDDEN
    pass

# ❌ FORBIDDEN - Local TypeVar definitions
TResult = TypeVar("TResult")  # FORBIDDEN - Use T from flext_core.typings
```

**Available Type Aliases** (in `t.Types` namespace):
- Configuration: `ConfigurationDict`, `ConfigurationMapping`, `StringConfigurationDictDict`
- String mappings: `StringDict`, `StringIntDict`, `StringFloatDict`, `StringBoolDict`
- Enum types: `StringStrEnumTypeDict`, `StringStrEnumInstanceDict`
- Exception types: `StringFlextExceptionTypeDict`
- Handler types: `StringHandlerCallableListDict`, `HandlerTypeDict`
- Settings types: `StringBaseSettingsTypeDict`
- Path types: `StringPathDict`
- And many more... (see `typings.py` for complete list)

**Available TypeVars** (from `flext_core.typings`):
- Core generics: `T`, `T_co` (covariant), `T_contra` (contravariant)
- Utilities: `U`, `R`, `E`
- ParamSpec: `P` (for decorators)
- Handlers: `MessageT_contra`, `ResultT`
- Config/Models: `T_Model`, `T_Namespace`, `T_Settings`

**Status**: ✅ **COMPLETED** (January 2025) - All 66 Python files in `src/` using centralized types and TypeVars.

### Protocol Usage Guidelines

**MANDATORY**: Use protocols extensively in type hints for interfaces:

```python
from flext_core.protocols import p

# ✅ CORRECT - Protocols in interface signatures
def create_service(config: p.Configuration.Config) -> p.Domain.Service[str]:
    """Use protocols, not concrete classes."""
    pass

# ✅ CORRECT - Protocols in return types
def get_logger() -> p.Infrastructure.Logger.StructlogLogger:
    """Return protocol type, not concrete class."""
    logger = FlextLogger(__name__)
    return cast("p.Infrastructure.Logger.StructlogLogger", logger)

# ❌ FORBIDDEN - Direct class references in interfaces
def create_service(config: FlextConfig) -> FlextService[str]:  # AVOID
    pass
```

**Protocol Rules**:
- ✅ All protocol imports at top of file (no lazy imports)
- ✅ No `TYPE_CHECKING` blocks for protocol imports
- ✅ Use protocols for all interface type hints
- ✅ Use concrete classes only for instantiation and inheritance

**Type System Rules**:
- ✅ Import: `from flext_core.typings import t, T, U` (NOT `from flext_core import typings as t`)
- ✅ All `dict[str, ...]` patterns MUST use `t.Types.*` aliases
- ✅ All TypeVars MUST be imported from `flext_core.typings` (no local definitions)
- ✅ Generic types like `dict[str, T]` where `T` is a type parameter are OK
- ✅ Zero tolerance for duplicate type/TypeVar definitions

### Short Aliases Pattern (ECOSYSTEM STANDARD)

**MANDATORY**: Use short aliases for frequently used types to keep code concise:

```python
# ✅ CORRECT - Import short aliases from their modules
from flext_core.result import r       # FlextResult alias
from flext_core.typings import t      # FlextTypes alias
from flext_core.constants import c    # FlextConstants alias
from flext_core.models import m       # FlextModels alias
from flext_core.protocols import p    # FlextProtocols alias
from flext_core.utilities import u    # FlextUtilities alias
from flext_core.exceptions import e   # FlextExceptions alias
from flext_core.context import x      # FlextContext alias

# ✅ CORRECT - Use short aliases in type hints and code
def validate_data(data: t.GeneralValueType) -> r[bool]:
    """Return FlextResult using r[T] alias."""
    if not data:
        return r[bool].fail("Data cannot be empty")
    return r[bool].ok(True)

# ✅ CORRECT - Access nested namespaces via short aliases
config_dict: t.Types.ConfigurationDict = {"key": "value"}
error_code = c.Errors.VALIDATION_ERROR
```

**FlextResult Creation** (MANDATORY):
```python
# ✅ CORRECT - Use r[T].ok() and r[T].fail() directly
def process(value: str) -> r[str]:
    if not value:
        return r[str].fail("Empty value")
    return r[str].ok(value.upper())

# ❌ FORBIDDEN - FlextRuntime.result_* (causes pyright errors)
def process(value: str) -> r[str]:
    return FlextRuntime.result_fail("Error")  # FORBIDDEN
```

**Lint Configuration**: The `PYI042` rule is ignored to allow type alias names without annotations.

See [`CLAUDE.md`](./CLAUDE.md) for detailed architecture guidelines.

## Documentation

Full documentation lives in [`docs/`](./docs/) and follows the standards in [`docs/standards/`](./docs/standards/). Notable entry points:

- [`docs/QUICK_START.md`](./docs/QUICK_START.md) — five-minute overview
- [`docs/architecture/overview.md`](./docs/architecture/overview.md) — layer summary
- [`docs/api-reference/`](./docs/api-reference/) — module-level API reference by layer
- [`examples/`](./examples/) — runnable examples demonstrating handlers, dispatcher flows, and supporting utilities

## Project Status

**Version**: 0.9.9 → 1.0.0 (December 2025)
**Python**: 3.13+ only
**Tests**: 2820 tests passing
**Coverage**: 81.41% (above 80% minimum)
**Files**: 66 Python files in `src/`

### Quality Gate Command

```bash
make validate  # Runs: lint + format-check + type-check + complexity + docstring-check + security + test
```

### Quality Metrics

| Category | Tool | Threshold | Status |
|----------|------|-----------|--------|
| **Coverage** | pytest-cov | 80% minimum | ✅ |
| **Type Checking** | Pyrefly (Pyright-based) | ZERO errors | ✅ |
| **Linting** | Ruff | ZERO violations | ✅ |
| **Security** | Bandit + detect-secrets | ZERO high/medium issues | ✅ |
| **Complexity** | Radon CC + MI | CC ≤ 10, MI ≥ A | ✅ |
| **Docstrings** | interrogate | 80% coverage | ✅ |

### Detailed Quality Status

- ✅ **Linting**: Ruff ZERO violations (both `src/flext_core/` and `src/flext_tests/`)
- ✅ **Type Checking**: Pyrefly ZERO errors (uses Pyright internally)
- ✅ **Security**: Bandit + detect-secrets with baseline (local + CI)
- ✅ **Complexity**: Radon CC + MI analysis passing
- ✅ **Circular Dependencies**: ZERO (verified by import tests)
- ✅ **API Compatibility**: Both `.data` and `.value` work (backward compatible)
- ✅ **Centralized Types**: ✅ **COMPLETED** - All modules using `t.Types.*` aliases
- ✅ **Centralized TypeVars**: ✅ **COMPLETED** - All TypeVars from `flext_core.typings`
- ✅ **Centralized Constants**: ✅ **COMPLETED** - All constants using `c.Namespace.CONSTANT`

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
[tool.pyrefly.errors]
    unknown-name = false         # Recursive type aliases (PEP 695)
    bad-return = false           # Complex generic type inference
    bad-assignment = false       # Union type inference
    no-matching-overload = false # Generic overload resolution
    bad-argument-type = false    # Complex argument typing
    not-iterable = false         # StrEnum iteration (valid Python 3.11+)
```

**PYI042**: Ignored globally to allow short alias names (`r`, `t`, `c`, `m`, `p`, `u`) without type annotations.

### Recent Improvements (January 2025)

**Centralized Type System** ✅ **COMPLETED**:
- ✅ All 66 Python files in `src/` using centralized types
- ✅ All TypeVars imported from `flext_core.typings` (T, U, T_co, T_contra, E, R, P, etc.)
- ✅ All complex types using `t.Types.*` aliases
- ✅ Zero local TypeVar definitions
- ✅ Zero duplicate type definitions

**Centralized Constants** ✅ **COMPLETED**:
- ✅ `FlextConstants` fully organized with 20+ namespaces
- ✅ All constants using `c.Namespace.CONSTANT` pattern
- ✅ Zero duplication, fully typed with `Final`

**flext_tests Module Standardization** ✅ **COMPLETED** (January 2025):
- ✅ All modules follow FLEXT patterns with Python 3.13+ syntax
- ✅ Short aliases work without lint complaints (`r`, `t`, `c`, `m`, `p`, `u`)
- ✅ Result patterns work with protocols without casts
- ✅ FlextRuntime and FlextResult patterns unified
- ✅ FlextTestsUtilities with comprehensive helper classes
- ✅ All 2561 tests passing with 81.40% coverage

**Type Safety Improvements** ✅ **COMPLETED** (January 2025):
- ✅ Pyrefly: 0 errors in `src/` and `tests/` (345 ignored - known limitations)
- ✅ Pyright: 0 errors in core modules (`configuration.py`, `cache.py`, `domain.py`, `context.py`)
- ✅ Fixed dynamic attribute access using `getattr` with `cast()`
- ✅ Fixed generic type annotations in collection utilities
- ✅ Fixed test class inheritance and method override signatures
- ✅ Tests incrementally using `tt, tf, tb, tv, tm` from `flext_tests` for simplification

**Key Patterns Established**:
- ✅ Direct import: `from flext_core.typings import t, T, U` (required for MyPy)
- ✅ All complex types use `t.Types.*` aliases
- ✅ All TypeVars from `flext_core.typings` (no local definitions)
- ✅ Generic types (`dict[str, T]` where `T` is a type parameter) remain as-is
- ✅ Zero tolerance for duplicate type/constant definitions
- ✅ BLE001 globally ignored for Railway-oriented programming pattern
- ✅ Pyrefly configured for known type system limitations

## Contributing and Support

- Review [`CLAUDE.md`](./CLAUDE.md) for project-specific guidelines and architecture patterns
- Review [`docs/development/contributing.md`](./docs/development/contributing.md) for coding, documentation, and testing expectations (PEP 8/257 compliant)
- Open issues or discussions on GitHub for bug reports or design questions
