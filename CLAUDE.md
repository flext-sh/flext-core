# CLAUDE

Project-level pointer file.

**See [../CLAUDE.md](../CLAUDE.md) for full workspace standards.**

## Zero Tolerance Rules (Completely Prohibited)

1. **Hacks**: ❌ PROHIBITED - `model_rebuild()`, `eval()`, `exec()`, and architectural `getattr()`.
1. **Inline/Lazy Imports**: ❌ PROHIBITED - No imports inside functions or `try / except ImportError:`.
1. **# type: ignore**: ❌ PROHIBITED COMPLETELY - Zero tolerance, no exceptions.
1. **Root Aliases**: ❌ PROHIBITED COMPLETELY - Always use complete namespace.
1. **cast()**: ❌ PROHIBITED - Replace with Models/Protocols/TypeGuards.
1. **Any**: ❌ PROHIBITED - Replace with specific types.
1. **TypeAlias→type reversion**: ❌ PROHIBITED - Non-recursive aliases in `typings.py` MUST use `X: TypeAlias = ...`. Converting to PEP 695 `type X = ...` breaks isinstance(). Run `make validate-typings` to verify.

## TypeAliasType isinstance Incompatibility (CRITICAL — Python 3.12+)

PEP 695 `type X = ...` creates `TypeAliasType` objects, NOT `UnionType`. This means `isinstance(val, X)` **FAILS at runtime** with `TypeError: isinstance() arg 2 cannot be a parameterized generic`.

### Mandatory Rules

1. **Non-recursive type aliases** (`Primitives`, `Scalar`, `Container`, `ConfigurationMapping`, etc.) MUST use `TypeAlias` annotation syntax: `X: TypeAlias = str | int | float | bool`. This creates a `UnionType` that works with `isinstance()`.
1. **Recursive type aliases** (`Serializable`, `ContainerValue`, `JsonValue`, `GeneralValueType`) MUST use PEP 695 `type X = ...` statement — it is the ONLY syntax that supports self-referencing types.
1. **NEVER use `isinstance()` directly on recursive type aliases** — they are `TypeAliasType` and will crash. Use `TypeGuard` functions in `guards.py` instead.
1. **`TypeAliasType.__value__` is transitively poisoned**: if `Scalar.__value__` contains `Primitives` which is itself a `TypeAliasType`, the `__value__` chain fails. Do NOT attempt `isinstance(val, X.__value__)` as a workaround.
1. **`TypeAdapter` is NOT a substitute for `isinstance()`**: benchmarked at 4x slower (1.03s vs 0.25s for 100K iterations). Do NOT use `TypeAdapter.validate_python()` for type checking — it is exception-driven control flow.

### Guards for Type Narrowing

Use `TypeGuard` functions from `_utilities/guards.py`:

```python
from flext_core._utilities.guards import is_primitive, is_scalar, is_flexible_value

if is_primitive(val):   # TypeGuard[str | int | float | bool]
    ...
if is_scalar(val):     # TypeGuard[str | int | float | bool | datetime]
    ...
```

### Quick Reference

| Syntax                      | Runtime Type    | isinstance safe? | Use for                |
| --------------------------- | --------------- | ---------------- | ---------------------- |
| `X: TypeAlias = str \| int` | `UnionType`     | ✅ YES           | Non-recursive aliases  |
| `type X = str \| int`       | `TypeAliasType` | ❌ NO            | Recursive aliases only |
| `TypeGuard[X]`              | N/A             | ✅ YES           | Any type narrowing     |

## API Consolidation — Canonical Method Names (v0.12.0+)

After Phase 2 API consolidation, these are the ONLY canonical method names. Old aliases are DELETED — do NOT use them.

### Removed Aliases → Canonical Methods

| ❌ DELETED (old alias)           | ✅ CANONICAL (use this)               | Module        |
| -------------------------------- | ------------------------------------- | ------------- |
| `.data`                          | `.value`                              | result.py     |
| `.and_then()`                    | `.flat_map()`                         | result.py     |
| `.alt()`                         | `.map_error()`                        | result.py     |
| `.warn()`                        | `.warning()`                          | loggings.py   |
| `.try_unbind()`                  | `.unbind(safe=True)`                  | loggings.py   |
| `.validate_command()`            | `.validate()`                         | handlers.py   |
| `.validate_query()`              | `.validate()`                         | handlers.py   |
| `.track_performance()`           | `.log_operation(track_perf=True)`     | decorators.py |
| `.create_error()`                | `.create()`                           | exceptions.py |
| `.set_all()`                     | `.set(data)`                          | context.py    |
| `.register_factory()`            | `.register(kind='factory')`           | container.py  |
| `.register_resource()`           | `.register(kind='resource')`          | container.py  |
| `.with_service()`                | `.register()`                         | container.py  |
| `.with_factory()`                | `.register(kind='factory')`           | container.py  |
| `.with_resource()`               | `.register(kind='resource')`          | container.py  |
| `.get_typed()`                   | `.get(type_cls=T)`                    | container.py  |
| `.with_config()`                 | `.configure()`                        | container.py  |
| `.reset_singleton_for_testing()` | `.reset_for_testing()`                | container.py  |
| `.get_global_instance()`         | `.get_global()`                       | settings.py   |
| `.materialize()`                 | `.get_global(overrides=...)`          | settings.py   |
| `.reset_global_instance()`       | `.reset_for_testing()`                | settings.py   |
| `.auto_register()`               | `.register_namespace(decorator=True)` | settings.py   |
| `.get_namespace_config()`        | `.get_namespace()`                    | settings.py   |
| `.register_class_plugin()`       | `.register_plugin(scope='class')`     | registry.py   |
| `.get_class_plugin()`            | `.get_plugin(scope='class')`          | registry.py   |
| `.list_class_plugins()`          | `.list_plugins(scope='class')`        | registry.py   |
| `.unregister_class_plugin()`     | `.unregister_plugin(scope='class')`   | registry.py   |

### Privatized Methods

| Old (public)                 | New (private)                 | Module        |
| ---------------------------- | ----------------------------- | ------------- |
| `prepare_exception_kwargs()` | `_prepare_exception_kwargs()` | exceptions.py |
| `extract_common_kwargs()`    | `_extract_common_kwargs()`    | exceptions.py |

## Agent Skill Enforcement (MANDATORY)

Any agent editing `typings.py`, or any file that imports `from flext_core import t` or `from flext_core.typings import`, MUST:

1. Load skill `flext-strict-typing` before starting work.
2. Read the `typings.py` inline warnings before making ANY changes.
3. Run `make validate-typings` after editing `typings.py`.
4. Non-recursive TypeAlias syntax is NON-NEGOTIABLE — violations cause runtime crashes.

Agent prompts MUST include:
```
⚠️ TYPINGS.PY RULE: Non-recursive aliases use `X: TypeAlias = ...` (NOT `type X = ...`).
Do NOT convert any TypeAlias to a type statement. See CLAUDE.md Zero Tolerance Rules.
```
