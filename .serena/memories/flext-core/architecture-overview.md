# flext-core: Core Architecture

**Version**: 0.12.0-dev | **Type**: Platform/Foundation | **Python**: 3.13+

## 5 Core Pillars

1. **Railway-Oriented Programming**: `r[T, E]` for error handling (no exceptions in business code)
2. **Dependency Injection**: `FlextContainer` singleton with scoped lifetimes & factory registration
3. **CQRS Dispatching**: `FlextDispatcher` routes typed commands/queries to handlers
4. **Settings & Config**: `FlextSettings` hierarchy via MRO; env prefix `FLEXT_*`
5. **Structured Logging**: `FlextLogger` wraps structlog; context propagated via `FlextContext`

## Alias System (10 Main)

| `r` | Result | `c` | Constants  | `m` | Models   | `t` | Types      | `p` | Protocols |
| --- | ------ | --- | ---------- | --- | -------- | --- | ---------- | --- | --------- |
| `u` | Utils  | `e` | Exceptions | `s` | Services | `d` | Dispatcher | `h` | Helpers   |

## Key Classes

- `FlextResult[T,E]`: Railway carrier
- `FlextContainer`: DI singleton (resolve, bind, factory, scope)
- `FlextDispatcher`: Command router (register_handler, dispatch)
- `FlextSettings[T]`: Config base with env override
- `FlextLogger`: Structured logging bridge
- `FlextContext`: Request context + correlation IDs

## Dependency Flow

**Inward only**: L3(App) → L2(Domain) → L1(Foundation) → L0(Contracts)
Bridge pattern: External infra accessed ONLY via flext-core facades.

**Last Updated**: 2026-04-14