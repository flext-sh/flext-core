# Using flext-core 0.12

<!-- TOC START -->

- [Current public API](#current-public-api)
- [Need -> capability -> benefit](#need-capability-benefit)
- [Construction and execution](#construction-and-execution)
- [What flext-core must not absorb](#what-flext-core-must-not-absorb)
- [Validation and release boundary](#validation-and-release-boundary)

<!-- TOC END -->

This guide describes the public interfaces present in the 0.12 development line.
Source availability is not evidence that the complete fleet or published artifacts
have passed release acceptance.

The [0.13 contracts](../architecture/v0.13-contracts.md) are a separate planning
document. Do not replace current imports with those proposed interfaces while
finishing 0.12.

## Current public API

The current lazy root exports include `c`, `t`, `p`, `m`, `u`, the operational
aliases `r`, `e`, `x`, `h`, `d`, `s`, and their declared long class names.
`core` currently resolves to the registry/dispatcher execution instance exposed
by `api.py`; it is not a module containing `Service` or `Result` classes.

Consumer code imports declared symbols from `flext_core`, not private parts.
Model fields and service metadata use the models owned by `m`; dependency and
return annotations use the relevant contracts from `p` and aliases from `t`.

## Need -> capability -> benefit

| Need | Capability | Benefit |
| --- | --- | --- |
| Service foundation | `s[T]` / `FlextService[T]` | Shared service validation and lifecycle; `T` is the result payload type, not the settings type |
| Railway results | `r[T]` / `FlextResult[T]` | One failure-preserving result implementation |
| Typed ports | `t.Port[p.X]` | Constructor-level dependency injection without service locators |
| Data validation | `m` presets and `u.validate_value` | Shared typed boundaries; choose a strict preset when strict validation is required |
| Configuration | `FlextConfig`, `FlextSettings`, `config`, `settings` | Existing configuration and settings contracts; application-specific objects remain consumer-owned |
| Handlers | `h` / `FlextHandlers` | Existing typed operation metadata and registration |
| Lazy exports | Declared root exports | Deferred loading of requested components; not a guarantee that accessing a component has no initialization effects |

## Construction and execution

```python
from flext_core import r

result = r[int].ok(7)
print(result.unwrap())
```

An application composition root constructs adapters and passes them to its
services. A dependency uses `t.Port[p.X]`, with a runtime-checkable protocol and
an excluded field declaration, so service construction validates the supplied
collaborator without serializing it into the request or response.

Service `execute()` returns `p.Result[T]`. Project-specific bases may add their
own settings and options through MRO; inheriting directly from `s` is not a
replacement for those local specializations.

The existing `fetch_global()` service accessor builds without arguments. It is
not a way to supply required ports. `with_settings()` creates an isolated
settings snapshot; it is not equivalent to retaining an injected settings object
by reference.

The [registry example](https://github.com/flext-sh/flext-core/blob/0.12.0-dev/examples/ex_12_flext_registry.py) exercises public
registration, discovery and handler behavior. Its
[service](https://github.com/flext-sh/flext-core/blob/0.12.0-dev/examples/ex_12_registry_support.py) consumes a
[registry port](https://github.com/flext-sh/flext-core/blob/0.12.0-dev/examples/_models/ex12.py); the
[composition example](https://github.com/flext-sh/flext-core/blob/0.12.0-dev/examples/ex_12_registry_plugins.py) demonstrates
shared and isolated registries. For the corresponding service and port rules,
see [service patterns](service-patterns.md).

## What flext-core must not absorb

- CLI frameworks and command rendering belong to `flext-cli`.
- Code generation, AST/Rope analysis, locks and Git belong to `flext-infra`.
- Pytest fixtures and matchers belong to `flext-tests`.
- Domain adapters and LDAP/LDIF/Meltano semantics belong to their packages.
- Application authorization and use cases belong to the application.

## Validation and release boundary

From the containing FLEXT workspace root, use the canonical member commands:

```bash
make -f flext-core/Makefile examples
make -f flext-core/Makefile docs
make -f flext-core/Makefile tests
```

Examples exercise the public runtime; docs checks validate the documentation
contract; tests provide additional behavioral evidence. Passing one does not
prove the others, the whole fleet, or an installed release candidate.

Removing operational aliases, changing `core` into a typed module namespace and
rewiring consumer bases are 0.13 proposals. They are not prerequisites to close
the 0.12 documentation slice or permission to publish the 0.12 release. Release
acceptance remains with Bead `flext-1wjg1`.
