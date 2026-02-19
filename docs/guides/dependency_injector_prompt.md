# dependency-injector pattern prompt


<!-- TOC START -->
- No sections found
<!-- TOC END -->

Use this prompt whenever you evolve **flext-core** or downstream projects to keep the dependency-injector integration pattern aligned (bridge L0.5/L1 + container L2 + handlers L3), without breaking the ABI or reintroducing ad-hoc registries.

```
You are responsible for applying and reinforcing the dependency-injector-based pattern across the codebase. Follow these rules:

Architectural context
- L0.5 (runtime bridge): `flext_core.runtime.FlextRuntime` is the single surface to access providers/containers/wiring (Provide, inject) and configuration helpers. Expand capabilities only by exposing new helpers here.
- L1 (DI integration): `FlextRuntime.DependencyIntegration` owns the declarative container, typed providers (Singleton/Factory/Resource), and `providers.Configuration` for defaults/overrides. Adjustments must preserve the public API exported in `flext_core.__init__`.
- L1.5 (service runtime bootstrap): `FlextRuntime.create_service_runtime` (inherited by `FlextService`) materializes config/context/container in one call with optional overrides, registrations, and wiring. Prefer overriding `FlextService._runtime_bootstrap_options` to pass parameters instead of duplicating setup logic.
- L2 (container): `FlextContainer` uses the bridge providers to register services, factories, and resources with generics, cloning them for scopes without manual dictionaries. The public API stays in `register`, `register_factory`, `register_resource`, `resolve`, `create_scope`, `configure`, etc.
- L3 (handlers/dispatcher): handlers are wired via `wire_modules`, and @inject/Provide decorators are re-exported by the runtime. Do not import dependency-injector directly in L3.

Implementation rules
- Prefer `providers.Resource` for external clients (DB/HTTP/queues), guaranteeing teardown/close and removing manual lifecycle boilerplate.
- Use `providers.Configuration` to synchronize defaults/overrides validated by `FlextSettings`; avoid manual merges and preserve override precedence.
- Prefer the parameterized `DependencyIntegration.create_container` helper when instantiating DI containers directly; it can bind configuration, register providers, and wire modules in one call, with caching controlled via parameters.
- Service classes should reuse the runtime automation by overriding `_runtime_bootstrap_options` on `FlextService` to feed parameters into `FlextRuntime.create_service_runtime` (config overrides, service/factory/resource seeding, wiring targets). Avoid duplicating container/context/config creation in constructors.
- Registrations must use typed providers (Generic[T]) to keep type-safety; avoid extra casts.
- When changing caching/lifecycles, preserve existing configuration and the public ABI (parameters and returns exposed to consumers).
- Any new DI surface must be exposed via the bridge/runtime or existing facades, never by direct dependency-injector imports in upper layers.
- For wiring, use `wire_modules` from the container/bridge and respect modules/classes/handlers already registered by the dispatcher.

Usage in downstream projects
- In projects consuming flext-core, inject services/resources via the same facades (`FlextContainer`, `wire_modules`, re-exported decorators). Do not create alternative containers or access dependency-injector directly.
- Honor inherited configuration contracts: defaults via `providers.Configuration` and user overrides applied by `configure(...)`.
- If you need new capabilities (e.g., a new provider), add it to the flext-core bridge and re-export it before using it in the dependent project.

Final checklist
- Ensure `tests/unit/test_container.py` and `tests/unit/test_runtime.py` continue to pass.
- Do not break the public ABI (imports documented in `flext_core/__init__.py`).
- Avoid regressions in automatic wiring within the dispatcher/handlers.
- Update architecture or guide docs in the same change when DI behavior/public usage changes.
- Keep examples aligned with root policy and avoid direct dependency-injector imports in upper layers.
```
