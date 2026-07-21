# flext-core Contract Budget (c/t/p/m/u)

**Owner**: this session
**Scope**: flext-core/src ONLY (tests/examples/docs follow after src is green)
**Gate order**: per-module ruff = 0 → pyrefly = 0 → next module. pytest ONLY after all src green.

## Rules (locked)

- One concept = one home. No aliases-of-aliases.
- No recursive types anywhere (`Recursive*` forbidden). Already-commented stay commented.
- `_typings/` = pure typing. No runtime constants, no concrete classes, no tuples of `type`.
- No `None` baked into type aliases. Callers compose `T | None` where needed.
- No compatibility shims, no flat re-exports from facades.
- **`m.*` never appears in type hints**: parameters, returns, field annotations must use `p.*` or `t.*`. `m.*` is only instantiation / validation / `model_validate`. This is the anti-cyclic-import rule.
- Custom validation logic in models → move to Pydantic v2 validators; if it cannot be expressed there, comment with `# REWIRE:` and migrate later.
- Protocols (`p.*`) = behaviour only; never export concretes as types.
- Utilities (`u.*`) = pure helpers. Narrowing ladders are forbidden — normalize-or-fail via Pydantic validators upstream.
- Constants (`c.*`) = constants only. Runtime tuples of `type` currently in `_typings/base.py` move here.

## Baseline

- ruff src/ = **0** (clean)
- pyrefly src/ = **380 errors**
  - ~181 `missing-attribute` for `RecursiveContainer*` / `MutableRecursiveContainer*` (correct: callers must be rewired, aliases stay deleted)
  - ~43 `ConfigMap`/`Dict` `TypeAlias` used as callable (legacy model-alias confusion in runtime.py)
  - 212 total `missing-attribute`
  - balance: mis-wired arguments, unknown-name, implicit-any — cleared as side-effect of A-D

## Budget: t.* (FlextTypingBase + FlextTypesServices + FlextTypingContainers + FlextTypesPydantic + FlextTypesAnnotateds)

### FlextTypingBase — KEEP (flat, composable)
- `Numeric`, `Primitives`, `Scalar`, `Container`
- `FlatScalarMapping`, `FlatScalarSequence`
- `StrMapping` / `MutableStrMapping`, `StrSequence`
- `ScalarMapping` / `MutableScalarMapping`, `ScalarList`
- `FlatContainerList`
- `IntMapping` / `BoolMapping` / `FrozensetMapping` / `StrSequenceMapping` (+ mutable twins)
- `AttributeMapping`, `ConfigValueMapping`, `HeaderMapping`, `FeatureFlagMapping` (+ mutable)
- `Pair/Triple/Quad/Quint/VariadicTuple/IntPair`
- `OpaqueValue = object` — KEEP for one specific boundary (TypeAdapter erased values) but audit callers.
- Concrete bases `ContainerMappingBase`, `ContainerListBase`, `MutableContainerMappingBase`, `MutableContainerListBase` — MOVE OUT to `_constants/` (they are runtime classes, not types).
- Tuples `PRIMITIVES_TYPES`/`NUMERIC_TYPES`/`SCALAR_TYPES`/`CONTAINER_TYPES`/`CONTAINER_AND_COLLECTION_TYPES` — MOVE to `_constants/` (runtime, not typing).

### FlextTypingBase — REMOVE / MIGRATE
- None yet (already lean).

### FlextTypesServices — KEEP (signature composition only)
- `RegistryDict`, `ModelCarrier`, `ContainerCarrier`, `ScalarOrModel`, `ModelClass`
- `LogArgument`, `LogValue`, `LogResult`
- `MetadataValue`, `MetadataData`, `MetadataData`
- `RuntimeData`, `RuntimeData`, `RuntimeData`, `PresentRuntimeData`, `BootstrapInput`
- `HandlerCallable`, `DispatchableHandler`, `HandlerProtocolVariant`, `ResolvedHandlerCallable`, `RoutedHandlerCallable`, `RegisteredHandler`, `AutoHandlerRegistration`, `MessageTypeSpecifier`
- `ConfigurationMapping`, `FlatContainerMapping` / `MutableFlatContainerMapping`, `ResultErrorData`
- `SortableObjectType`, `TypeHintSpecifier`, `ValueAdapter`, `GenericTypeArgument`, `IncEx`, `LazyImportIndex`
- `SettingsClass`, `RuntimeModule`, `LazyScalar`, `LazyCollection`, `ModuleExportValue`, `ModuleExport`, `LazyGetattr`, `LazyDir`, `LazyNamespaceValue`
- `ValidatorCallable`, `MapperCallable`, `MapperInput`, `StrictValue`, `PaginationMeta`
- `GuardInput` (already flat-rewritten), `UserOverridesMapping`
- `RegistrablePlugin`
- `ContextHookCallable` / `ContextHookMap`

### FlextTypesServices — REMOVE
- All `# DEPRECATED` comment blocks (dead code — delete, don't leave).
- `ProtocolModelCarrier = p.Model` — thin pass-through duplicate of `p.Model`; inline at call sites.
- `DomainModelCarrier = ModelCarrier | ProtocolModelCarrier` — collapse to `ModelCarrier` (pydantic `BaseModel` is the SSOT).
- `RegisterableService` gigantic union — tighten to `ModelCarrier | Callable[..., ModelCarrier]` and require structured payloads via `m.*`. Flat-value registration still via `t.ContainerCarrier`.
- `ResourceCallable` is identical to `FactoryCallable` — keep one, alias removed.
- `ServiceMap`, `FactoryMap`, `ResourceMap` — keep `ServiceMap` only. The factory/resource variants are not referenced outside container.
- `HandlerLike` duplicates `HandlerCallable` — remove.
- `TypeOriginSpecifier = TypeHintSpecifier` — remove.
- `ScopedContainerRegistry`, `ScopedScalarRegistry` — move to `m.*` (container model).

## Budget: p.* (FlextProtocols*)

Keep only behavioral:
- `Result`, `ResultLike`, `SuccessCheckable`, `StructuredError`, `ErrorDomainProtocol`
- `Model` (structural), `Routable`, `Dispatcher`, `Handle`, `Execute`, `AutoDiscoverableHandler`
- `Context`, `Container` (protocol, not the concrete class), `ProviderLike`
- `Settings`, `Configurable`
- `Logger`, `OutputLogger`, `Flushable`
- `Registry`, `RegistryBacked`
- `Service`, `DispatchableService`
- `ProjectMetadata`

Nothing else. All dict/list recursive hints in protocol signatures must switch to `t.FlatContainerMapping` / `t.StrMapping` / `m.*` / concrete payload models.

## Budget: m.* (FlextModels*)

SSOT for structured/validated payloads. Primary rewiring targets (replace recursive dicts):

1. **`FlextModelsContext`** — already exists. Ensure `ContextState` / `ContextSnapshot` carry `FlatContainerMapping` fields + validators; strip recursive hints.
2. **`FlextModelsDomainEvent`** — payload must be a Pydantic model (`EventPayload`), not `t.RecursiveContainerMapping`.
3. **`FlextModelsRegistry`** — `RegistryEntry` model for `(key, service, metadata: FlatContainerMapping)`.
4. **`FlextModelsHandler`** — `HandlerSpec` with registered callable + metadata model.
5. **`FlextModelsCollections`** — strip custom merge/diff code to validators; delegate to `u.collection` only where Pydantic can't express it.
6. **`FlextModelsContextScope`** — custom_fields/mapping becomes a tight Pydantic model (no recursion), serialization via model_serializer.
7. **Normalization** — the `_normalization`/`_lifecycle` utilities emit into concrete models.

Models flagged for validator-consolidation (move custom logic into `field_validator` / `model_validator` / computed_field):
- `handler.py` (511 LOC)
- `exception_params.py` (460 LOC)
- `dispatcher.py` (441 LOC)
- `base.py` (399 LOC)
- `container.py` (381 LOC)
- `collections.py` (376 LOC)

Each of these gets a pass in Stage B.

## Budget: u.* (FlextUtilities)

Keep only pure helpers:
- `collection` merge/diff ops (shrink — use `m.*` validators where possible)
- `mapper` normalization (strip `RecursiveContainer*` — use `FlatContainerMapping`)
- `checker` — audit, remove redundant narrowing
- `guards_type_*` — shrink: protocol guards only, drop recursive
- `generators`, `parser`, `text`, `enum`, `conversion`, `discovery`, `args`

Remove / collapse:
- `guards_ensure.py` redundant with pydantic validators — audit & shrink
- `context_*.py` move data-holder logic into `FlextModelsContext*`, leave `u.*` only as composition/reflect helpers
- `reliability.py` audit vs. actual callers
- `logging_context.py` redundant with `m.Context` — trim

## Budget: c.* (FlextConstants)

Keep domain constants and error codes. Accept runtime tuples relocated from `_typings/base.py`:
- `c.Typing.PRIMITIVES_TYPES`
- `c.Typing.NUMERIC_TYPES`
- `c.Typing.SCALAR_TYPES`
- `c.Typing.CONTAINER_TYPES`
- `c.Typing.CONTAINER_AND_COLLECTION_TYPES`

Concrete base classes (`ContainerMappingBase`, etc.) move to `_models/containers.py` as `m.Containers.*` OR to a new `_base_containers.py` in `_utilities/`. Decision: keep in `m.Containers.*` since they are structural bases.

## Execution order (strict)

1. **A1 — Freeze t.***:
   - Delete commented `# DEPRECATED` blocks in `_typings/services.py` and `_typings/core.py`.
   - Remove `ProtocolModelCarrier`, `DomainModelCarrier`, `ResourceCallable`, `HandlerLike`, `TypeOriginSpecifier`, `ServiceMap` dup variants, `ScopedContainerRegistry`, `ScopedScalarRegistry` — following grep check that each has no active caller or can be inlined.
   - Move `PRIMITIVES_TYPES`/etc. + concrete `*Base` classes out of `_typings/base.py` → `_constants/` or `_models/containers.py`.
   - Rewrite `typeadapters.py` to use `t.FlatContainerMapping` instead of `t.RecursiveContainerMapping`.
2. **A2 — Freeze p.***: audit every protocol signature; replace any `Mapping[str, Any-nested]` with `t.FlatContainerMapping` or concrete `m.*`.
3. **B — m.* absorbs validation**: starting with context + domain_event + registry + handler, embed validators; comment (`# REWIRE:`) any logic that doesn't fit Pydantic and plan its migration for Stage C.
4. **C — u.* trim**: rewire utilities to `t.FlatContainerMapping` / `m.*`. Strip narrowing. Each utility module ruff+pyrefly zero before moving on.
5. **D — Consumer rewiring**: `context.py`, `handlers.py`, `runtime.py`, `registry.py`, `service.py`, `mixins.py` → canonical paths only.
6. **Gate**: `ruff src/ == 0` + `pyrefly src/ == 0` for the WHOLE src before touching tests/examples/docs.
7. Tests → examples → docs → pytest.

## Invariants audited each module

- Zero new aliases introduced (count public objects before/after → equal or smaller).
- Zero runtime code in `_typings/`.
- Zero `None` in `t.*` alias RHS.
- Zero `# DEPRECATED` comment garbage.
- Zero `Recursive*` references.
- Zero pass-through wrappers.
