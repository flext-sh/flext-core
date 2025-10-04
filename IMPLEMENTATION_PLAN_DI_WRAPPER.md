# Implementation Plan: dependency-injector Internal Wrapper

**Branch**: `feature/di-internal-wrapper`
**Target Version**: v1.1.0
**Start Date**: 2025-10-04
**Status**: 🚧 IN PROGRESS

---

## 🎯 OBJECTIVE

Implement dependency-injector as internal implementation for FlextContainer while maintaining 100% external API compatibility and FlextResult railway pattern.

**Key Principles**:
- ✅ Zero breaking changes to public API
- ✅ FlextResult railway pattern preserved on all public methods
- ✅ All 670 existing usages continue working
- ✅ Maintain 1.0.0 stability guarantees
- ✅ Gradual feature exposure (auto-wiring, providers)

---

## 📋 TASK CHECKLIST

### Phase 0: Setup & Baseline
- [x] **TASK 0.1**: Create feature branch `feature/di-internal-wrapper` ✅
- [x] **TASK 0.2**: Add dependency-injector to pyproject.toml ✅
- [x] **TASK 0.3**: Install dependencies and verify baseline tests pass ✅
- [ ] **TASK 0.4**: Create Architecture Decision Record (ADR) ⏳

### Phase 1: Core Implementation
- [x] **TASK 1.1**: Implement internal DI container in FlextContainer ✅
  - Added `_di_container: DynamicContainer`
  - Maintained `_services` and `_factories` dicts for compatibility
  - Implemented sync between tracking dicts and DI container
- [x] **TASK 1.2**: Update `register()` method ✅
  - Stores in both `_services` dict AND DI container via `_store_service()`
  - Wrapped DI operations in FlextResult
  - Preserved existing error handling
- [x] **TASK 1.3**: Update `register_factory()` method ✅
  - Uses `providers.Factory` internally via `_store_factory()`
  - Stores in both `_factories` dict AND DI container
  - Maintained FlextResult return
- [x] **TASK 1.4**: Update `get()` method ✅
  - Resolves via DI container in `_resolve_service()`
  - Wraps result in FlextResult
  - Preserved error messages
- [x] **TASK 1.5**: Update `get_typed()` method ✅
  - Maintained type safety with DI resolution (unchanged, uses get() internally)
  - Preserved FlextResult[T] return type
- [ ] **TASK 1.6**: Update `batch_register()` and batch operations ⏳
  - Need to verify DI container sync for batch ops
  - Maintain atomic rollback on failure

### Phase 2: FlextConfig Integration
- [x] **TASK 2.1**: Add `to_di_config()` method to FlextConfig ✅
  - Implemented as `_sync_config_to_di()` in FlextContainer
- [x] **TASK 2.2**: Add `_sync_config_to_di()` method to FlextContainer ✅
  - Creates Configuration provider
  - Syncs all FlextConfig fields to DI container
- [x] **TASK 2.3**: Ensure environment variables flow to DI container ✅
  - FlextConfig reads from environment
  - _sync_config_to_di() mirrors to DI container
- [ ] **TASK 2.4**: Test bidirectional config sync ⏳
  - Need integration tests

### Phase 3: Testing
- [ ] **TASK 3.1**: Create `tests/unit/test_container_di_adapter.py`
  - Test DI container initialization
  - Test sync between dicts and DI container
  - Test FlextResult wrapping
  - Test exception translation
- [ ] **TASK 3.2**: Create `tests/unit/test_container_compatibility.py`
  - Test ALL existing API methods work unchanged
  - Test FlextResult railway pattern preserved
  - Test singleton pattern maintained
  - Test thread-safety preserved
- [ ] **TASK 3.3**: Create `tests/integration/test_di_container_integration.py`
  - Test FlextConfig + DI integration
  - Test service lifecycle
  - Test factory resolution
- [ ] **TASK 3.4**: Run full existing test suite
  - Must pass with 79%+ coverage
  - Zero test failures allowed

### Phase 4: Quality Gates
- [ ] **TASK 4.1**: Run `make lint` - must pass with zero violations
- [ ] **TASK 4.2**: Run `make type-check` - must pass with zero errors
- [ ] **TASK 4.3**: Run `make test` - must pass with 79%+ coverage
- [ ] **TASK 4.4**: Run `make security` - no critical vulnerabilities
- [ ] **TASK 4.5**: Run `make validate` - complete validation pipeline

### Phase 5: Ecosystem Validation
- [ ] **TASK 5.1**: Test flext-api project compatibility
- [ ] **TASK 5.2**: Test flext-cli project compatibility
- [ ] **TASK 5.3**: Test flext-ldap project compatibility
- [ ] **TASK 5.4**: Test flext-auth project compatibility
- [ ] **TASK 5.5**: Run flext-core examples (02, 08, etc.)

### Phase 6: Documentation
- [ ] **TASK 6.1**: Update container.py docstrings
- [ ] **TASK 6.2**: Update CHANGELOG.md for v1.1.0
- [ ] **TASK 6.3**: Update README.md with implementation note
- [ ] **TASK 6.4**: Create ADR document
- [ ] **TASK 6.5**: Update API_STABILITY.md if needed

### Phase 7: Review & Merge
- [ ] **TASK 7.1**: Create comprehensive PR description
- [ ] **TASK 7.2**: Self-review all changes
- [ ] **TASK 7.3**: Address any review comments
- [ ] **TASK 7.4**: Final validation before merge

---

## 📊 PROGRESS TRACKING

### Current Status: Phase 1 & 2 Mostly Complete, Starting Phase 3
- **Current Task**: TASK 3.1 - Creating adapter layer tests
- **Completed**: 11/35 tasks (31%)
- **Estimated Time Remaining**: 1-2 days

### Milestones
- [ ] **M1**: Setup complete (Phase 0)
- [ ] **M2**: Core implementation complete (Phase 1)
- [ ] **M3**: All tests passing (Phases 2-3)
- [ ] **M4**: Quality gates passed (Phase 4)
- [ ] **M5**: Ecosystem validated (Phase 5)
- [ ] **M6**: Documentation complete (Phase 6)
- [ ] **M7**: PR merged (Phase 7)

---

## 🔧 IMPLEMENTATION NOTES

### Key Design Decisions

#### 1. Wrapper Pattern
```python
class FlextContainer:
    def __init__(self):
        # Internal: dependency-injector DynamicContainer
        self._di_container = containers.DynamicContainer()

        # Compatibility: Keep existing tracking dicts
        self._services: FlextTypes.Dict = {}
        self._factories: FlextTypes.Dict = {}

        # Integration: FlextConfig sync
        self._flext_config = FlextConfig()
        self._sync_config_to_di()
```

#### 2. API Preservation
```python
# PUBLIC API - Unchanged
def register(self, name: str, service: object) -> FlextResult[None]:
    """FlextResult preserved, DI used internally."""
    try:
        # Store in both places
        self._services[name] = service
        self._di_container.set_provider(
            name,
            providers.Singleton(lambda s=service: s)
        )
        return FlextResult[None].ok(None)
    except Exception as e:
        return FlextResult[None].fail(str(e))
```

#### 3. FlextConfig Integration
```python
def _sync_config_to_di(self) -> None:
    """Sync FlextConfig to DI container Configuration."""
    config_provider = providers.Configuration()
    config_provider.from_dict({
        'environment': self._flext_config.environment,
        'debug': self._flext_config.debug,
        'log_level': self._flext_config.log_level,
        # ... all FlextConfig fields
    })
    self._di_container.config = config_provider
```

---

## ⚠️ RISKS & MITIGATION

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Performance regression | Medium | Benchmark before/after, <5% threshold | 📋 Planned |
| Hidden API dependencies | High | Comprehensive ecosystem testing | 📋 Planned |
| FlextConfig sync issues | Medium | Extensive integration tests | 📋 Planned |
| Thread safety problems | High | Concurrency stress tests | 📋 Planned |
| Breaking changes (unintended) | Critical | Run all 670 usage tests | 📋 Planned |

---

## 📈 SUCCESS METRICS

- ✅ **Zero breaking changes**: All existing API methods work unchanged
- ✅ **Test coverage**: 79%+ maintained (currently 79%)
- ✅ **Quality gates**: All pass (lint, type-check, security, test)
- ✅ **Ecosystem validation**: All dependent projects pass their tests
- ✅ **Performance**: <5% regression vs baseline
- ✅ **Documentation**: Complete and accurate

---

## 🔗 REFERENCES

- **ADR**: docs/architecture/adr/ADR-001-dependency-injector-wrapper.md (to be created)
- **dependency-injector docs**: https://python-dependency-injector.ets-labs.org/
- **FlextContainer current**: src/flext_core/container.py
- **Related discussions**: GitHub issue #XXX (to be created)

---

## 📝 EXECUTION LOG

### 2025-10-04 - Session 1
- ✅ Created implementation plan document
- ✅ TASK 0.1: Created feature branch `feature/di-internal-wrapper`
- ✅ TASK 0.2: Added dependency-injector>=4.41.0,<5.0.0 to pyproject.toml
- ✅ TASK 0.3: Installed dependency-injector v4.48.1, verified baseline working
- ✅ TASK 1.1-1.5: Implemented internal DI wrapper in container.py
  - Added DynamicContainer as `_di_container`
  - Updated register(), register_factory(), get(), unregister()
  - Implemented `_sync_config_to_di()` for FlextConfig integration
  - Fixed DynamicContainer attribute access pattern
  - Verified with smoke tests - all passing ✅
- 🚧 Next: Create comprehensive tests (Phase 3)

---

**Last Updated**: 2025-10-04
**Next Review**: After Phase 1 completion
