# FLEXT-CORE: Kwargs/Automation Architecture Plan

**Version**: 1.0
**Date**: 2025-11-20
**Status**: 🟢 APPROVED FOR IMPLEMENTATION
**Total Phases**: 8 (0-7)
**Estimated Duration**: 7-10 weeks
**Expected Impact**: -500 to -800 LOC, +advanced automation features

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Goals & Objectives](#goals--objectives)
3. [Architecture Principles](#architecture-principles)
4. [Phase Overview](#phase-overview)
5. [Quality Gates](#quality-gates)
6. [Risk Mitigation](#risk-mitigation)
7. [Success Metrics](#success-metrics)
8. [References](#references)

---

## 🎯 EXECUTIVE SUMMARY

This plan consolidates **three strategic initiatives** into a unified architecture evolution:

1. **Advanced Automation** (kwargs/dict/model/hybrid APIs with PEP 695 generics)
2. **Code Reduction** (eliminate duplication, consolidate helpers)
3. **Alignment with Existing Structure** (FlextService, FlextResult, FlextMixins, FlextModels, FlextProtocols)

### Key Outcomes

- **No feature loss**: 100% backward compatibility with deprecation warnings
- **No code growth**: Target reduction of 500-800 lines
- **Enhanced power**: Quad API (model/dict/kwargs/hybrid) for all config points
- **Zero circular imports**: Protocol-based dependency breaking
- **Zero business dicts**: All business logic uses Pydantic models
- **Zero TYPE_CHECKING for protocols**: Runtime-checkable protocols only

---

## 🎯 GOALS & OBJECTIVES

### Primary Goals

1. **🔧 Unified Automation API**
   - Single pattern for all configuration objects (RetryOptions, CacheOptions, etc.)
   - Quad API support: model/dict/kwargs/hybrid
   - Generic-first design with PEP 695 type parameters

2. **📉 Code Reduction Without Feature Loss**
   - Eliminate duplicated patterns (isinstance checks, conversions)
   - Consolidate helpers in FlextMixins
   - Replace 6-7 parameter methods with config objects
   - Convert inner classes to Pydantic models where appropriate

3. **🏗️ Architectural Integrity**
   - Break all circular dependencies with Protocols
   - Pure facade pattern for FlextModels
   - Module-level generics only
   - No business logic in dict[str, object]

### Secondary Goals

4. **📚 Enhanced Documentation**
   - Mini-RFC for automation patterns
   - Quad API usage examples
   - Migration guide from old patterns

5. **✅ Quality Assurance**
   - Automated cycle detection in CI
   - Automated business dict detection
   - LOC tracking per PR
   - Test coverage maintenance (≥79% for flext-core)

---

## 🏛️ ARCHITECTURE PRINCIPLES

### 1. Facade Purity (`FlextModels`)

**Rule**: `FlextModels` NEVER defines generics or business logic.

**Allowed**:
```python
# Thin wrappers for non-generics
class ExceptionContext(FlextModelsAutomation.ExceptionContext): ...

# Simple assignment for generics
Payload = FlextModelsBase.Payload
Categories = FlextModelsCollections.Categories
```

**Forbidden**:
```python
# NO generics defined here
class Payload[T](BaseModel): ...  # ❌ Wrong place!

# NO business logic
def validate_something(self): ...  # ❌ Not a facade!
```

### 2. Module-Level Generics (PEP 695)

**Rule**: ALL generics defined in `src/flext_core/_models/` modules.

```python
# ✅ In _models/base.py
class Payload[T](BaseModel):
    value: T
    metadata: dict[str, Any] = Field(default_factory=dict)

# ✅ In models.py (facade)
Payload = FlextModelsBase.Payload  # Simple assignment
```

### 3. Protocol-Based Dependency Breaking

**Rule**: Core modules (`exceptions`, `decorators`, `dispatcher`, `services`) NEVER import `FlextModels`.

**Use Protocols Instead**:
```python
from flext_core.protocols import (
    HasModelDump,
    ExceptionContextProtocol,
    RetryOptionsProtocol,
)

def my_function(context: ExceptionContextProtocol) -> None:
    data = context.model_dump()  # Protocol method
```

### 4. No TYPE_CHECKING for Protocols

**Rule**: With `from __future__ import annotations` + `@runtime_checkable`, import protocols directly.

```python
# ✅ Direct import
from flext_core.protocols import HasModelDump

@runtime_checkable
class MyProtocol(Protocol):
    def model_dump(self) -> dict[str, Any]: ...

# ❌ NO TYPE_CHECKING needed
if TYPE_CHECKING:  # DELETE THIS!
    from flext_core.protocols import HasModelDump
```

**Goal**: `grep -r "TYPE_CHECKING" src/` → **0 results**

### 5. Business Dicts Forbidden

**Rule**: `dict[str, object]` only for logging/tracing/free-form payloads.

**Business concepts → Pydantic Models**:
- ❌ `retry_config: dict[str, Any]`
- ✅ `retry_config: RetryOptions`

**Allowed Uses**:
- Logging context (dynamic, unstructured)
- Tracing metadata (key-value pairs)
- Generic payload wrappers (Payload[dict[str, Any]])

### 6. Helpers in FlextMixins

**Rule**: NO duplication of conversion patterns.

**Centralized Helpers**:
```python
# ✅ In FlextMixins.ModelConversion
@staticmethod
def to_dict(obj: BaseModel | dict[str, object] | None) -> dict[str, object]:
    if obj is None:
        return {}
    return obj.model_dump() if isinstance(obj, BaseModel) else obj

# ✅ In FlextMixins.ResultHandling
@staticmethod
def ensure_result[T](value: T | FlextResult[T]) -> FlextResult[T]:
    return value if isinstance(value, FlextResult) else FlextResult[T].ok(value)
```

**Usage**:
```python
# ❌ Before (duplicated everywhere)
if isinstance(request, BaseModel):
    data = request.model_dump()
elif isinstance(request, dict):
    data = request
else:
    data = {}

# ✅ After (single helper)
data = FlextMixins.ModelConversion.to_dict(request)
```

### 7. Config Objects > Many Parameters

**Rule**: Methods with 6+ parameters become config objects.

```python
# ❌ Before
def _execute_dispatch_attempt(
    self,
    handler_name: str,
    request: Any,
    timeout: float,
    mode: str,
    correlation_id: str,
    metadata: dict[str, Any],
) -> FlextResult[Any]: ...

# ✅ After
@dataclass
class DispatchAttemptConfig(BaseModel):
    handler_name: str
    request: Any
    timeout: float = 30.0
    mode: str = "async"
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)

def _execute_dispatch_attempt(
    self,
    config: DispatchAttemptConfig,
) -> FlextResult[Any]: ...
```

### 8. Pydantic for Infrastructure Classes

**Rule**: Inner classes with manual `__init__` → BaseModel + PrivateAttr.

**Benefits**:
- Automatic validation
- `model_dump()` for debugging
- Reduced boilerplate

**When to Apply**:
- Class has 5+ initialization parameters
- Class performs validation logic
- Class needs serialization/debugging

---

## 📊 PHASE OVERVIEW

| Phase | Name | Duration | LOC Impact | Risk |
|-------|------|----------|------------|------|
| [Phase 0](./epics/EPIC-00-baseline.md) | Baseline & Mapping | Week 0 | 0 | 🟢 Low |
| [Phase 1](./epics/EPIC-01-automation-core.md) | Automation Core + Helpers | Week 1 | -50 to -100 | 🟢 Low |
| [Phase 2](./epics/EPIC-02-models-facade.md) | Models + Facade | Week 1-2 | -30 to -50 | 🟢 Low |
| [Phase 3](./epics/EPIC-03-exceptions-quad.md) | Exceptions Quad API | Week 2 | +20 to -50 | 🟡 Medium |
| [Phase 4](./epics/EPIC-04-decorators.md) | Decorators Unified | Week 3 | -100 to -150 | 🟡 Medium |
| [Phase 5](./epics/EPIC-05-dispatcher-reduction.md) | Dispatcher Reduction | Week 4-5 | -200 to -300 | 🔴 High |
| [Phase 6](./epics/EPIC-06-dict-elimination.md) | Dict Elimination | Week 6 | -100 to -150 | 🟡 Medium |
| [Phase 7](./epics/EPIC-07-tests-docs-gates.md) | Tests/Docs/Gates | Week 7-10 | +50 (tests) | 🟢 Low |

**Total Expected Impact**: -500 to -800 LOC (net reduction)

---

## ✅ QUALITY GATES

### Per-PR Gates (Mandatory)

**Before Merge**:
1. ✅ `make lint` passes (Ruff: zero violations)
2. ✅ `make type-check` passes (MyPy strict: zero errors)
3. ✅ `make test` passes (all tests green)
4. ✅ Coverage ≥ baseline (≥79% for flext-core)
5. ✅ No new TYPE_CHECKING blocks
6. ✅ No new business dicts (`dict[str, object]` for config/options)
7. ✅ No nested generics created
8. ✅ No new isinstance duplication (use helpers)

**Automated Checks** (CI Integration):
```bash
# Add to CI pipeline
python scripts/detect_cycles.py        # Must exit 0
python scripts/analyze_dicts.py        # Must report 0 business dicts
grep -r "TYPE_CHECKING" src/ && exit 1 # Must find none
```

### Phase-Specific Gates

See individual epic files for detailed quality gates per phase.

---

## 🛡️ RISK MITIGATION

### High-Risk Areas

1. **Phase 5 - Dispatcher Refactoring**
   - **Risk**: Central component, touches many flows
   - **Mitigation**:
     - Incremental refactoring (one helper at a time)
     - Comprehensive integration tests
     - Feature flags for new code paths
     - Rollback plan prepared

2. **Phase 3 - Exception Breaking Changes**
   - **Risk**: Existing exception usage patterns
   - **Mitigation**:
     - Maintain backward compatibility
     - Add deprecation warnings
     - Gradual migration over 2-3 releases
     - Document migration path

3. **Phase 6 - Business Dict Elimination**
   - **Risk**: Hidden dict usages in satellite projects
   - **Mitigation**:
     - Baseline scan first (Phase 0)
     - Validate in 4+ satellite projects
     - Maintain dict support with deprecation
     - Clear migration guide

### Rollback Strategy

**Per Phase**:
- Each phase in separate PR/branch
- Tag before merge: `pre-phase-N`
- Easy revert if issues found in validation

**Emergency Rollback**:
```bash
git revert <phase-commit-sha>
make validate  # Ensure still passes
```

---

## 📈 SUCCESS METRICS

### Quantitative Metrics

1. **LOC Reduction**: -500 to -800 lines (target)
2. **Circular Imports**: 0 (from baseline count)
3. **Business Dicts**: 0 (from baseline count)
4. **TYPE_CHECKING**: 0 occurrences
5. **Test Coverage**: ≥79% maintained
6. **Duplicated isinstance**: -80% (consolidation to helpers)
7. **Methods with 6+ params**: -60% (config objects)

### Qualitative Metrics

1. **Developer Experience**:
   - Quad API usability feedback
   - Time to add new decorator/option
   - Onboarding time for new devs

2. **Maintainability**:
   - Time to fix bugs in dispatcher
   - Ease of adding new automation features
   - Code review feedback quality

3. **Ecosystem Validation**:
   - Zero breaks in flext-cli
   - Zero breaks in flext-ldap
   - Zero breaks in flext-ldif
   - Zero breaks in algar-oud-mig

### Tracking

```bash
# Baseline (Phase 0)
cloc src/ > docs/metrics/baseline.txt
python scripts/detect_cycles.py > docs/metrics/cycles_baseline.txt
python scripts/analyze_dicts.py > docs/metrics/dicts_baseline.txt

# After Each Phase
cloc src/ > docs/metrics/phase_N.txt
diff docs/metrics/baseline.txt docs/metrics/phase_N.txt

# Final Report (Phase 7)
python scripts/generate_metrics_report.py
```

---

## 📚 REFERENCES

### Internal Documentation

- [EPIC-00: Baseline & Mapping](./epics/EPIC-00-baseline.md)
- [EPIC-01: Automation Core](./epics/EPIC-01-automation-core.md)
- [EPIC-02: Models & Facade](./epics/EPIC-02-models-facade.md)
- [EPIC-03: Exceptions Quad API](./epics/EPIC-03-exceptions-quad.md)
- [EPIC-04: Decorators Unified](./epics/EPIC-04-decorators.md)
- [EPIC-05: Dispatcher Reduction](./epics/EPIC-05-dispatcher-reduction.md)
- [EPIC-06: Dict Elimination](./epics/EPIC-06-dict-elimination.md)
- [EPIC-07: Tests/Docs/Gates](./epics/EPIC-07-tests-docs-gates.md)

### External References

- [PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)
- [PEP 544 - Protocols](https://peps.python.org/pep-0544/)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

### Project Standards

- [FLEXT Development Standards](/home/marlonsc/CLAUDE.md)
- [FLEXT Core Architecture](./architecture/)
- [Quality Gates Documentation](./standards/)

---

## 🔄 MAINTENANCE

**Document Owner**: Core Team
**Review Cadence**: After each phase completion
**Update Triggers**:
- Phase completion
- Architecture decision changes
- Quality gate adjustments
- Metrics target revisions

**Last Updated**: 2025-11-20
**Next Review**: After Phase 0 completion

---

## ✍️ CHANGELOG

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-20 | 1.0 | Initial plan created from consolidated requirements | Claude |

---

**STATUS**: 🟢 Ready for Phase 0 execution
