# Architecture Decision Records (ADRs)

<!-- TOC START -->

- [ADR-001: 5-Layer Architecture](#adr-001-5-layer-architecture)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-002: Railway-Oriented Programming with FlextResult[T]](#adr-002-railway-oriented-programming-with-flextresultt)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-003: Pydantic v2 (No v1 Legacy)](#adr-003-pydantic-v2-no-v1-legacy)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-004: Single Class Per Module Pattern](#adr-004-single-class-per-module-pattern)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-005: Global Container Singleton Pattern](#adr-005-global-container-singleton-pattern)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-006: Ecosystem API Stability](#adr-006-ecosystem-api-stability)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-007: Type Safety: No `Any` Type](#adr-007-type-safety-no-any-type)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-008: Clean Layered Dependencies Only (No Shortcuts)](#adr-008-clean-layered-dependencies-only-no-shortcuts)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-009: Python 3.13+ Exclusive](#adr-009-python-313-exclusive)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [ADR-010: Domain Events Over Direct Calls](#adr-010-domain-events-over-direct-calls)
  - [Problem](#problem)
  - [Decision](#decision)
  - [Rationale](#rationale)
  - [Alternatives Considered](#alternatives-considered)
  - [Consequences](#consequences)
- [Decision Making Process](#decision-making-process)
- [Obsoleted Decisions](#obsoleted-decisions)
- [Future Decisions Needed](#future-decisions-needed)
- [Next Steps](#next-steps)
- [See Also](#see-also)
- [Viewing Decisions](#viewing-decisions)

<!-- TOC END -->

**Status**: Production Ready | **Version**: 0.10.0 | **Date**: 2025-12-07

Record of major architectural decisions made in FLEXT-Core development.

## ADR-001: 5-Layer Architecture

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-001

Need clear separation of concerns and dependency management for scalable, maintainable codebase supporting 32+ dependent projects.

### Decision - ADR-001

Implement strict 5-layer architecture with unidirectional dependencies:

- Layer 0: Pure Constants (zero dependencies)
- Layer 0.5: Integration Bridge (external libraries)
- Layer 1: Foundation (core primitives)
- Layer 2: Domain (business logic)
- Layer 3: Application (use cases)
- Layer 4: Infrastructure (external dependencies)

### Rationale - ADR-001

- Clear separation enables independent testing
- Unidirectional dependencies prevent circular imports
- Scalable: Each project follows same structure
- Maintainable: Changes localized to appropriate layer
- Stable API: Layer 0 and 1 form contract

### Alternatives Considered - ADR-001

- Hexagonal (Ports/Adapters): Less clear dependency flow
- Microkernel: Overkill for library
- Event-Driven: Works but less deterministic

### Consequences - ADR-001

- ✅ Clear code organization
- ✅ Testable in isolation
- ❌ Cannot use arbitrary imports
- ❌ Requires discipline

______________________________________________________________________

## ADR-002: Railway-Oriented Programming with FlextResult[T]

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-002

Need composable error handling without exceptions for explicit error flows.

### Decision - ADR-002

Implement `FlextResult[T]` monad supporting both success and failure states with monadic operations (map, flat_map, filter).

### Rationale - ADR-002

- Composable: Chain operations like railway tracks
- Type-safe: Compiler catches missing error handling
- Explicit: Error paths visible in code
- Functional: Pure functions without exceptions
- Legacy support: Both `.data` and `.value` work

### Alternatives Considered - ADR-002

- Exceptions: Implicit, implicit error paths
- Optional\[T\]: Only represents absence, not failures
- Result enums: Language-dependent, less composable

### Consequences - ADR-002

- ✅ Clear error handling
- ✅ No exception overhead
- ✅ Composable chains
- ❌ Learning curve for new developers
- ❌ Verbose in some cases

______________________________________________________________________

## ADR-003: Pydantic v2 (No v1 Legacy)

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-003

Need modern validation framework for Python 3.13+ projects.

### Decision - ADR-003

Use Pydantic v2 exclusively with zero v1 legacy code (.dict(), .parse_obj() forbidden).

### Rationale - ADR-003

- Modern: Built for Python 3.13+
- Performance: 2-5x faster than v1
- API: Cleaner, more intuitive
- Ecosystem: v2 is future direction
- Fresh start: No migration baggage

### Alternatives Considered - ADR-003

- Pydantic v1: Legacy, slower, sunset path
- Dataclasses: No validation
- Attrs: Limited validation

### Consequences - ADR-003

- ✅ Modern, fast validation
- ✅ Clear migration path for users
- ❌ Cannot use Pydantic v1 packages
- ❌ Requires Python 3.13+

______________________________________________________________________

## ADR-004: Single Class Per Module Pattern

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-004

Multiple top-level classes in modules causes circular dependencies and confuses API surface.

### Decision - ADR-004

ONE public class per module with `Flext` prefix. Nested helpers allowed inside main class.

```python
# ✅ CORRECT
class FlextResult:
    class _Implementation:  # Nested helper OK
        pass

# ❌ FORBIDDEN
class FlextResult:
    pass

class FlextContainer:  # Second top-level class
    pass
```

### Rationale - ADR-004

- Clear API surface: Users know main export
- Prevents circular deps: One class per module file
- Ecosystem scale: 32+ projects depend on this
- Discoverability: Root imports only

### Alternatives Considered - ADR-004

- Multiple classes: Causes confusion and circular deps
- Submodules: Breaks ecosystem import pattern

### Consequences - ADR-004

- ✅ Clear, predictable structure
- ✅ No circular dependencies
- ❌ Less flexible organization
- ❌ Requires discipline

______________________________________________________________________

## ADR-005: Global Container Singleton Pattern

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-005

Need centralized service management accessible from anywhere without passing through entire call chain.

### Decision - ADR-005

Implement `FlextContainer.get_global()` singleton for service registration and resolution.

### Rationale - ADR-005

- Accessibility: Available everywhere
- Simplicity: No constructor parameter threading
- Lifecycle: Centralized service management
- Testing: Can be cleared per test

### Alternatives Considered - ADR-005

- Constructor injection: Verbose in deep call chains
- Service locator pattern: Similar but less explicit
- Multiple containers: Complex state management

### Consequences - ADR-005

- ✅ Easy service access
- ✅ Centralized lifecycle
- ❌ Global state (testability requires cleanup)
- ❌ Can hide dependencies

______________________________________________________________________

## ADR-006: Ecosystem API Stability

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-006

32+ dependent projects need predictable, stable APIs to avoid cascading breaking changes.

### Decision - ADR-006

- Layer 0-1: NEVER break (core APIs guaranteed)
- Layer 2-3: Evolve carefully (deprecation cycles)
- Layer 4: Internal (can change)

Layer 1 contracts example:

```python
# GUARANTEED in 1.x
FlextResult[T].ok(value)
FlextResult[T].fail(error)
FlextResult[T].value
FlextResult[T].is_success
FlextResult[T].value  # `.data` remains available as a legacy alias
```

### Rationale - ADR-006

- Ecosystem resilience: Dependent projects stay compatible
- Versioning strategy: Semantic versioning with guarantees
- User confidence: Upgrade safely
- Long-term support: Stable foundation

### Alternatives Considered - ADR-006

- Breaking changes allowed: Breaks ecosystem
- No stability guarantees: Unpredictable maintenance

### Consequences - ADR-006

- ✅ Stable foundation
- ✅ Ecosystem resilience
- ❌ Limited evolution
- ❌ Deprecation cycles needed

______________________________________________________________________

## ADR-007: Type Safety: No `Any` Type

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-007

Need complete type safety without escape hatches that reduce code reliability.

### Decision - ADR-007

**Zero `Any` type usage** across flext-core. All code must have complete type annotations.

### Rationale - ADR-007

- Compiler checks: Catches errors early
- Documentation: Types document intent
- Refactoring: Safe renaming and changes
- IDE support: Better autocomplete and hints
- Example: Sets standard for ecosystem

### Alternatives Considered - ADR-007

- Allow `Any`: Reduces type safety benefit
- Partial typing: Still has gaps

### Consequences - ADR-007

- ✅ Complete type safety
- ✅ Excellent IDE support
- ❌ More complex types initially
- ❌ Learning curve for developers

______________________________________________________________________

## ADR-008: Clean Layered Dependencies Only (No Shortcuts)

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-008

Developers want to take shortcuts across layers to "just make it work" causing tight coupling.

### Decision - ADR-008

**Enforce strict layer boundaries** - can only import from lower layers. Violations cause immediate code review rejection.

### Rationale - ADR-008

- Maintainability: Decoupled layers stay decoupled
- Testability: Mock external layers easily
- Scalability: Add new features without refactoring
- Quality: Prevents technical debt

### Alternatives Considered - ADR-008

- Flexible boundaries: Enables shortcuts, creates debt
- Enforced at runtime: Too late, already broken

### Consequences - ADR-008

- ✅ Clean architecture maintained
- ✅ No circular dependencies
- ❌ Requires discipline
- ❌ Cannot take shortcuts

______________________________________________________________________

## ADR-009: Python 3.13+ Exclusive

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-009

Support multiple Python versions increases maintenance burden and limits modern features.

### Decision - ADR-009

**Python 3.13+ ONLY**. Do not support older versions.

Features enabled:

- New syntax features
- Performance improvements
- Modern type hints (PEP 696, etc.)
- Latest standard library

### Rationale - ADR-009

- Maintenance: Focus on current Python
- Performance: Use latest optimizations
- Ecosystem: Modern dependencies also target 3.13+
- Features: Access newest Python capabilities

### Alternatives Considered - ADR-009

- Support 3.10+: Maintenance burden
- Support 3.12: Still missing features

### Consequences - ADR-009

- ✅ Modern features available
- ✅ Reduced maintenance
- ❌ Cannot use on older Python
- ❌ Limits some user adoption

______________________________________________________________________

## ADR-010: Domain Events Over Direct Calls

**Status:** ACCEPTED | **Date:** 2025-12-07

### Problem - ADR-010

Need decoupled communication between domain objects without direct references.

### Decision - ADR-010

Use domain events (emitted from aggregates, published by infrastructure) for cross-aggregate communication.

```python
class OrderService(FlextService):
    def place_order(self, order: Order) -> FlextResult[Order]:
        # Business logic
        order.place()
        # Emit event for subscribers
        self.add_domain_event(OrderPlacedEvent(order.entity_id))
        return FlextResult[Order].ok(order)
```

### Rationale - ADR-010

- Decoupling: Services don't know about each other
- Async-friendly: Events can be processed asynchronously
- Audit trail: Record all important business events
- Scalability: Easy to add subscribers

### Alternatives Considered - ADR-010

- Direct calls: Creates coupling
- Callbacks: Less explicit than events

### Consequences - ADR-010

- ✅ Decoupled services
- ✅ Async processing capability
- ❌ Complexity in event orchestration
- ❌ Debugging trace requires event log

______________________________________________________________________

## Decision Making Process

For new ADRs:

1. **Problem**: What's the issue?
1. **Decision**: What are we choosing?
1. **Rationale**: Why this choice?
1. **Alternatives**: What else could work?
1. **Consequences**: Trade-offs?
1. **Status**: Proposed/Accepted/Deprecated

## Obsoleted Decisions

- ADR-XXXX (Date): Reason for change and replacement

(None currently)

## Future Decisions Needed

- API versioning strategy for major versions
- Multi-tenancy support patterns
- Distributed tracing standards
- Performance benchmarking targets

## Next Steps

1. **Architecture Overview**: See Architecture Overview for layer structure
1. **Clean Architecture**: Review Clean Architecture for dependency rules
1. **CQRS Patterns**: Explore CQRS Architecture for implementation details
1. **Architecture Patterns**: Check Architecture Patterns for common patterns
1. **Implementation Guides**: Review Getting Started for practical usage

## See Also

- Architecture Overview - Visual layer topology
- Clean Architecture - Dependency rules and layer responsibilities
- CQRS Architecture - Handler and dispatcher implementation
- Architecture Patterns - Common implementation patterns
- Getting Started Guide - Practical implementation
- **FLEXT CLAUDE.md**: Architecture principles and development workflow

## Viewing Decisions

Each ADR should be referenced when:

- Creating new features
- Making architectural changes
- Reviewing pull requests
- Training new developers

This ensures consistency and prevents re-deciding settled issues.
