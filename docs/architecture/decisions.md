# Architecture Decision Records (ADRs)

**Status**: Production Ready | **Version**: 0.10.0 | **Date**: 2025-12-07

Record of major architectural decisions made in FLEXT-Core development.

## ADR-001: 5-Layer Architecture

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Need clear separation of concerns and dependency management for scalable, maintainable codebase supporting 32+ dependent projects.

### Decision

Implement strict 5-layer architecture with unidirectional dependencies:

- Layer 0: Pure Constants (zero dependencies)
- Layer 0.5: Integration Bridge (external libraries)
- Layer 1: Foundation (core primitives)
- Layer 2: Domain (business logic)
- Layer 3: Application (use cases)
- Layer 4: Infrastructure (external dependencies)

### Rationale

- Clear separation enables independent testing
- Unidirectional dependencies prevent circular imports
- Scalable: Each project follows same structure
- Maintainable: Changes localized to appropriate layer
- Stable API: Layer 0 and 1 form contract

### Alternatives Considered

- Hexagonal (Ports/Adapters): Less clear dependency flow
- Microkernel: Overkill for library
- Event-Driven: Works but less deterministic

### Consequences

- ✅ Clear code organization
- ✅ Testable in isolation
- ❌ Cannot use arbitrary imports
- ❌ Requires discipline

---

## ADR-002: Railway-Oriented Programming with FlextResult[T]

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Need composable error handling without exceptions for explicit error flows.

### Decision

Implement `FlextResult[T]` monad supporting both success and failure states with monadic operations (map, flat_map, filter).

### Rationale

- Composable: Chain operations like railway tracks
- Type-safe: Compiler catches missing error handling
- Explicit: Error paths visible in code
- Functional: Pure functions without exceptions
- Legacy support: Both `.data` and `.value` work

### Alternatives Considered

- Exceptions: Implicit, implicit error paths
- Optional[T]: Only represents absence, not failures
- Result enums: Language-dependent, less composable

### Consequences

- ✅ Clear error handling
- ✅ No exception overhead
- ✅ Composable chains
- ❌ Learning curve for new developers
- ❌ Verbose in some cases

---

## ADR-003: Pydantic v2 (No v1 Legacy)

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Need modern validation framework for Python 3.13+ projects.

### Decision

Use Pydantic v2 exclusively with zero v1 legacy code (.dict(), .parse_obj() forbidden).

### Rationale

- Modern: Built for Python 3.13+
- Performance: 2-5x faster than v1
- API: Cleaner, more intuitive
- Ecosystem: v2 is future direction
- Fresh start: No migration baggage

### Alternatives Considered

- Pydantic v1: Legacy, slower, sunset path
- Dataclasses: No validation
- Attrs: Limited validation

### Consequences

- ✅ Modern, fast validation
- ✅ Clear migration path for users
- ❌ Cannot use Pydantic v1 packages
- ❌ Requires Python 3.13+

---

## ADR-004: Single Class Per Module Pattern

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Multiple top-level classes in modules causes circular dependencies and confuses API surface.

### Decision

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

### Rationale

- Clear API surface: Users know main export
- Prevents circular deps: One class per module file
- Ecosystem scale: 32+ projects depend on this
- Discoverability: Root imports only

### Alternatives Considered

- Multiple classes: Causes confusion and circular deps
- Submodules: Breaks ecosystem import pattern

### Consequences

- ✅ Clear, predictable structure
- ✅ No circular dependencies
- ❌ Less flexible organization
- ❌ Requires discipline

---

## ADR-005: Global Container Singleton Pattern

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Need centralized service management accessible from anywhere without passing through entire call chain.

### Decision

Implement `FlextContainer.get_global()` singleton for service registration and resolution.

### Rationale

- Accessibility: Available everywhere
- Simplicity: No constructor parameter threading
- Lifecycle: Centralized service management
- Testing: Can be cleared per test

### Alternatives Considered

- Constructor injection: Verbose in deep call chains
- Service locator pattern: Similar but less explicit
- Multiple containers: Complex state management

### Consequences

- ✅ Easy service access
- ✅ Centralized lifecycle
- ❌ Global state (testability requires cleanup)
- ❌ Can hide dependencies

---

## ADR-006: Ecosystem API Stability

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

32+ dependent projects need predictable, stable APIs to avoid cascading breaking changes.

### Decision

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
FlextResult[T].value  # AND .data
```

### Rationale

- Ecosystem resilience: Dependent projects stay compatible
- Versioning strategy: Semantic versioning with guarantees
- User confidence: Upgrade safely
- Long-term support: Stable foundation

### Alternatives Considered

- Breaking changes allowed: Breaks ecosystem
- No stability guarantees: Unpredictable maintenance

### Consequences

- ✅ Stable foundation
- ✅ Ecosystem resilience
- ❌ Limited evolution
- ❌ Deprecation cycles needed

---

## ADR-007: Type Safety: No `Any` Type

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Need complete type safety without escape hatches that reduce code reliability.

### Decision

**Zero `Any` type usage** across flext-core. All code must have complete type annotations.

### Rationale

- Compiler checks: Catches errors early
- Documentation: Types document intent
- Refactoring: Safe renaming and changes
- IDE support: Better autocomplete and hints
- Example: Sets standard for ecosystem

### Alternatives Considered

- Allow `Any`: Reduces type safety benefit
- Partial typing: Still has gaps

### Consequences

- ✅ Complete type safety
- ✅ Excellent IDE support
- ❌ More complex types initially
- ❌ Learning curve for developers

---

## ADR-008: Clean Layered Dependencies Only (No Shortcuts)

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Developers want to take shortcuts across layers to "just make it work" causing tight coupling.

### Decision

**Enforce strict layer boundaries** - can only import from lower layers. Violations cause immediate code review rejection.

### Rationale

- Maintainability: Decoupled layers stay decoupled
- Testability: Mock external layers easily
- Scalability: Add new features without refactoring
- Quality: Prevents technical debt

### Alternatives Considered

- Flexible boundaries: Enables shortcuts, creates debt
- Enforced at runtime: Too late, already broken

### Consequences

- ✅ Clean architecture maintained
- ✅ No circular dependencies
- ❌ Requires discipline
- ❌ Cannot take shortcuts

---

## ADR-009: Python 3.13+ Exclusive

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Support multiple Python versions increases maintenance burden and limits modern features.

### Decision

**Python 3.13+ ONLY**. Do not support older versions.

Features enabled:

- New syntax features
- Performance improvements
- Modern type hints (PEP 696, etc.)
- Latest standard library

### Rationale

- Maintenance: Focus on current Python
- Performance: Use latest optimizations
- Ecosystem: Modern dependencies also target 3.13+
- Features: Access newest Python capabilities

### Alternatives Considered

- Support 3.10+: Maintenance burden
- Support 3.12: Still missing features

### Consequences

- ✅ Modern features available
- ✅ Reduced maintenance
- ❌ Cannot use on older Python
- ❌ Limits some user adoption

---

## ADR-010: Domain Events Over Direct Calls

**Status:** ACCEPTED | **Date:** 2025-01-XX

### Problem

Need decoupled communication between domain objects without direct references.

### Decision

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

### Rationale

- Decoupling: Services don't know about each other
- Async-friendly: Events can be processed asynchronously
- Audit trail: Record all important business events
- Scalability: Easy to add subscribers

### Alternatives Considered

- Direct calls: Creates coupling
- Callbacks: Less explicit than events

### Consequences

- ✅ Decoupled services
- ✅ Async processing capability
- ❌ Complexity in event orchestration
- ❌ Debugging trace requires event log

---

## Decision Making Process

For new ADRs:

1. **Problem**: What's the issue?
2. **Decision**: What are we choosing?
3. **Rationale**: Why this choice?
4. **Alternatives**: What else could work?
5. **Consequences**: Trade-offs?
6. **Status**: Proposed/Accepted/Deprecated

## Obsoleted Decisions

- ADR-XXXX (Date): Reason for change and replacement

(None currently)

## Future Decisions Needed

- API versioning strategy for major versions
- Multi-tenancy support patterns
- Distributed tracing standards
- Performance benchmarking targets

## Next Steps

1. **Architecture Overview**: See [Architecture Overview](./overview.md) for layer structure
2. **Clean Architecture**: Review [Clean Architecture](./clean-architecture.md) for dependency rules
3. **CQRS Patterns**: Explore [CQRS Architecture](./cqrs.md) for implementation details
4. **Architecture Patterns**: Check [Architecture Patterns](./patterns.md) for common patterns
5. **Implementation Guides**: Review [Getting Started](../guides/getting-started.md) for practical usage

## See Also

- [Architecture Overview](./overview.md) - Visual layer topology
- [Clean Architecture](./clean-architecture.md) - Dependency rules and layer responsibilities
- [CQRS Architecture](./cqrs.md) - Handler and dispatcher implementation
- [Architecture Patterns](./patterns.md) - Common implementation patterns
- [Getting Started Guide](../guides/getting-started.md) - Practical implementation
- **FLEXT CLAUDE.md**: Architecture principles and development workflow

## Viewing Decisions

Each ADR should be referenced when:

- Creating new features
- Making architectural changes
- Reviewing pull requests
- Training new developers

This ensures consistency and prevents re-deciding settled issues.
