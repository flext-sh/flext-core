# FLEXT Core - Base Module Hierarchy

Validated against current source tree in `src/flext_core/`.

## Purpose

This document captures the layering and import rules for FLEXT Core's
"base" modules and surrounding layers, reflecting the actual modules in
this repository today.

## Layered Structure (Reality-Based)

### Level 0: Foundation (no dependencies)

- `constants.py` — Core enums and constants
- `__version__.py` — Version metadata and compatibility utilities

### Level 1: Railway Pattern

- `result.py` — FlextResult[T] (central error-handling pattern)

### Level 2: Dependency Injection

- `container.py` — FlextContainer and container utilities

### Level 3: Configuration

- `config.py` — FlextConfig and config management
- `config_base.py` — Base configuration building blocks
- `config_models.py` — Shared configuration models

### Level 4: Domain Model Layer

- `models.py` — Shared model primitives
- `entities.py` — FlextEntity
- `value_objects.py` — FlextValue
- `aggregate_root.py` — FlextAggregates
- `domain_services.py` — Domain service patterns

### Level 5: Architectural Patterns

- `commands.py` — FlextCommands namespace
- `handlers.py` — Handler patterns
- `handlers_base.py` — Base handler support
- `validation.py` — Validation system
- `validation_base.py` — Base validation support
- `protocols.py` — Protocol/interface definitions
- `guards.py` — Guard helpers and validators

### Level 6: Cross-Cutting Concerns

- `loggings.py` — Structured logging
- `decorators.py` — Decorator utilities
- `mixins.py` — Reusable mixins
- `fields.py` — Field metadata
- `utilities.py` — Utility helpers
- `observability.py` — Observability helpers
- `schema_processing.py` — Schema processing
- `payload.py` — Messaging primitives

### Level 7: Type System

- `typings.py` — Hierarchical types and protocols
- `types.py` — Thin compatibility surface (re-exports)

### Level 8: Base Implementations (Internal building blocks)

- `base_commands.py`
- `base_decorators.py`
- `base_exceptions.py`
- `base_handlers.py`
- `base_mixins.py`
- `base_testing.py`
- `base_utilities.py`
- `delegation_system.py`
- `legacy.py`

### Level 9: Integration / Composition

- `core.py` — Composition utilities for higher-level use
- `context.py` — Context management patterns
- `singer_base.py` — Singer-related base helpers
- `testing_utilities.py` — Testing helpers

## Import Rules (Enforced by Convention)

1. Lower levels must not import from higher levels.
2. Base modules (Level 8) are internal building blocks; avoid importing
   them from application code. Prefer public, higher-level modules.
3. Keep domain/business logic free of infrastructure concerns.
4. Favor `FlextResult` return types over exceptions in business logic.

## Rationale

- The order above mirrors real dependencies observed in the source code.
- Clean separation makes it easier to evolve advanced patterns (CQRS,
  plugins, event sourcing) without breaking the foundation.

## Status

- This hierarchy reflects the CURRENT repository contents and naming.
- Private legacy `_..._base.py` modules referenced in older docs no
  longer exist; the modern equivalents are the `base_*.py` modules
  listed under Level 8.
