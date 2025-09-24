# Source Layout

This directory implements the public API exported by `flext_core`. The structure mirrors the 1.0.0 modernization pillars – dispatcher unification, context-first observability, and aligned configuration/domain services.

---

## Foundations

| Module         | Responsibility                                        |
| -------------- | ----------------------------------------------------- |
| `result.py`    | `FlextResult` railway type and helpers                |
| `typings.py`   | Shared generics and aliases (no runtime dependencies) |
| `constants.py` | Canonical constants used across modules               |
| `version.py`   | Version helpers and release metadata                  |

## Runtime Surfaces

| Module         | Responsibility                                            |
| -------------- | --------------------------------------------------------- |
| `container.py` | Global dependency container with typed service keys       |
| `service.py`   | Base class for domain services (immutable, context-aware) |
| `models.py`    | Entities, values, aggregates, plus helper models          |
| `utilities.py` | Validation, ID generation, retry helpers                  |

## Execution Flow

| Module          | Responsibility                                     |
| --------------- | -------------------------------------------------- |
| `bus.py`        | Core command bus implementation                    |
| `dispatcher.py` | Dispatcher façade orchestrating bus + context      |
| `registry.py`   | Batch registration helpers for downstream packages |
| `handlers.py`   | Base classes for command/query handlers            |
| `processing.py` | Lightweight processing utilities and registries    |
| `cqrs.py`       | CQRS adapters built on top of the bus/handlers     |

## Context & Observability

| Module        | Responsibility                                                  |
| ------------- | --------------------------------------------------------------- |
| `context.py`  | Hierarchical context (correlation, request, performance scopes) |
| `loggings.py` | Structured logging integrated with `FlextContext`               |
| `mixins.py`   | Serialization/logging/timestamp mixins reused across modules    |

## Configuration

| Module      | Responsibility                                     |
| ----------- | -------------------------------------------------- |
| `config.py` | `FlextConfig` base with `.env`, YAML, TOML support |

---

All exported symbols are surfaced through `__init__.py` and protected by integration tests. When adding new modules or exports, update this document and the modernization plan notes accordingly.
