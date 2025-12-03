# FLEXT-Core Examples

Run examples with `python examples/<file>.py` from the project root (after `pip install -e .`).

## Foundation & Setup

- **00_single_import_demo.py** — Minimal import/verification helper.
- **01_basic_result.py** — Railway-oriented `FlextResult` patterns (map/and_then/fail paths).
- **02_dependency_injection.py** — `FlextContainer` usage and logger resolution.
- **03_models_basics.py** — Entities/Values/AggregateRoot basics with `FlextModels`.
- **04_config_basics.py** — `FlextConfig` settings loading and validation.

## Context, Utilities, and Logging

- **09_context_management.py** — `FlextContext` request/user/operation scopes.
- **12_utilities_comprehensive.py** — Validation/type-guard helpers from `_utilities`.
- **logging_config_once_pattern.py** — Idempotent logging configuration helper.

## Application-Layer Patterns

- **14_flext_handlers_complete.py** — Handler base class, validation hooks, and dispatcher-style execution.
- **16_layer3_advanced_processing.py** — Dispatcher reliability knobs (timeouts, retries, caching) in action.

## Cross-Cutting Automation

- **15_automation_showcase.py** — Context enrichment helpers and tracing-friendly execution wrappers.

> Each script is self-contained; activate the project's virtual environment or install in editable mode before running.
