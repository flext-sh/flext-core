# FLEXT-Core Examples

<!-- TOC START -->

- [Quick Start](#quick-start)
- [Example Categories](#example-categories)
  - [🚀 Foundation & Core Patterns](#-foundation-core-patterns)
  - [🔧 Advanced Patterns & Utilities](#-advanced-patterns-utilities)
  - [🏗️ Application Layer & Integration](#-application-layer-integration)
- [Example Guide Mapping](#example-guide-mapping)
- [Expected Output](#expected-output)
- [Requirements](#requirements)
- [Next Steps](#next-steps)
- [See Also](#see-also)

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

**Version**: 0.12.0-dev | **Status**: Production Ready

Comprehensive examples demonstrating FLEXT-Core patterns and best practices. All examples are self-contained and can be run independently.

## Quick Start

```bash
# Install FLEXT-Core in development mode
pip install -e .

# Run an example
python examples/01_basic_result.py

# Or from project root
cd /path/to/flext-core
python examples/01_basic_result.py
```

## Example Categories

### 🚀 Foundation & Core Patterns

**Basic Concepts**:

- **00_single_import_demo.py** — Minimal import verification helper. Demonstrates basic package import.
- **01_basic_result.py** — Railway-oriented `r` patterns (map, flat_map, fail paths). **Start here for ROP basics**.

**Dependency Injection & Configuration**:

- **02_dependency_injection.py** — `FlextContainer` usage, service registration, and logger resolution.
- **04_config_basics.py** — `FlextSettings` settings loading, validation, and environment-specific configurations.

**Domain Modeling**:

- **03_models_basics.py** — Entities, Values, and AggregateRoot basics with `FlextModels`. Domain-driven design fundamentals.

### 🔧 Advanced Patterns & Utilities

**Context & Utilities**:

- **09_context_management.py** — `FlextContext` request/user/operation scopes and correlation ID propagation.
- **12_utilities_comprehensive.py** — Validation, type guards, and helper utilities from `_utilities` module.
- **logging_config_once_pattern.py** — Idempotent logging configuration helper pattern.

**Decorators & Automation**:

- **05_utilities_advanced.py** — Advanced utility patterns and helper functions.
- **06_decorators_complete.py** — Complete decorator showcase: `@inject`, `@log_operation`, `@railway`, `@with_context`, `@combined`.

### 🏗️ Application Layer & Integration

**Handlers & Dispatchers**:

- **07_registry_dispatcher.py** — `FlextRegistry` and `FlextDispatcher` patterns for CQRS command/query routing.
- **14_flext_handlers_complete.py** — Handler base class, validation hooks, and dispatcher-style execution.

**Advanced Processing**:

- **08_integration_complete.py** — Complete integration example combining all FLEXT-Core patterns.
- **15_automation_showcase.py** — Context enrichment helpers and tracing-friendly execution wrappers.
- **16_layer3_advanced_processing.py** — Dispatcher reliability patterns (timeouts, retries, caching, circuit breakers).

## Example Guide Mapping

| Example File                 | Related Guide                                                                                                                          |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `01_basic_result.py`         | [Railway-Oriented Programming](../docs/guides/railway-oriented-programming.md)                                                         |
| `02_dependency_injection.py` | [Advanced Dependency Injection](../docs/guides/dependency-injection-advanced.md)                                                       |
| `03_models_basics.py`        | [Domain-Driven Design](../docs/guides/domain-driven-design.md)                                                                         |
| `04_settings_basics.py`      | [Settings Management](../docs/guides/settings.md)                                                                                      |
| `06_decorators_complete.py`  | [Railway-Oriented Programming - Decorators](../docs/guides/railway-oriented-programming.md#decorator-integration)                     |
| `07_registry_dispatcher.py`  | [Advanced Dependency Injection - Dispatcher](../docs/guides/dependency-injection-advanced.md#flextdispatcher-reliability-settings)     |
| `09_context_management.py`   | [Advanced Dependency Injection - Context](../docs/guides/dependency-injection-advanced.md)                                             |

## Expected Output

Each example produces structured output demonstrating the patterns. Examples include:

- ✅ Success indicators for successful operations
- ❌ Error indicators for failure cases
- 🔥 Exception handling demonstrations
- 📊 Metrics and performance tracking
- 🔗 Context propagation examples

## Requirements

- **Python**: 3.13+ (required)
- **Installation**: `pip install -e .` from project root
- **Dependencies**: All dependencies installed via `poetry install` or `pip install -e .`

## Next Steps

After running examples:

1. **Read Guides**: Explore [Documentation Guides](../docs/guides/) for detailed explanations
1. **API Reference**: Check [API Reference](../docs/api-reference/) for complete API documentation
1. **Patterns**: Review [Service Patterns](../docs/guides/service-patterns.md) for production patterns

## See Also

- [Getting Started Guide](../docs/guides/getting-started.md) - Quick start with FLEXT-Core
- [Documentation Index](../docs/INDEX.md) - Complete documentation navigation
- [API Reference](../docs/api-reference/) - Full API documentation
