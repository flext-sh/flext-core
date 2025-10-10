# FLEXT-Core Examples - Complete API Demonstrations

**Version**: 0.9.9 RC | **Coverage**: 100% of Public APIs | **Status**: Production Ready | **Phase 1**: Context Enrichment Completed

This directory contains comprehensive examples demonstrating ALL capabilities of the FLEXT-Core foundation library. Each example showcases complete API usage with best practices and deprecation warnings for anti-patterns.

## 📚 Example Organization

### Foundation Examples (01-05)

These examples demonstrate the core building blocks of the FLEXT ecosystem:

#### 01_basic_result.py - FlextCore.Result Railway Pattern

- **Purpose**: Complete FlextCore.Result[T] API demonstration
- **Concepts**: Railway-oriented programming, error handling without exceptions
- **Key Methods**: `ok()`, `fail()`, `map()`, `flat_map()`, `filter()`, `recover()`, `tap()`, `zip_with()`, `traverse()`
- **Run**: `python examples/01_basic_result.py`

#### 02_dependency_injection.py - FlextCore.Container DI

- **Purpose**: Complete dependency injection and service management
- **Concepts**: Service registration, resolution, auto-wiring, lifecycle management
- **Key Methods**: `register()`, `get()`, `register_factory()`, `register_singleton()`, `auto_wire()`, `batch_register()`
- **Run**: `python examples/02_dependency_injection.py`

#### 03_models_basics.py - FlextCore.Models DDD Patterns

- **Purpose**: Domain-driven design with entities, values, and aggregates
- **Concepts**: Value Objects, Entities, Aggregate Roots, Domain Events, Business Logic
- **Key Classes**: `FlextCore.Models.Value`, `FlextCore.Models.Entity`, `FlextCore.Models.AggregateRoot`
- **Run**: `python examples/03_models_basics.py`

#### 04_config_basics.py - FlextCore.Config Management

- **Purpose**: Environment-aware configuration with Pydantic Settings
- **Concepts**: Global singleton, environment detection, all configuration domains
- **Key Features**: Database, cache, API, logging, performance, CQRS settings
- **Run**: `python examples/04_config_basics.py`

#### 05_logging_basics.py - FlextCore.Logger Structured Logging

- **Purpose**: Complete structured logging with context management
- **Concepts**: Log levels, context binding, correlation tracking, performance metrics
- **Key Methods**: `bind()`, `unbind()`, `contextualize()`, `configure()`, child loggers
- **Run**: `python examples/05_logging_basics.py`

### Intermediate Examples (06-07)

These examples show specialized patterns and processing:

#### 06_messaging_patterns.py - Payload & Events

- **Purpose**: Message passing and domain event patterns
- **Concepts**: Generic payloads, domain events, message routing, correlation
- **Key Classes**: `FlextCore.Models.Payload[T]`, `FlextCore.Models.DomainEvent`
- **Run**: `python examples/06_messaging_patterns.py`

#### 07_processing_handlers.py - FlextCore.Processors Patterns

- **Purpose**: Handler pipelines and strategy patterns
- **Concepts**: Chain of responsibility, strategy pattern, registry, error recovery
- **Key Classes**: `FlextCore.Processors.Implementation.BasicHandler`
- **Run**: `python examples/07_processing_handlers.py`

### Integration Example (08)

#### 08_integration_complete.py - Complete E-Commerce System

- **Purpose**: All FLEXT components working together in a real-world scenario
- **Concepts**: Order processing with DDD, DI, handlers, events, and logging
- **Integration**: Demonstrates how all components interact seamlessly
- **Run**: `python examples/08_integration_complete.py`

### Phase 1 Examples (09+)

#### 15_automation_showcase.py - Phase 1 Context Enrichment

- **Purpose**: Complete demonstration of Phase 1 context enrichment capabilities
- **Concepts**: Zero-boilerplate context management, distributed tracing, audit trails
- **Key Methods**: `_with_correlation_id()`, `_with_user_context()`, `_with_operation_context()`, `execute_with_context_enrichment()`
- **Run**: `python examples/15_automation_showcase.py`

## 🚀 Running the Examples

### Prerequisites

1. Ensure flext-core is installed:

   ```bash
   cd flext-core
   pip install -e .
   ```

2. Or use the project's virtual environment:

   ```bash
   source .venv/bin/activate  # or equivalent for your shell
   ```

### Running Individual Examples

Each example is self-contained and can be run directly:

```bash
# Foundation patterns
python examples/01_basic_result.py
python examples/02_dependency_injection.py
python examples/03_models_basics.py
python examples/04_config_basics.py
python examples/05_logging_basics.py

# Intermediate patterns
python examples/06_messaging_patterns.py
python examples/07_processing_handlers.py

# Complete integration
python examples/08_integration_complete.py

# Phase 1 context enrichment
python examples/15_automation_showcase.py
```

### Running All Examples

```bash
# Run all examples in sequence
for f in examples/[0-9]*.py; do
    echo "=== Running $(basename $f) ==="
    python "$f"
    echo
done
```

## 📖 Learning Path

### For Beginners

1. Start with **01_basic_result.py** to understand railway-oriented error handling
2. Move to **02_dependency_injection.py** for service management
3. Study **03_models_basics.py** for domain modeling patterns
4. Review **04_config_basics.py** and **05_logging_basics.py** for infrastructure

### For Intermediate Users

1. Study **06_messaging_patterns.py** for event-driven patterns
2. Explore **07_processing_handlers.py** for processing pipelines
3. Dive into **08_integration_complete.py** to see everything working together

### For Advanced Users

1. Review **08_integration_complete.py** for architectural patterns
2. Study **15_automation_showcase.py** for Phase 1 context enrichment patterns
3. Study the deprecation warnings in each example for anti-patterns
4. Use these examples as templates for your own FLEXT applications

## 🏆 Best Practices Demonstrated

### ✅ DO (Shown in Examples)

- Use `FlextCore.Result[T]` for all error handling (no exceptions in business logic)
- Access configuration via `FlextCore.Config()` (direct instantiation)
- Use `FlextCore.Container.get_global()` for dependency injection
- Model domains with `FlextCore.Models.Entity/Value/AggregateRoot`
- Structure logs with `FlextCore.Logger(__name__)` and context binding
- Chain operations with railway pattern (`.flat_map()`, `.map()`)
- Use type hints everywhere for type safety

### ❌ DON'T (Shown as Deprecation Warnings)

- Don't use try/except for business logic (use FlextCore.Result)
- Don't use print() for logging (use FlextCore.Logger)
- Don't hard-code configuration (use FlextCore.Config)
- Don't use global variables (use FlextCore.Container)
- Don't mix infrastructure with domain logic (use DDD patterns)
- Don't use mutable default arguments (use Pydantic models)
- Don't ignore type hints (enable MyPy strict mode)

## 🧪 Testing the Examples

### Quality Gates

All examples pass the project's quality gates:

```bash
# Lint check
ruff check examples/

# Type check
mypy examples/ --strict

# Format check
black --check examples/
```

### Example Tests

The examples themselves serve as integration tests for the FLEXT Core API:

```bash
# Test that all examples run without errors
pytest tests/test_examples.py -v
```

## 📚 Additional Resources

### Support Files

- **shared_example_strategies.py** - Shared utilities for examples (not meant to be run directly)
- **.bak/** - Archived redundant examples for reference

### Documentation

- [FLEXT Core README](../README.md) - Main project documentation
- [FLEXT Core API Reference](../docs/api/) - Detailed API documentation
- [FLEXT Workspace Standards](../../CLAUDE.md) - Development standards

## 🎯 Key Takeaways

1. **FlextCore.Result eliminates exceptions** - All operations return results that can be composed
2. **FlextCore.Container manages dependencies** - Type-safe service injection without magic
3. **FlextCore.Models enforce business rules** - Domain logic lives in the models
4. **FlextCore.Config centralizes settings** - One source of truth for configuration
5. **FlextCore.Logger provides structure** - Context-aware logging with correlation
6. **Everything composes** - All patterns work together seamlessly

## 🔄 Version Compatibility

These examples are compatible with:

- **flext-core**: v0.9.9 RC (preparing for 1.0.0 stable)
- **Python**: 3.13+
- **Pydantic**: v2.0+
- **Test Coverage**: 80% (1,143 tests passing, 92 failures)
- **Phase 1**: Context enrichment completed

## 📝 Contributing

When adding new examples:

1. Follow the numbered naming convention (XX_description.py)
2. Include comprehensive docstrings
3. Demonstrate ALL methods of the component being showcased
4. Add deprecation warnings for anti-patterns
5. Ensure the example passes all quality gates
6. Update this README with the new example

---

_These examples represent the complete API surface and best practices of FLEXT Core v0.9.9 RC, serving as both learning materials and integration tests for the foundation library preparing for its 1.0.0 stable release. With 80% test coverage, 1,143 passing tests, and Phase 1 context enrichment completed, the foundation is solid for the upcoming 1.0.0 release._
