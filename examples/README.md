# FLEXT Core Examples

Comprehensive collection of working examples demonstrating FLEXT Core patterns and best practices.

## Overview

This directory contains 20+ practical examples showcasing FLEXT Core's capabilities, from basic patterns to advanced enterprise architectures. Each example is self-contained, fully documented, and demonstrates real-world usage scenarios.

## Example Structure

### Core Patterns (01-04)
**Foundation patterns every FLEXT application uses**

| File | Description | Key Concepts |
|------|-------------|-------------|
| `01_flext_result_railway_pattern.py` | Railway-oriented programming | Error handling, chaining, composition |
| `02_flext_container_dependency_injection.py` | Dependency injection patterns | Service registration, resolution, lifecycle |
| `03_flext_commands_cqrs_pattern.py` | CQRS implementation | Commands, queries, handlers |
| `04_flext_utilities_modular.py` | Utility functions | ID generation, helpers, common operations |

### Domain-Driven Design (05-08)
**Building rich domain models**

| File | Description | Key Concepts |
|------|-------------|-------------|
| `05_flext_validation_advanced_system.py` | Validation patterns | Business rules, constraints, validators |
| `06_flext_entity_valueobject_ddd_patterns.py` | DDD building blocks | Entities, value objects, aggregates |
| `07_flext_mixins_multiple_inheritance.py` | Behavior composition | Mixins, traits, composition |
| `08_flext_config_enterprise_configuration.py` | Configuration management | Settings, environments, validation |

### Enterprise Features (09-14)
**Production-ready patterns**

| File | Description | Key Concepts |
|------|-------------|-------------|
| `09_flext_decorators_enterprise_patterns.py` | Cross-cutting concerns | Logging, caching, retry, auth |
| `10_flext_payload_messaging_events.py` | Event-driven architecture | Events, messages, payloads |
| `11_flext_handlers_enterprise_patterns.py` | Request handling | Middleware, pipelines, processors |
| `12_flext_logging_structured_system.py` | Structured logging | Correlation IDs, context, metrics |
| `13_flext_interfaces_architecture_patterns.py` | Clean Architecture | Ports, adapters, boundaries |
| `14_flext_exceptions_enterprise_handling.py` | Error management | Exception hierarchy, recovery |

### Advanced Integration (15-20)
**Complex scenarios and integrations**

| File | Description | Key Concepts |
|------|-------------|-------------|
| `15_flext_advanced_examples.py` | Advanced patterns | Complex workflows, orchestration |
| `16_flext_integration_example.py` | System integration | External services, APIs |
| `17_flext_working_examples.py` | Complete applications | Full implementation examples |
| `18_flext_unified_semantic_patterns_example.py` | Semantic modeling | Domain semantics, ubiquitous language |
| `19_modern_patterns_showcase.py` | Modern architecture | Microservices, event sourcing |
| `20_boilerplate_reduction_example.py` | Code generation | Reducing boilerplate, metaprogramming |

### Supporting Modules

| Module | Purpose | Usage |
|--------|---------|-------|
| `shared_domain.py` | Shared domain models | Common entities, value objects, and test data |
| `utilities/` | Helper modules | Demonstration runners, formatters, validators |
| `utilities/demonstration_runner.py` | Example runner | Executes and displays example results |
| `utilities/validation_utilities.py` | Validation helpers | Common validation functions |
| `utilities/formatting_helpers.py` | Output formatting | Pretty printing and display utilities |

## Learning Path

### Beginner Path
**Start here if new to FLEXT Core**

1. **Railway Pattern** (`01_flext_result_railway_pattern.py`)
   - Learn error handling without exceptions
   - Understand result chaining and composition

2. **Dependency Injection** (`02_flext_container_dependency_injection.py`)
   - Service registration and resolution
   - Managing application dependencies

3. **Configuration** (`08_flext_config_enterprise_configuration.py`)
   - Environment-based settings
   - Type-safe configuration

### Intermediate Path
**For building domain models**

1. **Domain Entities** (`06_flext_entity_valueobject_ddd_patterns.py`)
   - Creating rich domain models
   - Business logic encapsulation

2. **Validation** (`05_flext_validation_advanced_system.py`)
   - Business rule validation
   - Complex constraints

3. **CQRS Pattern** (`03_flext_commands_cqrs_pattern.py`)
   - Separating reads and writes
   - Command and query handlers

### Advanced Path
**For enterprise applications**

1. **Event-Driven** (`10_flext_payload_messaging_events.py`)
   - Domain events and messaging
   - Event sourcing foundations

2. **Clean Architecture** (`13_flext_interfaces_architecture_patterns.py`)
   - Architectural boundaries
   - Dependency inversion

3. **Integration** (`16_flext_integration_example.py`)
   - External service integration
   - Cross-system communication

## Running Examples

### Prerequisites

```bash
# Ensure FLEXT Core is installed
pip install -e .

# Or with Poetry
poetry install
```

### Running Individual Examples

```bash
# Navigate to project root
cd flext-core

# Run specific example
python examples/01_flext_result_railway_pattern.py

# With verbose output
python -v examples/02_flext_container_dependency_injection.py

# Run with Poetry
poetry run python examples/06_flext_entity_valueobject_ddd_patterns.py
```

### Running All Examples

```bash
# Using shell script
./run_examples.sh

# Or manually
for file in examples/[0-9]*.py; do
    echo "\n=== Running $(basename $file) ==="
    python "$file"
    echo "=== Completed ==="
done

# Using Make
make run-examples
```

### Interactive Mode

```python
# Start Python REPL with examples loaded
python -i examples/01_flext_result_railway_pattern.py

# Now you can interact with the example
>>> result = divide(10, 2)
>>> print(result.unwrap())
5.0
```

## Example Guidelines

### Code Standards

**All examples follow these principles:**

- ✅ **Type Safety**: Full type annotations with MyPy strict mode
- ✅ **Error Handling**: FlextResult pattern, no raw exceptions
- ✅ **Documentation**: Docstrings for all functions and classes
- ✅ **Self-Contained**: Each example runs independently
- ✅ **Real-World**: Practical scenarios, not toy examples
- ✅ **PEP 8**: 79 character line limit, consistent formatting

### Example Structure

```python
#!/usr/bin/env python3
"""Example: [Pattern Name]

Demonstrates:
- Key concept 1
- Key concept 2
- Key concept 3
"""

from flext_core import FlextResult, FlextContainer
from typing import Optional

def main() -> None:
    """Main example function."""
    # Example implementation
    pass

if __name__ == "__main__":
    main()
```

### Quality Checks

```bash
# Lint examples
ruff check examples/

# Type check
mypy examples/ --strict

# Format check
black --check examples/
```

## Common Patterns Demonstrated

### Error Handling Patterns

- **Railway-oriented programming**: Chaining operations without exceptions
- **Error aggregation**: Collecting multiple errors
- **Graceful degradation**: Fallback strategies
- **Error recovery**: Retry and circuit breaker patterns

### Architectural Patterns

- **Clean Architecture**: Separation of concerns
- **Domain-Driven Design**: Rich domain models
- **CQRS**: Command/Query separation
- **Event Sourcing**: Event-driven state management
- **Dependency Injection**: Inversion of control

### Enterprise Patterns

- **Repository Pattern**: Data access abstraction
- **Unit of Work**: Transaction management
- **Specification Pattern**: Business rule composition
- **Strategy Pattern**: Algorithm selection
- **Observer Pattern**: Event notification

### Integration Patterns

- **Adapter Pattern**: External service integration
- **Facade Pattern**: Simplified interfaces
- **Gateway Pattern**: External communication
- **Anti-Corruption Layer**: Boundary protection

## Dependencies and Requirements

### Core Dependencies

```python
# Required (installed with flext-core)
pydantic >= 2.11.7
pydantic-settings >= 2.10.1
structlog >= 25.4.0

# Python version
python >= 3.13
```

### Optional Dependencies

```python
# For specific examples
typing-extensions  # Advanced type hints
python-dotenv     # Environment file loading
rich              # Pretty console output
```

### No External Services Required

All examples are self-contained:
- No database connections needed
- No external APIs required
- No network dependencies
- Mock data included

## Tips for Learning

### Start Simple

1. Run each example and observe the output
2. Modify examples to test different scenarios
3. Break examples to understand error handling
4. Combine patterns from different examples

### Understanding Output

Examples use clear output formatting:

```
=== Example: Railway Pattern ===
✅ Success: Operation completed
❌ Error: Invalid input
📊 Metrics: 10 operations, 2 failures
```

### Creating Your Own Examples

1. Copy an existing example as template
2. Import from `shared_domain` for common models
3. Follow the established patterns
4. Test with `make validate`

## Troubleshooting

### Import Errors

```bash
# If examples can't import flext_core
PYTHONPATH=. python examples/01_flext_result_railway_pattern.py

# Or install in development mode
pip install -e .
```

### Type Errors

```bash
# Check types for specific example
mypy examples/01_flext_result_railway_pattern.py

# Ignore type errors temporarily
python examples/01_flext_result_railway_pattern.py 2>/dev/null
```

## Contributing Examples

When adding new examples:

1. Follow the naming convention: `NN_flext_<pattern>_<description>.py`
2. Include comprehensive docstrings
3. Add to this README with description
4. Ensure it passes `make validate`
5. Test with Python 3.13+

## Related Resources

- **[API Documentation](../docs/api/core.md)**: Complete API reference
- **[Architecture Guide](../docs/architecture/overview.md)**: Design principles
- **[Testing Guide](../tests/README.md)**: Test examples and patterns
- **[Source Code](../src/flext_core/README.md)**: Implementation details
- **[Development Guide](../CONTRIBUTING.md)**: Contribution guidelines

---

These examples are actively maintained and updated with new patterns as FLEXT Core evolves.
