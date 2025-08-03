# FLEXT Core Examples

**Comprehensive Working Examples Demonstrating Enterprise Patterns**

This directory contains 17 comprehensive working examples that demonstrate proper usage of FLEXT Core patterns across all architectural layers, providing practical implementation guidance for the 32-project ecosystem.

## Example Organization

### Foundation & Core Patterns

- `01_flext_result_railway_pattern.py` - Railway-oriented programming with FlextResult[T]
- `02_flext_container_dependency_injection.py` - Enterprise dependency injection patterns

### CQRS & Command Patterns

- `03_flext_commands_cqrs_pattern.py` - Command Query Responsibility Segregation implementation

### Utility & Configuration

- `04_flext_utilities_modular.py` - Modular utility functions and generators
- `05_flext_validation_advanced_system.py` - Advanced validation system patterns
- `08_flext_config_enterprise_configuration.py` - Enterprise configuration management

### Domain-Driven Design

- `06_flext_entity_valueobject_ddd_patterns.py` - Entity and Value Object patterns
- `07_flext_mixins_multiple_inheritance.py` - Mixin composition patterns

### Enterprise Patterns

- `09_flext_decorators_enterprise_patterns.py` - Cross-cutting concern decorators
- `10_flext_payload_messaging_events.py` - Message and event payload patterns
- `11_flext_handlers_enterprise_patterns.py` - Handler and processor patterns
- `12_flext_logging_structured_system.py` - Structured logging implementation

### Architecture & Integration

- `13_flext_interfaces_architecture_patterns.py` - Clean Architecture interfaces
- `14_flext_exceptions_enterprise_handling.py` - Enterprise exception handling
- `15_flext_advanced_examples.py` - Advanced usage scenarios
- `16_flext_integration_example.py` - Cross-module integration patterns
- `17_flext_working_examples.py` - Real-world working implementations

### Supporting Files

- `shared_domain.py` - Shared domain models used across examples
- `shared_example_helpers.py` - Common helper functions
- `boilerplate_reduction_example.py` - Boilerplate reduction techniques

## Usage Patterns

### Basic Patterns

Examples 01-04 demonstrate fundamental FLEXT Core patterns that form the foundation for all enterprise applications.

### Domain Modeling

Examples 05-07 show proper domain-driven design implementation using FLEXT Core components.

### Enterprise Features

Examples 08-12 demonstrate enterprise-grade features including configuration, logging, and message handling.

### Advanced Integration

Examples 13-17 show advanced usage scenarios and real-world integration patterns.

## Running Examples

### Individual Examples

```bash
# Basic railway pattern
python examples/01_flext_result_railway_pattern.py

# Dependency injection
python examples/02_flext_container_dependency_injection.py

# Domain-driven design
python examples/06_flext_entity_valueobject_ddd_patterns.py
```

### All Examples

```bash
# Run all examples in sequence
for example in examples/*.py; do
    if [[ ! "$example" =~ (shared_|__) ]]; then
        echo "Running: $example"
        python "$example"
    fi
done
```

## Example Standards

### Code Quality

- **Type Safety**: All examples use strict type annotations
- **Error Handling**: Railway-oriented programming throughout
- **Documentation**: Comprehensive inline documentation
- **Realism**: Examples reflect real-world usage scenarios

### Educational Value

- **Progressive Complexity**: Examples build from simple to advanced
- **Best Practices**: Demonstrate enterprise patterns correctly
- **Common Scenarios**: Cover typical usage patterns
- **Integration**: Show how patterns work together

## Example Categories

### **Foundation Examples** (01-02)

Basic patterns that every FLEXT Core application uses.

### **Architectural Examples** (03-07)

CQRS, DDD, and architectural pattern implementations.

### **Enterprise Examples** (08-12)

Production-ready features for enterprise applications.

### **Advanced Examples** (13-17)

Complex scenarios and real-world integration patterns.

## Dependencies

Examples use only FLEXT Core components with minimal external dependencies:

- Standard library modules for demonstration
- No external services required
- Self-contained and executable

## Related Documentation

- [Source Code Organization](../src/flext_core/README.md)
- [Test Suite](../tests/README.md)
- [Development Guidelines](../CLAUDE.md)
