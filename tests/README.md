# FLEXT Core Test Suite

Professional test organization following industry best practices.

## Structure

```
tests/
├── conftest.py                 # Main test configuration
├── conftest_integration.py     # Integration-specific fixtures
├── unit/                       # Unit tests (isolated components)
│   ├── core/                   # Core module tests
│   │   ├── test_result.py      # FlextResult tests
│   │   ├── test_container.py   # FlextContainer tests
│   │   ├── test_types*.py      # Type system tests
│   │   ├── test_config.py      # Configuration tests
│   │   └── ...
│   ├── patterns/               # Pattern implementation tests
│   │   ├── test_patterns_commands.py
│   │   ├── test_patterns_handlers.py
│   │   └── ...
│   ├── domain/                 # DDD building block tests
│   │   ├── test_domain_entity.py
│   │   ├── test_domain_value_object.py
│   │   └── ...
│   ├── test_pep8_compliance.py # PEP8 validation tests
│   └── test_coverage_gaps.py   # Coverage analysis tests
├── integration/                # Integration tests (component interactions)
│   ├── services/               # Service integration tests
│   ├── containers/             # Container integration tests
│   ├── configs/                # Configuration integration tests
│   └── test_integration.py     # Main integration tests
└── e2e/                        # End-to-end tests (complete workflows)
    └── test_real_usage.py       # Real-world usage patterns
```

## Test Types

### Unit Tests (`tests/unit/`)

- **Purpose**: Test individual components in isolation
- **Scope**: Single classes, functions, or modules
- **Dependencies**: Minimal, use mocks when needed
- **Speed**: Fast (<100ms per test)
- **Coverage**: 95% minimum requirement

### Integration Tests (`tests/integration/`)

- **Purpose**: Test component interactions
- **Scope**: Multiple classes working together
- **Dependencies**: Real dependencies within FLEXT Core
- **Speed**: Medium (100ms-1s per test)
- **Coverage**: Critical integration points

### End-to-End Tests (`tests/e2e/`)

- **Purpose**: Test complete user workflows
- **Scope**: Full scenarios from start to finish
- **Dependencies**: Complete system
- **Speed**: Slow (1s+ per test)
- **Coverage**: User-facing scenarios

## Running Tests

### By Test Type

```bash
# Unit tests only
pytest tests/unit -m unit

# Integration tests only
pytest tests/integration -m integration

# End-to-end tests only
pytest tests/e2e -m e2e
```

### By Component

```bash
# Core module tests
pytest tests/unit/core/

# Pattern tests
pytest tests/unit/patterns/

# Domain tests
pytest tests/unit/domain/
```

### By Marker

```bash
# PEP8 compliance tests
pytest -m pep8

# Core functionality tests
pytest -m core

# DDD pattern tests
pytest -m ddd

# Architecture tests
pytest -m architecture
```

### Quality Gates

```bash
# Fast feedback loop
pytest tests/unit/core/test_result.py -v

# Full unit test suite
pytest tests/unit/ -v

# Complete test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/flext_core --cov-report=html
```

## Test Markers

Available pytest markers for selective test execution:

- `unit` - Unit tests (isolated components)
- `integration` - Integration tests (component interactions)
- `e2e` - End-to-end tests (complete workflows)
- `pep8` - PEP8 compliance validation
- `core` - Core framework functionality
- `architecture` - Architectural pattern tests
- `ddd` - Domain-driven design tests
- `slow` - Slow tests (can be excluded)

## Configuration

### Main Configuration (`conftest.py`)

- Test markers configuration
- Shared fixtures for all test types
- Environment setup and cleanup

### Integration Configuration (`conftest_integration.py`)

- Integration-specific fixtures
- Mock services and containers
- Database and service mocks

## Best Practices

1. **Test Organization**: Each module has corresponding test file(s)
2. **Naming Convention**: `test_{module_name}.py`
3. **Marker Usage**: Always mark tests with appropriate markers
4. **Fast Tests**: Keep unit tests under 100ms
5. **Isolation**: Unit tests should not depend on external services
6. **Coverage**: Maintain 95% minimum test coverage
7. **Documentation**: Document complex test scenarios

## Dependencies

Test dependencies are minimal and isolated:

- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- Standard library only for mocks

No external services required for unit tests.
