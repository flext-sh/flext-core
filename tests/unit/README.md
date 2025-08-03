# Unit Tests

**Isolated Component Testing with 95%+ Coverage Requirement**

Unit tests validate individual components in isolation, ensuring comprehensive coverage of all business logic, error paths, and edge cases across the FLEXT Core architectural layers.

## Test Organization

### Core Pattern Layer Tests (`core/`)

- `test_result.py` - FlextResult[T] railway-oriented programming validation
- `test_container.py` - FlextContainer dependency injection testing
- `test_exceptions.py` - Exception hierarchy and error handling validation
- `test_utilities.py` - Pure utility function testing with performance validation

### Domain Layer Tests (`domain/`)

- `test_domain_entity.py` - FlextEntity business logic and event testing
- `test_domain_value_object.py` - FlextValueObject immutability and equality
- `test_entities.py` - Entity behavior and lifecycle validation

### Pattern Implementation Tests (`patterns/`)

- `test_patterns_commands.py` - CQRS command pattern validation
- `test_patterns_handlers.py` - Message handler pattern testing
- `test_patterns_validation.py` - Validation pattern comprehensive testing

### Cross-Cutting Tests

- `test_pep8_compliance.py` - Code style and formatting validation
- `test_coverage_gaps.py` - Coverage analysis and gap identification

## Quality Standards

### Coverage Requirements

- **Minimum**: 95% line coverage per module
- **Critical Paths**: 100% coverage for FlextResult, FlextContainer
- **Error Paths**: All exception scenarios tested
- **Edge Cases**: Boundary conditions and invalid inputs

### Performance Requirements

- **Execution Time**: < 100ms per individual test
- **Suite Time**: < 10 seconds for complete unit test suite
- **Memory Usage**: No memory leaks in test execution

### Test Isolation

- **No External Dependencies**: All external services mocked
- **Clean State**: Each test starts with fresh state
- **No Side Effects**: Tests do not affect other tests
- **Deterministic**: Tests produce same results on every run

## Running Unit Tests

```bash
# All unit tests
pytest tests/unit/

# Specific test categories
pytest tests/unit/core/                # Core pattern tests
pytest tests/unit/domain/              # Domain layer tests
pytest tests/unit/patterns/            # Pattern implementation tests

# Fast feedback for development
pytest tests/unit/core/test_result.py -v
pytest -m "unit and not slow"
```

## Test Standards

### Test Structure

```python
def test_should_[behavior]_when_[condition](fixture_name: Type) -> None:
    """Test that [component] [behavior] when [condition].

    Validates [specific behavior] ensuring [business rule].
    Coverage: [coverage area] - [quality aspect validated]
    """
    # Arrange
    setup_test_data

    # Act
    result = component_under_test.method(parameters)

    # Assert
    assert result.is_success
    assert expected_behavior_occurred
```

### Mock Usage

- **External Dependencies**: Always mocked in unit tests
- **Time Dependencies**: Mock datetime.now() for deterministic testing
- **File System**: Mock file operations
- **Network**: Mock all HTTP and network calls

## Related Documentation

- [Integration Tests](../integration/README.md)
- [End-to-End Tests](../e2e/README.md)
- [Source Code Organization](../../src/flext_core/README.md)
