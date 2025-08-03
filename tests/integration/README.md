# Integration Tests

**Cross-Module Interaction Validation**

Integration tests validate interactions between FLEXT Core modules, ensuring proper component integration and real-world usage scenarios work correctly across architectural boundaries.

## Test Organization

### Configuration Integration (`configs/`)

- Real configuration loading and validation
- Environment variable integration testing
- Configuration hierarchy and inheritance validation

### Container Integration (`containers/`)

- Dependency injection with real services
- Service lifecycle management testing
- Container configuration and registration validation

### Service Integration (`services/`)

- Cross-service communication validation
- Service discovery and resolution testing
- Service composition and orchestration validation

### Main Integration Tests

- `test_integration.py` - Cross-module integration scenarios
- End-to-end component interaction validation
- Real-world usage pattern testing

## Integration Scope

### What We Test

- **Module Boundaries**: Interactions between architectural layers
- **Real Dependencies**: Actual service implementations
- **Configuration**: Real configuration loading and parsing
- **Error Propagation**: Error handling across module boundaries

### What We Mock

- **External Services**: Database connections, HTTP APIs
- **File System**: For test environment isolation
- **Time**: For deterministic time-based testing

## Quality Standards

### Performance Requirements

- **Execution Time**: 100ms - 1s per test
- **Setup Time**: Efficient test fixture initialization
- **Cleanup**: Proper resource cleanup after tests

### Reliability Standards

- **Deterministic**: Same results on every execution
- **Isolated**: Tests do not interfere with each other
- **Realistic**: Test scenarios reflect real usage patterns

## Running Integration Tests

```bash
# All integration tests
pytest tests/integration/

# Specific integration categories
pytest tests/integration/configs/      # Configuration integration
pytest tests/integration/containers/   # Container integration
pytest tests/integration/services/     # Service integration

# With specific markers
pytest -m integration
```

## Test Configuration

### Integration Fixtures (`conftest_integration.py`)

- `integration_container` - Container with real service registrations
- `integration_config` - Real configuration for testing
- `integration_logger` - Configured logger with test settings

### Real vs Mock Balance

- **Real**: FLEXT Core module interactions
- **Mock**: External dependencies (databases, APIs)
- **Configurable**: Test-specific service implementations

## Related Documentation

- [Unit Tests](../unit/README.md)
- [End-to-End Tests](../e2e/README.md)
- [Test Configuration](../README.md)
