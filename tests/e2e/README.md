# End-to-End Tests

**Complete Workflow Validation**

End-to-end tests validate complete user workflows and real-world usage scenarios, ensuring FLEXT Core functions correctly in production-like environments with realistic data and configurations.

## Test Organization

### Real Usage Scenarios (`test_real_usage.py`)

- Complete workflows from external perspective
- Production-like configuration and data
- Error recovery and resilience validation
- Performance under realistic load

## E2E Test Scope

### Complete Workflows

- **User Registration Flow**: Entity creation, validation, persistence
- **Data Processing Pipeline**: Input → Validation → Transformation → Output
- **Error Recovery**: Graceful handling of failures and recovery
- **Configuration Loading**: Complete application startup and initialization

### Production Simulation

- **Real Configuration**: Production-like settings and parameters
- **Realistic Data**: Data volumes and complexity matching production
- **Error Scenarios**: Network failures, invalid data, resource constraints
- **Performance**: Response times and resource usage validation

## Quality Standards

### Performance Requirements

- **Response Time**: Realistic response time expectations
- **Memory Usage**: Memory consumption within acceptable limits
- **Resource Cleanup**: Proper cleanup of all resources

### Reliability Standards

- **Error Recovery**: Graceful handling of all error scenarios
- **Data Integrity**: No data corruption under any circumstances
- **State Consistency**: System remains in valid state after operations

## Running E2E Tests

```bash
# All end-to-end tests
pytest tests/e2e/

# Specific test with verbose output
pytest tests/e2e/test_real_usage.py -v -s

# With performance monitoring
pytest tests/e2e/ --durations=10
```

## Test Configuration

### E2E Test Environment

- **Real Services**: FLEXT Core components run as in production
- **Mock External**: External dependencies mocked or stubbed
- **Test Data**: Realistic test data reflecting production scenarios

### Performance Monitoring

- **Response Times**: Track and validate response time requirements
- **Memory Usage**: Monitor memory consumption patterns
- **Resource Utilization**: Validate efficient resource usage

## Related Documentation

- [Unit Tests](../unit/README.md)
- [Integration Tests](../integration/README.md)
- [Performance Requirements](../../CLAUDE.md)
