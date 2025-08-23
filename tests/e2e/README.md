# End-to-End Tests

Complete workflow validation for FLEXT Core production scenarios.

## Overview

End-to-end tests validate complete user workflows and real-world usage scenarios, ensuring FLEXT Core functions correctly in production-like environments. These tests simulate actual usage patterns with realistic data, configurations, and error conditions.

## Test Organization

```
tests/e2e/
├── test_complete_workflows.py      # Full workflow scenarios
├── test_real_usage.py              # Production-like usage patterns
├── test_error_recovery.py          # Error handling and recovery
├── test_performance_scenarios.py   # Performance validation
└── fixtures/                       # E2E test data and configs
    ├── test_data.json
    └── test_config.yaml
```

## E2E Test Scenarios

### User Registration Workflow

```python
def test_complete_user_registration_flow():
    """Test complete user registration from start to finish."""
    # Initialize application
    app = Application()
    app.initialize()

    # Step 1: Validate input
    user_data = {
        "email": "newuser@example.com",
        "password": "SecurePass123!",
        "name": "John Doe",
        "age": 30
    }

    # Step 2: Create user through application layer
    result = app.register_user(user_data)
    assert result.success

    # Step 3: Verify user creation
    user = result.value
    assert user.id is not None
    assert user.email == "newuser@example.com"
    assert user.is_active is False

    # Step 4: Activate user
    activation_result = app.activate_user(user.id)
    assert activation_result.success

    # Step 5: Verify user can login
    login_result = app.authenticate(
        email="newuser@example.com",
        password="SecurePass123!"
    )
    assert login_result.success
    assert login_result.value.token is not None
```

### Data Processing Pipeline

```python
def test_data_processing_pipeline():
    """Test complete data processing pipeline."""
    # Setup pipeline
    pipeline = DataPipeline()
    pipeline.configure({
        "validation": {"strict": True},
        "transformation": {"format": "normalized"},
        "enrichment": {"add_metadata": True},
        "output": {"format": "json"}
    })

    # Load test data
    input_data = load_test_data("large_dataset.json")

    # Process through pipeline
    result = pipeline.process(input_data)

    # Verify all stages completed
    assert result.success
    output = result.value

    # Validate output
    assert output["status"] == "completed"
    assert output["records_processed"] == len(input_data)
    assert output["errors"] == []
    assert output["metadata"]["pipeline_version"] == "0.9.0"

    # Verify data integrity
    for record in output["data"]:
        assert record["validated"] is True
        assert record["transformed"] is True
        assert record["enriched"] is True
```

### Error Recovery Workflow

```python
def test_error_recovery_scenarios():
    """Test system recovery from various error conditions."""
    app = Application()

    # Scenario 1: Database connection failure
    with simulate_database_failure():
        result = app.save_data({"test": "data"})
        assert result.is_failure
        assert "database" in result.error.lower()

        # Verify graceful degradation
        assert app.is_running()
        assert app.health_check()["status"] == "degraded"

    # Scenario 2: Recovery after failure
    result = app.save_data({"test": "data"})
    assert result.success  # Should work after recovery

    # Scenario 3: Cascading failures
    with simulate_multiple_failures(["database", "cache", "queue"]):
        result = app.process_request({"action": "complex"})
        assert result.is_failure

        # Verify circuit breaker activated
        assert app.circuit_breaker.is_open

    # Scenario 4: Automatic recovery
    time.sleep(5)  # Wait for recovery
    assert app.circuit_breaker.is_closed
    result = app.process_request({"action": "simple"})
    assert result.success
```

## Running E2E Tests

### Basic Execution

```bash
# Run all E2E tests
pytest tests/e2e/

# Run with detailed output
pytest tests/e2e/ -v -s

# Run specific workflow
pytest tests/e2e/test_complete_workflows.py::test_user_registration

# With coverage
pytest tests/e2e/ --cov=src/flext_core --cov-report=html
```

### Performance Testing

```bash
# Run with performance metrics
pytest tests/e2e/ --durations=10

# With memory profiling
pytest tests/e2e/ --memprof

# Stress testing
pytest tests/e2e/test_performance_scenarios.py -k stress
```

### Environment Configuration

```bash
# Use production-like settings
E2E_ENV=production pytest tests/e2e/

# Custom configuration
pytest tests/e2e/ --config=tests/e2e/fixtures/prod_config.yaml

# With external services
USE_REAL_DB=true pytest tests/e2e/
```

## Writing E2E Tests

### Test Structure

```python
import pytest
from flext_core import Application, FlextResult
import time

class TestCompleteWorkflows:
    """Test complete application workflows."""

    @pytest.fixture(scope="class")
    def app(self):
        """Provide configured application instance."""
        app = Application()
        app.configure({
            "environment": "test",
            "debug": False,
            "database_url": "sqlite:///:memory:",
            "cache_enabled": True
        })
        app.initialize()
        yield app
        app.shutdown()

    def test_user_lifecycle(self, app):
        """Test complete user lifecycle."""
        # Create user
        create_result = app.create_user(
            email="test@example.com",
            password="secure123"
        )
        assert create_result.success
        user_id = create_result.value.id

        # Update user
        update_result = app.update_user(
            user_id,
            {"name": "Updated Name"}
        )
        assert update_result.success

        # Deactivate user
        deactivate_result = app.deactivate_user(user_id)
        assert deactivate_result.success

        # Verify final state
        user_result = app.get_user(user_id)
        assert user_result.success
        user = user_result.value
        assert user.name == "Updated Name"
        assert user.is_active is False
```

### Testing Production Scenarios

```python
def test_concurrent_operations():
    """Test system under concurrent load."""
    import concurrent.futures

    app = Application()
    results = []

    def create_user(index):
        """Create a user with index."""
        return app.create_user(
            email=f"user{index}@example.com",
            password="password123"
        )

    # Execute concurrent operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(create_user, i)
            for i in range(100)
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    # Verify all operations completed
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if r.is_failure)

    assert successful >= 95  # Allow 5% failure rate
    assert failed <= 5

    # Verify data consistency
    users_result = app.list_users()
    assert users_result.success
    all_users = users_result.value
    assert len(all_users) == successful
```

### Testing Error Scenarios

```python
def test_resilience_under_failures():
    """Test system resilience under various failures."""
    app = Application()

    # Test timeout handling
    with set_timeout(0.1):  # 100ms timeout
        result = app.slow_operation()
        assert result.is_failure
        assert "timeout" in result.error.lower()

    # Test rate limiting
    for i in range(100):
        result = app.api_call()
        if i > 10:  # After rate limit
            assert result.is_failure
            assert "rate limit" in result.error.lower()

    # Test memory pressure
    with limit_memory(100_000_000):  # 100MB limit
        result = app.memory_intensive_operation()
        if result.is_failure:
            assert "memory" in result.error.lower()
```

## Performance Validation

### Response Time Testing

```python
def test_response_time_requirements():
    """Validate response time meets requirements."""
    import time

    app = Application()
    operations = {
        "read": 0.1,    # 100ms max
        "write": 0.5,   # 500ms max
        "query": 1.0,   # 1s max
        "batch": 5.0    # 5s max
    }

    for operation, max_time in operations.items():
        start = time.time()
        result = getattr(app, f"{operation}_operation")()
        elapsed = time.time() - start

        assert result.success
        assert elapsed < max_time, f"{operation} took {elapsed}s (max: {max_time}s)"
```

### Load Testing

```python
def test_sustained_load():
    """Test system under sustained load."""
    app = Application()
    duration = 60  # 1 minute
    start_time = time.time()
    operations_count = 0
    errors_count = 0

    while time.time() - start_time < duration:
        result = app.process_request({
            "type": "standard",
            "data": generate_test_data()
        })

        operations_count += 1
        if result.is_failure:
            errors_count += 1

    # Calculate metrics
    error_rate = errors_count / operations_count
    ops_per_second = operations_count / duration

    # Verify requirements
    assert error_rate < 0.01  # Less than 1% error rate
    assert ops_per_second > 100  # At least 100 ops/sec
```

## Test Data Management

### Loading Test Data

```python
def load_test_data(filename):
    """Load test data from fixtures."""
    import json
from pathlib import Path

    data_path = Path(__file__).parent / "fixtures" / filename
    with open(data_path) as f:
        return json.load(f)

def generate_test_data(size="medium"):
    """Generate test data of specified size."""
    sizes = {
        "small": 10,
        "medium": 100,
        "large": 1000,
        "huge": 10000
    }

    count = sizes.get(size, 100)
    return [
        {
            "id": f"item_{i}",
            "value": i * 2,
            "timestamp": time.time() + i
        }
        for i in range(count)
    ]
```

### Test Configuration

```python
@pytest.fixture
def production_config():
    """Production-like configuration for E2E tests."""
    return {
        "debug": False,
        "log_level": "INFO",
        "database": {
            "url": "postgresql://localhost/test_db",
            "pool_size": 20,
            "timeout": 30
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "max_size": 1000
        },
        "api": {
            "rate_limit": 1000,
            "timeout": 60,
            "max_connections": 100
        }
    }
```

## Common Patterns

### State Verification

```python
def test_state_transitions():
    """Test entity state transitions."""
    app = Application()

    # Initial state
    order_result = app.create_order({"items": ["item1", "item2"]})
    assert order_result.success
    order = order_result.value
    assert order.status == "pending"

    # Process order
    app.process_order(order.id)
    order_result = app.get_order(order.id)
    assert order_result.success
    order = order_result.value
    assert order.status == "processing"

    # Complete order
    app.complete_order(order.id)
    order_result = app.get_order(order.id)
    assert order_result.success
    order = order_result.value
    assert order.status == "completed"

    # Verify state consistency
    assert order.completed_at is not None
    assert order.completed_at > order.created_at
```

### Cleanup and Teardown

```python
class TestWithCleanup:
    """Tests with proper cleanup."""

    def setup_method(self):
        """Setup before each test."""
        self.app = Application()
        self.created_resources = []

    def teardown_method(self):
        """Cleanup after each test."""
        # Clean up created resources
        for resource_id in self.created_resources:
            self.app.delete_resource(resource_id)

        # Shutdown application
        self.app.shutdown()

    def test_with_resources(self):
        """Test that creates resources."""
        result = self.app.create_resource({"name": "test"})
        if result.success:
            self.created_resources.append(result.value.id)

        # Test logic here
        assert result.success
```

## Troubleshooting

### Common Issues

**Slow E2E tests:**

```bash
# Profile slow tests
pytest tests/e2e/ --profile

# Run in parallel
pytest tests/e2e/ -n 4

# Skip slow tests
pytest tests/e2e/ -m "not slow"
```

**Flaky tests:**

```bash
# Retry flaky tests
pytest tests/e2e/ --reruns 3

# Increase timeouts
pytest tests/e2e/ --timeout=300
```

**Resource issues:**

```bash
# Monitor resource usage
pytest tests/e2e/ --monitor-resources

# Limit resource usage
ulimit -m 1000000  # Limit memory to 1GB
pytest tests/e2e/
```

## Best Practices

### DO

- ✅ Test complete workflows from user perspective
- ✅ Use realistic data volumes and patterns
- ✅ Verify error recovery and resilience
- ✅ Clean up all created resources
- ✅ Test performance requirements
- ✅ Validate state consistency

### DON'T

- ❌ Mock core FLEXT components
- ❌ Use unrealistic test data
- ❌ Skip error scenarios
- ❌ Leave resources after tests
- ❌ Ignore performance issues
- ❌ Test implementation details

## Related Documentation

- [Unit Tests](../unit/)
- [Integration Tests](../integration/)
- [Test Configuration](../conftest.py)
- [Performance Requirements](../../docs/architecture/overview.md)
- [Main Test Documentation](../README.md)
