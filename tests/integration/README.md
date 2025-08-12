# Integration Tests

Component interaction testing for FLEXT Core modules.

## Overview

Integration tests validate how FLEXT Core components work together, ensuring proper interaction between modules, services, and configurations. These tests use real implementations where possible while mocking external dependencies.

## Test Organization

```
tests/integration/
├── test_integration.py              # Main integration scenarios
├── test_integration_examples.py     # Example-based integration tests
├── configs/                         # Configuration integration (planned)
│   ├── test_config_loading.py      
│   └── test_config_inheritance.py  
├── containers/                      # Container integration (planned)
│   ├── test_container_lifecycle.py 
│   └── test_service_resolution.py  
└── services/                        # Service integration (planned)
    ├── test_service_communication.py
    └── test_service_orchestration.py
```

## Integration Test Scenarios

### Service Integration

```python
def test_service_with_container():
    """Test service registration and resolution."""
    # Setup
    container = FlextContainer()
    logger = FlextLogger("test")
    repository = UserRepository()
    
    # Register dependencies
    container.register("logger", logger)
    container.register("repository", repository)
    
    # Create service with dependencies
    service = UserService(
        logger=container.get("logger").unwrap(),
        repository=container.get("repository").unwrap()
    )
    
    # Test service operations
    result = service.create_user("test@example.com")
    assert result.success
    assert result.unwrap().email == "test@example.com"
```

### Configuration Integration

```python
def test_configuration_with_services():
    """Test configuration loading and service initialization."""
    # Load configuration
    config = AppSettings(_env_file=".env.test")
    
    # Initialize services with config
    database = DatabaseService(
        url=config.database_url,
        pool_size=config.database_pool_size
    )
    
    cache = CacheService(
        url=config.redis_url,
        ttl=config.cache_ttl
    )
    
    # Test integrated behavior
    result = database.connect()
    assert result.success
    
    cache_result = cache.set("key", "value")
    assert cache_result.success
```

### Error Propagation

```python
def test_error_propagation_across_layers():
    """Test error handling across module boundaries."""
    # Create layered architecture
    repository = UserRepository()
    service = UserService(repository)
    handler = UserHandler(service)
    
    # Test error propagation
    invalid_request = {"email": "invalid"}
    result = handler.handle(invalid_request)
    
    assert result.is_failure
    assert "validation" in result.error.lower()
    
    # Verify error context preserved
    assert result.metadata.get("layer") == "handler"
    assert result.metadata.get("original_error") is not None
```

## Running Integration Tests

### Run All Integration Tests

```bash
# Standard execution
pytest tests/integration/

# With coverage
pytest tests/integration/ --cov=src/flext_core

# Verbose output
pytest tests/integration/ -v

# With specific timeout
pytest tests/integration/ --timeout=10
```

### Run Specific Categories

```bash
# Configuration tests only
pytest tests/integration/configs/

# Container tests only
pytest tests/integration/containers/

# Service tests only
pytest tests/integration/services/

# Main integration scenarios
pytest tests/integration/test_integration.py
```

### Run by Marker

```bash
# Integration tests only
pytest -m integration

# Exclude slow integration tests
pytest -m "integration and not slow"

# Database integration tests
pytest -m "integration and database"
```

## Writing Integration Tests

### Test Structure

```python
import pytest
from flext_core import FlextContainer, FlextResult
from unittest.mock import Mock

class TestServiceIntegration:
    """Test service integration scenarios."""
    
    @pytest.fixture
    def container(self):
        """Provide configured container."""
        container = FlextContainer()
        # Register common services
        container.register("logger", Mock())
        container.register("config", {"debug": True})
        return container
    
    def test_service_initialization(self, container):
        """Test service initializes with dependencies."""
        # Arrange
        repository = UserRepository()
        container.register("repository", repository)
        
        # Act
        service = UserService.from_container(container)
        
        # Assert
        assert service is not None
        assert service.repository == repository
    
    def test_service_operation_flow(self, container):
        """Test complete service operation flow."""
        # Setup service chain
        repository = UserRepository()
        service = UserService(repository)
        handler = UserHandler(service)
        
        # Execute operation
        request = CreateUserRequest(
            email="user@example.com",
            name="Test User"
        )
        result = handler.handle(request)
        
        # Verify complete flow
        assert result.success
        user = result.unwrap()
        assert user.email == "user@example.com"
        assert user.is_active is False
```

### Testing Component Interaction

```python
def test_component_interaction():
    """Test multiple components working together."""
    # Initialize components
    validator = UserValidator()
    repository = UserRepository()
    notifier = EmailNotifier()
    
    # Create service with components
    service = UserRegistrationService(
        validator=validator,
        repository=repository,
        notifier=notifier
    )
    
    # Test integrated workflow
    result = service.register_user(
        email="new@example.com",
        password="secure123"
    )
    
    # Verify all components worked
    assert result.success
    assert repository.find_by_email("new@example.com").success
    assert notifier.last_sent_to == "new@example.com"
```

### Testing Configuration Loading

```python
def test_configuration_cascade():
    """Test configuration loading from multiple sources."""
    import os
    
    # Set environment variables
    os.environ["APP_DEBUG"] = "true"
    os.environ["APP_DATABASE_URL"] = "postgresql://test/db"
    
    try:
        # Load configuration
        config = AppSettings(
            _env_file=".env.test",
            database_url="sqlite:///:memory:"  # Override
        )
        
        # Verify configuration cascade
        assert config.debug is True  # From env var
        assert config.database_url == "sqlite:///:memory:"  # Override
        
        # Test with services
        app = Application(config)
        assert app.is_debug_mode is True
        
    finally:
        # Cleanup
        os.environ.pop("APP_DEBUG", None)
        os.environ.pop("APP_DATABASE_URL", None)
```

## Test Patterns

### Repository Pattern Testing

```python
def test_repository_with_unit_of_work():
    """Test repository pattern with unit of work."""
    # Setup
    uow = UnitOfWork()
    user_repo = UserRepository(uow)
    order_repo = OrderRepository(uow)
    
    # Begin transaction
    with uow:
        # Create user
        user = User(email="test@example.com")
        user_result = user_repo.add(user)
        assert user_result.success
        
        # Create order for user
        order = Order(user_id=user.id, total=100.00)
        order_result = order_repo.add(order)
        assert order_result.success
        
        # Commit transaction
        uow.commit()
    
    # Verify persistence
    found_user = user_repo.find(user.id)
    assert found_user.success
    assert found_user.unwrap().email == "test@example.com"
```

### Event-Driven Testing

```python
def test_event_driven_integration():
    """Test event-driven component interaction."""
    # Setup event bus
    event_bus = EventBus()
    
    # Register handlers
    email_handler = EmailNotificationHandler()
    audit_handler = AuditLogHandler()
    
    event_bus.subscribe("UserCreated", email_handler)
    event_bus.subscribe("UserCreated", audit_handler)
    
    # Create service that publishes events
    service = UserService(event_bus=event_bus)
    
    # Execute operation that triggers events
    result = service.create_user("test@example.com")
    
    # Verify event handling
    assert result.success
    assert email_handler.notifications_sent == 1
    assert audit_handler.events_logged == 1
```

### Pipeline Testing

```python
def test_processing_pipeline():
    """Test data processing pipeline."""
    # Build pipeline
    pipeline = (
        ProcessingPipeline()
        .add_stage(ValidationStage())
        .add_stage(TransformationStage())
        .add_stage(EnrichmentStage())
        .add_stage(PersistenceStage())
    )
    
    # Process data through pipeline
    input_data = {"user_id": "123", "action": "login"}
    result = pipeline.process(input_data)
    
    # Verify pipeline execution
    assert result.success
    output = result.unwrap()
    assert output["validated"] is True
    assert output["transformed"] is True
    assert output["enriched"] is True
    assert output["persisted"] is True
```

## Performance Considerations

### Test Execution Time

```python
import pytest
import time

@pytest.mark.timeout(1)  # 1 second timeout
def test_integration_performance():
    """Ensure integration test runs within time limit."""
    start = time.time()
    
    # Setup components
    container = FlextContainer()
    for i in range(100):
        container.register(f"service_{i}", f"value_{i}")
    
    # Test operations
    for i in range(100):
        result = container.get(f"service_{i}")
        assert result.success
    
    # Verify performance
    elapsed = time.time() - start
    assert elapsed < 0.5  # Should complete in 500ms
```

### Resource Management

```python
def test_resource_cleanup():
    """Test proper resource cleanup in integration."""
    resources_created = []
    
    try:
        # Create resources
        db_connection = DatabaseConnection()
        resources_created.append(db_connection)
        
        cache_client = CacheClient()
        resources_created.append(cache_client)
        
        # Use resources
        service = DataService(db_connection, cache_client)
        result = service.process_data()
        assert result.success
        
    finally:
        # Ensure cleanup
        for resource in resources_created:
            resource.close()
```

## Common Issues and Solutions

### Fixture Conflicts

```python
# Use unique fixture names for integration tests
@pytest.fixture
def integration_container():  # Not just 'container'
    """Integration-specific container fixture."""
    return FlextContainer()

@pytest.fixture
def integration_config():  # Not just 'config'
    """Integration-specific configuration."""
    return AppSettings(_env_file=".env.integration")
```

### Test Isolation

```python
def test_isolated_integration():
    """Ensure test isolation in integration scenarios."""
    # Create isolated instances
    container1 = FlextContainer()
    container2 = FlextContainer()
    
    # Register different values
    container1.register("value", "test1")
    container2.register("value", "test2")
    
    # Verify isolation
    assert container1.get("value").unwrap() == "test1"
    assert container2.get("value").unwrap() == "test2"
```

### Mock vs Real Boundaries

```python
def test_with_mock_boundaries():
    """Test with clear mock boundaries."""
    # Mock external dependencies
    mock_http_client = Mock(spec=HttpClient)
    mock_http_client.get.return_value = {"status": "ok"}
    
    # Use real internal components
    parser = ResponseParser()  # Real
    validator = ResponseValidator()  # Real
    
    # Integration with mixed components
    service = ExternalApiService(
        http_client=mock_http_client,  # Mocked
        parser=parser,  # Real
        validator=validator  # Real
    )
    
    # Test integration
    result = service.fetch_data()
    assert result.success
    mock_http_client.get.assert_called_once()
```

## Troubleshooting

### Debugging Integration Tests

```bash
# Run with verbose output
pytest tests/integration/ -vv

# Show print statements
pytest tests/integration/ -s

# Debug specific test
pytest tests/integration/test_integration.py::test_specific -vv --pdb

# Show slowest tests
pytest tests/integration/ --durations=10
```

### Common Errors

**Import errors:**

```bash
# Ensure proper Python path
PYTHONPATH=. pytest tests/integration/
```

**Timeout errors:**

```bash
# Increase timeout for slow tests
pytest tests/integration/ --timeout=30
```

**Resource errors:**

```bash
# Run tests individually to isolate resource issues
pytest tests/integration/test_integration.py -k test_name
```

## Related Documentation

- [Unit Tests](../unit/)
- [E2E Tests](../e2e/)
- [Test Configuration](../conftest.py)
- [Integration Fixtures](../conftest_integration.py)
- [Main Test Documentation](../README.md)
