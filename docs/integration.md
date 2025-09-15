# Integration Guide

**FLEXT-Core Ecosystem Integration Patterns**
**Date**: September 17, 2025 | **Version**: 0.9.0

---

## Overview

This guide covers integration patterns for using FLEXT-Core as the foundation library across the FLEXT ecosystem. These patterns ensure consistency and reliability across all 45+ dependent projects.

**Workspace Integration Documentation**: [FLEXT Integration Standards](../../docs/integration/)

---

## Foundation Integration Principles

### 1. Layered Architecture Integration

FLEXT-Core provides foundation patterns that organize naturally into Clean Architecture layers:

```python
# Foundation Layer (FLEXT-Core)
from flext_core import FlextResult, FlextContainer, FlextModels

# Domain Layer (Your Project)
class BusinessService(FlextModels.Entity):
    def perform_business_logic(self, data: dict) -> FlextResult[dict]:
        # Business rules with foundation error handling
        if not data:
            return FlextResult[dict].fail("Business data required")
        return FlextResult[dict].ok(processed_data)

# Application Layer (Your Project)
class ApplicationService:
    def __init__(self) -> None:
        self._container = FlextContainer.get_global()
        self._business_service = self._container.get("business").unwrap()

    def handle_request(self, request: dict) -> FlextResult[dict]:
        return self._business_service.perform_business_logic(request)

# Infrastructure Layer (Your Project)
from flext_core import FlextConfig

class ProjectConfig(FlextConfig):
    database_url: str
    api_key: str
    debug: bool = False
```

### 2. Error Handling Consistency

All FLEXT ecosystem projects use FlextResult[T] for consistent error handling:

```python
from flext_core import FlextResult

# ✅ Ecosystem Standard Pattern
def ecosystem_operation(input_data: dict) -> FlextResult[ProcessedData]:
    """Standard error handling pattern across all FLEXT projects."""
    if not input_data:
        return FlextResult[ProcessedData].fail("Input required")

    # Railway-oriented composition
    return (
        validate_input(input_data)
        .flat_map(process_data)
        .flat_map(store_result)
        .map_error(lambda e: f"Operation failed: {e}")
    )

# ✅ Consistent API access (both patterns supported)
result = ecosystem_operation(data)
if result.is_success:
    # Both access patterns work for ecosystem compatibility
    value = result.value    # New API
    value = result.data     # Legacy API (maintained for ecosystem)
    value = result.unwrap() # Explicit API
```

### 3. Dependency Injection Patterns

Singleton container pattern used across all FLEXT services:

```python
from flext_core import FlextContainer

# Service registration at application startup
def initialize_services() -> FlextResult[None]:
    container = FlextContainer.get_global()

    # Register core services
    container.register("logger", create_logger())
    container.register("database", create_database_service())
    container.register("cache", create_cache_service())

    return FlextResult[None].ok(None)

# Service consumption throughout application
class ProjectService:
    def __init__(self) -> None:
        container = FlextContainer.get_global()
        self._logger = container.get("logger").unwrap()
        self._db = container.get("database").unwrap()

    def perform_operation(self) -> FlextResult[dict]:
        self._logger.info("Starting operation")
        # Use services with error handling
        return self._db.query("SELECT * FROM table")
```

---

## Project-Specific Integration Patterns

### CLI Projects Integration

```python
# DO NOT import click directly - use flext-cli
from flext_cli import FlextCliApi, FlextCliMain
from flext_core import FlextResult, FlextLogger

class ProjectCli:
    def __init__(self) -> None:
        self._cli = FlextCliApi()
        self._logger = FlextLogger(__name__)

    def create_command_interface(self) -> FlextResult[FlextCliMain]:
        main_cli = FlextCliMain(name="project-cli")

        # Register commands using flext-cli patterns
        main_cli.add_command("process", self.process_command)

        return FlextResult[FlextCliMain].ok(main_cli)

    def process_command(self, args: dict) -> FlextResult[None]:
        self._logger.info("Processing command", extra=args)
        # Business logic with foundation patterns
        return FlextResult[None].ok(None)
```

### API Projects Integration

```python
from flext_core import FlextResult, FlextContainer, FlextConfig
from flext_api import FlextApiMain, FlextApiConfig

class ProjectApiConfig(FlextConfig):
    host: str = "localhost"
    port: int = 8000
    database_url: str

class ProjectApi:
    def __init__(self, config: ProjectApiConfig) -> None:
        self._config = config
        self._container = FlextContainer.get_global()
        self._api = FlextApiMain(config=config)

    def create_endpoints(self) -> FlextResult[None]:
        # Register API endpoints with foundation error handling
        self._api.add_route("/health", self.health_check)
        self._api.add_route("/process", self.process_request)

        return FlextResult[None].ok(None)

    def process_request(self, request_data: dict) -> FlextResult[dict]:
        # Use foundation patterns for request processing
        return (
            self.validate_request(request_data)
            .flat_map(self.process_business_logic)
            .map_error(lambda e: f"Request failed: {e}")
        )
```

### Data Integration Projects

```python
from flext_core import FlextResult, FlextModels, FlextValidations

# Domain modeling for data integration
class DataRecord(FlextModels.Entity):
    source_id: str
    data: dict
    status: str = "pending"

    def validate_data(self) -> FlextResult[None]:
        """Validate data record using foundation patterns."""
        if not self.data:
            return FlextResult[None].fail("Data cannot be empty")

        # Use FlextValidations for complex validation
        validation_result = FlextValidations.validate_required_fields(
            self.data, ["id", "timestamp"]
        )

        if validation_result.is_failure:
            return FlextResult[None].fail(f"Validation failed: {validation_result.error}")

        return FlextResult[None].ok(None)

    def process(self) -> FlextResult[None]:
        """Process data record with business logic."""
        validation_result = self.validate_data()
        if validation_result.is_failure:
            return validation_result

        self.status = "processed"
        self.add_domain_event("DataProcessed", {"record_id": self.id})

        return FlextResult[None].ok(None)

# ETL Pipeline integration
class DataPipeline:
    def __init__(self) -> None:
        self._container = FlextContainer.get_global()
        self._logger = self._container.get("logger").unwrap()

    def process_batch(self, records: list[dict]) -> FlextResult[list[DataRecord]]:
        """Process batch of records using foundation patterns."""
        processed_records = []

        for record_data in records:
            record = DataRecord(**record_data)
            process_result = record.process()

            if process_result.is_failure:
                self._logger.error(f"Record processing failed: {process_result.error}")
                return FlextResult[list[DataRecord]].fail(f"Batch failed: {process_result.error}")

            processed_records.append(record)

        return FlextResult[list[DataRecord]].ok(processed_records)
```

---

## Database Integration Patterns

### Oracle Integration (flext-db-oracle)

```python
from flext_core import FlextResult, FlextConfig
from flext_db_oracle import FlextOracleClient

class OracleIntegration:
    def __init__(self, config: FlextConfig) -> None:
        self._config = config
        self._client = FlextOracleClient(config)

    def query_with_error_handling(self, sql: str) -> FlextResult[list[dict]]:
        """Execute Oracle query with foundation error handling."""
        try:
            connection_result = self._client.connect()
            if connection_result.is_failure:
                return FlextResult[list[dict]].fail(f"Connection failed: {connection_result.error}")

            query_result = self._client.execute(sql)
            return query_result  # Returns FlextResult[list[dict]]

        except Exception as e:
            return FlextResult[list[dict]].fail(f"Query execution failed: {str(e)}")
```

### LDAP Integration (flext-ldap)

```python
from flext_core import FlextResult, FlextModels
from flext_ldap import FlextLdapClient

class LdapUser(FlextModels.Entity):
    dn: str
    attributes: dict

    def update_attribute(self, name: str, value: str) -> FlextResult[None]:
        """Update LDAP attribute using foundation patterns."""
        if not name or not value:
            return FlextResult[None].fail("Attribute name and value required")

        self.attributes[name] = value
        self.add_domain_event("AttributeUpdated", {"attribute": name, "value": value})

        return FlextResult[None].ok(None)

class LdapIntegration:
    def __init__(self, client: FlextLdapClient) -> None:
        self._client = client

    def search_users(self, filter_expr: str) -> FlextResult[list[LdapUser]]:
        """Search LDAP users with foundation error handling."""
        search_result = self._client.search(filter_expr)

        if search_result.is_failure:
            return FlextResult[list[LdapUser]].fail(f"LDAP search failed: {search_result.error}")

        # Convert to domain models
        users = []
        for entry in search_result.unwrap():
            user = LdapUser(dn=entry["dn"], attributes=entry["attributes"])
            users.append(user)

        return FlextResult[list[LdapUser]].ok(users)
```

---

## Testing Integration Patterns

### Foundation Test Support

```python
from flext_core import FlextResult, FlextContainer
import pytest

# Test fixtures for foundation patterns
@pytest.fixture
def clean_container():
    """Provide clean container for each test."""
    container = FlextContainer.get_global()
    container.clear()  # Reset for test isolation
    yield container
    container.clear()  # Cleanup after test

@pytest.fixture
def test_config():
    """Provide test configuration."""
    from flext_core import FlextConfig

    class TestConfig(FlextConfig):
        test_mode: bool = True
        database_url: str = "sqlite:///:memory:"

    return TestConfig()

# Test patterns for FlextResult
def test_result_composition():
    """Test railway-oriented composition patterns."""
    def step1(x: int) -> FlextResult[int]:
        return FlextResult[int].ok(x + 1)

    def step2(x: int) -> FlextResult[int]:
        return FlextResult[int].ok(x * 2)

    def step3(x: int) -> FlextResult[str]:
        return FlextResult[str].ok(f"Result: {x}")

    # Test composition chain
    result = (
        FlextResult[int].ok(5)
        .flat_map(step1)
        .flat_map(step2)
        .flat_map(step3)
    )

    assert result.is_success
    assert result.unwrap() == "Result: 12"

# Test patterns for domain models
def test_domain_model_integration():
    """Test domain model patterns with events."""
    from flext_core import FlextModels

    class TestEntity(FlextModels.Entity):
        name: str

        def change_name(self, new_name: str) -> FlextResult[None]:
            if not new_name:
                return FlextResult[None].fail("Name required")

            self.name = new_name
            self.add_domain_event("NameChanged", {"new_name": new_name})
            return FlextResult[None].ok(None)

    entity = TestEntity(name="Original")
    result = entity.change_name("Updated")

    assert result.is_success
    assert entity.name == "Updated"
    assert len(entity._domain_events) == 1
    assert entity._domain_events[0]["event_type"] == "NameChanged"
```

### Integration Test Patterns

```python
import pytest
from flext_core import FlextResult, FlextContainer

class TestProjectIntegration:
    """Integration tests using foundation patterns."""

    def test_full_service_integration(self, clean_container, test_config):
        """Test complete service integration with foundation patterns."""
        # Register services
        clean_container.register("config", test_config)
        clean_container.register("logger", create_test_logger())

        # Test service integration
        service = ProjectService()
        result = service.perform_full_operation({"test": "data"})

        assert result.is_success
        processed_data = result.unwrap()
        assert processed_data["processed"] is True

    def test_error_propagation(self, clean_container):
        """Test error propagation through service layers."""
        service = ProjectService()

        # Test error handling
        result = service.perform_full_operation({})  # Empty data should fail

        assert result.is_failure
        assert "required" in result.error.lower()
```

---

## Configuration Integration

### Environment-Aware Configuration

```python
from flext_core import FlextConfig
from typing import Optional

class ProjectConfig(FlextConfig):
    """Project configuration using foundation patterns."""

    # Required settings
    database_url: str
    api_key: str

    # Optional with defaults
    debug: bool = False
    port: int = 8000
    timeout: int = 30
    log_level: str = "INFO"

    # Environment-specific settings
    redis_url: Optional[str] = None
    cache_ttl: int = 3600

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = "PROJECT_"

# Usage in service initialization
def initialize_project(config_overrides: Optional[dict] = None) -> FlextResult[None]:
    """Initialize project with configuration."""
    try:
        # Load configuration with optional overrides
        config_data = config_overrides or {}
        config = ProjectConfig(**config_data)

        # Register in container for global access
        container = FlextContainer.get_global()
        container.register("config", config)

        return FlextResult[None].ok(None)

    except Exception as e:
        return FlextResult[None].fail(f"Configuration failed: {str(e)}")
```

### Multi-Environment Support

```python
from flext_core import FlextConfig, FlextResult
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class EnvironmentConfig(FlextConfig):
    environment: Environment = Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    def validate_environment(self) -> FlextResult[None]:
        """Validate environment-specific requirements."""
        if self.is_production and self.debug:
            return FlextResult[None].fail("Debug mode not allowed in production")

        if self.is_production and not self.api_key:
            return FlextResult[None].fail("API key required in production")

        return FlextResult[None].ok(None)
```

---

## Performance Integration Patterns

### Caching Integration

```python
from flext_core import FlextResult, FlextContainer
from typing import Optional, Any

class CacheService:
    """Cache service using foundation patterns."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, key: str) -> FlextResult[Optional[Any]]:
        """Get cached value with foundation error handling."""
        if not key:
            return FlextResult[Optional[Any]].fail("Cache key required")

        value = self._cache.get(key)
        return FlextResult[Optional[Any]].ok(value)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> FlextResult[None]:
        """Set cached value with foundation error handling."""
        if not key:
            return FlextResult[None].fail("Cache key required")

        self._cache[key] = value
        return FlextResult[None].ok(None)

# Service with caching integration
class CachedService:
    def __init__(self) -> None:
        container = FlextContainer.get_global()
        self._cache = container.get("cache").unwrap()

    def get_expensive_data(self, id: str) -> FlextResult[dict]:
        """Get data with caching using foundation patterns."""
        cache_key = f"expensive_data:{id}"

        # Try cache first
        cache_result = self._cache.get(cache_key)
        if cache_result.is_success and cache_result.unwrap() is not None:
            return FlextResult[dict].ok(cache_result.unwrap())

        # Compute expensive operation
        data_result = self._compute_expensive_data(id)
        if data_result.is_failure:
            return data_result

        # Cache the result
        self._cache.set(cache_key, data_result.unwrap())

        return data_result
```

---

## Monitoring and Observability Integration

### Structured Logging Integration

```python
from flext_core import FlextLogger, FlextResult
import structlog

class ObservableService:
    """Service with structured logging using foundation patterns."""

    def __init__(self) -> None:
        self._logger = FlextLogger(__name__)

    def process_with_observability(self, data: dict) -> FlextResult[dict]:
        """Process data with comprehensive logging."""
        operation_id = self._generate_operation_id()

        self._logger.info(
            "Starting data processing",
            extra={
                "operation_id": operation_id,
                "data_size": len(data),
                "data_keys": list(data.keys())
            }
        )

        try:
            # Business logic with detailed logging
            validation_result = self._validate_data(data)
            if validation_result.is_failure:
                self._logger.error(
                    "Data validation failed",
                    extra={
                        "operation_id": operation_id,
                        "error": validation_result.error
                    }
                )
                return validation_result

            # Process data
            processed_data = self._process_data(data)

            self._logger.info(
                "Data processing completed",
                extra={
                    "operation_id": operation_id,
                    "processed_size": len(processed_data)
                }
            )

            return FlextResult[dict].ok(processed_data)

        except Exception as e:
            self._logger.exception(
                "Unexpected error during processing",
                extra={
                    "operation_id": operation_id,
                    "error": str(e)
                }
            )
            return FlextResult[dict].fail(f"Processing failed: {str(e)}")
```

---

## Migration Patterns

### Gradual Migration to Foundation Patterns

```python
# Phase 1: Wrapper pattern for gradual migration
from flext_core import FlextResult
from typing import Union

def migrate_existing_function(old_function) -> callable:
    """Wrapper to migrate existing functions to FlextResult patterns."""

    def wrapper(*args, **kwargs) -> FlextResult[any]:
        try:
            result = old_function(*args, **kwargs)
            return FlextResult[any].ok(result)
        except Exception as e:
            return FlextResult[any].fail(str(e))

    return wrapper

# Phase 2: Service layer migration
class MigrationService:
    """Service to help migrate from old patterns to foundation patterns."""

    def migrate_error_handling(self, old_result: Union[dict, Exception]) -> FlextResult[dict]:
        """Migrate old error handling to FlextResult pattern."""
        if isinstance(old_result, Exception):
            return FlextResult[dict].fail(str(old_result))

        if isinstance(old_result, dict) and old_result.get("error"):
            return FlextResult[dict].fail(old_result["error"])

        return FlextResult[dict].ok(old_result)
```

---

## Best Practices Summary

### Integration Checklist

- [ ] Use FlextResult[T] for all operations that can fail
- [ ] Access FlextContainer via `.get_global()` for dependency injection
- [ ] Inherit from FlextModels for domain entities
- [ ] Use FlextConfig for environment-aware configuration
- [ ] Implement structured logging with FlextLogger
- [ ] Test with foundation patterns and clean container fixtures
- [ ] Maintain API compatibility with both `.data` and `.value` access
- [ ] Follow Clean Architecture layering with foundation at the base

### Common Anti-Patterns to Avoid

- ❌ Direct exception handling instead of FlextResult patterns
- ❌ Multiple dependency injection approaches in the same project
- ❌ Bypassing foundation exports with internal imports
- ❌ Inconsistent error handling across service layers
- ❌ Configuration scattered across multiple mechanisms
- ❌ Domain logic mixed with infrastructure concerns

---

**Integration Authority**: Patterns for consistent FLEXT ecosystem integration
**Foundation Compatibility**: All patterns tested with FLEXT-Core v0.9.0
**Ecosystem Proven**: Used across 45+ FLEXT projects successfully