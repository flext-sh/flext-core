# Configuration Guide

**FLEXT-Core v0.9.0** configuration management with environment-aware settings using Pydantic integration.

**Last Updated**: September 17, 2025

---

## Overview

FLEXT-Core provides `FlextConfig` base class for environment-aware configuration management with automatic environment variable loading, type validation, and configuration inheritance.

### Key Features

- **Environment Integration** - Automatic loading from environment variables and .env files
- **Type Safety** - Pydantic-based validation with Python 3.13+ type annotations
- **Validation** - Built-in validation with custom validators
- **Inheritance** - Configuration composition and inheritance patterns
- **Ecosystem Standard** - Used across all 45+ FLEXT projects

---

## Basic Configuration

### Simple Configuration

```python
from flext_core import FlextConfig

class AppConfig(FlextConfig):
    """Basic application configuration."""
    database_url: str
    api_key: str
    debug: bool = False
    timeout: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False

# Automatically loads from environment variables or .env file
config = AppConfig()
```

### Environment Variable Mapping

Environment variables are automatically mapped to configuration fields:

```bash
# .env file or environment variables
DATABASE_URL=postgresql://localhost:5432/mydb
API_KEY=secret-key-here
DEBUG=true
TIMEOUT=60
```

---

## Advanced Configuration Patterns

### Prefix-based Configuration

```python
class DatabaseConfig(FlextConfig):
    """Database configuration with prefix."""
    host: str = "localhost"
    port: int = 5432
    username: str
    password: str
    database: str

    class Config:
        env_prefix = "DB_"
        env_file = ".env"

# Maps to environment variables:
# DB_HOST, DB_PORT, DB_USERNAME, DB_PASSWORD, DB_DATABASE
```

### Nested Configuration

```python
class RedisConfig(FlextConfig):
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0

    class Config:
        env_prefix = "REDIS_"

class LoggingConfig(FlextConfig):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"

    class Config:
        env_prefix = "LOG_"

class AppConfig(FlextConfig):
    """Composed application configuration."""
    app_name: str = "flext-core"
    version: str = "0.9.0"

    # Nested configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    logging: LoggingConfig = LoggingConfig()

    class Config:
        env_file = ".env"
```

---

## Configuration Validation

### Built-in Validators

```python
from pydantic import validator, Field
from flext_core import FlextConfig

class ValidatedConfig(FlextConfig):
    """Configuration with validation."""

    # Field-level validation
    port: int = Field(ge=1, le=65535, description="Valid port number")
    email: str = Field(regex=r'^[^@]+@[^@]+\.[^@]+$', description="Valid email address")

    # Custom validation
    database_url: str

    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith(('postgresql://', 'mysql://', 'sqlite://')):
            raise ValueError('Invalid database URL scheme')
        return v

    # Computed properties
    @property
    def connection_string(self) -> str:
        """Build connection string from components."""
        return f"{self.database_url}?timeout={self.timeout}"
```

### FlextResult Integration

```python
from flext_core import FlextConfig, FlextResult

class SafeConfig(FlextConfig):
    """Configuration with FlextResult validation."""

    def validate_configuration(self) -> FlextResult[None]:
        """Validate configuration state."""
        if not self.database_url:
            return FlextResult[None].fail("Database URL is required")

        if self.timeout <= 0:
            return FlextResult[None].fail("Timeout must be positive")

        return FlextResult[None].ok(None)

# Usage with validation
config = SafeConfig()
validation_result = config.validate_configuration()
if validation_result.is_failure:
    print(f"Configuration error: {validation_result.error}")
```

---

## Environment Management

### Multiple Environment Support

```python
from enum import Enum
from flext_core import FlextConfig

class Environment(str, Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class EnvironmentConfig(FlextConfig):
    """Environment-aware configuration."""
    environment: Environment = Environment.DEVELOPMENT

    # Environment-specific settings
    debug: bool = False
    log_level: str = "INFO"

    @validator('debug', pre=True, always=True)
    def set_debug_from_environment(cls, v, values):
        env = values.get('environment', Environment.DEVELOPMENT)
        if env == Environment.DEVELOPMENT:
            return True
        return v

    @validator('log_level', pre=True, always=True)
    def set_log_level_from_environment(cls, v, values):
        env = values.get('environment', Environment.DEVELOPMENT)
        if env == Environment.DEVELOPMENT:
            return "DEBUG"
        elif env == Environment.PRODUCTION:
            return "WARNING"
        return v
```

### Configuration Loading Strategy

```python
from pathlib import Path
from flext_core import FlextConfig, FlextResult

class ConfigLoader:
    """Configuration loading with multiple sources."""

    @staticmethod
    def load_config() -> FlextResult[AppConfig]:
        """Load configuration from multiple sources."""

        # Try loading from different sources
        config_paths = [
            ".internal.invalid",       # Local overrides
            ".env",             # Default environment
            "/etc/app/.env",    # System-wide
        ]

        for path in config_paths:
            if Path(path).exists():
                try:
                    config = AppConfig(_env_file=path)
                    return FlextResult[AppConfig].ok(config)
                except Exception as e:
                    continue

        # Fall back to environment variables only
        try:
            config = AppConfig()
            return FlextResult[AppConfig].ok(config)
        except Exception as e:
            return FlextResult[AppConfig].fail(f"Configuration loading failed: {e}")
```

---

## FLEXT Ecosystem Integration

### Service Configuration Pattern

```python
from flext_core import FlextConfig, FlextContainer, FlextResult

class ServiceConfig(FlextConfig):
    """Base configuration for FLEXT services."""
    service_name: str
    version: str = "0.9.0"

    # Common service settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # FLEXT patterns
    enable_metrics: bool = True
    enable_tracing: bool = True

    class Config:
        env_file = ".env"

def register_config(container: FlextContainer, config: ServiceConfig) -> FlextResult[None]:
    """Register configuration in container."""
    register_result = container.register("config", config)
    if register_result.is_failure:
        return FlextResult[None].fail(f"Config registration failed: {register_result.error}")

    # Register individual config sections
    if hasattr(config, 'database'):
        db_result = container.register("database_config", config.database)
        if db_result.is_failure:
            return FlextResult[None].fail(f"Database config registration failed: {db_result.error}")

    return FlextResult[None].ok(None)
```

### Multi-Project Configuration

```python
class FlextEcosystemConfig(FlextConfig):
    """Configuration for multi-project FLEXT deployments."""

    # Core services
    core_api_url: str = "http://localhost:8000"
    auth_service_url: str = "http://localhost:8001"

    # Data services
    oracle_host: str = "localhost"
    oracle_port: int = 1521
    ldap_host: str = "localhost"
    ldap_port: int = 389

    # Integration settings
    singer_state_dir: str = "/tmp/singer-state"
    meltano_project_dir: str = "./meltano"

    # Observability
    metrics_endpoint: str = "http://localhost:9090"
    tracing_endpoint: str = "http://localhost:14268"

    class Config:
        env_prefix = "FLEXT_"
        env_file = ".env"
```

---

## Configuration Patterns

### Factory Pattern

```python
from flext_core import FlextConfig, FlextResult

class ConfigFactory:
    """Factory for creating configurations."""

    @staticmethod
    def create_database_config(environment: str) -> FlextResult[DatabaseConfig]:
        """Create database configuration for environment."""

        config_map = {
            "development": {
                "host": "localhost",
                "port": 5432,
                "database": "flext_dev"
            },
            "testing": {
                "host": "localhost",
                "port": 5432,
                "database": "flext_test"
            },
            "production": {
                "host": "prod-db.company.com",
                "port": 5432,
                "database": "flext_prod"
            }
        }

        if environment not in config_map:
            return FlextResult[DatabaseConfig].fail(f"Unknown environment: {environment}")

        try:
            config = DatabaseConfig(**config_map[environment])
            return FlextResult[DatabaseConfig].ok(config)
        except Exception as e:
            return FlextResult[DatabaseConfig].fail(f"Config creation failed: {e}")
```

### Configuration Composition

```python
class CompositeConfig(FlextConfig):
    """Composite configuration pattern."""

    def __init__(self, **data):
        # Load base configuration
        super().__init__(**data)

        # Apply configuration overlays
        self._apply_environment_overlay()
        self._apply_feature_flags()

    def _apply_environment_overlay(self):
        """Apply environment-specific overrides."""
        if self.environment == "production":
            self.debug = False
            self.log_level = "WARNING"

    def _apply_feature_flags(self):
        """Apply feature flag configuration."""
        # Could integrate with feature flag services
        pass

    def get_effective_config(self) -> dict:
        """Get final effective configuration."""
        return self.dict(exclude_none=True)
```

---

## Testing Configuration

### Test Configuration

```python
class TestConfig(FlextConfig):
    """Configuration for testing environments."""

    # Override for testing
    database_url: str = "sqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/15"  # Test database

    # Testing specific
    test_mode: bool = True
    mock_external_services: bool = True

    class Config:
        env_prefix = "TEST_"

# Test fixtures
import pytest

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig()

@pytest.fixture
def app_config():
    """Provide application configuration for testing."""
    return AppConfig(
        database_url="sqlite:///:memory:",
        debug=True
    )
```

### Configuration Validation Tests

```python
import pytest
from flext_core import FlextResult

def test_config_validation():
    """Test configuration validation."""

    # Valid configuration
    config = AppConfig(
        database_url="postgresql://localhost:5432/test",
        api_key="valid-key"
    )

    validation_result = config.validate_configuration()
    assert validation_result.is_success

    # Invalid configuration
    invalid_config = AppConfig(
        database_url="",  # Invalid
        api_key="valid-key"
    )

    validation_result = invalid_config.validate_configuration()
    assert validation_result.is_failure
    assert "Database URL is required" in validation_result.error
```

---

## Security Considerations

### Secret Management

```python
from pydantic import SecretStr
from flext_core import FlextConfig

class SecureConfig(FlextConfig):
    """Configuration with secret handling."""

    # Secrets are masked in logs and repr
    database_password: SecretStr
    api_secret_key: SecretStr

    # Regular fields
    database_host: str
    database_user: str

    def get_database_url(self) -> str:
        """Build database URL with secret."""
        password = self.database_password.get_secret_value()
        return f"postgresql://{self.database_user}:{password}@{self.database_host}/db"

    class Config:
        # Don't include secrets in JSON serialization
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None
        }
```

### Environment Isolation

```python
class ProductionConfig(FlextConfig):
    """Production-specific security configuration."""

    # Require explicit production settings
    environment: str = Field(..., regex="^production$")

    # Security settings
    allowed_hosts: list[str] = Field(min_items=1)
    cors_origins: list[str] = []

    # TLS configuration
    tls_cert_path: str
    tls_key_path: str

    @validator('debug', pre=True, always=True)
    def disable_debug_in_production(cls, v):
        return False  # Always False in production
```

---

## Troubleshooting

### Common Configuration Issues

1. **Environment Variable Not Loading**
   ```python
   # Check if .env file exists and is readable
   from pathlib import Path
   env_file = Path(".env")
   print(f"Env file exists: {env_file.exists()}")
   print(f"Env file readable: {env_file.is_file()}")
   ```

2. **Type Conversion Errors**
   ```python
   # Debug type conversion
   import os
   print(f"PORT env var: '{os.getenv('PORT')}'")
   print(f"PORT type: {type(os.getenv('PORT'))}")

   # Explicit conversion in config
   port: int = Field(default=8000, description="Server port")
   ```

3. **Validation Failures**
   ```python
   # Use FlextResult for graceful validation
   def safe_config_load() -> FlextResult[AppConfig]:
       try:
           config = AppConfig()
           return FlextResult[AppConfig].ok(config)
       except ValidationError as e:
           return FlextResult[AppConfig].fail(f"Config validation failed: {e}")
   ```

---

**Configuration Guide Complete** - For integration with dependency injection, see [API Reference](api-reference.md). For service patterns, see [Architecture](architecture.md).