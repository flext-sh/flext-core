# Configuration with FLEXT-Core

**Configuration management patterns for FLEXT ecosystem applications**

---

## Overview

FLEXT-Core provides configuration management through the `FlextConfig` class, which integrates with the foundation patterns for type-safe configuration handling.

## Basic Configuration

### Using FlextConfig

```python
from flext_core import FlextConfig
from pydantic import BaseSettings, Field

class AppConfig(BaseSettings):
    """Application configuration with environment support."""

    # Database settings
    database_url: str = Field("postgresql://localhost/app", description="Database URL")
    database_pool_size: int = Field(10, description="Database pool size")

    # API settings
    api_host: str = Field("localhost", description="API host")
    api_port: int = Field(8000, description="API port")
    debug: bool = Field(False, description="Debug mode")

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = "APP_"

# Usage
config = AppConfig()
print(f"Database URL: {config.database_url}")
print(f"Running on {config.api_host}:{config.api_port}")
```

### Environment Variables

Configuration automatically loads from environment variables:

```bash
# .env file or environment
APP_DATABASE_URL=postgresql://prod.example.com/app
APP_DATABASE_POOL_SIZE=25
APP_API_HOST=0.0.0.0
APP_API_PORT=80
APP_DEBUG=false
```

## Integration with FlextResult

Configuration operations can use FlextResult for error handling:

```python
from flext_core import FlextConfig, FlextResult

class DatabaseConfig(BaseSettings):
    database_url: str = Field(..., description="Database URL")
    timeout: int = Field(30, description="Connection timeout")

    def validate_connection(self) -> FlextResult[bool]:
        """Validate database configuration."""
        if not self.database_url.startswith(("postgresql://", "mysql://")):
            return FlextResult[bool].fail("Invalid database URL format")

        if self.timeout <= 0:
            return FlextResult[bool].fail("Timeout must be positive")

        return FlextResult[bool].ok(True)

# Usage with error handling
config = DatabaseConfig()
validation_result = config.validate_connection()
if validation_result.is_failure:
    print(f"Configuration error: {validation_result.error}")
```

## Service Configuration Pattern

Configuration classes work with dependency injection:

```python
from flext_core import FlextConfig, FlextContainer, FlextDomainService

class ServiceConfig(BaseSettings):
    """Service-specific configuration."""
    redis_url: str = Field("redis://localhost:6379", description="Redis URL")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")

    class Config:
        env_prefix = "CACHE_"

class CacheService(FlextDomainService):
    """Cache service using configuration."""

    def __init__(self) -> None:
        super().__init__()
        self._config = ServiceConfig()
        self._container = FlextContainer.get_global()

    def get_cache_ttl(self) -> int:
        return self._config.cache_ttl

# Register configuration
container = FlextContainer.get_global()
container.register("cache_config", ServiceConfig())
```

## Development vs Production

Different configurations for different environments:

```python
import os
from flext_core import FlextConfig

class AppConfig(BaseSettings):
    """Environment-aware configuration."""

    environment: str = Field("development", description="Environment")
    database_url: str = Field(..., description="Database URL")
    log_level: str = Field("INFO", description="Log level")

    class Config:
        env_file = ".env"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

# Usage
config = AppConfig()
if config.is_development:
    print("Running in development mode")
    # Development-specific settings
elif config.is_production:
    print("Running in production mode")
    # Production-specific settings
```

## Validation Patterns

Use Pydantic validators for configuration validation:

```python
from pydantic import validator, BaseSettings

class ValidatedConfig(BaseSettings):
    """Configuration with custom validation."""

    port: int = Field(..., description="Server port")
    timeout: float = Field(..., description="Request timeout")

    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v

    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v
```

## Testing Configuration

Mock configuration for testing:

```python
import pytest
from unittest.mock import patch

def test_config_loading():
    """Test configuration loading with environment variables."""

    test_env = {
        "APP_DATABASE_URL": "postgresql://test.db/app",
        "APP_DEBUG": "true"
    }

    with patch.dict(os.environ, test_env):
        config = AppConfig()
        assert config.database_url == "postgresql://test.db/app"
        assert config.debug is True

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return AppConfig(
        database_url="sqlite:///test.db",
        debug=True,
        api_port=8080
    )
```

---

**Configuration Management**: FLEXT-Core v0.9.0 provides foundation patterns for type-safe, environment-aware configuration management throughout the FLEXT ecosystem.