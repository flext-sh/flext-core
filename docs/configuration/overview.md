# Configuration Management Overview

Guide to FLEXT Core's type-safe configuration system.

For verified project capabilities and accurate status information, see [ACTUAL_CAPABILITIES.md](../ACTUAL_CAPABILITIES.md).

## Overview

FLEXT Core provides configuration management through Pydantic v2 integration, offering type safety, environment variable support, and validation out of the box. The configuration system is designed to support both simple applications and complex multi-service architectures.

## Features

### Core Capabilities

- **Type Safety**: Full Pydantic v2 validation with type hints
- **Environment Variables**: Automatic loading with customizable prefixes
- **Validation**: Built-in and custom validators for data integrity
- **Multi-Environment**: Support for development, staging, and production configs
- **Composability**: Nested configurations and inheritance patterns

### Integration Support

- **Singer SDK**: Configuration for data extraction/loading
- **CLI Applications**: Command-line tool configuration
- **Web Services**: FastAPI and Flask configuration patterns
- **Database Connections**: Connection pooling and SSL configuration
- **Cache Services**: Redis and memcached configuration

## Core API

### Available Imports

```python
# Primary import for most use cases
from flext_core import FlextConfig

# Advanced configuration imports
from flext_core.config import FlextConfig, FlextBaseSettings
from flext_core.config_models import (
    FlextDatabaseConfig,
    FlextRedisConfig,
    FlextServiceConfig
)
from flext_core.config_base import ConfigurationManager
```

### Basic Usage

```python
from flext_core import FlextConfig
from typing import Optional

class AppSettings(FlextConfig):
    """Application configuration with sensible defaults."""

    # Application metadata
    app_name: str = "My FLEXT App"
    version: str = "0.9.0"
    debug: bool = False
    environment: str = "development"

    # Database configuration
    database_url: str = "sqlite:///app.db"
    database_pool_size: int = 5
    database_timeout: int = 30

    # API configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 1

    # Optional features
    redis_url: Optional[str] = None
    enable_metrics: bool = False
    enable_tracing: bool = False

    class Config:
        env_prefix = "MYAPP_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage example
def initialize_app():
    """Initialize application with configuration."""
    settings = AppSettings()

    # Access configuration values
    print(f"Starting {settings.app_name} v{settings.version}")
    print(f"Environment: {settings.environment}")
    print(f"API endpoint: http://{settings.api_host}:{settings.api_port}")

    # Environment-specific behavior
    if settings.environment == "production":
        settings.database_pool_size = 20
        settings.api_workers = 4
    elif settings.environment == "staging":
        settings.database_pool_size = 10
        settings.api_workers = 2

    # Optional feature flags
    if settings.enable_metrics:
        print("Metrics collection enabled")

    if settings.enable_tracing:
        print("Distributed tracing enabled")

    return settings
```

### Environment Variables

How to configure environment variables:

```bash
# Basic settings
export MYAPP_APP_NAME="Production App"
export MYAPP_DEBUG=false

# Database settings
export MYAPP_DATABASE_URL="postgresql://user:pass@localhost/db"
export MYAPP_DATABASE_POOL_SIZE=20

# API settings
export MYAPP_API_HOST="0.0.0.0"
export MYAPP_API_PORT=9000

# Optional features
export MYAPP_REDIS_URL="redis://localhost:6379"
export MYAPP_ENABLE_METRICS=true
```

### Advanced Validation

```python
from flext_core import FlextConfig
from pydantic import field_validator, Field, model_validator
from typing import Optional
import re

class DatabaseSettings(FlextConfig):
    """Database configuration with validation."""

    # Connection parameters
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, ge=1, le=65535, description="Database port")
    name: str = Field("myapp", pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    user: str = Field("postgres", min_length=1, max_length=63)
    password: str = Field("", description="Database password")

    # Connection pool settings
    pool_min_size: int = Field(2, ge=1, le=100)
    pool_max_size: int = Field(10, ge=1, le=100)
    pool_timeout: int = Field(30, ge=1, le=300)

    # SSL configuration
    ssl_mode: str = Field("prefer", description="SSL mode")
    ssl_cert_path: Optional[str] = Field(None, description="SSL certificate path")
    ssl_key_path: Optional[str] = Field(None, description="SSL key path")

    @field_validator("ssl_mode")
    @classmethod
    def validate_ssl_mode(cls, v: str) -> str:
        """Validate SSL mode."""
        valid_modes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if v not in valid_modes:
            raise ValueError(f"SSL mode must be one of: {', '.join(valid_modes)}")
        return v

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password strength in production."""
        if v and len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @model_validator(mode='after')
    def validate_pool_sizes(self) -> 'DatabaseSettings':
        """Ensure pool_max_size >= pool_min_size."""
        if self.pool_max_size < self.pool_min_size:
            raise ValueError("pool_max_size must be >= pool_min_size")
        return self

    @model_validator(mode='after')
    def validate_ssl_paths(self) -> 'DatabaseSettings':
        """Validate SSL certificate paths."""
        if self.ssl_mode in {"verify-ca", "verify-full"}:
            if not self.ssl_cert_path:
                raise ValueError(f"ssl_cert_path required for ssl_mode={self.ssl_mode}")
        return self

    @property
    def connection_url(self) -> str:
        """Build database connection URL with parameters."""
        auth = f"{self.user}:{self.password}@" if self.password else f"{self.user}@"
        base_url = f"postgresql://{auth}{self.host}:{self.port}/{self.name}"

        # Add SSL parameters if needed
        params = []
        if self.ssl_mode != "disable":
            params.append(f"sslmode={self.ssl_mode}")
        if self.ssl_cert_path:
            params.append(f"sslcert={self.ssl_cert_path}")
        if self.ssl_key_path:
            params.append(f"sslkey={self.ssl_key_path}")

        if params:
            return f"{base_url}?{'&'.join(params)}"
        return base_url

    def get_async_url(self) -> str:
        """Get async connection URL for asyncpg."""
        return self.connection_url.replace("postgresql://", "postgresql+asyncpg://")

    def get_pool_config(self) -> dict:
        """Get connection pool configuration."""
        return {
            "min_size": self.pool_min_size,
            "max_size": self.pool_max_size,
            "timeout": self.pool_timeout,
            "command_timeout": 60,
            "max_queries": 50000,
            "max_inactive_connection_lifetime": 300
        }

# Usage example
def setup_database():
    """Setup database with validated configuration."""
    try:
        db_config = DatabaseSettings()
        print(f"Database URL: {db_config.connection_url}")
        print(f"Async URL: {db_config.get_async_url()}")
        print(f"Pool config: {db_config.get_pool_config()}")
        return db_config
    except ValueError as e:
        print(f"Configuration error: {e}")
        raise
```

### Multi-Service Architecture

```python
"""
Configuration for multiple services.
"""

from flext_core import FlextConfig
from pydantic import Field
from typing import Optional

class ServiceConfig(FlextConfig):
    """Base configuration for services."""

    service_name: str = Field(..., description="Service name")
    log_level: str = Field("INFO", description="Logging level")
    metrics_enabled: bool = Field(False, description="Enable metrics")

    class Config:
        env_prefix = "SERVICE_"

class APIServiceConfig(ServiceConfig):
    """Configuration for API service."""

    service_name: str = "api-service"
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    workers: int = Field(1, description="Number of workers")

    # Security
    secret_key: str = Field(..., description="JWT secret key")
    cors_origins: FlextTypes.Core.StringList = Field(default_factory=list, description="CORS origins")

    class Config:
        env_prefix = "API_"

class DatabaseServiceConfig(ServiceConfig):
    """Configuration for database service."""

    service_name: str = "database-service"
    connection_url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(10, description="Connection pool size")
    timeout: int = Field(30, description="Connection timeout")

    class Config:
        env_prefix = "DB_"

class CacheServiceConfig(ServiceConfig):
    """Configuration for cache service."""

    service_name: str = "cache-service"
    redis_url: str = Field("redis://localhost:6379", description="Redis URL")
    ttl: int = Field(3600, description="Default TTL in seconds")
    max_connections: int = Field(10, description="Max Redis connections")

    class Config:
        env_prefix = "CACHE_"

# Application configuration
class AppConfiguration:
    """Main application configuration."""

    def __init__(self):
        self.api = APIServiceConfig()
        self.database = DatabaseServiceConfig()
        self.cache = CacheServiceConfig()

    def validate_all(self) -> bool:
        """Validate all configurations."""
        try:
            # All configs are validated on instantiation
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def print_summary(self) -> None:
        """Print configuration summary."""
        print("=== Application Configuration ===")
        print(f"API Service: {self.api.host}:{self.api.port}")
        print(f"Database: {self.database.connection_url}")
        print(f"Cache: {self.cache.redis_url}")
        print(f"Log Level: {self.api.log_level}")
        print(f"Metrics: {self.api.metrics_enabled}")

# Usage
def main():
    """Load and validate all configurations."""
    config = AppConfiguration()

    if config.validate_all():
        config.print_summary()
        print("✅ All configurations valid")
    else:
        print("❌ Configuration validation failed")
        exit(1)

if __name__ == "__main__":
    main()
```

## Testing Configuration

```python
"""
Testing configuration patterns.
"""

import pytest
import os
from flext_core import FlextConfig

def test_default_configuration():
    """Test default configuration values."""

    class TestSettings(FlextConfig):
        app_name: str = "test-app"
        debug: bool = True
        port: int = 8000

    settings = TestSettings()
    assert settings.app_name == "test-app"
    assert settings.debug is True
    assert settings.port == 8000

def test_environment_override():
    """Test environment variable override."""

    class TestSettings(FlextConfig):
        app_name: str = "test-app"
        port: int = 8000

        class Config:
            env_prefix = "TEST_"

    # Set environment variables
    os.environ["TEST_APP_NAME"] = "overridden-app"
    os.environ["TEST_PORT"] = "9000"

    try:
        settings = TestSettings()
        assert settings.app_name == "overridden-app"
        assert settings.port == 9000
    finally:
        # Cleanup
        os.environ.pop("TEST_APP_NAME", None)
        os.environ.pop("TEST_PORT", None)

def test_validation_error():
    """Test configuration validation."""

    class TestSettings(FlextConfig):
        port: int = Field(..., ge=1, le=65535)

    # This should fail validation
    os.environ["PORT"] = "70000"  # Invalid port

    try:
        with pytest.raises(ValueError):
            TestSettings()
    finally:
        os.environ.pop("PORT", None)

if __name__ == "__main__":
    test_default_configuration()
    test_environment_override()
    print("✅ All configuration tests passed")
```

## Best Practices

### 1. Environment-Specific Files

Organize configuration files by environment:

```
config/
├── .env.development    # Development defaults
├── .env.staging       # Staging configuration
├── .env.production    # Production configuration
└── .env.local         # Local overrides (gitignored)
```

### 2. Secret Management

```python
from flext_core import FlextConfig
from pydantic import Field, SecretStr

class SecureSettings(FlextConfig):
    """Configuration with secure secret handling."""

    # Sensitive values use SecretStr
    api_key: SecretStr = Field(..., description="API key")
    database_password: SecretStr = Field(..., description="DB password")
    jwt_secret: SecretStr = Field(..., description="JWT secret")

    class Config:
        env_prefix = "SECURE_"

    def get_api_key(self) -> str:
        """Get API key value (careful with logging)."""
        return self.api_key.get_secret_value()
```

### 3. Configuration Validation

```python
def validate_configuration(settings: FlextConfig) -> FlextResult[None]:
    """Validate configuration at startup."""
    errors = []

    # Check required services
    if settings.database_url == "sqlite:///app.db" and settings.environment == "production":
        errors.append("SQLite not recommended for production")

    # Validate external connections
    if settings.redis_url and not can_connect_redis(settings.redis_url):
        errors.append("Cannot connect to Redis")

    if errors:
        return FlextResult[None].fail("; ".join(errors))

    return FlextResult[None].ok(None)
```

### 4. Configuration Documentation

Always document configuration options:

```python
class DocumentedSettings(FlextConfig):
    """Application settings with documentation.

    Environment Variables:
        APP_NAME: Application name for logging and metrics
        APP_DEBUG: Enable debug mode (verbose logging)
        APP_PORT: HTTP server port (1-65535)
        APP_WORKERS: Number of worker processes

    Example:
        export APP_NAME="my-service"
        export APP_DEBUG=false
        export APP_PORT=8080
        export APP_WORKERS=4
    """

    app_name: str = Field(
        "my-app",
        description="Application identifier used in logs and metrics"
    )
    debug: bool = Field(
        False,
        description="Enable debug mode with verbose logging"
    )
    port: int = Field(
        8000,
        ge=1,
        le=65535,
        description="HTTP server port"
    )
    workers: int = Field(
        1,
        ge=1,
        le=16,
        description="Number of worker processes"
    )
```

## Common Patterns

### Service Discovery

```python
class ServiceDiscoverySettings(FlextConfig):
    """Configuration for service discovery."""

    # Service registry
    consul_host: str = "localhost"
    consul_port: int = 8500

    # Service metadata
    service_name: str
    service_id: str
    service_tags: FlextTypes.Core.StringList = []

    # Health check
    health_check_interval: int = 10
    health_check_timeout: int = 5

    def get_consul_url(self) -> str:
        """Get Consul API URL."""
        return f"http://{self.consul_host}:{self.consul_port}"
```

### Feature Flags

```python
class FeatureFlags(FlextConfig):
    """Feature flag configuration."""

    # Feature toggles
    enable_new_ui: bool = False
    enable_beta_features: bool = False
    enable_analytics: bool = True

    # Rollout percentages
    new_algorithm_rollout: int = Field(0, ge=0, le=100)

    # A/B testing
    ab_test_groups: dict[str, int] = {}

    def is_feature_enabled(self, feature: str, user_id: str = None) -> bool:
        """Check if feature is enabled for user."""
        # Implement percentage-based rollout logic
        pass
```

## Troubleshooting

### Common Issues

1. **Environment variables not loading**:

   - Check prefix matches (case-sensitive on Linux/Mac)
   - Verify .env file location and encoding
   - Use `export` command on Unix systems

2. **Validation errors**:

   - Enable debug logging to see actual values
   - Check field constraints (min/max, patterns)
   - Verify required fields have values

3. **Type conversion errors**:
   - Ensure correct types in environment variables
   - Use "true"/"false" for booleans (not 1/0)
   - Lists should be comma-separated

### Debug Configuration Loading

```python
import os
from flext_core import FlextConfig

def debug_configuration():
    """Debug configuration loading."""

    # Show all environment variables
    print("Environment variables:")
    for key, value in os.environ.items():
        if key.startswith("MYAPP_"):
            print(f"  {key}={value}")

    # Try loading configuration
    try:
        settings = AppSettings()
        print("\nLoaded configuration:")
        print(settings.model_dump_json(indent=2))
    except Exception as e:
        print(f"\nConfiguration error: {e}")
        import traceback
        traceback.print_exc()
```

---

For implementation examples, see [Examples Guide](../examples/overview.md).
