# Configuration Management Overview

FLEXT Core provides a sophisticated, type-safe configuration system built on top of Pydantic V2, designed for enterprise applications with complex configuration requirements.

## Key Features

- **Type Safety**: Full Pydantic validation with IDE support
- **Environment Variable Support**: Automatic env var loading with prefixes
- **Multi-Environment**: Development, testing, staging, production
- **Framework Integration**: Singer, Django, CLI adapters
- **Advanced Features**: Multi-file configs, secret management
- **Zero Dependencies**: Uses core Python and Pydantic only

## Configuration Architecture

```mermaid
graph TB
    A[FlextCoreSettings] --> B[BaseSettings]
    B --> C[Pydantic BaseSettings]
    A --> D[Environment Detection]
    A --> E[Validation System]
    A --> F[Secret Management]

    G[Your App Settings] --> A
    G --> H[Custom Validators]
    G --> I[Framework Adapters]

    I --> J[Singer Config]
    I --> K[Django Settings]
    I --> L[CLI Config]
```

## Basic Usage

### Simple Application Configuration

```python
from flext_core.config import FlextCoreSettings
from pydantic import Field

class MyAppSettings(FlextCoreSettings):
    """Application configuration with type safety."""

    # Database configuration
    database_url: str = Field(
        "sqlite:///app.db",
        description="Database connection URL"
    )

    # API configuration
    api_host: str = Field("127.0.0.1", description="API host")
    api_port: int = Field(8000, description="API port")

    # Feature flags
    enable_auth: bool = Field(True, description="Enable authentication")
    enable_metrics: bool = Field(False, description="Enable metrics collection")

    # Environment-specific defaults
    @property
    def is_production(self) -> bool:
        return self.environment == FlextEnvironment.PRODUCTION

    @property
    def database_pool_size(self) -> int:
        return 20 if self.is_production else 5

# Load configuration
settings = MyAppSettings()

# Access with full type safety
print(f"Database: {settings.database_url}")
print(f"API: http://{settings.api_host}:{settings.api_port}")
print(f"Pool size: {settings.database_pool_size}")
```

### Environment Variable Integration

```python
# Set environment variables
# FLEXT_MYAPP_DATABASE_URL=postgresql://user:pass@host/db
# FLEXT_MYAPP_API_PORT=9000
# FLEXT_ENVIRONMENT=production

class MyAppSettings(FlextCoreSettings):
    model_config = {
        **FlextCoreSettings.model_config,
        "env_prefix": "FLEXT_MYAPP_",  # Custom prefix
    }

    database_url: str = Field(...)  # Required from env
    api_port: int = Field(8000)     # Default with env override

settings = MyAppSettings()
# Automatically loads from environment variables
```

## Advanced Configuration Patterns

### Multi-Environment Configuration

```python
from flext_core.constants import FlextEnvironment

class DatabaseSettings(FlextCoreSettings):
    """Database configuration with environment-specific settings."""

    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    name: str = Field("myapp", description="Database name")
    user: str = Field("postgres", description="Database user")
    password: str = Field("", description="Database password")

    # SSL configuration
    ssl_mode: str = Field("prefer", description="SSL mode")
    ssl_cert_path: str | None = Field(None, description="SSL certificate path")

    @property
    def connection_url(self) -> str:
        """Build database connection URL."""
        auth = f"{self.user}:{self.password}@" if self.password else f"{self.user}@"
        ssl_params = f"?sslmode={self.ssl_mode}"
        if self.ssl_cert_path:
            ssl_params += f"&sslcert={self.ssl_cert_path}"

        return f"postgresql://{auth}{self.host}:{self.port}/{self.name}{ssl_params}"

    def get_pool_settings(self) -> dict[str, int]:
        """Get connection pool settings based on environment."""
        if self.environment == FlextEnvironment.PRODUCTION:
            return {
                "min_size": 10,
                "max_size": 20,
                "max_queries": 50000,
                "max_inactive_connection_lifetime": 300.0,
            }
        elif self.environment == FlextEnvironment.STAGING:
            return {
                "min_size": 5,
                "max_size": 10,
                "max_queries": 10000,
                "max_inactive_connection_lifetime": 600.0,
            }
        else:  # Development/Testing
            return {
                "min_size": 2,
                "max_size": 5,
                "max_queries": 1000,
                "max_inactive_connection_lifetime": 1200.0,
            }
```

### Secret Management

```python
from pydantic import SecretStr, field_validator

class SecureSettings(FlextCoreSettings):
    """Configuration with secret management."""

    # Secret fields are never logged or displayed
    database_password: SecretStr = Field(..., description="Database password")
    api_key: SecretStr = Field(..., description="External API key")
    jwt_secret: SecretStr = Field(..., description="JWT signing secret")

    # Public fields
    service_name: str = Field("myapp", description="Service name")

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: SecretStr) -> SecretStr:
        """Ensure JWT secret is strong enough."""
        secret = v.get_secret_value()
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v

    def get_database_url(self) -> str:
        """Build database URL with secret password."""
        password = self.database_password.get_secret_value()
        return f"postgresql://user:{password}@host/db"

    def __repr__(self) -> str:
        """Safe representation hiding secrets."""
        return f"<{self.__class__.__name__} service={self.service_name}>"

# Usage
settings = SecureSettings(
    database_password="super_secret_password",
    api_key="api_key_12345",
    jwt_secret="very_long_and_secure_jwt_secret_key_here"
)

print(settings)  # <SecureSettings service=myapp>
# Secrets are never exposed in logs or string representations
```

### Validation and Business Rules

```python
from pydantic import field_validator, model_validator
from typing import Self

class APISettings(FlextCoreSettings):
    """API configuration with business rule validation."""

    # Basic settings
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    workers: int = Field(1, description="Number of workers")

    # Rate limiting
    rate_limit_rpm: int = Field(1000, description="Requests per minute limit")
    rate_limit_burst: int = Field(100, description="Burst limit")

    # Timeouts
    request_timeout: int = Field(30, description="Request timeout in seconds")
    keepalive_timeout: int = Field(5, description="Keep-alive timeout")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        """Validate worker count based on environment."""
        if v < 1:
            raise ValueError("Must have at least 1 worker")
        if v > 32:  # Reasonable upper limit
            raise ValueError("Too many workers (max 32)")
        return v

    @model_validator(mode="after")
    def validate_rate_limits(self) -> Self:
        """Validate rate limiting configuration makes sense."""
        if self.rate_limit_burst > self.rate_limit_rpm:
            raise ValueError("Burst limit cannot exceed RPM limit")

        if self.keepalive_timeout >= self.request_timeout:
            raise ValueError("Keep-alive timeout must be less than request timeout")

        return self

    @property
    def bind_address(self) -> str:
        """Get the complete bind address."""
        return f"{self.host}:{self.port}"

    def get_worker_count(self) -> int:
        """Get optimal worker count based on environment."""
        if self.environment == FlextEnvironment.PRODUCTION:
            return max(self.workers, 4)  # At least 4 in production
        return self.workers
```

## Framework Integration Examples

### Singer Tap/Target Configuration

```python
from flext_core.config import FlextCoreSettings

class SingerTapConfig(FlextCoreSettings):
    """Configuration for Singer tap implementation."""

    # API connection
    api_url: str = Field(..., description="API base URL")
    api_key: SecretStr = Field(..., description="API authentication key")
    api_version: str = Field("v1", description="API version")

    # Extraction settings
    start_date: str = Field(..., description="Start date for data extraction")
    page_size: int = Field(100, description="API page size")
    request_timeout: int = Field(60, description="Request timeout")

    # Output settings
    batch_size: int = Field(1000, description="Batch size for records")
    max_records: int | None = Field(None, description="Maximum records to extract")

    @field_validator("start_date")
    @classmethod
    def validate_start_date(cls, v: str) -> str:
        """Validate start date format."""
        from datetime import datetime
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}") from e

    def get_api_headers(self) -> dict[str, str]:
        """Get API headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Accept": "application/json",
            "User-Agent": f"flext-tap/{self.api_version}"
        }

# Usage in Singer tap
def main():
    config = SingerTapConfig()
    headers = config.get_api_headers()

    # Use in Singer tap logic
    import singer
    singer.write_schema("users", schema, ["id"])
```

### Django Integration

```python
class DjangoSettings(FlextCoreSettings):
    """Django-compatible settings."""

    # Django basics
    secret_key: SecretStr = Field(..., description="Django secret key")
    debug: bool = Field(False, description="Debug mode")
    allowed_hosts: list[str] = Field(default_factory=list, description="Allowed hosts")

    # Database
    database_url: str = Field(..., description="Database URL")

    # Static files
    static_url: str = Field("/static/", description="Static files URL")
    static_root: str = Field("staticfiles/", description="Static files root")

    # Security
    secure_ssl_redirect: bool = Field(False, description="Force HTTPS redirect")
    session_cookie_secure: bool = Field(False, description="Secure session cookies")

    @property
    def django_database_config(self) -> dict:
        """Convert database URL to Django format."""
        # Parse database URL and return Django database dict
        # Implementation would parse the URL
        return {
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": "extracted_from_url",
                # ... other settings
            }
        }

    def apply_to_django_settings(self, django_settings: dict) -> dict:
        """Apply configuration to Django settings dict."""
        django_settings.update({
            "SECRET_KEY": self.secret_key.get_secret_value(),
            "DEBUG": self.debug,
            "ALLOWED_HOSTS": self.allowed_hosts,
            "DATABASES": self.django_database_config,
            "STATIC_URL": self.static_url,
            "STATIC_ROOT": self.static_root,
            "SECURE_SSL_REDIRECT": self.secure_ssl_redirect,
            "SESSION_COOKIE_SECURE": self.session_cookie_secure,
        })
        return django_settings

# In Django settings.py
django_config = DjangoSettings()
globals().update(django_config.apply_to_django_settings(globals()))
```

## Testing Configuration

```python
import pytest
from flext_core.config import FlextCoreSettings
from flext_core.constants import FlextEnvironment

class TestSettings(FlextCoreSettings):
    """Test-specific configuration."""

    database_url: str = Field("sqlite:///:memory:", description="Test database")
    api_port: int = Field(0, description="Random port for testing")
    enable_auth: bool = Field(False, description="Disable auth in tests")

def test_configuration_loading(monkeypatch):
    """Test configuration loading with environment variables."""
    # Set test environment variables
    monkeypatch.setenv("FLEXT_ENVIRONMENT", "testing")
    monkeypatch.setenv("FLEXT_DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("FLEXT_API_PORT", "9999")

    settings = TestSettings()

    assert settings.environment == FlextEnvironment.TESTING
    assert settings.database_url == "sqlite:///test.db"
    assert settings.api_port == 9999
    assert settings.enable_auth is False

def test_validation_errors():
    """Test configuration validation."""
    with pytest.raises(ValueError, match="Port must be between"):
        TestSettings(api_port=70000)

    with pytest.raises(ValueError, match="Database URL required"):
        TestSettings(database_url="")

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestSettings(
        database_url="sqlite:///:memory:",
        api_port=0,
        enable_auth=False
    )

def test_with_config(test_config):
    """Test using configuration fixture."""
    assert test_config.database_url == "sqlite:///:memory:"
    assert test_config.enable_auth is False
```

## Migration from Legacy Systems

### From Environment Variables

```python
# Before (manual env parsing)
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite3")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# After (type-safe configuration)
class AppSettings(FlextCoreSettings):
    database_url: str = Field("sqlite:///db.sqlite3")
    api_port: int = Field(8000)
    debug: bool = Field(False)

settings = AppSettings()
```

### From Configuration Files

```python
# Before (manual file parsing)
import yaml

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

config = load_config()
api_timeout = config.get("api", {}).get("timeout", 30)

# After (structured configuration)
class APIConfig(FlextCoreSettings):
    timeout: int = Field(30, description="API timeout")
    retries: int = Field(3, description="API retries")
    base_url: str = Field(..., description="API base URL")

# Can load from file if needed, but env vars are preferred
settings = APIConfig()
```

## Best Practices

### 1. Use Type-Safe Defaults

```python
# Good - type-safe with validation
class Settings(FlextCoreSettings):
    timeout: int = Field(30, ge=1, le=300, description="Timeout in seconds")

# Avoid - stringly typed
timeout = int(os.getenv("TIMEOUT", "30"))
```

### 2. Group Related Settings

```python
class DatabaseSettings(FlextCoreSettings):
    """Database-related configuration."""
    url: str = Field(...)
    pool_size: int = Field(10)
    timeout: int = Field(30)

class CacheSettings(FlextCoreSettings):
    """Cache-related configuration."""
    redis_url: str = Field(...)
    ttl: int = Field(3600)
```

### 3. Use Environment-Specific Logic

```python
class Settings(FlextCoreSettings):
    @property
    def log_level(self) -> str:
        return "DEBUG" if self.environment.is_development else "INFO"

    @property
    def enable_debug_toolbar(self) -> bool:
        return self.environment.is_development and self.debug
```

### 4. Validate Business Rules

```python
class Settings(FlextCoreSettings):
    min_workers: int = Field(1)
    max_workers: int = Field(10)

    @model_validator(mode="after")
    def validate_worker_range(self) -> Self:
        if self.min_workers > self.max_workers:
            raise ValueError("min_workers cannot exceed max_workers")
        return self
```

## Common Patterns and Troubleshooting

See the [Configuration Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

## Advanced Topics

- [Multi-Environment Configuration](multi-environment.md)
- [Secret Management](secrets.md)
- [Framework Integration](framework-integration.md)
- [Custom Validation](validation.md)
