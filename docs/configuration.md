# Configuration Management with FlextConfig

Comprehensive guide to configuration management in FLEXT-Core v0.9.9 using FlextConfig with Pydantic Settings for environment-based configuration, type validation, and multi-source loading.

---

## Overview

FlextConfig provides production-ready configuration management with:

- ✅ **Pydantic Settings v2** - Type-safe configuration with validation
- ✅ **Environment Variables** - Automatic loading with prefix support
- ✅ **Multiple Sources** - .env files, TOML, YAML, JSON support
- ✅ **Type Safety** - Complete type validation with helpful error messages
- ✅ **Container Integration** - Works seamlessly with FlextContainer
- ✅ **Environment Profiles** - Development, testing, production support

**Coverage**: 90% test coverage (src/flext_core/config.py)

---

## Basic Configuration

### Simple Configuration Class

```python
from flext_core import FlextConfig

class AppConfig(FlextConfig):
    """Application configuration with environment variable support."""
    app_name: str = "myapp"
    debug: bool = False
    database_url: str
    max_connections: int = 100
    timeout_seconds: float = 30.0

# Load from environment variables
config = AppConfig()

# Access configuration values
print(f"App: {config.app_name}")
print(f"Debug: {config.debug}")
print(f"Database: {config.database_url}")
```

### Environment Variable Loading

FlextConfig automatically loads environment variables with optional prefix:

```python
class DatabaseConfig(FlextConfig):
    """Database configuration with prefix."""
    host: str = "localhost"
    port: int = 5432
    username: str
    password: str
    database: str

    model_config = {
        "env_prefix": "DB_",  # Environment variables: DB_HOST, DB_PORT, etc.
        "case_sensitive": False,  # Case-insensitive matching
    }

# Reads: DB_HOST, DB_PORT, DB_USERNAME, DB_PASSWORD, DB_DATABASE
config = DatabaseConfig()
```

**Environment Setup**:

```bash
export DB_HOST=postgres.example.com
export DB_PORT=5432
export DB_USERNAME=app_user
export DB_PASSWORD=secret123
export DB_DATABASE=production_db
```

---

## Configuration Sources

### .env File Loading

```python
class ServiceConfig(FlextConfig):
    """Load configuration from .env file."""
    api_key: str
    api_secret: str
    base_url: str = "https://api.example.com"

    model_config = {
        "env_file": ".env",  # Load from .env file
        "env_file_encoding": "utf-8",
    }

# Automatically loads .env if present
config = ServiceConfig()
```

**.env file**:

```bash
API_KEY=your_api_key_here
API_SECRET=your_secret_here
BASE_URL=https://production.api.example.com
```

### Multiple Environment Files

```python
class MultiEnvConfig(FlextConfig):
    """Load from multiple environment files."""
    app_env: str = "development"
    api_url: str
    feature_flags: dict = {}

    model_config = {
        "env_file": [".env", ".env.local"],  # Load multiple files
        "env_file_encoding": "utf-8",
    }
```

**File Priority**: Later files override earlier ones (.env.local overrides .env)

### TOML Configuration

```python
from flext_core import FlextConfig
import toml

class TomlConfig(FlextConfig):
    """Load configuration from TOML file."""
    server_host: str
    server_port: int
    workers: int = 4

    @classmethod
    def from_toml(cls, path: str) -> "TomlConfig":
        """Load configuration from TOML file."""
        with open(path) as f:
            data = toml.load(f)
        return cls(**data.get("server", {}))

# Load from TOML
config = TomlConfig.from_toml("config.toml")
```

**config.toml**:

```toml
[server]
server_host = "0.0.0.0"
server_port = 8000
workers = 8
```

### YAML Configuration

```python
from flext_core import FlextConfig
import yaml

class YamlConfig(FlextConfig):
    """Load configuration from YAML file."""
    service_name: str
    log_level: str = "INFO"
    features: dict = {}

    @classmethod
    def from_yaml(cls, path: str) -> "YamlConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Load from YAML
config = YamlConfig.from_yaml("config.yaml")
```

**config.yaml**:

```yaml
service_name: myapp
log_level: DEBUG
features:
  feature_a: true
  feature_b: false
```

---

## Environment Profiles

### Profile-Based Configuration

```python
from flext_core import FlextConfig, FlextResult

class ProfileConfig(FlextConfig):
    """Configuration with environment profiles."""
    environment: str = "development"
    debug: bool = True
    database_pool_size: int = 5
    cache_enabled: bool = False

    @classmethod
    def create_for_environment(
        cls,
        env: str,
        **overrides
    ) -> "ProfileConfig":
        """Create configuration for specific environment."""
        if env == "production":
            return cls(
                environment="production",
                debug=False,
                database_pool_size=20,
                cache_enabled=True,
                **overrides
            )
        elif env == "testing":
            return cls(
                environment="testing",
                debug=True,
                database_pool_size=1,
                cache_enabled=False,
                **overrides
            )
        else:  # development
            return cls(
                environment="development",
                debug=True,
                database_pool_size=5,
                cache_enabled=False,
                **overrides
            )

# Create environment-specific configuration
dev_config = ProfileConfig.create_for_environment("development")
prod_config = ProfileConfig.create_for_environment("production", database_pool_size=50)
test_config = ProfileConfig.create_for_environment("testing")
```

---

## Type Validation

### Automatic Type Validation

```python
from flext_core import FlextConfig
from pydantic import Field, field_validator

class ValidatedConfig(FlextConfig):
    """Configuration with type validation."""
    port: int = Field(ge=1, le=65535, description="Server port number")
    workers: int = Field(ge=1, le=100, description="Worker count")
    timeout: float = Field(ge=0.1, description="Timeout in seconds")
    email: str = Field(pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is not in privileged range in production."""
        if v < 1024:
            raise ValueError("Port must be >= 1024 for non-privileged users")
        return v

# Valid configuration
config = ValidatedConfig(
    port=8000,
    workers=4,
    timeout=30.0,
    email="user@example.com"
)

# Invalid configuration raises ValidationError
try:
    invalid_config = ValidatedConfig(
        port=999,  # Below 1024
        workers=4,
        timeout=30.0,
        email="invalid-email"
    )
except Exception as e:
    print(f"Validation error: {e}")
```

### Complex Type Validation

```python
from flext_core import FlextConfig
from pydantic import Field, field_validator, model_validator
from typing import Any

class AdvancedConfig(FlextConfig):
    """Configuration with complex validation."""
    redis_url: str
    redis_max_connections: int = Field(ge=1, le=1000)
    cache_ttl_seconds: int = Field(ge=60)
    feature_flags: dict[str, bool] = {}

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("Redis URL must start with redis://")
        return v

    @model_validator(mode="after")
    def validate_cache_config(self) -> "AdvancedConfig":
        """Validate cache configuration consistency."""
        if self.redis_max_connections > 100 and self.cache_ttl_seconds < 300:
            raise ValueError(
                "High connection count requires higher TTL for efficiency"
            )
        return self
```

---

## Container Integration

### Registering Configuration

```python
from flext_core import FlextConfig, FlextContainer, FlextResult

class AppConfig(FlextConfig):
    """Application configuration."""
    app_name: str = "myapp"
    debug: bool = False
    api_key: str

# Create configuration
config = AppConfig()

# Register with container
container = FlextContainer.get_global()
register_result = container.register("config", config)

if register_result.is_success:
    print("Configuration registered successfully")

# Retrieve from container
config_result = container.get("config")
if config_result.is_success:
    retrieved_config = config_result.unwrap()
    print(f"Retrieved config: {retrieved_config.app_name}")
```

### Service Integration

```python
from flext_core import FlextService, FlextResult, FlextLogger

class EmailService(FlextService):
    """Email service using configuration."""

    def __init__(self) -> None:
        super().__init__()
        self._logger = FlextLogger(__name__)
        self._container = FlextContainer.get_global()

    def send_email(self, to: str, subject: str, body: str) -> FlextResult[None]:
        """Send email using configured settings."""
        # Get configuration from container
        config_result = self._container.get("config")
        if config_result.is_failure:
            return FlextResult[None].fail("Configuration not available")

        config = config_result.unwrap()

        # Use configuration
        if config.debug:
            self._logger.info(
                "Email sending (debug mode)",
                extra={"to": to, "subject": subject}
            )
            return FlextResult[None].ok(None)

        # Actual email sending logic here
        self._logger.info("Email sent", extra={"to": to})
        return FlextResult[None].ok(None)
```

---

## Configuration Patterns

### Factory Pattern

```python
from flext_core import FlextConfig, FlextResult

class DatabaseConfig(FlextConfig):
    """Database configuration with factory methods."""
    host: str
    port: int
    username: str
    password: str
    database: str
    pool_size: int = 5

    @classmethod
    def create_postgres(cls, **overrides) -> "DatabaseConfig":
        """Create PostgreSQL configuration."""
        defaults = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
        }
        return cls(**{**defaults, **overrides})

    @classmethod
    def create_mysql(cls, **overrides) -> "DatabaseConfig":
        """Create MySQL configuration."""
        defaults = {
            "host": "localhost",
            "port": 3306,
            "database": "mysql",
        }
        return cls(**{**defaults, **overrides})

# Use factory methods
pg_config = DatabaseConfig.create_postgres(
    username="postgres",
    password="secret"
)

mysql_config = DatabaseConfig.create_mysql(
    username="root",
    password="secret"
)
```

### Nested Configuration

```python
from flext_core import FlextConfig
from pydantic import BaseModel

class LogConfig(BaseModel):
    """Logging configuration section."""
    level: str = "INFO"
    format: str = "json"
    file: str | None = None

class CacheConfig(BaseModel):
    """Cache configuration section."""
    enabled: bool = True
    ttl_seconds: int = 300
    max_size: int = 1000

class ApplicationConfig(FlextConfig):
    """Application configuration with nested sections."""
    app_name: str = "myapp"
    debug: bool = False
    log: LogConfig = LogConfig()
    cache: CacheConfig = CacheConfig()

# Create with nested configuration
config = ApplicationConfig(
    app_name="production-app",
    debug=False,
    log=LogConfig(level="WARNING", format="json", file="/var/log/app.log"),
    cache=CacheConfig(enabled=True, ttl_seconds=600, max_size=5000)
)

# Access nested configuration
print(f"Log level: {config.log.level}")
print(f"Cache TTL: {config.cache.ttl_seconds}")
```

---

## Testing with Configuration

### Test Configuration Overrides

```python
from flext_core import FlextConfig

class TestConfig(FlextConfig):
    """Test configuration with safe defaults."""
    database_url: str = "sqlite:///:memory:"
    debug: bool = True
    cache_enabled: bool = False
    external_api_enabled: bool = False

# Use in tests
def test_with_config():
    """Test with specific configuration."""
    config = TestConfig(
        database_url="sqlite:///test.db",
        external_api_enabled=False  # Disable external calls in tests
    )
    # Test code here
```

### Configuration Fixtures

```python
import pytest
from flext_core import FlextConfig, FlextContainer

@pytest.fixture
def test_config():
    """Provide test configuration."""
    class TestAppConfig(FlextConfig):
        app_name: str = "test-app"
        debug: bool = True
        database_url: str = "sqlite:///:memory:"

    return TestAppConfig()

@pytest.fixture
def configured_container(test_config):
    """Provide container with test configuration."""
    container = FlextContainer.get_global()
    container.register("config", test_config)
    yield container
    container.reset()  # Clean up after test

def test_service_with_config(configured_container):
    """Test service with configured container."""
    config_result = configured_container.get("config")
    assert config_result.is_success
    config = config_result.unwrap()
    assert config.app_name == "test-app"
```

---

## Best Practices

### 1. Use Type Hints

```python
from flext_core import FlextConfig

class GoodConfig(FlextConfig):
    """Configuration with proper type hints."""
    host: str
    port: int
    enabled: bool
    timeout: float
    options: dict[str, Any]
    tags: list[str]
```

### 2. Provide Sensible Defaults

```python
class DefaultsConfig(FlextConfig):
    """Configuration with sensible defaults."""
    debug: bool = False  # Production-safe default
    timeout_seconds: int = 30
    max_retries: int = 3
    log_level: str = "INFO"
```

### 3. Document Configuration

```python
from pydantic import Field

class DocumentedConfig(FlextConfig):
    """Well-documented configuration class."""
    api_key: str = Field(
        description="API key for external service authentication"
    )
    max_connections: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of concurrent connections"
    )
    timeout: float = Field(
        default=30.0,
        ge=0.1,
        description="Request timeout in seconds"
    )
```

### 4. Validate Critical Settings

```python
from pydantic import field_validator

class SecureConfig(FlextConfig):
    """Configuration with security validation."""
    api_key: str
    secret_key: str
    allowed_hosts: list[str]

    @field_validator("api_key", "secret_key")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Ensure secrets are not empty."""
        if not v or len(v) < 10:
            raise ValueError("Secret must be at least 10 characters")
        return v

    @field_validator("allowed_hosts")
    @classmethod
    def validate_hosts(cls, v: list[str]) -> list[str]:
        """Ensure at least one host is configured."""
        if not v:
            raise ValueError("At least one allowed host required")
        return v
```

---

## Common Patterns

### Pattern 1: Configuration Loading with Validation

```python
from flext_core import FlextConfig, FlextResult, FlextLogger

def load_config_safe(env: str = "development") -> FlextResult[FlextConfig]:
    """Load configuration with error handling."""
    logger = FlextLogger(__name__)

    try:
        config = AppConfig.create_for_environment(env)
        logger.info("Configuration loaded", extra={"environment": env})
        return FlextResult[FlextConfig].ok(config)
    except Exception as e:
        logger.error("Configuration load failed", extra={"error": str(e)})
        return FlextResult[FlextConfig].fail(f"Configuration error: {e}")
```

### Pattern 2: Configuration with Container Bootstrap

```python
def bootstrap_application() -> FlextResult[FlextContainer]:
    """Bootstrap application with configuration."""
    # Load configuration
    config_result = load_config_safe()
    if config_result.is_failure:
        return FlextResult[FlextContainer].fail(
            f"Config load failed: {config_result.error}"
        )

    config = config_result.unwrap()

    # Setup container
    container = FlextContainer.get_global()
    register_result = container.register("config", config)

    if register_result.is_failure:
        return FlextResult[FlextContainer].fail(
            f"Registration failed: {register_result.error}"
        )

    return FlextResult[FlextContainer].ok(container)
```

### Pattern 3: Environment-Specific Overrides

```python
class FlexibleConfig(FlextConfig):
    """Configuration with environment-specific behavior."""
    environment: str = "development"
    debug: bool = True

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def is_testing(self) -> bool:
        """Check if running in tests."""
        return self.environment == "testing"

    def get_log_level(self) -> str:
        """Get appropriate log level for environment."""
        if self.is_production():
            return "WARNING"
        return "DEBUG" if self.debug else "INFO"
```

---

## Troubleshooting

### Missing Environment Variables

```python
from pydantic import Field

class RequiredConfig(FlextConfig):
    """Configuration with required fields."""
    api_key: str = Field(
        description="API key (set API_KEY environment variable)"
    )

try:
    config = RequiredConfig()
except Exception as e:
    print(f"Missing required configuration: {e}")
    # Output: Field required [api_key]
```

### Type Conversion Errors

```python
# Environment: PORT=abc (invalid integer)
try:
    config = AppConfig()
except Exception as e:
    print(f"Type conversion error: {e}")
    # Output: Input should be a valid integer
```

### Validation Errors

```python
try:
    config = ValidatedConfig(port=999)  # Below minimum
except Exception as e:
    print(f"Validation error: {e}")
    # Output: Port must be >= 1024
```

---

## Next Steps

- **Logging Integration**: See [docs/logging.md](logging.md) for FlextLogger with configuration
- **Container Patterns**: See [docs/dependency-injection.md](dependency-injection.md) for advanced DI
- **Domain Services**: See [docs/services.md](services.md) for service patterns
- **Testing**: See [tests/unit/test_config.py](../tests/unit/test_config.py) for examples

---

**FLEXT-Core Configuration** - Type-safe, validated, environment-aware configuration management with Pydantic Settings.
