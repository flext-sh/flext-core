# Configuration Management Guide

**Status**: Production Ready | **Version**: 0.10.0 | **Pattern**: Layered Configuration

Comprehensive guide to managing application configuration with FlextSettings in FLEXT-Core.

## Overview

**FlextSettings** provides a layered configuration system that supports multiple sources (environment variables, files, programmatic overrides) with type-safe access and validation.

**Key Features:**

- Multiple configuration sources (files, environment variables, programmatic)
- Type-safe configuration with Pydantic v2 validation
- Environment-specific configurations
- Secrets management with sanitization
- Zero dependencies for Layer 0 configuration constants

## Configuration Sources Hierarchy

Configuration is loaded with the following priority (highest to lowest):

1. **Programmatic overrides** - Passed directly in code
2. **Environment variables** - System environment
3. **Configuration files** - `.toml`, `.yaml`, `.env`
4. **Defaults** - Built-in defaults

Higher priority sources override lower priority ones.

## Basic Usage

### Creating a Configuration Instance

```python
from flext_core import FlextSettings

# Create with default settings
config = FlextSettings()

# Create with specific environment
config = FlextSettings(environment='production')

# Create with config files
config = FlextSettings(
    config_files=['config.toml', 'secrets.env'],
    environment='production'
)

# Create with programmatic overrides
config = FlextSettings(
    config_files=['config.toml'],
    overrides={'debug': False, 'log_level': 'WARNING'}
)
```

### Accessing Configuration Values

```python
from flext_core import FlextSettings

config = FlextSettings()

# Get configuration value with default
database_url = config.get('database.url', default='sqlite:///:memory:')

# Get required configuration (raises error if missing)
api_key = config.get('api.key', required=True)

# Get with type casting
max_connections = config.get('database.max_connections', cast=int, default=10)
debug_mode = config.get('debug', cast=bool, default=False)

# Get configuration section
database_config = config.get_section('database')
if database_config:
    host = database_config.get('host')
    port = database_config.get('port', cast=int, default=5432)
```

## Configuration File Formats

### TOML Configuration

**config.toml:**

```toml
[application]
name = "myapp"
version = "1.0.0"
debug = false

[database]
host = "localhost"
port = 5432
name = "myapp_db"
pool_size = 10
pool_recycle = 3600

[api]
host = "0.0.0.0"
port = 8000
timeout = 30
max_request_size = 1048576  # 1MB

[logging]
level = "INFO"
format = "json"
output = "stdout"

[email]
smtp_host = "smtp.example.com"
smtp_port = 587
smtp_user = "${EMAIL_SMTP_USER}"
```

### Environment File (.env)

**.env:**

```env
# Database secrets
DATABASE_PASSWORD=secure_password_here
DATABASE_CONNECTION_STRING=postgresql://user:${DATABASE_PASSWORD}@localhost/myapp

# API keys
API_KEY=your_api_key_here
API_SECRET=your_secret_here

# Email configuration
EMAIL_SMTP_USER=noreply@example.com
EMAIL_SMTP_PASSWORD=email_password_here

# Logging
LOG_LEVEL=DEBUG
```

## Environment-Specific Configurations

### Load Configuration by Environment

```python
from flext_core import FlextSettings
import os

# Detect environment
env = os.getenv('APP_ENV', 'development')

# Load environment-specific config
config_files = [
    'config.toml',              # Base configuration
    f'config.{env}.toml'        # Environment-specific overrides
]

config = FlextSettings(
    config_files=config_files,
    environment=env
)

# Access environment-aware values
debug = config.get('debug')
log_level = config.get('logging.level', default='INFO')
database_url = config.get('database.url')
```

### Development vs Production

**config.development.toml:**

```toml
[application]
debug = true

[logging]
level = "DEBUG"

[database]
url = "sqlite:///./dev.db"
```

**config.production.toml:**

```toml
[application]
debug = false

[logging]
level = "WARNING"

[database]
url = "${DATABASE_URL}"
```

## Type-Safe Configuration

### Using Pydantic Models for Type Safety

```python
from pydantic import BaseModel, Field
from flext_core import FlextSettings
from typing import Optional

class DatabaseConfig(BaseModel):
    """Database configuration with validation."""
    host: str = "localhost"
    port: int = Field(ge=1, le=65535, default=5432)
    database: str
    username: str
    password: str
    pool_size: int = Field(ge=1, le=100, default=10)
    pool_recycle: int = Field(ge=60, default=3600)

class AppConfig(BaseModel):
    """Application configuration with validation."""
    name: str
    version: str
    debug: bool = False
    database: DatabaseConfig

# Load and validate configuration
config = FlextSettings(config_files=['config.toml'])

# Parse into typed model
try:
    app_config = AppConfig(
        name=config.get('application.name'),
        version=config.get('application.version'),
        debug=config.get('application.debug', cast=bool),
        database=DatabaseConfig(
            host=config.get('database.host'),
            port=config.get('database.port', cast=int),
            database=config.get('database.name'),
            username=config.get('database.username'),
            password=config.get('database.password'),
            pool_size=config.get('database.pool_size', cast=int),
        )
    )
    print(f"✅ Configuration valid: {app_config}")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
```

## Secrets Management

### Handling Sensitive Data

```python
from flext_core import FlextSettings
import os

config = FlextSettings(
    config_files=['config.toml', '.env'],
    environment=os.getenv('APP_ENV', 'development')
)

# Access secrets (never log these!)
database_password = config.get('database.password', required=True)
api_secret = config.get('api.secret', required=True)
smtp_password = config.get('email.smtp_password', required=True)

# ✅ CORRECT - Use secrets, don't log them
db_url = f"postgresql://user:{database_password}@localhost/myapp"

# ❌ WRONG - Never log secrets
logger.info(f"Database password: {database_password}")

# ✅ CORRECT - Log only non-sensitive info
logger.info("Connecting to database", extra={
    "database": "myapp",
    "host": config.get('database.host')
})
```

### Environment Variable Substitution

**config.toml:**

```toml
[database]
# Environment variables in config automatically expand
connection_string = "${DATABASE_CONNECTION_STRING}"
password = "${DATABASE_PASSWORD}"

[api]
api_key = "${API_KEY}"
api_secret = "${API_SECRET}"
```

**Python code:**

```python
from flext_core import FlextSettings

# Set environment variables
import os
os.environ['DATABASE_PASSWORD'] = 'secure_password'
os.environ['API_KEY'] = 'your_api_key'

config = FlextSettings(config_files=['config.toml'])

# Values automatically expand
db_password = config.get('database.password')  # "secure_password"
api_key = config.get('api.api_key')  # "your_api_key"
```

## Integration with Dependency Injection

### Register Configuration in Container

```python
from flext_core import FlextSettings, FlextContainer

# Create and validate configuration
config = FlextSettings(
    config_files=['config.toml'],
    environment='production'
)

# Register in global container
container = FlextContainer.get_global()
container.register("config", config, singleton=True)

# Access from anywhere in application
def get_database_url() -> str:
    config_result = container.get("config")
    if config_result.is_failure:
        raise RuntimeError("Configuration not available")

    config = config_result.value
    return config.get('database.url', required=True)
```

## Best Practices

### 1. Never Hardcode Configuration

```python
# ❌ WRONG - Hardcoded configuration
database_url = "postgresql://user:password@localhost/myapp"

# ✅ CORRECT - Load from configuration
config = FlextSettings(config_files=['config.toml'])
database_url = config.get('database.url', required=True)
```

### 2. Use Environment-Specific Files

```python
# ✅ CORRECT - Different configs per environment
config_files = [
    'config.toml',                    # Shared config
    f'config.{os.getenv("ENV")}.toml' # Environment-specific
]
config = FlextSettings(config_files=config_files)
```

### 3. Validate Configuration Early

```python
# ✅ CORRECT - Validate during startup
from pydantic import BaseModel

class AppConfig(BaseModel):
    database_url: str
    api_key: str
    debug: bool = False

try:
    config = FlextSettings(config_files=['config.toml'])
    app_config = AppConfig(
        database_url=config.get('database.url', required=True),
        api_key=config.get('api.key', required=True),
        debug=config.get('debug', cast=bool, default=False)
    )
except ValueError as e:
    raise RuntimeError(f"Invalid configuration: {e}")
```

### 4. Use Type Casting for Safety

```python
# ✅ CORRECT - Type casting with defaults
port = config.get('api.port', cast=int, default=8000)
debug = config.get('debug', cast=bool, default=False)
timeout = config.get('timeout', cast=float, default=30.0)

# ❌ WRONG - Assuming types
port = config.get('api.port')  # Could be string!
```

### 5. Separate Secrets from Configuration

```python
# ✅ CORRECT - Secrets in .env, config in .toml
# config.toml
[database]
host = "localhost"
port = 5432

# .env (git-ignored)
DATABASE_PASSWORD=secret_password

# Python
config = FlextSettings(config_files=['config.toml', '.env'])
db_host = config.get('database.host')
db_password = config.get('database.password')
```

## Common Patterns

### Application Configuration Class

```python
from flext_core import FlextSettings
from dataclasses import dataclass
from typing import Optional

@dataclass
class AppConfiguration:
    """Immutable application configuration."""
    app_name: str
    app_version: str
    debug: bool
    database_url: str
    api_port: int
    log_level: str

    @classmethod
    def from_config(cls, config: FlextSettings) -> 'AppConfiguration':
        """Create configuration from FlextSettings."""
        return cls(
            app_name=config.get('app.name', required=True),
            app_version=config.get('app.version', required=True),
            debug=config.get('app.debug', cast=bool, default=False),
            database_url=config.get('database.url', required=True),
            api_port=config.get('api.port', cast=int, default=8000),
            log_level=config.get('logging.level', default='INFO')
        )

# Usage
config = FlextSettings(config_files=['config.toml'])
app_config = AppConfiguration.from_config(config)
```

### Configuration with Sections

```python
from flext_core import FlextSettings

config = FlextSettings(config_files=['config.toml'])

# Get database section
db_section = config.get_section('database')
if db_section:
    db_host = db_section.get('host')
    db_port = db_section.get('port', cast=int)
    db_name = db_section.get('database')

# Get API section
api_section = config.get_section('api')
if api_section:
    api_host = api_section.get('host')
    api_port = api_section.get('port', cast=int)
    api_timeout = api_section.get('timeout', cast=int)
```

## Troubleshooting

### Missing Configuration File

```python
from flext_core import FlextSettings
import os

# ✅ CORRECT - Handle missing files gracefully
if os.path.exists('config.toml'):
    config = FlextSettings(config_files=['config.toml'])
else:
    # Use defaults or raise error
    config = FlextSettings()
    logger.warning("config.toml not found, using defaults")
```

### Unresolved Configuration Value

```python
# ✅ CORRECT - Check for required values
config = FlextSettings(config_files=['config.toml'])

try:
    database_url = config.get('database.url', required=True)
except KeyError as e:
    raise RuntimeError(f"Required configuration missing: {e}")
```

### Type Casting Errors

```python
# ✅ CORRECT - Handle type casting errors
try:
    port = config.get('api.port', cast=int)
except ValueError as e:
    logger.error(f"Invalid port configuration: {e}")
    port = 8000  # Use default
```

## Summary

FlextSettings provides:

- ✅ Multiple configuration sources with priority hierarchy
- ✅ Type-safe access with Pydantic validation
- ✅ Environment-specific configurations
- ✅ Secrets management with environment variables
- ✅ Integration with FLEXT-Core dependency injection
- ✅ Production-ready error handling

Use FlextSettings to manage all application configuration with confidence and safety.

## Next Steps

1. **Dependency Injection**: See [Advanced Dependency Injection](./dependency-injection-advanced.md) for registering config in container
2. **Error Handling**: Check [Error Handling Guide](./error-handling.md) for configuration error patterns
3. **Testing**: Review [Testing Guide](./testing.md) for configuration testing strategies
4. **Troubleshooting**: See [Troubleshooting Guide](./troubleshooting.md) for configuration debugging
5. **API Reference**: Consult [API Reference: FlextSettings](../api-reference/infrastructure.md#flextconfig) for complete API

## See Also

- [Dependency Injection Advanced](./dependency-injection-advanced.md) - Registering configuration in container
- [Error Handling Guide](./error-handling.md) - Handling configuration errors
- [Testing Guide](./testing.md) - Testing with configuration
- [Troubleshooting Guide](./troubleshooting.md) - Configuration troubleshooting
- [API Reference: FlextSettings](../api-reference/infrastructure.md#flextconfig) - Complete configuration API
- **FLEXT CLAUDE.md**: Architecture principles and development workflow

---

**Example from FLEXT Ecosystem**: See `src/flext_tests/test_config.py` for comprehensive configuration testing examples.
