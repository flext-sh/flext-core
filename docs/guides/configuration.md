<!-- Generated from docs/guides/configuration.md for flext-core. -->
<!-- Source of truth: workspace docs/guides/. -->

# flext-core - FLEXT Configuration Guide

> Project profile: `flext-core`







This guide covers how to configure FLEXT for your specific environment and requirements.

## Overview

FLEXT uses a hierarchical configuration system that supports environment variables, configuration files,
and programmatic configuration. All configuration is validated using Pydantic v2 models for type safety and validation.

## Configuration Sources

FLEXT loads configuration in the following order (later sources override earlier ones):

1. **Default values** in Pydantic models
2. **Environment variables** (prefixed with `FLEXT_`)
3. **Configuration files** (YAML, JSON, or TOML)
4. **Programmatic configuration** in code

## Basic Configuration

### Environment Variables

Set configuration using environment variables with the `FLEXT_` prefix:

```bash
# Core configuration
export FLEXT_LOG_LEVEL=INFO
export FLEXT_DEBUG=false
export FLEXT_ENVIRONMENT=production

# LDIF processing
export FLEXT_LDIF_DEFAULT_ENCODING=utf-8
export FLEXT_LDIF_STRICT_VALIDATION=true
export FLEXT_LDIF_SERVERS_ENABLED=true

# API configuration
export FLEXT_API_BASE_URL=https://api.example.com
export FLEXT_API_TIMEOUT=30
```

### Configuration Files

Create configuration files in YAML, JSON, or TOML format:

**config.YAML:**

```yaml
# FLEXT Configuration
log_level: INFO
debug: false
environment: production

# LDIF Processing
ldif:
  default_encoding: utf-8
  strict_validation: true
  servers_enabled: true
  batch_size: 1000

# API Configuration
api:
  base_url: https://api.example.com
  timeout: 30
  retry_attempts: 3
```

### Programmatic Configuration

Configure FLEXT programmatically in your code:

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u
from flext_ldif import FlextLdifSettings

# Core configuration
config = FlextSettings(
    log_level="INFO",
    debug=False,
    environment="production"
)

# LDIF configuration
ldif_config = FlextLdifSettings(
    default_encoding="utf-8",
    strict_validation=True,
    servers_enabled=True,
    batch_size=1000
)
```

## Project-Specific Configuration

### flext-ldif Configuration

```python
from flext_ldif import FlextLdifSettings

config = FlextLdifSettings(
    # Server-specific settings
    source_server="oid",
    target_server="oud",

    # Migration options
    preserve_oid_modifiers=True,
    handle_schema_extensions=True,
    validate_entries=True,

    # Performance settings
    batch_size=1000,
    parallel_processing=True,
    max_workers=4
)
```

### flext-api Configuration

```python
from flext_api import FlextApiSettings

config = FlextApiSettings(
    base_url="https://api.example.com",
    timeout=30,
    retry_attempts=3,
    verify_ssl=True,
    headers={
        "User-Agent": "FLEXT-API/1.0"
    }
)
```

### flext-auth Configuration

```python
from flext_auth import FlextAuthSettings

config = FlextAuthSettings(
    secret_key="your-secret-key",
    algorithm="HS256",
    access_token_expire_minutes=30,
    refresh_token_expire_days=7
)
```

## Environment-Specific Configuration

### Development Environment

```yaml
# config.dev.yaml
log_level: DEBUG
debug: true
environment: development

ldif:
  strict_validation: false
  servers_enabled: false

api:
  base_url: http://localhost:8000
  timeout: 60
```

### Production Environment

```yaml
# config.prod.yaml
log_level: WARNING
debug: false
environment: production

ldif:
  strict_validation: true
  servers_enabled: true
  batch_size: 5000

api:
  base_url: https://api.production.com
  timeout: 30
  retry_attempts: 5
```

## Configuration Validation

All configuration is validated using Pydantic v2 models:

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

try:
    config = FlextSettings(
        log_level="INVALID_LEVEL"  # This will raise ValidationError
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## Configuration Inheritance

FLEXT supports configuration inheritance for complex setups:

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

# Base configuration
base_config = FlextSettings(
    log_level="INFO",
    environment="production"
)

# Extended configuration
extended_config = FlextSettings(
    **base_config.dict(),
    debug=True,  # Override for development
    custom_setting="value"
)
```

## Best Practices

### 1. Use Environment Variables for Secrets

```bash
# Never put secrets in configuration files
export FLEXT_DATABASE_PASSWORD=secret_password
export FLEXT_API_KEY=your_api_key
```

### 2. Validate Configuration Early

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

def main():
    # Validate configuration at startup
    config = FlextSettings()

    if not config.is_valid():
        print("Invalid configuration")
        return 1

    # Continue with application logic
    return 0
```

### 3. Use Configuration Classes

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

class MyAppConfig(FlextSettings):
    custom_setting: str = "default_value"
    another_setting: int = 42

    @field_validator('another_setting')
    @classmethod
    def validate_another_setting(cls, v):
        if v < 0:
            raise ValueError('another_setting must be positive')
        return v
```

### 4. Document Configuration Options

```python
class FlextLdifSettings(BaseModel):
    """Configuration for LDIF processing."""

    default_encoding: str = Field(
        default="utf-8",
        description="Default encoding for LDIF files"
    )

    strict_validation: bool = Field(
        default=True,
        description="Enable strict RFC validation"
    )
```

## Troubleshooting

### Common Configuration Issues

1. **Environment Variables Not Loading**
   - Ensure variables are prefixed with `FLEXT_`
   - Check for typos in variable names
   - Verify environment is set before running application

2. **Configuration File Not Found**
   - Check file path is correct
   - Ensure file has proper permissions
   - Verify file format (YAML, JSON, or TOML)

3. **Validation Errors**
   - Check Pydantic model field types
   - Verify required fields are provided
   - Review field validators for constraints

### Debug Configuration

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

# Enable debug logging
config = FlextSettings(debug=True)

# Print configuration
print(config.dict())

# Validate configuration
if config.is_valid():
    print("Configuration is valid")
else:
    print("Configuration has errors")
```

## Examples

### Complete Configuration Example

```python
#!/usr/bin/env python3
"""Complete FLEXT configuration example."""

import os
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u
from flext_ldif import FlextLdifSettings
from flext_api import FlextApiSettings

def main():
    # Load configuration from environment
    config = FlextSettings()

    # Configure LDIF processing
    ldif_config = FlextLdifSettings(
        source_server=os.getenv("FLEXT_SOURCE_SERVER", "oid"),
        target_server=os.getenv("FLEXT_TARGET_SERVER", "oud"),
        batch_size=int(os.getenv("FLEXT_BATCH_SIZE", "1000"))
    )

    # Configure API client
    api_config = FlextApiSettings(
        base_url=os.getenv("FLEXT_API_URL", "http://localhost:8000"),
        timeout=int(os.getenv("FLEXT_API_TIMEOUT", "30"))
    )

    print("Configuration loaded successfully")
    print(f"Log level: {config.log_level}")
    print(f"LDIF batch size: {ldif_config.batch_size}")
    print(f"API base URL: {api_config.base_url}")

if __name__ == "__main__":
    main()
```

## Reference

- FLEXT Core Configuration
- Environment Variables
- [Pydantic v2 Documentation](https://docs.pydantic.dev/2.0/)
- Configuration Best Practices
