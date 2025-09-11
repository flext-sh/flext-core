# Configuration Guide - FLEXT Core

**Version**: 0.9.1  
**Last Updated**: 2025-01-07  
**Authority**: flext-core Pydantic v2.11 Unification

## Table of Contents

1. [Overview](#overview)
2. [Configuration Architecture](#configuration-architecture)
3. [Using SystemConfigs Models](#using-systemconfigs-models)
4. [Settings Registry Pattern](#settings-registry-pattern)
5. [Migration from Dict-based Configuration](#migration-from-dict-based-configuration)
6. [Extending Configuration in Subprojects](#extending-configuration-in-subprojects)
7. [Best Practices](#best-practices)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

## Overview

FLEXT Core uses Pydantic v2.11 for all configuration management, providing:

- **Type Safety**: Full type validation at configuration time
- **Centralized Validation**: All rules in one place (no scattered checks)
- **Backward Compatibility**: Dict interfaces preserved at API boundaries
- **Environment-aware**: Automatic adjustments based on environment
- **Extensible**: Easy to extend in subprojects

### Key Principles

1. **Models Internally, Dicts at Borders**: Use Pydantic models for internal validation, expose dicts for compatibility
2. **Single Source of Truth**: All validation rules in `FlextModels.SystemConfigs`
3. **FlextResult Pattern**: All configuration operations return `FlextResult[T]`
4. **No Manual Validation**: Let Pydantic handle all validation automatically

## Configuration Architecture

### Layer Structure

```
┌─────────────────────────────────────┐
│         API Layer (Dicts)           │  ← External interfaces
├─────────────────────────────────────┤
│     Configuration Functions         │  ← configure_*_system()
├─────────────────────────────────────┤
│      Pydantic Models Layer          │  ← SystemConfigs.*
├─────────────────────────────────────┤
│        Base Configuration           │  ← BaseSystemConfig
├─────────────────────────────────────┤
│      Constants & Enums Layer        │  ← FlextConstants
└─────────────────────────────────────┘
```

### Core Components

#### 1. BaseSystemConfig

The foundation for all system configurations:

```python
from flext_core.models import FlextModels
from pydantic import Field

class BaseSystemConfig(FlextModels.Config):
    """Base configuration with common fields."""
    
    environment: str = Field(
        default="development",
        description="Deployment environment"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode flag"
    )
    # ... other common fields
```

#### 2. Specialized Configurations

Each subsystem has its own configuration model:

```python
# Commands Configuration
class CommandsConfig(BaseSystemConfig):
    """Configuration for CQRS commands system."""
    enable_command_validation: bool = True
    command_timeout_seconds: int = 30
    max_command_retries: int = 3

# Domain Services Configuration  
class DomainServicesConfig(BaseSystemConfig):
    """Configuration for domain services."""
    service_level: str = "standard"
    enable_service_monitoring: bool = True
    max_concurrent_operations: int = 10
```

## Using SystemConfigs Models

### Basic Usage

```python
from flext_core import FlextModels, FlextResult

# Create configuration from dict
config_dict = {
    "environment": "production",
    "log_level": "INFO",
    "enable_command_validation": True
}

# Validate and create model
try:
    config = FlextModels.SystemConfigs.CommandsConfig.model_validate(config_dict)
    # Use the validated configuration
    print(f"Environment: {config.environment}")
    print(f"Debug mode: {config.debug}")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Using with FlextResult

```python
def configure_system(config_dict: dict) -> FlextResult[dict]:
    """Configure system with validation."""
    try:
        # Validate configuration
        config = FlextModels.SystemConfigs.CommandsConfig.model_validate(config_dict)
        
        # Apply configuration logic
        if config.environment == "production":
            config.debug = False
            config.log_level = "WARNING"
        
        # Return as dict for compatibility
        return FlextResult.ok(config.model_dump())
        
    except ValidationError as e:
        return FlextResult.fail(
            f"Invalid configuration: {e}",
            error_code="CONFIG_ERROR"
        )
```

### Environment-Specific Configuration

```python
from flext_core.constants import FlextConstants

class MyConfig(BaseSystemConfig):
    """Custom configuration with environment awareness."""
    
    @model_validator(mode="after")
    def adjust_for_environment(self) -> "MyConfig":
        """Auto-adjust settings based on environment."""
        if self.environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION:
            self.debug = False
            self.enable_monitoring = True
            self.cache_enabled = True
        elif self.environment == internal.invalid:
            self.debug = True
            self.enable_monitoring = False
            self.cache_enabled = False
        return self
```

## Settings Registry Pattern

The Settings Registry provides centralized management of all configuration models:

### Registry Usage

```python
from flext_core import FlextConfig

# Register a settings instance
settings = FlextConfig.Settings(
    app_name="my-service",
    environment="production"
)
FlextConfig.SettingsRegistry.register("my-service", settings)

# Retrieve settings
retrieved = FlextConfig.SettingsRegistry.get("my-service")
if retrieved:
    config = retrieved.to_config()
```

### Dynamic Field Updates

```python
# Update runtime fields
FlextConfig.SettingsRegistry.update_runtime("my-service", {
    "log_level": "DEBUG",
    "cache_enabled": False
})

# Get all dynamic fields
dynamic_fields = FlextConfig.SettingsRegistry.get_dynamic_fields()
```

### Reload from Sources

```python
# Reload configuration from environment/files
result = FlextConfig.SettingsRegistry.reload_from_sources(
    "my-service",
    env_prefix="MYAPP_",
    json_file="config.json"
)

if result.success:
    print("Configuration reloaded successfully")
```

## Migration from Dict-based Configuration

### Before (Dict-based)

```python
def configure_old_system(config: dict) -> FlextResult[dict]:
    """Old manual validation approach."""
    # Manual validation
    if "environment" not in config:
        return FlextResult.fail("environment required")
    
    env = config["environment"]
    valid_envs = ["development", "staging", "production"]
    if env not in valid_envs:
        return FlextResult.fail(f"Invalid environment: {env}")
    
    # Manual type checking
    if "port" in config:
        if not isinstance(config["port"], int):
            return FlextResult.fail("port must be integer")
        if not 1 <= config["port"] <= 65535:
            return FlextResult.fail("port out of range")
    
    # Manual defaults
    config.setdefault("log_level", "INFO")
    config.setdefault("debug", False)
    
    return FlextResult.ok(config)
```

### After (Pydantic-based)

```python
from flext_core.models import FlextModels
from pydantic import Field, ValidationError

class SystemConfig(BaseSystemConfig):
    """Pydantic model with automatic validation."""
    port: int = Field(default=8080, ge=1, le=65535)
    # All validation automatic!

def configure_new_system(config: dict) -> FlextResult[dict]:
    """New Pydantic-based approach."""
    try:
        # All validation happens automatically
        validated = SystemConfig.model_validate(config)
        return FlextResult.ok(validated.model_dump())
    except ValidationError as e:
        return FlextResult.fail(str(e))
```

### Migration Steps

1. **Identify Manual Validations**: Search for patterns like:
   ```bash
   grep -r "if .* not in\|isinstance\|setdefault" src/
   ```

2. **Create Pydantic Model**: Define fields with constraints:
   ```python
   class MyConfig(BaseSystemConfig):
       my_field: str = Field(..., min_length=1, max_length=100)
       my_number: int = Field(default=10, ge=0, le=1000)
   ```

3. **Replace Validation Logic**: Use `model_validate()`:
   ```python
   config = MyConfig.model_validate(input_dict)
   ```

4. **Maintain Compatibility**: Return dict at boundaries:
   ```python
   return FlextResult.ok(config.model_dump())
   ```

## Extending Configuration in Subprojects

### Example: flext-api Extension

Create a custom configuration that extends the base:

```python
# flext-api/src/flext_api/config.py
from flext_core.models import FlextModels
from flext_core.config import FlextConfig
from pydantic import Field, field_validator
from typing import Optional

class ApiConfig(FlextModels.SystemConfigs.BaseSystemConfig):
    """API-specific configuration."""
    
    # API-specific fields
    api_key: Optional[str] = Field(None, description="API authentication key")
    rate_limit: int = Field(default=100, ge=1, le=10000)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    base_url: str = Field(default="https://api.example.com")
    
    # Nested configuration
    retry_config: dict = Field(
        default_factory=lambda: {
            "max_retries": 3,
            "backoff_factor": 2.0,
            "max_wait": 60
        }
    )
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate API key format."""
        if v and not v.startswith("sk_"):
            raise ValueError("API key must start with 'sk_'")
        return v
    
    @model_validator(mode="after")
    def adjust_for_production(self) -> "ApiConfig":
        """Production adjustments."""
        if self.environment == "production":
            if not self.api_key:
                raise ValueError("API key required in production")
            self.timeout_seconds = max(self.timeout_seconds, 60)
        return self

class ApiSettings(FlextConfig.Settings):
    """Extended settings for API project."""
    
    api_key: Optional[str] = None
    rate_limit: int = 100
    base_url: str = "https://api.example.com"
    
    def to_config(self) -> ApiConfig:
        """Convert to ApiConfig model."""
        return ApiConfig(
            environment=self.environment,
            log_level=self.log_level,
            debug=self.debug,
            api_key=self.api_key,
            rate_limit=self.rate_limit,
            base_url=self.base_url
        )
```

### Using the Extended Configuration

```python
# flext-api/src/flext_api/client.py
from flext_api.config import ApiConfig, ApiSettings
from flext_core import FlextResult, FlextConfig

class ApiClient:
    """API client with configuration support."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize with configuration."""
        self.config = self._load_config(config)
    
    def _load_config(self, config_dict: Optional[dict]) -> ApiConfig:
        """Load and validate configuration."""
        if config_dict:
            # From provided dict
            return ApiConfig.model_validate(config_dict)
        else:
            # From environment/files
            settings = ApiSettings.from_sources(
                env_prefix="FLEXT_API_",
                json_file="api_config.json"
            )
            return settings.to_config()
    
    @classmethod
    def from_environment(cls) -> "ApiClient":
        """Create client from environment variables."""
        settings = ApiSettings.from_sources(env_prefix="FLEXT_API_")
        return cls(settings.to_config().model_dump())
```

## Best Practices

### 1. Validation at Model Level

**Do**: Define all validation in the Pydantic model
```python
class MyConfig(BaseSystemConfig):
    email: EmailStr  # Automatic email validation
    port: int = Field(ge=1, le=65535)  # Range validation
    
    @field_validator("custom_field")
    @classmethod
    def validate_custom(cls, v):
        # Custom validation logic
        return v
```

**Don't**: Validate after model creation
```python
# Bad: Manual validation after model
config = MyConfig.model_validate(data)
if config.port < 1 or config.port > 65535:  # Don't do this!
    raise ValueError("Invalid port")
```

### 2. Use Enums from FlextConstants

**Do**: Use centralized enums
```python
from flext_core.constants import FlextConstants

class MyConfig(BaseSystemConfig):
    environment: str = Field(default=FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value)
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        valid = [e.value for e in FlextConstants.Config.ConfigEnvironment]
        if v not in valid:
            raise ValueError(f"Invalid environment: {v}")
        return v
```

**Don't**: Define local enums or string literals
```python
# Bad: Local definition
VALID_ENVIRONMENTS = ["dev", "prod"]  # Don't do this!
```

### 3. Return Dicts at API Boundaries

**Do**: Use models internally, return dicts externally
```python
def configure_system(config_dict: dict) -> FlextResult[dict]:
    try:
        config = SystemConfig.model_validate(config_dict)  # Model internally
        # ... process with model ...
        return FlextResult.ok(config.model_dump())  # Dict at boundary
    except ValidationError as e:
        return FlextResult.fail(str(e))
```

**Don't**: Expose Pydantic models in public APIs
```python
# Bad: Exposing model directly
def configure_system(config_dict: dict) -> FlextResult[SystemConfig]:  # Don't!
    return FlextResult.ok(SystemConfig.model_validate(config_dict))
```

### 4. Handle Custom Values for Compatibility

**Do**: Preserve non-standard values when needed
```python
def configure_with_compatibility(config: dict) -> FlextResult[dict]:
    # Preserve custom log level for backward compatibility
    custom_log_level = None
    if "log_level" in config:
        if config["log_level"] not in standard_levels:
            custom_log_level = config["log_level"]
            config["log_level"] = "INFO"  # Temporary for validation
    
    validated = MyConfig.model_validate(config)
    result = validated.model_dump()
    
    if custom_log_level:
        result["log_level"] = custom_log_level  # Restore custom value
    
    return FlextResult.ok(result)
```

## Common Patterns

### Pattern 1: Factory Methods

```python
class MyConfig(BaseSystemConfig):
    """Configuration with factory methods."""
    
    @classmethod
    def for_production(cls) -> "MyConfig":
        """Create production configuration."""
        return cls(
            environment="production",
            log_level="WARNING",
            debug=False,
            enable_monitoring=True
        )
    
    @classmethod
    def for_testing(cls) -> "MyConfig":
        """Create test configuration."""
        return cls(
            environment="test",
            log_level="DEBUG",
            debug=True,
            enable_monitoring=False
        )
```

### Pattern 2: Configuration Merging

```python
def merge_configurations(*configs: dict) -> FlextResult[dict]:
    """Merge multiple configuration dicts."""
    merged = {}
    for config in configs:
        merged.update(config)
    
    try:
        validated = MyConfig.model_validate(merged)
        return FlextResult.ok(validated.model_dump())
    except ValidationError as e:
        return FlextResult.fail(f"Merge failed: {e}")
```

### Pattern 3: Conditional Fields

```python
class ConditionalConfig(BaseSystemConfig):
    """Config with conditional requirements."""
    
    mode: str = Field(default="basic")
    advanced_option: Optional[str] = None
    
    @model_validator(mode="after")
    def validate_conditional(self) -> "ConditionalConfig":
        """Validate conditional requirements."""
        if self.mode == "advanced" and not self.advanced_option:
            raise ValueError("advanced_option required in advanced mode")
        return self
```

### Pattern 4: Nested Configuration

```python
class DatabaseConfig(BaseModel):
    """Nested database configuration."""
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = "mydb"

class AppConfig(BaseSystemConfig):
    """Application config with nested database config."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    @classmethod
    def from_flat_dict(cls, data: dict) -> "AppConfig":
        """Create from flat dictionary."""
        # Extract database fields
        db_config = {
            k.replace("db_", ""): v
            for k, v in data.items()
            if k.startswith("db_")
        }
        
        # Create nested structure
        if db_config:
            data["database"] = db_config
        
        return cls.model_validate(data)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. ValidationError: field required

**Problem**: Missing required field
```python
ValidationError: 1 validation error for MyConfig
environment
  Field required [type=missing, input_value={}, input_type=dict]
```

**Solution**: Provide default or make optional
```python
class MyConfig(BaseSystemConfig):
    environment: str = Field(default="development")  # Add default
    # OR
    environment: Optional[str] = None  # Make optional
```

#### 2. Custom validation not triggering

**Problem**: Validator not being called
```python
@field_validator("my_field")
def validate_my_field(cls, v):  # Missing @classmethod
    return v
```

**Solution**: Add @classmethod decorator
```python
@field_validator("my_field")
@classmethod  # Required for Pydantic v2
def validate_my_field(cls, v):
    return v
```

#### 3. Circular import errors

**Problem**: Circular imports when using models
```python
# In config.py
from myapp.models import MyModel  # Causes circular import
```

**Solution**: Import inside function or use TYPE_CHECKING
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myapp.models import MyModel

def configure() -> FlextResult[dict]:
    from myapp.models import MyModel  # Import when needed
    # ...
```

#### 4. Type errors with model_dump()

**Problem**: Type checker doesn't recognize dict return
```python
config: MyConfig = MyConfig()
result: dict = config.model_dump()
```

**Solution**: Add type annotation
```python


result: dict[str, object] = config.model_dump()
```

#### 5. Environment variables not loading

**Problem**: Settings not picking up env vars
```python
settings = Settings.from_sources(env_prefix="MYAPP_")
# Environment variables not loaded
```

**Solution**: Check prefix and format
```bash
# Correct format: PREFIX_FIELD_NAME
export MYAPP_LOG_LEVEL=DEBUG
export MYAPP_DATABASE__HOST=localhost  # Nested: use double underscore
```

### Debug Tips

1. **Enable Pydantic Debug Mode**:
   ```python
   class MyConfig(BaseModel):
       model_config = ConfigDict(validate_assignment=True, extra="forbid")
   ```

2. **Check Model Schema**:
   ```python
   print(MyConfig.model_json_schema())  # See all fields and validation
   ```

3. **Validate Incrementally**:
   ```python
   # Validate field by field for debugging
   try:
       MyConfig.model_validate({"field1": value1})
   except ValidationError as e:
       print(f"field1 error: {e}")
   ```

4. **Use Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # Pydantic will log validation details
   ```

## Summary

The Pydantic v2.11 configuration system in FLEXT Core provides:

- ✅ **Type Safety**: Automatic validation of all configuration
- ✅ **Centralization**: All validation rules in one place
- ✅ **Extensibility**: Easy to extend in subprojects
- ✅ **Compatibility**: Dict interfaces preserved
- ✅ **Environment Awareness**: Auto-adjustments based on deployment

By following these patterns and best practices, you can:
- Eliminate manual validation code
- Reduce configuration errors
- Improve maintainability
- Ensure type safety throughout your application

For more examples, see the [examples directory](../examples/) and the [test suite](../../tests/unit/).
