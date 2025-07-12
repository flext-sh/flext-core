# FLEXT Configuration Migration Guide

This guide helps you migrate your FLEXT projects to use the standardized configuration system based on Pydantic, pydantic-settings, and Dynaconf.

## Overview

The new configuration system provides:

- **Type safety** with Pydantic models
- **Environment variable support** via pydantic-settings
- **Advanced features** through Dynaconf bridge (multi-file, environments, etc)
- **Framework adapters** for Singer, Django, and CLI projects

## Migration Steps

### 1. Update Dependencies

Add to your `pyproject.toml`:

```toml
[project]
dependencies = [
    "flext-core>=0.6.0",  # Includes config system
    # Remove direct dynaconf, pydantic-settings dependencies
]
```

### 2. Basic Python Project

**Before (using Python-dotenv):**

```python
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite3")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
```

**After (using flext-core config):**

```python
from flext_core import BaseSettings, get_settings
from pydantic import Field

class AppSettings(BaseSettings):
    model_config = {
        **BaseSettings.model_config,
        "env_prefix": "FLEXT_MYAPP_",
    }

    database_url: str = Field(
        "sqlite:///db.sqlite3",
        description="Database connection URL"
    )
    api_port: int = Field(8000, description="API port")
    debug: bool = Field(False, description="Debug mode")

# Load settings
settings = get_settings(AppSettings)

# Use settings
print(f"Database: {settings.database_url}")
print(f"Port: {settings.api_port}")
print(f"Debug: {settings.debug}")
```

### 3. Singer Tap/Target Project

**Before (custom config):**

```python
# config.py
config_jsonschema = {
    "type": "object",
    "properties": {
        "api_url": {"type": "string"},
        "api_key": {"type": "string", "secret": True},
        "page_size": {"type": "integer", "default": 100},
    },
    "required": ["api_url", "api_key"]
}

# tap.py
def discover_streams(config):
    api_url = config["api_url"]
    page_size = config.get("page_size", 100)
```

**After (using flext-core config):**

```python
from flext_core import SingerConfig, get_config
from pydantic import Field

class TapConfig(SingerConfig):
    api_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    page_size: int = Field(100, description="Page size")

# In tap.py
def discover_streams(raw_config):
    # Convert raw dict to typed config
    config = get_config(TapConfig, raw_config)

    # Use with type safety
    api_url = config.api_url
    page_size = config.page_size
```

### 4. Django Project

**Before (settings.py):**

```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "insecure")
DEBUG = os.environ.get("DEBUG", "False") == "True"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "").split(",")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "myapp"),
        "USER": os.environ.get("DB_USER", "postgres"),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
    }
}
```

**After (using flext-core config):**

```python
# settings_config.py
from flext_core import DjangoSettings, django_settings_adapter

class MyDjangoSettings(DjangoSettings):
    model_config = {
        **DjangoSettings.model_config,
        "env_prefix": "MYAPP_",
    }

# settings.py
from pathlib import Path
from .settings_config import MyDjangoSettings
from flext_core import django_settings_adapter

BASE_DIR = Path(__file__).resolve().parent.parent

# Load settings
settings = MyDjangoSettings()
django_config = django_settings_adapter(settings, {
    "BASE_DIR": BASE_DIR,
    "INSTALLED_APPS": [
        # Your apps here
    ],
})

# Apply all settings
globals().update(django_config)
```

### 5. CLI Project

**Before (custom YAML config):**

```python
import yaml

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

config = load_config()
output_format = config.get("output_format", "text")
```

**After (using flext-core config):**

```python
from flext_core import CLIConfig, get_config

class MyCLIConfig(CLIConfig):
    # Inherits output_format, verbose, quiet
    # Add your custom fields
    api_endpoint: str = Field(
        "https://api.example.com",
        description="API endpoint"
    )

# Load config
config = get_config(MyCLIConfig)

# Use in flext-cli
from flext_core import cli_config_to_dict
cli_dict = cli_config_to_dict(config)
```

### 6. Using Dynaconf Features

For advanced configuration needs:

```python
from flext_core import DynaconfSettings, DynaconfBridge
from pathlib import Path
from pydantic import Field

class AdvancedSettings(DynaconfSettings):
    model_config = {
        **DynaconfSettings.model_config,
        "env_prefix": "MYAPP_",
    }

    # Your fields
    feature_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags"
    )

    # Dynaconf-specific settings
    _dynaconf_settings_files = ["settings.toml", ".secrets.toml"]
    _dynaconf_environments = True

# Load with Dynaconf
settings = AdvancedSettings.from_dynaconf()

# Or use bridge for more control
bridge = DynaconfBridge(
    settings_class=AdvancedSettings,
    settings_files=["settings.yaml", "settings.prod.yaml"],
    root_path=Path.cwd()
)

settings = bridge.load_settings()
```

## Common Patterns

### Environment-Specific Defaults

```python
class AppSettings(BaseSettings):
    @property
    def database_pool_size(self) -> int:
        if self.environment == "production":
            return 20
        return 5
```

### Validation

```python
from flext_core.config.validators import validate_url
from pydantic import field_validator

class APISettings(BaseSettings):
    webhook_url: str

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook(cls, v: str) -> str:
        return validate_url(v, require_tld=True)
```

### Secret Management

```python
class SecureSettings(BaseSettings):
    api_key: str = Field(..., description="API key")

    def __repr__(self):
        # Hide sensitive values
        return f"<{self.__class__.__name__} project={self.project_name}>"
```

## Testing

```python
import pytest
from flext_core import get_settings

def test_settings(monkeypatch):
    # Set test environment variables
    monkeypatch.setenv("FLEXT_MYAPP_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("FLEXT_MYAPP_DEBUG", "true")

    settings = get_settings(MyAppSettings)
    assert settings.database_url == "sqlite:///:memory:"
    assert settings.debug is True
```

## Troubleshooting

### Import Errors

- Ensure `flext-core>=0.6.0` is installed
- Remove direct `dynaconf` and `pydantic-settings` dependencies

### Environment Variables Not Loading

- Check your `env_prefix` in `model_config`
- Verify `.env` file is in the correct location
- Use `settings.model_dump()` to see all loaded values

### Validation Errors

- Pydantic provides detailed error messages
- Check required fields have defaults or environment values
- Use `Field(...)` for required fields without defaults

## Benefits of Migration

1. **Type Safety**: Full IDE support and runtime validation
2. **Consistency**: Same configuration approach across all FLEXT projects
3. **Flexibility**: Use simple env vars or advanced Dynaconf features
4. **Maintainability**: Clear configuration schema in code
5. **Testing**: Easy to mock and test configurations
