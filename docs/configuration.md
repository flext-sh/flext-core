# Configuration System

**Foundation for FLEXT Framework Configuration**

FLEXT Core provides the configuration foundation used by all FLEXT modules. Built on Pydantic with workspace-wide environment variable support, validation, and type safety.

## FLEXT Workspace Configuration

### BaseSettings Foundation

The `BaseSettings` class is inherited by **all FLEXT modules** and provides automatic environment variable loading with the `FLEXT_` prefix:

```python
from flext_core.config.base import BaseSettings

class AppSettings(BaseSettings):
    database_url: str = "sqlite:///app.db"
    debug: bool = False
    log_level: str = "INFO"
    api_timeout: int = 30

# Automatically reads environment variables (used by all FLEXT modules):
# FLEXT_DATABASE_URL, FLEXT_DEBUG, FLEXT_LOG_LEVEL, FLEXT_API_TIMEOUT
settings = AppSettings()

# This pattern is used in:
# - flext-api for API server configuration
# - flext-web for Django settings
# - flext-meltano for ETL configuration
# - flext-auth for authentication settings
```

### BaseConfig Class

For configuration objects without environment variable loading:

```python
from flext_core.config.base import BaseConfig

class DatabaseConfig(BaseConfig):
    host: str = "localhost"
    port: int = 5432
    name: str = "myapp"
    
    def get_url(self) -> str:
        return f"postgresql://{self.host}:{self.port}/{self.name}"

config = DatabaseConfig(host="prod-db", port=5433)
```

## Environment Variables

### Automatic Loading

Environment variables are automatically loaded with the `FLEXT_` prefix:

```bash
# Set environment variables
export FLEXT_DATABASE_URL="postgresql://localhost/prod"
export FLEXT_DEBUG="true"
export FLEXT_LOG_LEVEL="DEBUG"
```

```python
class Settings(BaseSettings):
    database_url: str = "sqlite:///default.db"
    debug: bool = False
    log_level: str = "INFO"

settings = Settings()
# Will use environment values if set, defaults otherwise
```

### FLEXT Module-Specific Configuration

FLEXT modules extend Core's BaseSettings with module-specific prefixes:

```python
from pydantic_settings import SettingsConfigDict
from flext_core.config.base import BaseSettings

# flext-api configuration
class FlextAPISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLEXT_API_")
    
    host: str = "localhost"
    port: int = 8000
    workers: int = 4

# flext-web configuration
class FlextWebSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLEXT_WEB_")
    
    django_secret_key: str
    allowed_hosts: list[str] = ["localhost"]

# Environment variables:
# FLEXT_API_HOST, FLEXT_API_PORT, FLEXT_API_WORKERS
# FLEXT_WEB_DJANGO_SECRET_KEY, FLEXT_WEB_ALLOWED_HOSTS
```

### Environment Files

Load configuration from `.env` files:

```python
# .env file
FLEXT_DATABASE_URL=postgresql://localhost/myapp
FLEXT_DEBUG=true
FLEXT_API_KEY=secret-key

# Python code
settings = Settings.from_env(".env")
```

## Validation

### Built-in Validators

FLEXT Core includes validators for common configuration values:

```python
from flext_core.config.validators import validate_url, validate_port, validate_database_url

class Settings(BaseSettings):
    api_url: str = "https://api.example.com"
    port: int = 8080
    database_url: str = "postgresql://localhost/app"
    
    def __post_init__(self):
        # Validate configuration values
        self.api_url = validate_url(self.api_url)
        self.port = validate_port(self.port)
        self.database_url = validate_database_url(self.database_url)
```

### Pydantic Validators

Use Pydantic's field validators for custom validation:

```python
from pydantic import field_validator, ValidationError

class Settings(BaseSettings):
    api_key: str
    timeout: int = 30
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if len(v) < 10:
            raise ValueError('API key must be at least 10 characters')
        return v
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v < 1 or v > 300:
            raise ValueError('Timeout must be between 1 and 300 seconds')
        return v
```

## Configuration Patterns

### Environment-Specific Settings

```python
import os
from flext_core.config.base import BaseSettings

class Settings(BaseSettings):
    environment: str = "development"
    database_url: str = "sqlite:///dev.db"
    debug: bool = True
    
    @classmethod
    def for_environment(cls, env: str = None):
        env = env or os.getenv("ENVIRONMENT", "development")
        
        if env == "production":
            return cls(
                environment="production",
                database_url=os.getenv("DATABASE_URL"),
                debug=False
            )
        elif env == "test":
            return cls(
                environment="test",
                database_url="sqlite:///:memory:",
                debug=True
            )
        
        return cls()  # development defaults

# Usage
settings = Settings.for_environment()
```

### Nested Configuration

```python
class DatabaseConfig(BaseConfig):
    host: str = "localhost"
    port: int = 5432
    name: str = "app"
    username: str = "user"
    password: str = "password"

class RedisConfig(BaseConfig):
    host: str = "localhost" 
    port: int = 6379
    db: int = 0

class AppSettings(BaseSettings):
    debug: bool = False
    secret_key: str = "change-me"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load nested configurations
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "app"),
        )
        
        self.redis = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
        )
```

### Configuration Factory

```python
from typing import Type, TypeVar

T = TypeVar('T', bound=BaseSettings)

class ConfigFactory:
    """Factory for creating environment-specific configurations."""
    
    @staticmethod
    def create_settings(settings_class: Type[T], env: str = None) -> T:
        env = env or os.getenv("ENVIRONMENT", "development")
        
        # Load environment-specific .env file
        env_file = f".env.{env}"
        if os.path.exists(env_file):
            return settings_class.from_env(env_file)
        
        return settings_class()

# Usage
settings = ConfigFactory.create_settings(AppSettings, "production")
```

## Testing Configurations

### Test Settings

```python
class TestSettings(BaseSettings):
    """Settings specifically for testing."""
    
    database_url: str = "sqlite:///:memory:"
    debug: bool = True
    testing: bool = True
    api_timeout: int = 5
    
    # Disable external services in tests
    external_api_enabled: bool = False
    email_enabled: bool = False

# In conftest.py
@pytest.fixture
def test_settings():
    return TestSettings()
```

### Mocking Configuration

```python
import pytest
from unittest.mock import patch

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch.dict(os.environ, {
        "FLEXT_DATABASE_URL": "sqlite:///:memory:",
        "FLEXT_DEBUG": "true",
        "FLEXT_API_KEY": "test-key"
    }):
        yield Settings()

def test_with_mock_settings(mock_settings):
    assert mock_settings.database_url == "sqlite:///:memory:"
    assert mock_settings.debug is True
```

## FLEXT Workspace Best Practices

### 1. Follow FLEXT Module Conventions

All FLEXT modules should follow consistent configuration patterns:

```python
from flext_core.config.base import BaseSettings

# FLEXT module configuration template
class FlextModuleSettings(BaseSettings):
    # Required settings (no defaults)
    module_name: str
    version: str
    
    # Optional settings (with sensible defaults)
    debug: bool = False
    log_level: str = "INFO"
    max_connections: int = 100
```

### 2. FLEXT Workspace Environment Variables

Coordinate environment variables across FLEXT modules:

```bash
# Workspace-wide FLEXT configuration
export FLEXT_WORKSPACE_ROOT="/home/marlonsc/flext"
export FLEXT_PYTHON_VERSION="3.13"
export FLEXT_ENV="development"

# Core module configuration
export FLEXT_DEBUG="true"
export FLEXT_LOG_LEVEL="DEBUG"

# Module-specific configuration
export FLEXT_API_HOST="localhost"
export FLEXT_API_PORT="8000"
export FLEXT_WEB_DEBUG="true"
export FLEXT_MELTANO_STATE_DIR="/tmp/meltano"
```

### 3. FLEXT Cross-Module Validation

Validate configuration consistency across FLEXT modules:

```python
from flext_core.config.validators import validate_url, validate_database_url
from flext_core.config.base import BaseSettings

class FlextWorkspaceSettings(BaseSettings):
    """Workspace-wide configuration validation."""
    
    # Shared database (used by flext-web, flext-auth)
    shared_database_url: str
    
    # API endpoints (used by flext-web to connect to flext-api)
    api_base_url: str = "http://localhost:8000"
    
    def __post_init__(self):
        # Validate URLs used across modules
        self.shared_database_url = validate_database_url(self.shared_database_url)
        self.api_base_url = validate_url(self.api_base_url)
        
        # Ensure module compatibility
        if self.shared_database_url.startswith("sqlite") and "production" in self.api_base_url:
            raise ValueError("SQLite not allowed in production with external API")
```

### 4. Use Configuration Sections

Group related settings together:

```python
class APIConfig(BaseConfig):
    url: str
    timeout: int = 30
    retries: int = 3

class DatabaseConfig(BaseConfig):
    url: str
    pool_size: int = 10
    timeout: int = 30

class Settings(BaseSettings):
    debug: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api = APIConfig(url=os.getenv("API_URL"))
        self.database = DatabaseConfig(url=os.getenv("DATABASE_URL"))
```

### 5. Document FLEXT Workspace Variables

Document environment variables with workspace context:

```python
class FlextCoreSettings(BaseSettings):
    """FLEXT Core configuration - Foundation for all FLEXT modules.
    
    Workspace Environment Variables:
        FLEXT_WORKSPACE_ROOT: Path to FLEXT workspace (/home/marlonsc/flext)
        FLEXT_DEBUG: Debug mode for all FLEXT modules (default: false)
        FLEXT_LOG_LEVEL: Logging level for FLEXT ecosystem (default: INFO)
        FLEXT_DATABASE_URL: Shared database for flext-web, flext-auth
        FLEXT_REDIS_URL: Shared cache for session management
        FLEXT_SECRET_KEY: JWT secret shared across flext-api, flext-auth
    
    Module Integration:
        - Used by flext-api for server configuration
        - Extended by flext-web for Django settings
        - Inherited by flext-meltano for ETL configuration
        - Referenced by flext-cli for workspace commands
    """
    
    workspace_root: str = "/home/marlonsc/flext"
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    secret_key: str
    debug: bool = False
    log_level: str = "INFO"
```
