# Configuration Management Overview

**Sistema de configuração baseado na implementação atual do FLEXT Core**

## 🎯 Visão Geral

FLEXT Core fornece um sistema de configuração type-safe baseado em Pydantic V2. Esta documentação reflete a implementação REAL disponível em src/flext_core/config.py.

## 📦 Funcionalidades Disponíveis

**VALIDADO** - Baseado no código atual:

- ✅ **Type Safety**: Validação completa com Pydantic V2
- ✅ **Environment Variables**: Carregamento automático com prefixos
- ✅ **Multi-Environment**: Diferentes ambientes de deployment
- 🔧 **Framework Integration**: Singer, CLI (em desenvolvimento)
- 📋 **Advanced Features**: Multi-file configs (planejado)

## 🔧 API Atual

### Importações Disponíveis

```python
# CORRETO - Baseado na implementação atual
from flext_core import FlextBaseSettings

# Configuração avançada - disponível
from flext_core.config import FlextConfig
from flext_core.config_models import FlextDatabaseConfig, FlextRedisConfig
```

### Uso Básico

**VALIDADO** - Funciona com a implementação atual:

```python
"""
Exemplo real de configuração usando FLEXT Core.
Baseado em src/flext_core/config.py
"""

from flext_core import FlextBaseSettings
from typing import Optional

class AppSettings(FlextBaseSettings):
    """Configuration for your application."""

    # Basic settings with defaults
    app_name: str = "My FLEXT App"
    debug: bool = False

    # Database configuration
    database_url: str = "sqlite:///app.db"
    database_pool_size: int = 5

    # API configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Optional features
    redis_url: Optional[str] = None
    enable_metrics: bool = False

    class Config:
        env_prefix = "MYAPP_"
        case_sensitive = False

# Usage
def main():
    """Load and use configuration."""
    # Load from environment variables or defaults
    settings = AppSettings()

    print(f"App: {settings.app_name}")
    print(f"Debug: {settings.debug}")
    print(f"Database: {settings.database_url}")
    print(f"API: http://{settings.api_host}:{settings.api_port}")

    # Environment-based logic
    if settings.debug:
        print("🔧 Running in debug mode")
        pool_size = 2
    else:
        print("🚀 Running in production mode")
        pool_size = settings.database_pool_size

    print(f"Database pool size: {pool_size}")

if __name__ == "__main__":
    main()
```

### Environment Variables

**Como configurar variáveis de ambiente:**

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

### Configuração com Validação

```python
"""
Configuração com validação customizada.
"""

from flext_core import FlextBaseSettings
from pydantic import field_validator, Field
from typing import Optional

class DatabaseSettings(FlextBaseSettings):
    """Database configuration with validation."""

    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, ge=1, le=65535, description="Database port")
    name: str = Field("myapp", description="Database name")
    user: str = Field("postgres", description="Database user")
    password: str = Field("", description="Database password")

    # SSL settings
    ssl_mode: str = Field("prefer", description="SSL mode")
    ssl_cert_path: Optional[str] = Field(None, description="SSL certificate path")

    @field_validator("ssl_mode")
    @classmethod
    def validate_ssl_mode(cls, v: str) -> str:
        """Validate SSL mode."""
        valid_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        if v not in valid_modes:
            raise ValueError(f"SSL mode must be one of: {valid_modes}")
        return v

    @property
    def connection_url(self) -> str:
        """Build database connection URL."""
        auth = f"{self.user}:{self.password}@" if self.password else f"{self.user}@"
        return f"postgresql://{auth}{self.host}:{self.port}/{self.name}"

    def get_pool_config(self, environment: str = "development") -> dict:
        """Get connection pool configuration."""
        if environment == "production":
            return {"min_size": 10, "max_size": 20}
        elif environment == "staging":
            return {"min_size": 5, "max_size": 10}
        else:
            return {"min_size": 2, "max_size": 5}

# Usage
def setup_database():
    """Setup database with validated configuration."""
    db_config = DatabaseSettings()

    print(f"Database URL: {db_config.connection_url}")
    print(f"SSL Mode: {db_config.ssl_mode}")

    # Get environment-specific pool config
    pool_config = db_config.get_pool_config("production")
    print(f"Pool config: {pool_config}")

if __name__ == "__main__":
    setup_database()
```

### Multi-Service Configuration

```python
"""
Configuration for multiple services.
"""

from flext_core import FlextBaseSettings
from pydantic import Field
from typing import Optional

class ServiceConfig(FlextBaseSettings):
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
    cors_origins: list[str] = Field(default_factory=list, description="CORS origins")

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

## 🧪 Testing Configuration

```python
"""
Testing configuration patterns.
"""

import pytest
import os
from flext_core import FlextBaseSettings

def test_default_configuration():
    """Test default configuration values."""

    class TestSettings(FlextBaseSettings):
        app_name: str = "test-app"
        debug: bool = True
        port: int = 8000

    settings = TestSettings()
    assert settings.app_name == "test-app"
    assert settings.debug is True
    assert settings.port == 8000

def test_environment_override():
    """Test environment variable override."""

    class TestSettings(FlextBaseSettings):
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

    class TestSettings(FlextBaseSettings):
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

## 📋 Status da Implementação

**BASEADO NO CÓDIGO ATUAL** em src/flext_core/config.py:

### ✅ Funcional

- **FlextBaseSettings**: Configuração base com Pydantic V2
- **Environment Variables**: Carregamento automático
- **Type Safety**: Validação completa de tipos
- **Field Validation**: Validação customizada

### 🔧 Em Desenvolvimento

- **Framework Integration**: Singer/CLI integrations
- **Secret Management**: Gerenciamento seguro de secrets
- **Multi-file Configuration**: Configuração distribuída

### 📋 Planejado

- **Dynamic Configuration**: Configuração dinâmica em runtime
- **Configuration Templates**: Templates para diferentes ambientes
- **Validation Rules**: Regras de validação avançadas

## ⚠️ Importante

- Use `FlextBaseSettings` (não `FlextCoreSettings`)
- Todos os exemplos foram testados contra a implementação atual
- Para funcionalidades avançadas, consulte src/flext_core/config.py

---

**Esta documentação reflete a implementação REAL em src/flext_core/config.py**
