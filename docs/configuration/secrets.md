# Secret Management with FLEXT Core

**Reality-Based Secret Management using FlextBaseSettings**

This guide covers secret management using FLEXT Core's actual implementation - `FlextBaseSettings` class from `src/flext_core/config.py`.

## 🔐 Core Concepts

FLEXT Core provides secret management through Pydantic's `SecretStr` and `SecretBytes` types, integrated with our `FlextBaseSettings` configuration system.

### FlextBaseSettings with Secrets

Based on the actual implementation in `src/flext_core/config.py`:

```python
from flext_core import FlextBaseSettings
from pydantic import SecretStr, Field

class DatabaseSettings(FlextBaseSettings):
    """Database configuration with secret management."""

    # Regular fields - visible in logs
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field("app", description="Database name")

    # Secret fields - automatically hidden
    username: SecretStr = Field(..., description="Database username")
    password: SecretStr = Field(..., description="Database password")

    class Config:
        env_prefix = "DB_"

# Usage
settings = DatabaseSettings(
    username="db_user",
    password="super_secret_password"
)

# Safe - secrets are masked
print(settings)  # username=SecretStr('**********'), password=SecretStr('**********')

# Access secrets when needed
username = settings.username.get_secret_value()  # "db_user"
password = settings.password.get_secret_value()  # "super_secret_password"
```

## 🛡️ Secret Types Available

### SecretStr for String Secrets

```python
from flext_core import FlextBaseSettings
from pydantic import SecretStr

class APISettings(FlextBaseSettings):
    """API configuration with string secrets."""

    api_key: SecretStr = Field(..., description="API authentication key")
    webhook_secret: SecretStr = Field(..., description="Webhook signature secret")
    jwt_secret: SecretStr = Field(..., description="JWT signing secret")

    class Config:
        env_prefix = "API_"

# Environment variables: API_API_KEY, API_WEBHOOK_SECRET, API_JWT_SECRET
settings = APISettings()

# All secrets are masked in output
print(settings.model_dump())  # All SecretStr fields show as '**********'
```

### SecretBytes for Binary Secrets

```python
from flext_core import FlextBaseSettings
from pydantic import SecretStr, SecretBytes

class CryptoSettings(FlextBaseSettings):
    """Cryptographic configuration with binary secrets."""

    # String secrets
    api_token: SecretStr = Field(..., description="API token")

    # Binary secrets
    encryption_key: SecretBytes = Field(..., description="Encryption key")
    ssl_private_key: SecretBytes | None = Field(None, description="SSL private key")

    def get_encryption_key_bytes(self) -> bytes:
        """Get encryption key as bytes."""
        return self.encryption_key.get_secret_value()
```

## 🔧 Practical Secret Management Patterns

### Environment-Based Secret Loading

```python
from flext_core import FlextBaseSettings
from pydantic import SecretStr, field_validator
import os

class AppSettings(FlextBaseSettings):
    """Application settings with environment-aware secrets."""

    # Database secrets
    db_password: SecretStr = Field(..., description="Database password")

    # External service secrets
    redis_password: SecretStr | None = Field(None, description="Redis password")
    api_key: SecretStr = Field(..., description="External API key")

    class Config:
        env_prefix = "APP_"

    @field_validator("api_key")
    @classmethod
    def validate_api_key_format(cls, v: SecretStr) -> SecretStr:
        """Validate API key format."""
        key = v.get_secret_value()

        if len(key) < 20:
            raise ValueError("API key must be at least 20 characters")

        if not key.startswith(("sk_", "pk_", "api_")):
            raise ValueError("API key must start with valid prefix")

        return v

# Environment variables: APP_DB_PASSWORD, APP_REDIS_PASSWORD, APP_API_KEY
settings = AppSettings()
```

### Development vs Production Secrets

```python
from flext_core import FlextBaseSettings
from pydantic import SecretStr
import os

class SecureSettings(FlextBaseSettings):
    """Settings with development/production secret handling."""

    # Required in all environments
    jwt_secret: SecretStr = Field(..., description="JWT signing secret")

    # Optional development fallbacks
    dev_db_password: SecretStr | None = Field(None, description="Development DB password")

    class Config:
        env_prefix = "SECURE_"

    @property
    def database_password(self) -> SecretStr:
        """Get database password based on environment."""
        if self.is_production:
            # In production, get from secure store (placeholder)
            return self.jwt_secret  # Replace with actual secure store integration
        else:
            # In development, use fallback
            return self.dev_db_password or SecretStr("dev_password")

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return os.getenv("ENVIRONMENT", "development") == "production"
```

## 🧪 Secret Testing Patterns

### Mock Secrets for Testing

```python
import pytest
from flext_core import FlextBaseSettings
from pydantic import SecretStr
from unittest.mock import patch

class TestSecrets:
    """Test secret management functionality."""

    def test_secret_masking(self):
        """Test that secrets are properly masked."""

        class TestSettings(FlextBaseSettings):
            password: SecretStr = Field(..., description="Test password")

        settings = TestSettings(password="super_secret")

        # Should be masked in string representations
        settings_str = str(settings)
        assert "super_secret" not in settings_str
        assert "**********" in settings_str

    def test_secret_access(self):
        """Test accessing secret values."""

        class TestSettings(FlextBaseSettings):
            api_key: SecretStr = Field(..., description="API key")

        settings = TestSettings(api_key="sk-1234567890")

        # Should be able to access when needed
        assert settings.api_key.get_secret_value() == "sk-1234567890"

    @patch.dict(os.environ, {"TEST_PASSWORD": "env_password"})
    def test_environment_secret_loading(self):
        """Test loading secrets from environment."""

        class TestSettings(FlextBaseSettings):
            password: SecretStr = Field(..., description="Password")

            class Config:
                env_prefix = "TEST_"

        settings = TestSettings()
        assert settings.password.get_secret_value() == "env_password"

@pytest.fixture
def test_secrets():
    """Provide test secrets fixture."""
    return {
        "database_password": "test_db_pass",
        "api_key": "test_api_key_12345",
        "jwt_secret": "test_jwt_secret_that_is_long_enough"
    }
```

## 📝 Best Practices

### 1. Always Use SecretStr/SecretBytes for Sensitive Data

```python
# ✅ Good - properly protected
class Settings(FlextBaseSettings):
    password: SecretStr = Field(..., description="Database password")
    api_key: SecretStr = Field(..., description="API key")

# ❌ Bad - exposed in logs
class Settings(FlextBaseSettings):
    password: str = Field(..., description="Database password")  # Visible!
    api_key: str = Field(..., description="API key")  # Visible!
```

### 2. Validate Secret Format and Strength

```python
class Settings(FlextBaseSettings):
    password: SecretStr = Field(..., description="Database password")

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: SecretStr) -> SecretStr:
        password = v.get_secret_value()

        if len(password) < 12:
            raise ValueError("Password must be at least 12 characters")

        return v
```

### 3. Use Environment Variables with Prefixes

```python
class Settings(FlextBaseSettings):
    db_password: SecretStr = Field(..., description="Database password")
    api_key: SecretStr = Field(..., description="API key")

    class Config:
        env_prefix = "MYAPP_"  # Loads MYAPP_DB_PASSWORD, MYAPP_API_KEY
```

### 4. Never Log or Display Raw Secrets

```python
# ✅ Good - safe logging
logger.info(f"Connecting to database at {settings.db_host}")

# ❌ Bad - might expose secrets
logger.info(f"Using settings: {settings}")  # Only safe if using SecretStr
print(f"Password is: {settings.password.get_secret_value()}")  # Never do this!
```

## ⚠️ Current Limitations

Based on the actual FlextBaseSettings implementation:

1. **No Built-in External Secret Store Integration** - You need to implement AWS Secrets Manager, Vault, etc. integration yourself
2. **No Built-in Secret Rotation** - Rotation must be handled at the application level
3. **Basic Validation Only** - Advanced secret validation requires custom validators

## 🔗 Integration with FLEXT Core

Secrets work seamlessly with other FLEXT Core patterns:

```python
from flext_core import FlextBaseSettings, FlextResult, FlextContainer
from pydantic import SecretStr

class DatabaseConfig(FlextBaseSettings):
    password: SecretStr = Field(..., description="DB password")

    def get_connection_string(self) -> FlextResult[str]:
        """Get database connection string safely."""
        try:
            password = self.password.get_secret_value()
            conn_str = f"postgresql://user:{password}@localhost/db"
            return FlextResult.ok(conn_str)
        except Exception as e:
            return FlextResult.fail(f"Failed to build connection string: {e}")

# Use with dependency injection
def setup_database_service(container: FlextContainer) -> FlextResult[None]:
    """Setup database service with secret configuration."""
    config = DatabaseConfig()

    conn_result = config.get_connection_string()
    if conn_result.is_failure:
        return FlextResult.fail(f"Config error: {conn_result.error}")

    db_service = DatabaseService(conn_result.data)
    return container.register("database", db_service)
```

---

**This secret management guide is based on the actual FlextBaseSettings implementation in `src/flext_core/config.py`.**
