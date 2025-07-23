# Secret Management Guide

FLEXT Core provides enterprise-grade secret management capabilities built on Pydantic's `SecretStr` and `SecretBytes` types, ensuring sensitive data is never accidentally exposed in logs, error messages, or debugging output.

## Core Concepts

### SecretStr and SecretBytes

Pydantic's secret types automatically hide sensitive values while preserving functionality:

```python
from pydantic import SecretStr, SecretBytes
from flext_core.config import FlextCoreSettings

class DatabaseSettings(FlextCoreSettings):
    """Database configuration with secret management."""
    
    # Regular field - visible in logs
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field("myapp", description="Database name")
    
    # Secret fields - automatically hidden
    username: SecretStr = Field(..., description="Database username")
    password: SecretStr = Field(..., description="Database password")
    ssl_key: SecretBytes | None = Field(None, description="SSL private key")

# Usage
settings = DatabaseSettings(
    username="db_user",
    password="super_secret_password",
    ssl_key=b"-----BEGIN PRIVATE KEY-----\n..."
)

# Safe operations
print(settings)  # SecretStr values show as '**********'
print(repr(settings))  # Same protection in repr()

# Accessing secret values when needed
username = settings.username.get_secret_value()  # str
password = settings.password.get_secret_value()  # str
ssl_key = settings.ssl_key.get_secret_value() if settings.ssl_key else None  # bytes
```

### Automatic Masking

Secret types are automatically masked in all output contexts:

```python
from flext_core.config import FlextCoreSettings
from pydantic import SecretStr

class APISettings(FlextCoreSettings):
    api_key: SecretStr = Field(..., description="API authentication key")
    webhook_secret: SecretStr = Field(..., description="Webhook signature secret")
    service_name: str = Field("myapp", description="Service name")

settings = APISettings(
    api_key="sk-1234567890abcdef",
    webhook_secret="whsec_very_secret_key",
    service_name="payment-service"
)

# All of these hide the secrets:
print(settings)
# APISettings(api_key=SecretStr('**********'), webhook_secret=SecretStr('**********'), service_name='payment-service')

print(settings.model_dump())
# {'api_key': SecretStr('**********'), 'webhook_secret': SecretStr('**********'), 'service_name': 'payment-service'}

print(settings.model_dump_json())
# {"api_key":"**********","webhook_secret":"**********","service_name":"payment-service"}

# Only when explicitly requested:
secret_key = settings.api_key.get_secret_value()  # "sk-1234567890abcdef"
```

## Advanced Secret Management Patterns

### Environment-Specific Secret Handling

```python
from flext_core.config import FlextCoreSettings
from flext_core.constants import FlextEnvironment
from pydantic import SecretStr, field_validator

class SecureAppSettings(FlextCoreSettings):
    """Application settings with environment-aware secret handling."""
    
    # Database secrets
    db_password: SecretStr = Field(..., description="Database password")
    db_ssl_cert: SecretStr | None = Field(None, description="Database SSL certificate")
    
    # API secrets
    jwt_secret: SecretStr = Field(..., description="JWT signing secret")
    api_key: SecretStr = Field(..., description="External API key")
    
    # Encryption secrets
    encryption_key: SecretStr = Field(..., description="Data encryption key")
    
    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret_strength(cls, v: SecretStr) -> SecretStr:
        """Ensure JWT secret meets security requirements."""
        secret = v.get_secret_value()
        
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        
        # In production, require even stronger secrets
        if len(secret) < 64:
            import os
            env = os.getenv("FLEXT_ENVIRONMENT", "development")
            if env == "production":
                raise ValueError("Production JWT secret must be at least 64 characters")
        
        return v
    
    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key(cls, v: SecretStr) -> SecretStr:
        """Validate encryption key format and strength."""
        key = v.get_secret_value()
        
        # Check for proper base64 encoding (common for encryption keys)
        import base64
        try:
            decoded = base64.b64decode(key, validate=True)
            if len(decoded) not in [16, 24, 32]:  # AES key sizes
                raise ValueError("Encryption key must be 16, 24, or 32 bytes when decoded")
        except Exception as e:
            raise ValueError(f"Invalid encryption key format: {e}") from e
        
        return v
    
    def get_database_url(self, include_ssl: bool = True) -> str:
        """Build database URL with secret password."""
        password = self.db_password.get_secret_value()
        base_url = f"postgresql://user:{password}@{self.db_host}:{self.db_port}/{self.db_name}"
        
        if include_ssl and self.db_ssl_cert:
            # SSL certificate handling would go here
            base_url += "?sslmode=require"
        
        return base_url
    
    def get_jwt_config(self) -> dict[str, str]:
        """Get JWT configuration with secret key."""
        return {
            "secret_key": self.jwt_secret.get_secret_value(),
            "algorithm": "HS256",
            "access_token_expire_minutes": 30 if self.environment == FlextEnvironment.PRODUCTION else 1440,
        }
```

### Secret Rotation Support

```python
from datetime import datetime, timedelta
from typing import Optional
from pydantic import SecretStr, Field, field_validator

class RotatingSecretSettings(FlextCoreSettings):
    """Settings with secret rotation support."""
    
    # Current active secrets
    current_api_key: SecretStr = Field(..., description="Current API key")
    current_jwt_secret: SecretStr = Field(..., description="Current JWT secret")
    
    # Previous secrets for graceful rotation
    previous_api_key: SecretStr | None = Field(None, description="Previous API key (for rotation)")
    previous_jwt_secret: SecretStr | None = Field(None, description="Previous JWT secret (for rotation)")
    
    # Rotation metadata
    secret_rotation_date: datetime | None = Field(None, description="Last secret rotation date")
    rotation_grace_period_hours: int = Field(24, description="Grace period for old secrets")
    
    def is_in_rotation_grace_period(self) -> bool:
        """Check if we're in the grace period for secret rotation."""
        if not self.secret_rotation_date:
            return False
        
        grace_period = timedelta(hours=self.rotation_grace_period_hours)
        return datetime.utcnow() - self.secret_rotation_date < grace_period
    
    def get_valid_api_keys(self) -> list[str]:
        """Get all currently valid API keys (current + previous if in grace period)."""
        keys = [self.current_api_key.get_secret_value()]
        
        if self.previous_api_key and self.is_in_rotation_grace_period():
            keys.append(self.previous_api_key.get_secret_value())
        
        return keys
    
    def get_valid_jwt_secrets(self) -> list[str]:
        """Get all currently valid JWT secrets for token validation."""
        secrets = [self.current_jwt_secret.get_secret_value()]
        
        if self.previous_jwt_secret and self.is_in_rotation_grace_period():
            secrets.append(self.previous_jwt_secret.get_secret_value())
        
        return secrets
    
    def rotate_secrets(
        self, 
        new_api_key: str, 
        new_jwt_secret: str
    ) -> "RotatingSecretSettings":
        """Rotate secrets, keeping old ones for grace period."""
        return self.__class__(
            # Move current to previous
            previous_api_key=self.current_api_key,
            previous_jwt_secret=self.current_jwt_secret,
            
            # Set new current
            current_api_key=new_api_key,
            current_jwt_secret=new_jwt_secret,
            
            # Update rotation metadata
            secret_rotation_date=datetime.utcnow(),
            rotation_grace_period_hours=self.rotation_grace_period_hours,
            
            # Copy other settings
            environment=self.environment,
        )
```

### Hierarchical Secret Management

```python
from typing import Dict, Any
from pydantic import SecretStr, Field

class HierarchicalSecrets(FlextCoreSettings):
    """Settings with hierarchical secret organization."""
    
    # Service-level secrets
    service_master_key: SecretStr = Field(..., description="Master service encryption key")
    
    # Database secrets
    primary_db_password: SecretStr = Field(..., description="Primary database password")
    replica_db_password: SecretStr = Field(..., description="Read replica password")
    cache_password: SecretStr = Field(..., description="Cache (Redis) password")
    
    # External service secrets
    payment_api_key: SecretStr = Field(..., description="Payment provider API key")
    email_api_key: SecretStr = Field(..., description="Email service API key")
    monitoring_token: SecretStr = Field(..., description="Monitoring service token")
    
    # Feature-specific secrets
    oauth_client_secret: SecretStr = Field(..., description="OAuth client secret")
    webhook_signing_secret: SecretStr = Field(..., description="Webhook signature secret")
    
    def get_database_secrets(self) -> Dict[str, str]:
        """Get all database-related secrets."""
        return {
            "primary": self.primary_db_password.get_secret_value(),
            "replica": self.replica_db_password.get_secret_value(),
            "cache": self.cache_password.get_secret_value(),
        }
    
    def get_external_api_secrets(self) -> Dict[str, str]:
        """Get external service API secrets."""
        return {
            "payment": self.payment_api_key.get_secret_value(),
            "email": self.email_api_key.get_secret_value(),
            "monitoring": self.monitoring_token.get_secret_value(),
        }
    
    def get_oauth_config(self) -> Dict[str, str]:
        """Get OAuth configuration with secrets."""
        return {
            "client_secret": self.oauth_client_secret.get_secret_value(),
            "redirect_uri": f"https://{self.service_host}/auth/callback",
        }
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt data using the master key (example implementation)."""
        master_key = self.service_master_key.get_secret_value()
        # Implementation would use proper encryption library
        # This is just a placeholder showing the pattern
        import hashlib
        return hashlib.sha256(f"{master_key}:{data}".encode()).hexdigest()
```

## Secret Validation Patterns

### Custom Secret Validators

```python
import re
from pydantic import SecretStr, field_validator

class ValidatedSecrets(FlextCoreSettings):
    """Settings with comprehensive secret validation."""
    
    # API key with format validation
    stripe_api_key: SecretStr = Field(..., description="Stripe API key")
    github_token: SecretStr = Field(..., description="GitHub personal access token")
    aws_secret_key: SecretStr = Field(..., description="AWS secret access key")
    
    @field_validator("stripe_api_key")
    @classmethod
    def validate_stripe_key(cls, v: SecretStr) -> SecretStr:
        """Validate Stripe API key format."""
        key = v.get_secret_value()
        
        # Stripe keys have specific prefixes
        if not (key.startswith("sk_") or key.startswith("pk_")):
            raise ValueError("Stripe API key must start with 'sk_' or 'pk_'")
        
        # Check length (Stripe keys are typically around 107 characters)
        if len(key) < 50:
            raise ValueError("Stripe API key appears to be too short")
        
        return v
    
    @field_validator("github_token")
    @classmethod
    def validate_github_token(cls, v: SecretStr) -> SecretStr:
        """Validate GitHub token format."""
        token = v.get_secret_value()
        
        # GitHub personal access tokens start with 'ghp_'
        # GitHub app tokens start with 'ghs_'
        if not (token.startswith("ghp_") or token.startswith("ghs_")):
            raise ValueError("GitHub token must start with 'ghp_' or 'ghs_'")
        
        # Tokens are typically 40 characters after prefix
        if len(token) != 40:
            raise ValueError("GitHub token must be exactly 40 characters")
        
        return v
    
    @field_validator("aws_secret_key")
    @classmethod
    def validate_aws_secret(cls, v: SecretStr) -> SecretStr:
        """Validate AWS secret access key format."""
        secret = v.get_secret_value()
        
        # AWS secret keys are base64-like strings, 40 characters
        if len(secret) != 40:
            raise ValueError("AWS secret access key must be 40 characters")
        
        # Should only contain base64 characters
        if not re.match(r'^[A-Za-z0-9+/]+$', secret):
            raise ValueError("AWS secret key contains invalid characters")
        
        return v
```

### Runtime Secret Validation

```python
from typing import Callable, Any
from pydantic import SecretStr, Field

class RuntimeValidatedSecrets(FlextCoreSettings):
    """Settings with runtime secret validation."""
    
    database_password: SecretStr = Field(..., description="Database password")
    api_key: SecretStr = Field(..., description="External API key")
    
    def validate_database_connection(self) -> bool:
        """Validate database password by attempting connection."""
        password = self.database_password.get_secret_value()
        
        # Placeholder for actual database connection test
        try:
            # import psycopg2
            # conn = psycopg2.connect(
            #     host=self.db_host,
            #     database=self.db_name,
            #     user=self.db_user,
            #     password=password
            # )
            # conn.close()
            return True
        except Exception:
            return False
    
    def validate_api_key_permissions(self) -> bool:
        """Validate API key has required permissions."""
        api_key = self.api_key.get_secret_value()
        
        # Placeholder for actual API validation
        try:
            # import requests
            # response = requests.get(
            #     "https://api.example.com/permissions",
            #     headers={"Authorization": f"Bearer {api_key}"}
            # )
            # return response.status_code == 200
            return True
        except Exception:
            return False
    
    def validate_all_secrets(self) -> Dict[str, bool]:
        """Validate all secrets at runtime."""
        return {
            "database": self.validate_database_connection(),
            "api_key": self.validate_api_key_permissions(),
        }
```

## Integration with External Secret Stores

### AWS Secrets Manager Integration

```python
import boto3
from botocore.exceptions import ClientError
from pydantic import SecretStr, Field

class AWSSecretsSettings(FlextCoreSettings):
    """Settings that load secrets from AWS Secrets Manager."""
    
    # AWS configuration
    aws_region: str = Field("us-east-1", description="AWS region")
    secret_name_prefix: str = Field("myapp/", description="Secret name prefix")
    
    # Local fallbacks (for development)
    local_db_password: SecretStr | None = Field(None, description="Local database password")
    local_api_key: SecretStr | None = Field(None, description="Local API key")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._secrets_client = None
        self._cached_secrets = {}
    
    @property
    def secrets_client(self):
        """Lazy AWS Secrets Manager client."""
        if self._secrets_client is None:
            self._secrets_client = boto3.client(
                'secretsmanager',
                region_name=self.aws_region
            )
        return self._secrets_client
    
    def get_secret_from_aws(self, secret_name: str) -> str | None:
        """Retrieve secret from AWS Secrets Manager."""
        if secret_name in self._cached_secrets:
            return self._cached_secrets[secret_name]
        
        try:
            full_secret_name = f"{self.secret_name_prefix}{secret_name}"
            response = self.secrets_client.get_secret_value(SecretId=full_secret_name)
            secret_value = response['SecretString']
            
            # Cache the secret
            self._cached_secrets[secret_name] = secret_value
            return secret_value
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'DecryptionFailureException':
                raise e
            elif e.response['Error']['Code'] == 'InternalServiceErrorException':
                raise e
            elif e.response['Error']['Code'] == 'InvalidParameterException':
                raise e
            elif e.response['Error']['Code'] == 'InvalidRequestException':
                raise e
            elif e.response['Error']['Code'] == 'ResourceNotFoundException':
                return None
        except Exception:
            return None
    
    @property
    def database_password(self) -> SecretStr:
        """Get database password from AWS or local fallback."""
        if self.environment == FlextEnvironment.PRODUCTION:
            aws_secret = self.get_secret_from_aws("database/password")
            if aws_secret:
                return SecretStr(aws_secret)
        
        if self.local_db_password:
            return self.local_db_password
        
        raise ValueError("Database password not available from AWS or local config")
    
    @property
    def api_key(self) -> SecretStr:
        """Get API key from AWS or local fallback."""
        if self.environment == FlextEnvironment.PRODUCTION:
            aws_secret = self.get_secret_from_aws("external-api/key")
            if aws_secret:
                return SecretStr(aws_secret)
        
        if self.local_api_key:
            return self.local_api_key
        
        raise ValueError("API key not available from AWS or local config")
```

### HashiCorp Vault Integration

```python
import hvac
from pydantic import SecretStr, Field

class VaultSecrets(FlextCoreSettings):
    """Settings that integrate with HashiCorp Vault."""
    
    # Vault configuration
    vault_url: str = Field("http://localhost:8200", description="Vault server URL")
    vault_token: SecretStr | None = Field(None, description="Vault authentication token")
    vault_path_prefix: str = Field("secret/myapp/", description="Vault secret path prefix")
    
    # Vault auth method
    vault_role_id: str | None = Field(None, description="AppRole role ID")
    vault_secret_id: SecretStr | None = Field(None, description="AppRole secret ID")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vault_client = None
        self._authenticated = False
    
    @property
    def vault_client(self):
        """Lazy Vault client with authentication."""
        if self._vault_client is None:
            self._vault_client = hvac.Client(url=self.vault_url)
            self._authenticate_vault()
        return self._vault_client
    
    def _authenticate_vault(self):
        """Authenticate with Vault using available method."""
        if not self._authenticated:
            if self.vault_token:
                # Token authentication
                self._vault_client.token = self.vault_token.get_secret_value()
            elif self.vault_role_id and self.vault_secret_id:
                # AppRole authentication
                auth_response = self._vault_client.auth.approle.login(
                    role_id=self.vault_role_id,
                    secret_id=self.vault_secret_id.get_secret_value()
                )
                self._vault_client.token = auth_response['auth']['client_token']
            else:
                raise ValueError("No Vault authentication method configured")
            
            self._authenticated = True
    
    def get_vault_secret(self, secret_path: str, key: str) -> str | None:
        """Get secret from Vault."""
        try:
            full_path = f"{self.vault_path_prefix}{secret_path}"
            response = self.vault_client.secrets.kv.v2.read_secret_version(path=full_path)
            return response['data']['data'].get(key)
        except Exception:
            return None
    
    @property
    def database_password(self) -> SecretStr:
        """Get database password from Vault."""
        password = self.get_vault_secret("database", "password")
        if not password:
            raise ValueError("Database password not found in Vault")
        return SecretStr(password)
    
    @property 
    def encryption_key(self) -> SecretStr:
        """Get encryption key from Vault."""
        key = self.get_vault_secret("encryption", "key")
        if not key:
            raise ValueError("Encryption key not found in Vault")
        return SecretStr(key)
```

## Testing with Secrets

### Mock Secrets for Testing

```python
import pytest
from unittest.mock import patch, MagicMock
from pydantic import SecretStr

class TestSecrets:
    """Test cases for secret management."""
    
    def test_secret_masking(self):
        """Test that secrets are properly masked."""
        from myapp.config import AppSettings
        
        settings = AppSettings(
            database_password="super_secret",
            api_key="sk-1234567890"
        )
        
        # Should be masked in string representations
        settings_str = str(settings)
        assert "super_secret" not in settings_str
        assert "sk-1234567890" not in settings_str
        assert "**********" in settings_str
    
    def test_secret_access(self):
        """Test accessing secret values."""
        from myapp.config import AppSettings
        
        settings = AppSettings(
            database_password="super_secret",
            api_key="sk-1234567890"
        )
        
        # Should be able to access when needed
        assert settings.database_password.get_secret_value() == "super_secret"
        assert settings.api_key.get_secret_value() == "sk-1234567890"
    
    @patch.dict('os.environ', {
        'FLEXT_DATABASE_PASSWORD': 'env_password',
        'FLEXT_API_KEY': 'env_api_key'
    })
    def test_secrets_from_environment(self):
        """Test loading secrets from environment variables."""
        from myapp.config import AppSettings
        
        settings = AppSettings()
        
        assert settings.database_password.get_secret_value() == "env_password"
        assert settings.api_key.get_secret_value() == "env_api_key"
    
    def test_secret_validation(self):
        """Test secret validation rules."""
        from myapp.config import AppSettings
        
        # Should reject weak secrets
        with pytest.raises(ValueError, match="at least 32 characters"):
            AppSettings(
                database_password="short",
                api_key="sk-1234567890",
                jwt_secret="weak"  # Too short
            )
    
    @patch('boto3.client')
    def test_aws_secrets_integration(self, mock_boto):
        """Test AWS Secrets Manager integration."""
        from myapp.config import AWSSecretsSettings
        
        # Mock AWS response
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            'SecretString': 'aws_secret_value'
        }
        
        settings = AWSSecretsSettings(
            aws_region="us-west-2",
            secret_name_prefix="test/"
        )
        
        # Should retrieve from AWS
        secret = settings.get_secret_from_aws("database/password")
        assert secret == "aws_secret_value"
        
        mock_client.get_secret_value.assert_called_once_with(
            SecretId="test/database/password"
        )

@pytest.fixture
def test_secrets():
    """Provide test secrets configuration."""
    return {
        "database_password": "test_db_password",
        "api_key": "test_api_key",
        "jwt_secret": "test_jwt_secret_that_is_long_enough_for_validation"
    }

def test_with_secrets(test_secrets):
    """Test using secrets fixture."""
    from myapp.config import AppSettings
    
    settings = AppSettings(**test_secrets)
    
    assert settings.database_password.get_secret_value() == "test_db_password"
    assert settings.api_key.get_secret_value() == "test_api_key"
```

## Best Practices

### 1. Always Use Secret Types for Sensitive Data

```python
# Good - properly protected
class Settings(FlextCoreSettings):
    password: SecretStr = Field(..., description="Database password")
    api_key: SecretStr = Field(..., description="API key")

# Bad - exposed in logs
class Settings(FlextCoreSettings):
    password: str = Field(..., description="Database password")
    api_key: str = Field(..., description="API key")
```

### 2. Validate Secret Strength and Format

```python
class Settings(FlextCoreSettings):
    jwt_secret: SecretStr = Field(..., description="JWT secret")
    
    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: SecretStr) -> SecretStr:
        secret = v.get_secret_value()
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v
```

### 3. Support Secret Rotation

```python
class Settings(FlextCoreSettings):
    current_key: SecretStr = Field(..., description="Current encryption key")
    previous_key: SecretStr | None = Field(None, description="Previous key for rotation")
    
    def get_decryption_keys(self) -> list[str]:
        """Get keys for decryption (current + previous)."""
        keys = [self.current_key.get_secret_value()]
        if self.previous_key:
            keys.append(self.previous_key.get_secret_value())
        return keys
```

### 4. Use External Secret Stores in Production

```python
class Settings(FlextCoreSettings):
    @property
    def database_password(self) -> SecretStr:
        if self.environment == FlextEnvironment.PRODUCTION:
            # Load from external store
            return self._load_from_vault("db_password")
        else:
            # Use local env var for development
            return self.local_db_password
```

### 5. Never Log or Display Secrets

```python
# Good - safe logging
logger.info(f"Connecting to database at {settings.db_host}")

# Bad - might expose secrets
logger.info(f"Using config: {settings}")  # Could expose secrets if not using SecretStr
```

See the [Configuration Troubleshooting Guide](troubleshooting.md) for common secret management issues and solutions.
