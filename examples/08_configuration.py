#!/usr/bin/env python3
"""Enterprise configuration management with FlextConfig.

Demonstrates advanced configuration patterns for enterprise applications:
- Environment variable integration
- File-based configuration loading
- Configuration validation with business rules
- Multi-layer configuration merging
- Security-aware configuration handling

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import cast

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core import FlextConfig, FlextResult

# =============================================================================
# DEMONSTRATION CONSTANTS
# =============================================================================

# Example security keys for demonstration purposes only - NOT FOR PRODUCTION
_DEMO_SECRET_KEY_1 = os.getenv(
    "FLEXT_DEMO_SECRET_KEY_1",
    "ValidSecretKey123WithComplexity456",
)
_DEMO_SECRET_KEY_2 = os.getenv(
    "FLEXT_DEMO_SECRET_KEY_2",
    "ValidSecretKey123WithComplexityAndLength",
)
_LOCALHOST_IP = "127.0.0.1"  # Secure localhost binding

# =============================================================================
# ENTERPRISE DATABASE CONFIGURATION
# =============================================================================


class DatabaseConfig(FlextConfig):
    """Database configuration with enterprise validation."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="app_db", min_length=1, description="Database name")
    user: str = Field(default="app_user", min_length=1, description="Database user")
    password: str = Field(default="default_password", description="Database password")
    pool_size: int = Field(default=20, ge=1, le=100, description="Connection pool size")
    timeout: int = Field(default=30, ge=5, le=300, description="Connection timeout")
    ssl_enabled: bool = Field(default=False, description="Enable SSL connections")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate database configuration business rules."""
        if self.host == "localhost" and not self.password:
            return FlextResult[None].fail("Password required for localhost connections")

        # SSH port not allowed for database connections
        ssh_port = 22
        if self.port == ssh_port:
            return FlextResult[None].fail(
                "SSH port not allowed for database connections",
            )

        return FlextResult[None].ok(None)

    def get_connection_url(self) -> str:
        """Get database connection URL."""
        protocol = "postgresql+psycopg2"
        return f"{protocol}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    def get_masked_url(self) -> str:
        """Get connection URL with masked password for safe logging."""
        protocol = "postgresql+psycopg2"
        return f"{protocol}://{self.user}:***@{self.host}:{self.port}/{self.name}"


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================


class CacheConfig(FlextConfig):
    """Cache configuration supporting Redis and Memcached."""

    provider: str = Field(default="redis", pattern="^(redis|memcached|memory)$")
    host: str = Field(default="localhost", min_length=1)
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: str = Field(default="")
    timeout: int = Field(default=5, ge=1, le=60)
    max_connections: int = Field(default=50, ge=1, le=1000)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate cache configuration."""
        # Redis supports database indices 0-15
        max_redis_db_index = 15
        if self.provider == "redis" and self.db > max_redis_db_index:
            return FlextResult[None].fail("Redis database index must be 0-15")

        if self.provider == "memcached" and self.db != 0:
            return FlextResult[None].fail(
                "Memcached does not support database selection",
            )

        return FlextResult[None].ok(None)

    def get_connection_url(self) -> str:
        """Get cache connection URL."""
        if self.provider == "redis":
            if self.password:
                return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
            return f"redis://{self.host}:{self.port}/{self.db}"
        if self.provider == "memcached":
            return f"memcached://{self.host}:{self.port}"
        return "memory://"


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================


class SecurityConfig(FlextConfig):
    """Security configuration with comprehensive validation."""

    secret_key: str = Field(
        default=_DEMO_SECRET_KEY_1,
        min_length=32,
        description="Application secret key",
    )
    token_expiry_hours: int = Field(
        default=24,
        ge=1,
        le=8760,
        description="Token expiry in hours",
    )
    max_login_attempts: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max login attempts",
    )
    session_timeout_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Session timeout",
    )
    enable_2fa: bool = Field(
        default=False,
        description="Enable two-factor authentication",
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key complexity."""
        # Minimum secure key length
        min_secret_key_length = 32
        if len(v) < min_secret_key_length:
            msg = "Secret key must be at least 32 characters"
            raise ValueError(msg)

        # Check complexity requirements
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)

        if not (has_upper and has_lower and has_digit):
            msg = "Secret key must contain uppercase, lowercase, and digits"
            raise ValueError(msg)

        return v

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate security configuration rules."""
        # Conservative max login attempts when 2FA is enabled
        max_2fa_login_attempts = 10
        if self.enable_2fa and self.max_login_attempts > max_2fa_login_attempts:
            return FlextResult[None].fail(
                "Reduce max login attempts when 2FA is enabled",
            )

        if not self.cors_origins:
            return FlextResult[None].fail("CORS origins must be specified")

        return FlextResult[None].ok(None)


# =============================================================================
# MAIN ENTERPRISE CONFIGURATION
# =============================================================================


class EnterpriseConfig(BaseSettings):
    """Main enterprise configuration combining all components."""

    # Application metadata
    app_name: str = Field(default="Enterprise Application")
    version: str = Field(default="1.0.0")
    environment: str = Field(
        default="development",
        pattern="^(development|staging|production)$",
    )
    debug: bool = Field(default=False)

    # Server settings
    host: str = Field(default=_LOCALHOST_IP)
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)

    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Feature flags
    feature_flags: dict[str, bool] = Field(
        default_factory=lambda: {
            "new_ui": False,
            "advanced_analytics": False,
            "beta_features": False,
        },
    )

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="forbid",
    )

    def validate_all_components(self) -> FlextResult[None]:
        """Validate all configuration components."""
        components = [
            ("database", self.database),
            ("cache", self.cache),
            ("security", self.security),
        ]

        for name, component in components:
            if hasattr(component, "validate_business_rules"):
                validation = component.validate_business_rules()
                if not validation.success:
                    return FlextResult[None].fail(f"{name}: {validation.error}")

        # Global validations
        if self.environment == "production":
            if self.debug:
                return FlextResult[None].fail(
                    "Debug mode must be disabled in production",
                )
            if not self.security.secret_key:
                return FlextResult[None].fail("Secret key required in production")

        return FlextResult[None].ok(None)

    def get_summary(self) -> dict[str, object]:
        """Get configuration summary safe for logging."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "debug": self.debug,
            "api_endpoint": f"{self.host}:{self.port}",
            "database_host": self.database.host,
            "cache_provider": self.cache.provider,
            "security_2fa": self.security.enable_2fa,
            "feature_flags": self.feature_flags,
        }


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================


def create_enterprise_config(config_data: dict[str, object]) -> EnterpriseConfig:
    """Create EnterpriseConfig instance from dictionary data.

    Helper function to properly handle model validation with type safety.
    """
    # Construct directly for proper typing support with mypy and pydantic
    # Pydantic will validate nested models on initialization
    return EnterpriseConfig(**config_data)


def load_config_from_file(file_path: str | Path) -> FlextResult[dict[str, object]]:
    """Load configuration from JSON file."""
    try:
        path = Path(file_path)
        if not path.exists():
            return FlextResult[dict[str, object]].fail(
                f"Configuration file not found: {path}",
            )

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return FlextResult[dict[str, object]].fail(
                "Configuration must be a JSON object",
            )

        return FlextResult[dict[str, object]].ok(dict(data))

    except json.JSONDecodeError as e:
        return FlextResult[dict[str, object]].fail(f"Invalid JSON: {e}")
    except Exception as e:
        return FlextResult[dict[str, object]].fail(f"Error loading file: {e}")


def save_config_to_file(
    config_data: dict[str, object],
    file_path: str | Path,
) -> FlextResult[None]:
    """Save configuration to JSON file."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, default=str)

        return FlextResult[None].ok(None)

    except Exception as e:
        return FlextResult[None].fail(f"Error saving file: {e}")


def merge_configurations(
    base: dict[str, object],
    override: dict[str, object],
) -> dict[str, object]:
    """Deep merge two configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Cast to dict for type safety
            result_dict = cast("dict[str, object]", result[key])
            value_dict = cast("dict[str, object]", value)
            result[key] = merge_configurations(
                result_dict,
                value_dict,
            )
        else:
            result[key] = value

    return result


def mask_sensitive_data(config_data: dict[str, object]) -> dict[str, object]:
    """Mask sensitive values for safe logging."""
    sensitive_keys = {"password", "secret", "key", "token", "auth"}

    def mask_recursive(obj: object) -> object:
        if isinstance(obj, dict):
            result: dict[str, object] = {}
            for key, value in obj.items():
                key_str = str(key)
                if any(sensitive in key_str.lower() for sensitive in sensitive_keys):
                    result[key_str] = "***"
                else:
                    result[key_str] = mask_recursive(value)
            return result
        if isinstance(obj, list):
            return [mask_recursive(item) for item in obj]
        return obj

    # Cast result to expected return type
    masked_result = mask_recursive(config_data)
    return cast("dict[str, object]", masked_result)


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================


def demonstrate_basic_configuration() -> FlextResult[None]:
    """Demonstrate basic configuration creation and validation."""
    print("\n" + "=" * 60)
    print("🔧 Basic Configuration Demonstration")
    print("=" * 60)

    try:
        # Create default configuration
        config = EnterpriseConfig()

        print(f"Application: {config.app_name} v{config.version}")
        print(f"Environment: {config.environment}")
        print(f"Debug Mode: {config.debug}")
        print(f"API Server: {config.host}:{config.port}")
        print(f"Database: {config.database.get_masked_url()}")
        print(f"Cache: {config.cache.get_connection_url()}")

        # Validate configuration
        validation = config.validate_all_components()
        if validation.success:
            print("✅ Configuration is valid")
        else:
            print(f"❌ Configuration validation failed: {validation.error}")
            return validation

        return FlextResult[None].ok(None)

    except Exception as e:
        return FlextResult[None].fail(f"Basic configuration failed: {e}")


def demonstrate_environment_configuration() -> FlextResult[None]:
    """Demonstrate environment variable integration."""
    print("\n" + "=" * 60)
    print("🌍 Environment Configuration Demonstration")
    print("=" * 60)

    # Set test environment variables
    test_env = {
        "APP_APP_NAME": "Production Enterprise App",
        "APP_ENVIRONMENT": "production",
        "APP_DEBUG": "false",
        "APP_PORT": "8080",
        "APP_DATABASE__HOST": "prod-db.company.com",
        "APP_DATABASE__PASSWORD": "prod_secure_password_123",
        "APP_CACHE__PROVIDER": "redis",
        "APP_CACHE__HOST": "redis-cluster.company.com",
        "APP_SECURITY__SECRET_KEY": "ProductionSecretKey123WithComplexity456789",
        "APP_SECURITY__ENABLE_2FA": "true",
    }

    # Apply test environment
    original_env: dict[str, str | None] = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Create configuration from environment
        config = EnterpriseConfig()

        print(f"Application: {config.app_name}")
        print(f"Environment: {config.environment}")
        print(f"API Port: {config.port}")
        print(f"Database Host: {config.database.host}")
        print(f"Cache Provider: {config.cache.provider}")
        print(f"2FA Enabled: {config.security.enable_2fa}")

        # Validate configuration
        validation = config.validate_all_components()
        if validation.success:
            print("✅ Environment configuration is valid")

            # Show summary
            summary = config.get_summary()
            print("\n📋 Configuration Summary:")
            if isinstance(summary, dict):
                # Handle dict summary
                for summary_key, summary_value in summary.items():
                    print(f"  {summary_key}: {summary_value}")
        else:
            print(f"❌ Environment configuration invalid: {validation.error}")
            return validation

        return FlextResult[None].ok(None)

    except Exception as e:
        return FlextResult[None].fail(f"Environment configuration failed: {e}")

    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(original_value)


def demonstrate_file_configuration() -> FlextResult[None]:
    """Demonstrate file-based configuration loading."""
    print("\n" + "=" * 60)
    print("📁 File Configuration Demonstration")
    print("=" * 60)

    try:
        # Create test configuration data
        config_data: dict[str, object] = {
            "app_name": "File-Based Enterprise App",
            "environment": "staging",
            "debug": True,
            "port": 8080,
            "database": {
                "host": "staging-db.company.com",
                "port": 5433,
                "name": "staging_db",
                "user": "staging_user",
                "password": "staging_password_123",
                "pool_size": 30,
            },
            "security": {
                "secret_key": "StagingSecretKey123WithNumbers456AndMore789",
                "token_expiry_hours": 8,
                "enable_2fa": False,
            },
            "feature_flags": {
                "new_ui": True,
                "beta_features": True,
            },
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(config_data, f, indent=2)
            config_file = f.name

        try:
            # Load configuration from file
            loaded_result = load_config_from_file(config_file)
            if not loaded_result.success:
                return FlextResult[None].fail(
                    loaded_result.error or "Failed to load configuration",
                )

            loaded_data = loaded_result.value
            print(f"✅ Configuration loaded from: {config_file}")
            print(f"Loaded keys: {list(loaded_data.keys())}")

            # Create configuration with loaded data using helper function
            config = create_enterprise_config(loaded_data)

            validation = config.validate_all_components()
            if validation.success:
                print("✅ File configuration is valid")

                # Show masked configuration for safe logging
                masked = mask_sensitive_data(config_data)
                print("\n🔒 Masked Configuration (safe for logging):")
                print(json.dumps(masked, indent=2))
            else:
                print(f"❌ File configuration invalid: {validation.error}")
                return FlextResult[None].fail(
                    validation.error or "Configuration validation failed",
                )

            return FlextResult[None].ok(None)

        finally:
            # Clean up temp file
            Path(config_file).unlink(missing_ok=True)

    except Exception as e:
        return FlextResult[None].fail(f"File configuration failed: {e}")


def demonstrate_configuration_merging() -> FlextResult[None]:
    """Demonstrate configuration merging and hierarchies."""
    print("\n" + "=" * 60)
    print("🔀 Configuration Merging Demonstration")
    print("=" * 60)

    try:
        # Base configuration (defaults)
        base_config: dict[str, object] = {
            "app_name": "Base Enterprise App",
            "debug": True,
            "workers": 2,
            "database": {
                "host": "localhost",
                "port": 5432,
                "pool_size": 10,
            },
            "cache": {
                "provider": "memory",
            },
            "feature_flags": {
                "new_ui": False,
                "beta_features": False,
            },
        }

        # Environment-specific overrides
        env_overrides: dict[str, object] = {
            "environment": "production",
            "debug": False,
            "workers": 8,
            "database": {
                "host": "prod-db.company.com",
                "pool_size": 50,
                "ssl_enabled": True,
            },
            "cache": {
                "provider": "redis",
                "host": "redis-prod.company.com",
            },
        }

        # Local/deployment specific overrides
        local_overrides: dict[str, object] = {
            "app_name": "Customized Production App",
            "port": 9000,
            "security": {
                "secret_key": "LocalCustomSecretKey123WithComplexity456",
            },
            "feature_flags": {
                "new_ui": True,
            },
        }

        # Merge configurations step by step
        print("📋 Merging configurations...")
        print("  1. Base + Environment overrides")
        step1 = merge_configurations(base_config, env_overrides)

        print("  2. Previous + Local overrides")
        final_config = merge_configurations(step1, local_overrides)

        print("\n🔧 Final merged configuration structure:")
        print(f"Keys: {list(final_config.keys())}")

        # Create and validate final configuration
        config = create_enterprise_config(final_config)

        validation = config.validate_all_components()
        if validation.success:
            print("✅ Merged configuration is valid")

            # Show final summary
            summary = config.get_summary()
            print("\n📊 Final Configuration Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        else:
            print(f"❌ Merged configuration invalid: {validation.error}")
            return FlextResult[None].fail(
                validation.error or "Configuration validation failed",
            )

        return FlextResult[None].ok(None)

    except Exception as e:
        return FlextResult[None].fail(f"Configuration merging failed: {e}")


def demonstrate_validation_scenarios() -> FlextResult[None]:
    """Demonstrate various configuration validation scenarios."""
    print("\n" + "=" * 60)
    print("🔍 Configuration Validation Scenarios")
    print("=" * 60)

    scenarios: list[tuple[str, dict[str, object]]] = [
        (
            "Invalid database port",
            {
                "database": DatabaseConfig(
                    host="localhost",
                    port=22,
                ),  # SSH port not allowed
            },
        ),
        (
            "2FA with too many login attempts",
            {
                "security": SecurityConfig(
                    secret_key=_DEMO_SECRET_KEY_1,
                    enable_2fa=True,
                    max_login_attempts=15,  # Too high for 2FA
                ),
            },
        ),
    ]

    for scenario_name, config_data in scenarios:
        print(f"\nTesting: {scenario_name}")
        try:
            config = create_enterprise_config(config_data)
            validation = config.validate_all_components()

            if not validation.success:
                print(f"  ✅ Correctly rejected: {validation.error}")
            else:
                print("  ❌ Should have been rejected")

        except Exception as e:
            print(f"  ✅ Correctly rejected during creation: {e}")

    # Test valid configuration
    print("\nTesting: Valid configuration")
    try:
        valid_config_data: dict[str, object] = {
            "app_name": "Valid Test App",
            "environment": "development",
            "security": SecurityConfig(secret_key=_DEMO_SECRET_KEY_2),
        }
        valid_config = create_enterprise_config(valid_config_data)
        validation = valid_config.validate_all_components()

        if validation.success:
            print("  ✅ Valid configuration accepted")
        else:
            print(f"  ❌ Valid configuration rejected: {validation.error}")
            return FlextResult[None].fail(
                validation.error or "Configuration validation failed",
            )

    except Exception as e:
        print(f"  ❌ Valid configuration failed: {e}")
        return FlextResult[None].fail(f"Valid configuration failed: {e}")

    return FlextResult[None].ok(None)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> int:
    """Main demonstration function."""
    print("🎯 Enterprise Configuration Management Demo")
    print("Comprehensive configuration patterns with FLEXT")

    demonstrations = [
        ("Basic Configuration", demonstrate_basic_configuration),
        ("Environment Configuration", demonstrate_environment_configuration),
        ("File Configuration", demonstrate_file_configuration),
        ("Configuration Merging", demonstrate_configuration_merging),
        ("Validation Scenarios", demonstrate_validation_scenarios),
    ]

    for demo_name, demo_func in demonstrations:
        try:
            print(f"\n🎮 Running: {demo_name}")
            result = demo_func()

            if result.success:
                print(f"✅ {demo_name} completed successfully")
            else:
                print(f"❌ {demo_name} failed: {result.error}")
                return 1

        except Exception as e:
            print(f"❌ {demo_name} crashed: {e}")
            return 1

    print("\n🎉 All configuration demonstrations completed successfully!")
    print("\n📈 Enterprise Configuration Features Demonstrated:")
    print("   ✅ Environment variable integration with nested settings")
    print("   ✅ File-based configuration loading and validation")
    print("   ✅ Comprehensive business rule validation")
    print("   ✅ Security-aware configuration (password masking)")
    print("   ✅ Multi-layer configuration composition and merging")
    print("   ✅ Type-safe configuration with Pydantic validation")
    print("   ✅ Production-ready validation scenarios")
    print("   ✅ Enterprise patterns and best practices")

    return 0


if __name__ == "__main__":
    sys.exit(main())
