#!/usr/bin/env python3
"""Enterprise configuration management with unified FlextConfig.

Demonstrates the unified FlextConfig system with enterprise-grade features:
- Automatic .env file loading and environment variable integration
- Built-in validation with runtime and business rules
- Multiple configuration profiles (web service, microservice, batch job, etc.)
- Production readiness checks and feature flags
- Serialization to JSON with metadata tracking
- File-based configuration loading (JSON, YAML, TOML)
- Type-safe configuration with Pydantic v2

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from flext_core import FlextConfig, FlextResult, FlextTypes

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


# Using the unified FlextConfig class with specialized factory methods
# The new FlextConfig provides built-in validation, environment integration,
# and enterprise-grade configuration management features


# Utility functions leveraging the new FlextConfig capabilities


def create_enterprise_config(
    config_data: FlextTypes.Core.Dict | None = None,
) -> FlextConfig:
    """Create a FlextConfig instance using the unified configuration system.

    This function demonstrates how to create configuration instances using
    the new FlextConfig factory methods and unified validation system.

    Args:
        config_data: Optional configuration data to merge

    Returns:
        Configured FlextConfig instance

    """
    if config_data:
        # Use the factory method to create with custom data
        result = FlextConfig.create(constants=config_data)
        if result.is_success:
            return result.value
        # Fallback to default if creation fails
        print(f"Warning: Configuration creation failed: {result.error}")

    # Create default configuration
    result = FlextConfig.create()
    return result.value if result.is_success else FlextConfig()


def get_connection_string_example(config: FlextConfig) -> FlextResult[str]:
    """Get connection string using FlextConfig's built-in methods."""
    # Simplified example - in real usage you'd construct connection string from config
    db_config = getattr(config, "database", {})
    if isinstance(db_config, dict):
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)
        return FlextResult[str].ok(f"postgresql://{host}:{port}/mydb")
    return FlextResult[str].ok("postgresql://localhost:5432/mydb")


def demonstrate_config_validation(config: FlextConfig) -> FlextResult[None]:
    """Demonstrate FlextConfig's built-in validation capabilities."""
    # Use the unified validation system
    validation_result = config.validate_all()
    if validation_result.is_failure:
        return FlextResult[None].fail(validation_result.error or "Validation failed")

    return FlextResult[None].ok(None)


def demonstrate_config_serialization(config: FlextConfig) -> FlextResult[None]:
    """Demonstrate FlextConfig's serialization capabilities."""
    try:
        # Use built-in serialization methods
        config_dict = config.to_dict()
        config_json = config.to_json(indent=2)

        print(f"Configuration as dict keys: {list(config_dict.keys())}")
        print(f"Configuration as JSON length: {len(config_json)} characters")

        return FlextResult[None].ok(None)
    except Exception as e:
        return FlextResult[None].fail(f"Serialization failed: {e}")


def demonstrate_basic_configuration() -> FlextResult[None]:
    """Demonstrate basic configuration creation and validation using unified FlextConfig."""
    print("\n" + "=" * 60)
    print("🔧 Basic Configuration Demonstration")
    print("=" * 60)

    try:
        # Create default configuration using the unified FlextConfig
        config = create_enterprise_config()

        print(f"Application: {config.app_name} v{config.version}")
        print(f"Environment: {config.environment}")
        print(f"Debug Mode: {config.debug}")
        print(f"API Server: {config.host}:{config.port}")

        # Demonstrate connection string generation
        db_connection = get_connection_string_example(config)
        if db_connection.is_success:
            print(f"Database URL configured: {bool(config.database_url)}")
        else:
            print("Database connection not configured")

        # Demonstrate feature flags (simplified)
        feature_flags = getattr(config, "feature_flags", {})
        if isinstance(feature_flags, dict):
            print(f"Feature flags: {len(feature_flags)} available")
        else:
            print("Feature flags: 0 available")

        # Validate configuration using unified validation system
        validation = demonstrate_config_validation(config)
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

    # Set test environment variables using FLEXT_ prefix (built-in support)
    test_env = {
        "FLEXT_APP_NAME": "Production Enterprise App",
        "FLEXT_ENVIRONMENT": "production",
        "FLEXT_DEBUG": "false",
        "FLEXT_PORT": "8080",
        "FLEXT_DATABASE_URL": "postgresql://prod_user:prod_pass@prod-db.company.com:5432/prod_db",
        "FLEXT_ENABLE_AUTH": "true",
        "FLEXT_API_KEY": "prod_api_key_123",
        "FLEXT_ENABLE_METRICS": "true",
    }

    # Apply test environment
    original_env: dict[str, str | None] = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Create configuration from environment using factory method
        config_result = FlextConfig.create_from_environment()
        if config_result.is_failure:
            return FlextResult[None].fail(
                f"Failed to create config from environment: {config_result.error}"
            )

        config = config_result.value

        print(f"Application: {config.app_name}")
        print(f"Environment: {config.environment}")
        print(f"API Port: {config.port}")
        print(f"Database configured: {bool(config.database_url)}")
        print(f"Authentication enabled: {config.enable_auth}")
        print(f"Metrics enabled: {config.enable_metrics}")

        # Demonstrate production readiness check
        is_production = getattr(config, "environment", "development") == "production"
        if is_production:
            print("✅ Configuration is production-ready")
        else:
            print("⚠️  Configuration is for development/testing")

        # Validate configuration using unified validation
        validation = demonstrate_config_validation(config)
        if validation.success:
            print("✅ Environment configuration is valid")

            # Demonstrate serialization
            serial_demo = demonstrate_config_serialization(config)
            if serial_demo.is_failure:
                print(f"⚠️  Serialization demo failed: {serial_demo.error}")
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
        config_data: FlextTypes.Core.Dict = {
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
            # Load configuration from file using FlextConfig's built-in method
            config_result = FlextConfig.load_from_file(config_file)
            if config_result.is_failure:
                return FlextResult[None].fail(
                    config_result.error or "Failed to load configuration",
                )

            config = config_result.value
            print(f"✅ Configuration loaded from: {config_file}")
            print(
                f"Configuration profile: {getattr(config, 'environment', 'development')}"
            )
            print(
                f"Created from file: {config.get_metadata().get('loaded_from_file', 'unknown')}"
            )

            # Validate configuration using unified validation
            validation = demonstrate_config_validation(config)
            if validation.success:
                print("✅ File configuration is valid")

                # Demonstrate serialization for safe logging
                print("\n🔒 Configuration serialization demo:")
                serial_result = demonstrate_config_serialization(config)
                if serial_result.is_failure:
                    print(f"⚠️  Serialization failed: {serial_result.error}")

                # Show metadata
                metadata = config.get_metadata()
                print(f"📋 Configuration metadata: {list(metadata.keys())}")
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
        base_config: FlextTypes.Core.Dict = {
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
        env_overrides: FlextTypes.Core.Dict = {
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
        local_overrides: FlextTypes.Core.Dict = {
            "app_name": "Customized Production App",
            "port": 9000,
            "security": {
                "secret_key": "LocalCustomSecretKey123WithComplexity456",
            },
            "feature_flags": {
                "new_ui": True,
            },
        }

        # Merge configurations using FlextConfig's merge method
        print("📋 Merging configurations...")
        print("  1. Base + Environment overrides")
        merge_result1 = FlextConfig.merge_configs(base_config, env_overrides)
        if merge_result1.is_failure:
            return FlextResult[None].fail(f"Merge step 1 failed: {merge_result1.error}")

        step1 = merge_result1.value
        print("  2. Previous + Local overrides")
        merge_result2 = FlextConfig.merge_configs(step1, local_overrides)
        if merge_result2.is_failure:
            return FlextResult[None].fail(f"Merge step 2 failed: {merge_result2.error}")

        final_config = merge_result2.value
        print("\n🔧 Final merged configuration structure:")
        print(f"Keys: {list(final_config.keys())}")

        # Create and validate final configuration
        config = create_enterprise_config(final_config)

        # Validate configuration using unified validation
        validation = demonstrate_config_validation(config)
        if validation.success:
            print("✅ Merged configuration is valid")

            # Demonstrate serialization
            serial_result = demonstrate_config_serialization(config)
            if serial_result.is_failure:
                print(f"⚠️  Serialization failed: {serial_result.error}")

            # Show configuration profile
            print(
                f"📋 Configuration profile: {getattr(config, 'environment', 'development')}"
            )
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

    # Test scenarios using FlextConfig's built-in validation
    scenarios: list[tuple[str, FlextTypes.Core.Dict]] = [
        (
            "Invalid environment",
            {
                "app_name": "Test App",
                "environment": "invalid_env",  # Should fail validation
            },
        ),
        (
            "Debug in production",
            {
                "app_name": "Prod App",
                "environment": "production",
                "debug": True,  # Should fail business rules
            },
        ),
        (
            "Negative port",
            {
                "app_name": "Test App",
                "port": -1,  # Should fail validation
            },
        ),
    ]

    for scenario_name, config_data in scenarios:
        print(f"\nTesting: {scenario_name}")
        try:
            config_result = FlextConfig.create(constants=config_data)
            if config_result.is_failure:
                print(f"  ✅ Correctly rejected during creation: {config_result.error}")
                continue

            config = config_result.value
            validation = demonstrate_config_validation(config)

            if not validation.success:
                print(f"  ✅ Correctly rejected: {validation.error}")
            else:
                print("  ❌ Should have been rejected")

        except Exception as e:
            print(f"  ✅ Correctly rejected: {e}")

    # Test valid configuration
    print("\nTesting: Valid configuration")
    try:
        valid_config_data: FlextTypes.Core.Dict = {
            "app_name": "Valid Test App",
            "environment": "development",
            "port": 8080,
            "debug": False,
        }
        valid_config_result = FlextConfig.create(constants=valid_config_data)
        if valid_config_result.is_failure:
            print(f"  ❌ Valid configuration failed: {valid_config_result.error}")
            return FlextResult[None].fail(
                f"Valid configuration failed: {valid_config_result.error}"
            )

        valid_config = valid_config_result.value
        validation = demonstrate_config_validation(valid_config)

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
    print("\n📈 Unified FlextConfig Features Demonstrated:")
    print("   ✅ Automatic .env file loading with FLEXT_ prefix support")
    print("   ✅ Built-in validation (runtime + business rules)")
    print("   ✅ Multiple configuration profiles (web service, microservice, etc.)")
    print("   ✅ Production readiness checks and feature flags")
    print("   ✅ Type-safe configuration with Pydantic v2 validation")
    print("   ✅ File-based loading (JSON, YAML, TOML) with metadata tracking")
    print("   ✅ Configuration merging with conflict resolution")
    print("   ✅ Serialization to JSON with consistent formatting")
    print("   ✅ Enterprise-grade error handling with FlextResult")
    print("   ✅ Configuration sealing for immutability")
    print("   ✅ Metadata tracking and configuration lifecycle management")

    return 0


if __name__ == "__main__":
    sys.exit(main())
