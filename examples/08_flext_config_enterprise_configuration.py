#!/usr/bin/env python3
"""FLEXT Config Enterprise Configuration Example.

Comprehensive demonstration of FlextConfig system showing enterprise-grade
configuration management with environment integration, validation, and merging.

Features demonstrated:
    - Environment variable configuration loading
    - Configuration validation with type safety
    - Configuration merging and override patterns
    - Default value management and fallbacks
    - File-based configuration loading
    - Configuration validation and error handling
    - Enterprise configuration patterns
    - Configuration hierarchies and inheritance

Key Components:
    - FlextConfig: Main configuration management class
    - FlextBaseSettings: Pydantic-based settings with environment loading
    - FlextConfigDefaults: Default configuration management
    - FlextConfigOps: Configuration operations and transformations
    - Configuration validation with comprehensive error reporting

This example shows real-world enterprise configuration scenarios
demonstrating the power and flexibility of the FlextConfig system.
"""

import json
import os
import tempfile
import traceback
from pathlib import Path

from flext_core.config import (
    FlextBaseSettings,
    FlextConfig,
    FlextConfigDefaults,
    FlextConfigOps,
    FlextConfigValidation,
    merge_configs,
)


def demonstrate_basic_configuration() -> None:
    """Demonstrate basic configuration patterns with FlextConfig."""
    print("\n" + "=" * 80)
    print("🔧 BASIC CONFIGURATION PATTERNS")
    print("=" * 80)

    # 1. Basic configuration creation
    print("\n1. Creating basic configuration:")
    config_data = {
        "app_name": "MyApp",
        "debug": True,
        "max_connections": 100,
        "timeout": 30.0,
    }

    config_result = FlextConfig.create_complete_config(config_data)
    if config_result.is_success:
        config = config_result.data
        print(f"✅ Config created: {config}")
        print(f"   App name: {config.get('app_name')}")
        print(f"   Debug mode: {config.get('debug')}")
        print(f"   Max connections: {config.get('max_connections')}")
    else:
        print(f"❌ Config creation failed: {config_result.error}")

    # 2. Configuration validation
    print("\n2. Configuration validation:")
    validation_result = FlextConfigValidation.validate_config_type(
        config_data.get("app_name"),
        str,
        "app_name",
    )
    if validation_result.is_success:
        print("✅ Configuration type validation is valid")
    else:
        print(f"❌ Configuration validation failed: {validation_result.error}")

    # 3. Configuration defaults
    print("\n3. Configuration defaults:")
    defaults = {"debug": False, "timeout": 30, "port": 8000, "max_retries": 3}
    print(f"📋 Default configuration: {defaults}")

    # 4. Applying defaults
    defaults_result = FlextConfigDefaults.apply_defaults(config_data, defaults)
    if defaults_result.is_success:
        config_with_defaults = defaults_result.data
        print("🔄 Config with defaults applied:")
        for key, value in config_with_defaults.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ Applying defaults failed: {defaults_result.error}")


def demonstrate_environment_integration() -> None:
    """Demonstrate environment variable integration."""
    print("\n" + "=" * 80)
    print("🌍 ENVIRONMENT VARIABLE INTEGRATION")
    print("=" * 80)

    # 1. Set environment variables for testing
    test_env_vars = {
        "MYAPP_DEBUG": "false",
        "MYAPP_MAX_CONNECTIONS": "200",
        "MYAPP_DATABASE_URL": "postgresql://localhost:5432/myapp",
        "MYAPP_API_KEY": "super-secret-api-key",
        "MYAPP_TIMEOUT": "45.5",
    }

    print("1. Setting test environment variables:")
    for key, value in test_env_vars.items():
        os.environ[key] = value
        print(f"   {key} = {value}")

    # 2. Create settings class with environment loading
    class MyAppSettings(FlextBaseSettings):
        """Application settings with environment loading."""

        debug: bool = True
        max_connections: int = 100
        database_url: str = "sqlite:///app.db"
        api_key: str = "default-key"
        timeout: float = 30.0

        class Config:
            """Pydantic configuration."""

            env_prefix = "MYAPP_"
            case_sensitive = False
            env_file = ".env"
            extra = "ignore"  # Ignore extra environment variables

    # 3. Load configuration from environment
    print("\n2. Loading configuration from environment:")
    settings = None
    try:
        settings = MyAppSettings()
        print("✅ Settings loaded successfully:")
        print(f"   Debug: {settings.debug}")
        print(f"   Max connections: {settings.max_connections}")
        print(f"   Database URL: {settings.database_url}")
        print(f"   API key: {settings.api_key[:10]}...")  # Masked for security
        print(f"   Timeout: {settings.timeout}")
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")

    # 4. Configuration operations with environment data
    print("\n3. Configuration operations:")
    if settings:
        env_config = {
            "debug": settings.debug,
            "max_connections": settings.max_connections,
            "database_url": settings.database_url,
            "timeout": settings.timeout,
        }
    else:
        # Use fallback config if settings failed
        env_config = {
            "debug": False,
            "max_connections": 200,
            "database_url": "postgresql://localhost:5432/myapp",
            "timeout": 45.5,
        }
        print("Using fallback configuration due to settings load failure")

    # Filter configuration keys
    filtered_result = FlextConfigDefaults.filter_config_keys(
        env_config,
        ["debug", "max_connections", "timeout"],
    )
    if filtered_result.is_success:
        print(f"🔄 Filtered config: {filtered_result.data}")
    else:
        print(f"❌ Filter failed: {filtered_result.error}")

    # Clean up environment variables
    for key in test_env_vars:
        os.environ.pop(key, None)


def demonstrate_configuration_merging() -> None:
    """Demonstrate configuration merging and override patterns."""
    print("\n" + "=" * 80)
    print("🔀 CONFIGURATION MERGING AND OVERRIDES")
    print("=" * 80)

    # 1. Base configuration (defaults)
    base_config = {
        "app_name": "BaseApp",
        "debug": False,
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "base_db",
            "pool_size": 10,
        },
        "logging": {"level": "INFO", "format": "json"},
        "features": {"feature_a": True, "feature_b": False},
    }

    # 2. Development configuration (overrides)
    dev_config = {
        "debug": True,
        "database": {"name": "dev_db", "pool_size": 5},
        "logging": {"level": "DEBUG"},
        "features": {"feature_b": True, "feature_c": True},
    }

    # 3. Production configuration (overrides)
    prod_config = {
        "app_name": "ProductionApp",
        "database": {
            "host": "prod-db.example.com",
            "port": 5433,
            "name": "prod_db",
            "pool_size": 50,
        },
        "logging": {"level": "WARNING", "format": "structured"},
        "features": {"feature_a": True, "feature_b": True, "feature_c": False},
    }

    print("1. Base configuration:")
    print(f"📋 {base_config}")

    print("\n2. Development overrides:")
    print(f"🔧 {dev_config}")

    print("\n3. Production overrides:")
    print(f"🏭 {prod_config}")

    # 4. Merge configurations
    print("\n4. Configuration merging:")

    # Development configuration (base + dev)
    dev_merged = merge_configs(base_config, dev_config)
    print(f"🔧 Development config: {dev_merged}")

    # Production configuration (base + prod)
    prod_merged = merge_configs(base_config, prod_config)
    print(f"🏭 Production config: {prod_merged}")

    # 5. Deep merge validation
    print("\n5. Deep merge validation:")
    print(f"   Dev database config: {dev_merged.get('database')}")
    print(f"   Prod database config: {prod_merged.get('database')}")
    print(f"   Dev features: {dev_merged.get('features')}")
    print(f"   Prod features: {prod_merged.get('features')}")


def demonstrate_file_configuration() -> None:
    """Demonstrate file-based configuration loading."""
    print("\n" + "=" * 80)
    print("📁 FILE-BASED CONFIGURATION LOADING")
    print("=" * 80)

    # 1. Create temporary configuration files
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)

        # Create JSON configuration file
        json_config = {
            "service_name": "FileConfigService",
            "version": "1.0.0",
            "api": {"port": 8080, "host": "0.0.0.0", "cors": True},
            "security": {"jwt_secret": "secure-jwt-secret", "token_expiry": 3600},
            "monitoring": {"metrics_enabled": True, "health_check_interval": 30},
        }

        json_file = config_dir / "config.json"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_config, f, indent=2)

        print(f"1. Created configuration file: {json_file}")
        print(f"📄 Content: {json_config}")

        # 2. Load configuration from file
        print("\n2. Loading configuration from file:")
        try:
            loaded_result = FlextConfigOps.safe_load_json_file(str(json_file))
            if loaded_result.is_success:
                loaded_config = loaded_result.data
                print("✅ Configuration loaded successfully:")
                print(f"   Service: {loaded_config.get('service_name')}")
                print(f"   Version: {loaded_config.get('version')}")
                print(f"   API config: {loaded_config.get('api')}")
                print(f"   Security config: {loaded_config.get('security')}")
                print(f"   Monitoring: {loaded_config.get('monitoring')}")
            else:
                print(f"❌ Failed to load config: {loaded_result.error}")
        except Exception as e:
            print(f"❌ Exception loading config: {e}")

        # 3. Configuration validation
        print("\n3. File configuration validation:")
        if "loaded_config" in locals() and loaded_result.is_success:
            # Validate service name
            service_validation = FlextConfigValidation.validate_config_type(
                loaded_config.get("service_name"),
                str,
                "service_name",
            )
            if service_validation.is_success:
                print("✅ Service name validation passed")
            else:
                print(f"❌ Service validation failed: {service_validation.error}")

            # Validate version
            version_validation = FlextConfigValidation.validate_config_type(
                loaded_config.get("version"),
                str,
                "version",
            )
            if version_validation.is_success:
                print("✅ Version validation passed")
            else:
                print(f"❌ Version validation failed: {version_validation.error}")


def demonstrate_configuration_hierarchies() -> None:
    """Demonstrate configuration hierarchies and inheritance."""
    print("\n" + "=" * 80)
    print("🏗️ CONFIGURATION HIERARCHIES AND INHERITANCE")
    print("=" * 80)

    # 1. Global configuration
    global_config = {
        "organization": "ACME Corp",
        "timezone": "UTC",
        "logging": {"level": "INFO", "output": "console"},
        "security": {"encryption": "AES256", "audit": True},
    }

    # 2. Application configuration
    app_config = {
        "app_name": "UserService",
        "version": "2.1.0",
        "database": {"type": "postgresql", "timeout": 30},
        "cache": {"type": "redis", "ttl": 300},
    }

    # 3. Environment-specific configuration
    env_configs = {
        "development": {
            "logging": {"level": "DEBUG"},
            "database": {"host": "localhost", "debug": True},
            "cache": {"host": "localhost", "debug": True},
        },
        "staging": {
            "logging": {"level": "INFO"},
            "database": {"host": "staging-db.internal", "pool_size": 10},
            "cache": {"host": "staging-cache.internal"},
        },
        "production": {
            "logging": {"level": "WARNING", "output": "file"},
            "database": {"host": "prod-db.internal", "pool_size": 50},
            "cache": {"host": "prod-cache.internal", "cluster": True},
            "security": {"audit": True, "monitoring": True},
        },
    }

    print("1. Configuration hierarchy:")
    print(f"🌍 Global: {global_config}")
    print(f"📱 Application: {app_config}")
    print(f"🔧 Environment configs available: {list(env_configs.keys())}")

    # 4. Build complete configurations for each environment
    print("\n2. Building complete configurations:")
    for env_name, env_config in env_configs.items():
        print(f"\n🏷️ Environment: {env_name.upper()}")

        # Merge: Global -> App -> Environment
        step1 = merge_configs(global_config, app_config)
        complete_config = merge_configs(step1, env_config)

        print(f"   Organization: {complete_config.get('organization')}")
        print(
            f"   App: {complete_config.get('app_name')} v{complete_config.get('version')}",
        )
        print(f"   Logging: {complete_config.get('logging')}")
        print(f"   Database: {complete_config.get('database')}")
        print(f"   Cache: {complete_config.get('cache')}")
        print(f"   Security: {complete_config.get('security')}")

    # 5. Configuration inheritance validation
    print("\n3. Configuration inheritance validation:")
    prod_config = merge_configs(
        merge_configs(global_config, app_config),
        env_configs["production"],
    )

    # Check if all global keys are present in production config
    inheritance_valid = all(key in prod_config for key in global_config)
    if inheritance_valid:
        print("✅ Configuration inheritance is valid")
    else:
        print("❌ Configuration inheritance failed: missing global keys")


def demonstrate_advanced_configuration_patterns() -> None:
    """Demonstrate advanced configuration patterns and operations."""
    print("\n" + "=" * 80)
    print("🚀 ADVANCED CONFIGURATION PATTERNS")
    print("=" * 80)

    # 1. Configuration templates
    print("1. Configuration templates:")
    template_config = {
        "service_name": "${SERVICE_NAME}",
        "database_url": "${DATABASE_URL}",
        "api_url": "https://${HOST}:${PORT}/api",
        "debug": "${DEBUG:false}",  # Default value
        "max_connections": "${MAX_CONNECTIONS:100}",
    }

    variables = {
        "SERVICE_NAME": "AdvancedService",
        "DATABASE_URL": "postgresql://localhost/advanced_db",
        "HOST": "api.example.com",
        "PORT": "443",
        "DEBUG": "true",
    }

    print(f"📋 Template: {template_config}")
    print(f"🔧 Variables: {variables}")

    # Simple template substitution (manual for demo)
    resolved_config = {}
    for key, value in template_config.items():
        if isinstance(value, str) and value.startswith("${"):
            var_name = value[2:-1].split(":")[0]  # Remove ${} and get variable name
            resolved_config[key] = variables.get(var_name, value)
        else:
            resolved_config[key] = value
    print(f"✅ Resolved config: {resolved_config}")

    # 2. Configuration validation with rules
    print("\n2. Advanced configuration validation:")
    validation_rules = {
        "service_name": {"required": True, "type": str, "min_length": 3},
        "database_url": {"required": True, "type": str, "pattern": r"postgresql://.*"},
        "api_url": {"required": True, "type": str, "pattern": r"https://.*"},
        "max_connections": {"required": False, "type": int, "min": 1, "max": 1000},
    }

    # Manual validation for demo
    validation_errors = []
    for key, rules in validation_rules.items():
        value = resolved_config.get(key)
        if rules.get("required") and value is None:
            validation_errors.append(f"{key} is required")
        if value and "type" in rules and not isinstance(value, rules["type"]):
            validation_errors.append(f"{key} must be {rules['type'].__name__}")

    if not validation_errors:
        print("✅ Advanced validation passed")
    else:
        print(f"❌ Advanced validation failed: {'; '.join(validation_errors)}")

    # 3. Configuration serialization and export
    print("\n3. Configuration serialization:")

    serialized = json.dumps(resolved_config, indent=2)
    print(f"📤 Serialized config: {serialized}")

    # 4. Configuration diff analysis
    print("\n4. Configuration diff analysis:")
    old_config = {"debug": False, "max_connections": 50, "timeout": 30}
    new_config = {"debug": True, "max_connections": 100, "timeout": 45, "cache": True}

    # Simple diff analysis for demo
    added = {k: v for k, v in new_config.items() if k not in old_config}
    changed = {
        k: (old_config[k], new_config[k])
        for k in old_config.keys() & new_config.keys()
        if old_config[k] != new_config[k]
    }
    removed = {k: old_config[k] for k in old_config.keys() - new_config.keys()}

    diff = {"added": added, "changed": changed, "removed": removed}
    print(f"🔄 Configuration diff: {diff}")


def main() -> None:
    """Execute all FlextConfig demonstrations."""
    print("🚀 FLEXT CONFIG - ENTERPRISE CONFIGURATION MANAGEMENT EXAMPLE")
    print("Demonstrating comprehensive configuration patterns and operations")

    try:
        demonstrate_basic_configuration()
        demonstrate_environment_integration()
        demonstrate_configuration_merging()
        demonstrate_file_configuration()
        demonstrate_configuration_hierarchies()
        demonstrate_advanced_configuration_patterns()

        print("\n" + "=" * 80)
        print("✅ ALL FLEXT CONFIG DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Summary of capabilities demonstrated:")
        print("   🔧 Basic configuration creation and management")
        print("   🌍 Environment variable integration")
        print("   🔀 Configuration merging and overrides")
        print("   📁 File-based configuration loading")
        print("   🏗️ Configuration hierarchies and inheritance")
        print("   🚀 Advanced patterns (templates, validation, diff)")
        print("\n💡 FlextConfig provides enterprise-grade configuration management")
        print("   with type safety, validation, and flexible integration patterns!")

    except Exception as e:
        print(f"\n❌ Error during FlextConfig demonstration: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
