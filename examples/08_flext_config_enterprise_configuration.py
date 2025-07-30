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
    - Maximum type safety using flext_core.types

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

# Import shared domain models to integrate domain validation patterns
from shared_domain import (
    SharedDomainFactory,
    log_domain_operation,
)

from flext_core import (
    FlextBaseSettings,
    FlextConfig,
    FlextConfigDefaults,
    FlextConfigValidation,
    FlextTypes,
    TConfigDict,
    TErrorMessage,
    TLogMessage,
    merge_configs,
)


def demonstrate_basic_configuration() -> None:
    """Demonstrate basic configuration patterns with FlextConfig.

    Using flext_core.types for type safety.
    """
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🔧 BASIC CONFIGURATION PATTERNS")
    print("=" * 80)

    # 1. Basic configuration creation
    log_message = "\n1. Creating basic configuration:"
    print(log_message)
    config_data: TConfigDict = {
        "app_name": "MyApp",
        "debug": True,
        "max_connections": 100,
        "timeout": 30.0,
    }

    config_result = FlextConfig.create_complete_config(config_data)
    if config_result.is_success:
        config = config_result.data
        if config is not None:
            log_message = f"✅ Config created: {config}"
            print(log_message)
            log_message = f"   App name: {config.get('app_name')}"
            print(log_message)
            log_message = f"   Debug mode: {config.get('debug')}"
            print(log_message)
            log_message = f"   Max connections: {config.get('max_connections')}"
            print(log_message)
    else:
        error_message: TErrorMessage = f"Config creation failed: {config_result.error}"
        print(f"❌ {error_message}")

    # 2. Configuration validation
    log_message = "\n2. Configuration validation:"
    print(log_message)
    validation_result = FlextConfigValidation.validate_config_type(
        config_data.get("app_name"),
        str,
        "app_name",
    )
    if validation_result.is_success:
        log_message = "✅ Configuration type validation is valid"
        print(log_message)
    else:
        error_message = f"Configuration validation failed: {validation_result.error}"
        print(f"❌ {error_message}")

    # 3. Configuration defaults
    log_message = "\n3. Configuration defaults:"
    print(log_message)
    defaults: TConfigDict = {
        "debug": False,
        "timeout": 30,
        "port": 8000,
        "max_retries": 3,
    }
    log_message = f"📋 Default configuration: {defaults}"
    print(log_message)

    # 4. Applying defaults
    defaults_result = FlextConfigDefaults.apply_defaults(config_data, defaults)
    if defaults_result.is_success:
        config_with_defaults = defaults_result.data
        if config_with_defaults is not None:
            log_message = "🔄 Config with defaults applied:"
            print(log_message)
            for key, value in config_with_defaults.items():
                log_message = f"   {key}: {value}"
                print(log_message)
    else:
        error_message = f"Applying defaults failed: {defaults_result.error}"
        print(f"❌ {error_message}")


def demonstrate_environment_integration() -> None:
    """Demonstrate environment variable integration using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🌍 ENVIRONMENT INTEGRATION")
    print("=" * 80)

    # 1. Environment-based settings
    log_message = "\n1. Environment-based settings:"
    print(log_message)

    class MyAppSettings(FlextBaseSettings):
        """Application settings with environment loading using flext_core.types."""

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
            extra = "ignore"  # Skip extra environment variables not defined in model

    # 2. Simulate environment variables
    log_message = "\n2. Simulating environment variables:"
    print(log_message)

    # Set environment variables for demonstration
    os.environ["MYAPP_DEBUG"] = "false"
    os.environ["MYAPP_MAX_CONNECTIONS"] = "200"
    os.environ["MYAPP_DATABASE_URL"] = "postgresql://localhost:5432/mydb"
    os.environ["MYAPP_API_KEY"] = "env-api-key-123"

    try:
        settings = MyAppSettings()
        log_message = "✅ Settings loaded from environment:"
        print(log_message)
        log_message = f"   Debug: {settings.debug}"
        print(log_message)
        log_message = f"   Max connections: {settings.max_connections}"
        print(log_message)
        log_message = f"   Database URL: {settings.database_url}"
        print(log_message)
        log_message = f"   API Key: {settings.api_key[:10]}..."
        print(log_message)
        log_message = f"   Timeout: {settings.timeout}"
        print(log_message)

    except (RuntimeError, ValueError, TypeError) as e:
        error_message: TErrorMessage = f"Failed to load settings: {e}"
        print(f"❌ {error_message}")

    # 3. Environment variable validation
    log_message = "\n3. Environment variable validation:"
    print(log_message)

    # Test invalid environment variable
    os.environ["MYAPP_MAX_CONNECTIONS"] = "invalid_number"

    try:
        settings = MyAppSettings()
        log_message = "✅ Settings loaded successfully"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Validation error (expected): {e}"
        print(f"❌ {error_message}")

    # Clean up environment variables
    for key in [
        "MYAPP_DEBUG",
        "MYAPP_MAX_CONNECTIONS",
        "MYAPP_DATABASE_URL",
        "MYAPP_API_KEY",
    ]:
        os.environ.pop(key, None)


def demonstrate_configuration_merging() -> None:
    """Demonstrate configuration merging patterns using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🔄 CONFIGURATION MERGING")
    print("=" * 80)

    # 1. Basic configuration merging
    log_message = "\n1. Basic configuration merging:"
    print(log_message)

    base_config: TConfigDict = {
        "app_name": "MyApp",
        "debug": False,
        "port": 8000,
        "database": {
            "url": "sqlite:///app.db",
            "pool_size": 10,
        },
    }

    override_config: TConfigDict = {
        "debug": True,
        "port": 9000,
        "database": {
            "url": "postgresql://localhost:5432/prod",
            "pool_size": 20,
        },
        "new_setting": "value",
    }

    log_message = f"📋 Base config: {base_config}"
    print(log_message)
    log_message = f"📋 Override config: {override_config}"
    print(log_message)

    # Merge configurations
    merged_config = merge_configs(base_config, override_config)
    log_message = "✅ Merged configuration:"
    print(log_message)
    for key, value in merged_config.items():
        log_message = f"   {key}: {value}"
        print(log_message)

    # 2. Deep merging demonstration
    log_message = "\n2. Deep merging demonstration:"
    print(log_message)

    deep_base: TConfigDict = {
        "services": {
            "auth": {
                "enabled": True,
                "timeout": 30,
            },
            "cache": {
                "enabled": False,
                "ttl": 3600,
            },
        },
    }

    deep_override: TConfigDict = {
        "services": {
            "auth": {
                "timeout": 60,
            },
            "cache": {
                "enabled": True,
            },
            "new_service": {
                "enabled": True,
            },
        },
    }

    deep_merged = merge_configs(deep_base, deep_override)
    log_message = "✅ Deep merged configuration:"
    print(log_message)
    log_message = f"   Services: {deep_merged.get('services')}"
    print(log_message)


def demonstrate_file_configuration() -> None:
    """Demonstrate file-based configuration using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("📁 FILE CONFIGURATION")
    print("=" * 80)

    # 1. Create temporary configuration file
    log_message = "\n1. Creating configuration file:"
    print(log_message)

    config_data: TConfigDict = {
        "app": {
            "name": "FileConfigApp",
            "version": "0.9.0",
            "debug": True,
        },
        "database": {
            "url": "postgresql://localhost:5432/fileconfig",
            "pool_size": 15,
            "timeout": 30,
        },
        "features": {
            "caching": True,
            "logging": True,
            "metrics": False,
        },
    }

    # Create temporary file
    with tempfile.NamedTemporaryFile(
        encoding="utf-8",
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(config_data, f, indent=2)
        temp_file_path = f.name

    log_message = f"📄 Created config file: {temp_file_path}"
    print(log_message)

    # 2. Load configuration from file
    log_message = "\n2. Loading configuration from file:"
    print(log_message)

    try:
        with open(temp_file_path, encoding="utf-8") as f:  # noqa: PTH123
            loaded_config: TConfigDict = json.load(f)

        log_message = "✅ Configuration loaded from file:"
        print(log_message)
        log_message = f"   App name: {loaded_config.get('app', {}).get('name')}"
        print(log_message)
        log_message = f"   Database URL: {loaded_config.get('database', {}).get('url')}"
        print(log_message)
        log_message = f"   Features: {loaded_config.get('features')}"
        print(log_message)

    except (RuntimeError, ValueError, TypeError) as e:
        error_message: TErrorMessage = f"Failed to load config file: {e}"
        print(f"❌ {error_message}")

    # 3. Configuration validation
    log_message = "\n3. Configuration validation:"
    print(log_message)

    # Validate app name
    app_name = loaded_config.get("app", {}).get("name")
    validation_result = FlextConfigValidation.validate_config_type(
        app_name,
        str,
        "app.name",
    )

    if validation_result.is_success:
        log_message = "✅ App name validation passed"
        print(log_message)
    else:
        error_message = f"App name validation failed: {validation_result.error}"
        print(f"❌ {error_message}")

    # Clean up
    try:
        os.unlink(temp_file_path)  # noqa: PTH108
        log_message = "🗑️ Temporary file cleaned up"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Failed to clean up file: {e}"
        print(f"⚠️ {error_message}")


def demonstrate_configuration_hierarchies() -> None:
    """Demonstrate configuration hierarchies using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🏗️ CONFIGURATION HIERARCHIES")
    print("=" * 80)

    # 1. Configuration hierarchy levels
    log_message = "\n1. Configuration hierarchy levels:"
    print(log_message)

    # Base configuration
    base_config: TConfigDict = {
        "app": {
            "name": "HierarchyApp",
            "version": "0.9.0",
        },
        "database": {
            "pool_size": 10,
            "timeout": 30,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
        },
    }

    # Environment-specific overrides
    dev_config: TConfigDict = {
        "app": {
            "debug": True,
        },
        "database": {
            "url": "sqlite:///dev.db",
        },
        "logging": {
            "level": "DEBUG",
        },
    }

    prod_config: TConfigDict = {
        "app": {
            "debug": False,
        },
        "database": {
            "url": "postgresql://prod-server:5432/prod",
            "pool_size": 50,
        },
        "logging": {
            "level": "WARNING",
        },
    }

    log_message = "📋 Configuration hierarchy:"
    print(log_message)
    log_message = "   Base → Development → Production"
    print(log_message)

    # 2. Merge hierarchy
    log_message = "\n2. Merging configuration hierarchy:"
    print(log_message)

    # Merge base + dev
    dev_merged = merge_configs(base_config, dev_config)
    log_message = "✅ Development configuration:"
    print(log_message)
    log_message = f"   Debug: {dev_merged.get('app', {}).get('debug')}"
    print(log_message)
    log_message = f"   Database: {dev_merged.get('database', {}).get('url')}"
    print(log_message)
    log_message = f"   Log level: {dev_merged.get('logging', {}).get('level')}"
    print(log_message)

    # Merge base + prod
    prod_merged = merge_configs(base_config, prod_config)
    log_message = "✅ Production configuration:"
    print(log_message)
    log_message = f"   Debug: {prod_merged.get('app', {}).get('debug')}"
    print(log_message)
    log_message = f"   Database: {prod_merged.get('database', {}).get('url')}"
    print(log_message)
    log_message = f"   Pool size: {prod_merged.get('database', {}).get('pool_size')}"
    print(log_message)

    # 3. Configuration inheritance patterns
    log_message = "\n3. Configuration inheritance patterns:"
    print(log_message)

    # Feature flags configuration
    feature_config: TConfigDict = {
        "features": {
            "new_ui": {
                "enabled": True,
                "beta": True,
            },
            "analytics": {
                "enabled": True,
                "tracking_id": "UA-123456",
            },
            "caching": {
                "enabled": False,
                "ttl": 3600,
            },
        },
    }

    # Environment-specific feature overrides
    dev_features: TConfigDict = {
        "features": {
            "new_ui": {
                "enabled": True,
                "beta": True,
            },
            "analytics": {
                "enabled": False,  # Disable in dev
            },
            "caching": {
                "enabled": True,
                "ttl": 60,  # Short TTL for dev
            },
        },
    }

    # Merge feature configurations
    feature_merged = merge_configs(feature_config, dev_features)
    log_message = "✅ Feature configuration:"
    print(log_message)
    features = feature_merged.get("features", {})
    for feature_name, feature_config in features.items():
        enabled = feature_config.get("enabled", False)
        status = "✅ Enabled" if enabled else "❌ Disabled"
        log_message = f"   {feature_name}: {status}"
        print(log_message)


def demonstrate_advanced_configuration_patterns() -> None:  # noqa: PLR0912, PLR0915
    """Demonstrate advanced configuration patterns using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🚀 ADVANCED CONFIGURATION PATTERNS")
    print("=" * 80)

    # 1. Configuration validation patterns
    log_message = "\n1. Configuration validation patterns:"
    print(log_message)

    # Complex configuration with validation
    complex_config: TConfigDict = {
        "server": {
            "host": "localhost",
            "port": 8080,
            "workers": 4,
        },
        "security": {
            "ssl_enabled": True,
            "cert_path": "/path/to/cert.pem",
            "key_path": "/path/to/key.pem",
        },
        "monitoring": {
            "enabled": True,
            "metrics_port": 9090,
            "health_check_interval": 30,
        },
    }

    # Validate configuration structure
    log_message = "🔍 Validating configuration structure:"
    print(log_message)

    required_fields = [
        ("server.host", str),
        ("server.port", int),
        ("server.workers", int),
        ("security.ssl_enabled", bool),
        ("monitoring.enabled", bool),
    ]

    validation_errors: list[TErrorMessage] = []

    for field_path, expected_type in required_fields:
        # Navigate to nested field
        value = complex_config
        for key in field_path.split("."):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break

        # Validate type
        if not FlextTypes.TypeGuards.is_instance_of(value, expected_type):
            error_message = (
                f"Field '{field_path}' must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            validation_errors.append(error_message)

    if validation_errors:
        log_message = "❌ Configuration validation failed:"
        print(log_message)
        for error in validation_errors:
            log_message = f"   - {error}"
            print(log_message)
    else:
        log_message = "✅ Configuration validation passed"
        print(log_message)

    # 2. Configuration transformation patterns
    log_message = "\n2. Configuration transformation patterns:"
    print(log_message)

    # Transform configuration for different environments
    transform_config: TConfigDict = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp",
            "user": "app_user",
            "password": "secret",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
        },
    }

    # Transform to connection strings
    transformed_config: TConfigDict = {}

    # Database connection string
    db_config = transform_config.get("database", {})
    if all(key in db_config for key in ["host", "port", "name", "user", "password"]):
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
        transformed_config["database_url"] = db_url

    # Redis connection string
    redis_config = transform_config.get("redis", {})
    if all(key in redis_config for key in ["host", "port", "db"]):
        redis_url = f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
        transformed_config["redis_url"] = redis_url

    log_message = "🔄 Transformed configuration:"
    print(log_message)
    for key, value in transformed_config.items():
        # Mask sensitive information
        if "password" in str(value):
            masked_value = str(value).replace(db_config.get("password", ""), "***")
            log_message = f"   {key}: {masked_value}"
        else:
            log_message = f"   {key}: {value}"
        print(log_message)

    # 3. Configuration composition patterns
    log_message = "\n3. Configuration composition patterns:"
    print(log_message)

    # Compose configuration from multiple sources
    config_sources: list[tuple[str, TConfigDict]] = [
        ("defaults", {"timeout": 30, "retries": 3, "debug": False}),
        ("environment", {"timeout": 60, "debug": True}),
        ("user_preferences", {"retries": 5}),
    ]

    composed_config: TConfigDict = {}

    for source_name, source_config in config_sources:
        log_message = f"📋 Applying {source_name} configuration:"
        print(log_message)
        for key, value in source_config.items():
            composed_config[key] = value
            log_message = f"   {key}: {value}"
            print(log_message)

    log_message = "✅ Final composed configuration:"
    print(log_message)
    for key, value in composed_config.items():
        log_message = f"   {key}: {value}"
        print(log_message)


def demonstrate_domain_configuration_integration() -> None:  # noqa: PLR0912, PLR0915
    """Demonstrate configuration integration with shared domain models."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🏢 DOMAIN MODEL CONFIGURATION INTEGRATION")
    print("=" * 80)

    # 1. Configuration with domain model validation
    log_message = "\n1. Configuration with domain model validation:"
    print(log_message)

    # Configuration for user management service
    user_service_config: TConfigDict = {
        "service_name": "user_management",
        "REDACTED_LDAP_BIND_PASSWORD_users": [
            {
                "name": "REDACTED_LDAP_BIND_PASSWORD",
                "email": "REDACTED_LDAP_BIND_PASSWORD@company.com",
                "age": 30,
            },
            {
                "name": "moderator",
                "email": "mod@company.com",
                "age": 25,
            },
        ],
        "user_validation": {
            "min_age": 18,
            "max_age": 120,
            "require_email": True,
        },
        "features": {
            "email_notifications": True,
            "user_audit_log": True,
            "auto_suspend_inactive": False,
        },
    }

    log_message = "📋 User service configuration loaded"
    print(log_message)

    # 2. Validate REDACTED_LDAP_BIND_PASSWORD users using shared domain models
    log_message = "\n2. Validating REDACTED_LDAP_BIND_PASSWORD users with shared domain models:"
    print(log_message)

    REDACTED_LDAP_BIND_PASSWORD_users = user_service_config.get("REDACTED_LDAP_BIND_PASSWORD_users", [])
    validated_users = []

    for user_data in REDACTED_LDAP_BIND_PASSWORD_users:
        if isinstance(user_data, dict):
            name = user_data.get("name", "")
            email = user_data.get("email", "")
            age = user_data.get("age", 0)

            # Use SharedDomainFactory for validation
            user_result = SharedDomainFactory.create_user(name, email, age)
            if user_result.is_success:
                user = user_result.data
                validated_users.append(user)
                log_message = (
                    f"✅ Admin user validated: {user.name} ({user.email_address.email})"
                )
                print(log_message)

                # Log domain operation
                log_domain_operation(
                    "REDACTED_LDAP_BIND_PASSWORD_user_configured",
                    "SharedUser",
                    user.id,
                    service="user_management",
                    config_role="REDACTED_LDAP_BIND_PASSWORD",
                )
            else:
                log_message = f"❌ Invalid REDACTED_LDAP_BIND_PASSWORD user config: {user_result.error}"
                print(log_message)

    log_message = f"📊 Validated {len(validated_users)}/{len(REDACTED_LDAP_BIND_PASSWORD_users)} REDACTED_LDAP_BIND_PASSWORD users"
    print(log_message)

    # 3. Configuration-driven user creation
    log_message = "\n3. Configuration-driven user creation:"
    print(log_message)

    # Test configuration
    test_user_configs = [
        {"name": "test_user_1", "email": "test1@example.com", "age": 25},
        {"name": "test_user_2", "email": "test2@example.com", "age": 35},
        {"name": "", "email": "invalid", "age": 15},  # Invalid user
    ]

    created_users = []

    for config in test_user_configs:
        if isinstance(config, dict):
            user_result = SharedDomainFactory.create_user(
                config.get("name", ""),
                config.get("email", ""),
                config.get("age", 0),
            )

            if user_result.is_success:
                user = user_result.data
                created_users.append(user)
                log_message = f"✅ User created from config: {user.name}"
                print(log_message)
            else:
                log_message = (
                    f"❌ Failed to create user from config: {user_result.error}"
                )
                print(log_message)

    # 4. Configuration validation using domain rules
    log_message = "\n4. Configuration validation using domain rules:"
    print(log_message)

    validation_config = user_service_config.get("user_validation", {})
    min_age = validation_config.get("min_age", 18)
    max_age = validation_config.get("max_age", 120)

    log_message = "📋 Domain validation rules from config:"
    print(log_message)
    log_message = f"   Min age: {min_age}"
    print(log_message)
    log_message = f"   Max age: {max_age}"
    print(log_message)

    # Test configuration against domain models
    test_ages = [17, 25, 130]  # Below min, valid, above max

    for test_age in test_ages:
        try:
            user_result = SharedDomainFactory.create_user(
                "test",
                "test@example.com",
                test_age,
            )
            if user_result.is_success:
                log_message = f"✅ Age {test_age}: Valid according to domain rules"
                print(log_message)
            else:
                log_message = f"❌ Age {test_age}: Invalid - {user_result.error}"
                print(log_message)
        except (RuntimeError, ValueError, TypeError) as e:
            log_message = f"❌ Age {test_age}: Validation error - {e}"
            print(log_message)

    # 5. Configuration-based feature flags with domain context
    log_message = "\n5. Configuration-based feature flags with domain context:"
    print(log_message)

    features = user_service_config.get("features", {})

    for feature_name, enabled in features.items():
        if isinstance(enabled, bool):
            status = "✅ Enabled" if enabled else "❌ Disabled"
            log_message = f"   {feature_name}: {status}"
            print(log_message)

            # Log feature configuration as domain operation
            log_domain_operation(
                "feature_configured",
                "FeatureFlag",
                feature_name,
                enabled=enabled,
                service="user_management",
            )

    log_message = (
        f"📊 Successfully validated configuration with "
        f"{len(created_users)} domain objects"
    )
    print(log_message)


def main() -> None:
    """Run comprehensive FlextConfig demonstration with maximum type safety."""
    print("=" * 80)
    print("🚀 FLEXT CONFIG - ENTERPRISE CONFIGURATION DEMONSTRATION")
    print("=" * 80)

    # Run all demonstrations
    demonstrate_basic_configuration()
    demonstrate_environment_integration()
    demonstrate_configuration_merging()
    demonstrate_file_configuration()
    demonstrate_configuration_hierarchies()
    demonstrate_advanced_configuration_patterns()
    demonstrate_domain_configuration_integration()

    print("\n" + "=" * 80)
    print("🎉 FLEXT CONFIG DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
