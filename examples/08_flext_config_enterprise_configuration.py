#!/usr/bin/env python3
"""Enterprise configuration management with FlextConfig.

Demonstrates environment integration, validation, merging,
and override patterns for configuration management.
    - Default value management and fallbacks
    - File-based configuration loading
    - Configuration validation and error handling
    - Enterprise configuration patterns
    - Configuration hierarchies and inheritance
    - Maximum type safety using flext_core.typings

Key Components:
    - FlextConfig: Main configuration management class
    - FlextSettings: Pydantic-based settings with environment loading
    - FlextConfigDefaults: Default configuration management
    - FlextConfigOps: Configuration operations and transformations
    - Configuration validation with comprehensive error reporting

This example shows real-world enterprise configuration scenarios
demonstrating the power and flexibility of the FlextConfig system.
"""

import json
import os
import pathlib
import sys as _sys
import tempfile
from pathlib import Path as _Path
from typing import cast

from flext_core import (
    FlextConstants,
    FlextResult,
    FlextSettings,
    TAnyDict,
    TConfigDict,
    TErrorMessage,
    TLogMessage,
    merge_configs,
)

_project_root = _Path(__file__).resolve().parents[1]
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

from examples.shared_domain import (
    SharedDemonstrationPattern,
    SharedDomainFactory,
    log_domain_operation,
)


def demonstrate_basic_configuration() -> None:
    """Demonstrate basic configuration patterns with FlextConfig.

    Using flext_core.typings for type safety.
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

    # Cast TConfigDict to TAnyDict for method compatibility
    # Create basic configuration using available methods
    config_result = FlextResult.ok(dict(config_data))  # Simplified for demonstration
    if config_result.success:
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

    # Simple validation without using FlextConfigValidation methods that MyPy doesn't see
    app_name = config_data.get("app_name")
    if isinstance(app_name, str) and app_name:
        log_message = "✅ Configuration type validation is valid"
        print(log_message)
    else:
        error_message = (
            "Configuration validation failed: app_name must be a non-empty string"
        )
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
    # Simple merge without using FlextConfigDefaults methods that MyPy doesn't see
    config_with_defaults = dict(defaults)  # Start with defaults
    config_with_defaults.update(config_data)  # Override with actual config

    log_message = "🔄 Config with defaults applied:"
    print(log_message)
    for key, value in config_with_defaults.items():
        log_message = f"   {key}: {value}"
        print(log_message)


def demonstrate_environment_integration() -> None:
    """Demonstrate environment variable integration using flext_core.typings."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🌍 ENVIRONMENT INTEGRATION")
    print("=" * 80)

    # 1. Environment-based settings
    log_message = "\n1. Environment-based settings:"
    print(log_message)

    class MyAppSettings(FlextSettings):
        """Application settings with environment loading using flext_core.typings."""

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
    """Demonstrate configuration merging patterns using flext_core.typings."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🔄 CONFIGURATION MERGING")
    print("=" * 80)

    # 1. Basic configuration merging
    log_message = "\n1. Basic configuration merging:"
    print(log_message)

    # Use TAnyDict for configs with nested structures
    base_config: TAnyDict = {
        "app_name": "MyApp",
        "debug": False,
        "port": 8000,
        "database": {
            "url": "sqlite:///app.db",
            "pool_size": 10,
        },
    }

    # Use TAnyDict for configs with nested structures
    override_config: TAnyDict = {
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

    # Use TAnyDict for configs with nested structures
    deep_base: TAnyDict = {
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

    # Use TAnyDict for configs with nested structures
    deep_override: TAnyDict = {
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
    """Demonstrate file-based configuration using railway-oriented programming."""
    _print_file_config_section_header("📁 FILE CONFIGURATION")

    # Chain all file configuration operations using single-responsibility methods
    (
        _create_configuration_file()
        .flat_map(_load_configuration_from_file)
        .flat_map(_validate_loaded_configuration)
        .flat_map(_cleanup_configuration_file)
    )


def _print_file_config_section_header(title: str) -> None:
    """Print formatted file configuration section header."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print(title)
    print("=" * 80)


def _create_configuration_file() -> FlextResult[str]:
    """Create temporary configuration file and return file path."""
    print("\n1. Creating configuration file:")

    config_data = _build_file_configuration_data()

    try:
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
        return FlextResult.ok(temp_file_path)

    except (RuntimeError, ValueError, TypeError) as e:
        error_message: TErrorMessage = f"Failed to create config file: {e}"
        return FlextResult.fail(error_message)


def _build_file_configuration_data() -> TAnyDict:
    """Build configuration data for file demonstration."""
    return {
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


def _load_configuration_from_file(
    temp_file_path: str,
) -> FlextResult[tuple[str, TAnyDict]]:
    """Load configuration from file and display contents."""
    print("\n2. Loading configuration from file:")

    try:
        with pathlib.Path(temp_file_path).open(encoding="utf-8") as f:
            loaded_config: TAnyDict = json.load(f)

        print("✅ Configuration loaded from file:")
        _display_loaded_configuration(loaded_config)

        return FlextResult.ok((temp_file_path, loaded_config))

    except (RuntimeError, ValueError, TypeError) as e:
        error_message: TErrorMessage = f"Failed to load config file: {e}"
        print(f"❌ {error_message}")
        return FlextResult.fail(error_message)


def _display_loaded_configuration(loaded_config: TAnyDict) -> None:
    """Display loaded configuration with type guards."""
    # Type guard for loaded_config
    if isinstance(loaded_config, dict):
        app_dict = loaded_config.get("app", {})
        app_name = app_dict.get("name") if isinstance(app_dict, dict) else None
        log_message = f"   App name: {app_name}"
        print(log_message)

        db_dict = loaded_config.get("database", {})
        db_url = db_dict.get("url") if isinstance(db_dict, dict) else None
        log_message = f"   Database URL: {db_url}"
        print(log_message)

        features = loaded_config.get("features")
        log_message = f"   Features: {features}"
        print(log_message)


def _validate_loaded_configuration(
    config_data: tuple[str, TAnyDict],
) -> FlextResult[str]:
    """Validate loaded configuration and return file path for cleanup."""
    print("\n3. Configuration validation:")

    temp_file_path, loaded_config = config_data

    # Validate app name with type guard
    app_name = _extract_app_name_from_config(loaded_config)

    # Simple validation without using FlextConfigValidation methods that MyPy doesn't see
    if isinstance(app_name, str) and app_name:
        log_message = "✅ App name validation passed"
        print(log_message)
    else:
        error_message = (
            "App name validation failed: app.name must be a non-empty string"
        )
        print(f"❌ {error_message}")

    return FlextResult.ok(temp_file_path)


def _extract_app_name_from_config(loaded_config: TAnyDict) -> str | None:
    """Extract app name from loaded configuration with type safety."""
    if isinstance(loaded_config, dict):
        app_dict = loaded_config.get("app", {})
        if isinstance(app_dict, dict):
            return app_dict.get("name")
    return None


def _cleanup_configuration_file(temp_file_path: str) -> FlextResult[None]:
    """Clean up temporary configuration file."""
    try:
        pathlib.Path(temp_file_path).unlink()
        log_message = "🗑️ Temporary file cleaned up"
        print(log_message)
        return FlextResult.ok(None)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Failed to clean up file: {e}"
        print(f"⚠️ {error_message}")
        return FlextResult.fail(error_message)


def demonstrate_configuration_hierarchies() -> None:
    """Demonstrate configuration hierarchies.

    Shows how defaults, environment and user configs merge.
    """
    _print_config_section_header("🏗️ CONFIGURATION HIERARCHIES")
    base_config = _build_base_hierarchy_config()
    dev_config = _build_dev_hierarchy_config()
    prod_config = _build_prod_hierarchy_config()

    _print_config_hierarchy_overview()
    _print_merged_config("Development", merge_configs(base_config, dev_config))
    _print_merged_config("Production", merge_configs(base_config, prod_config))
    _print_feature_hierarchy_demo()


def _build_base_hierarchy_config() -> TAnyDict:
    return {
        "app": {"name": "HierarchyApp", "version": "0.9.0"},
        "database": {"pool_size": 10, "timeout": 30},
        "logging": {"level": "INFO", "format": "json"},
    }


def _build_dev_hierarchy_config() -> TAnyDict:
    return {
        "app": {"debug": True},
        "database": {"url": "sqlite:///dev.db"},
        "logging": {"level": "DEBUG"},
    }


def _build_prod_hierarchy_config() -> TAnyDict:
    return {
        "app": {"debug": False},
        "database": {
            "url": "postgresql://prod-server:5432/prod",
            "pool_size": 50,
        },
        "logging": {"level": "WARNING"},
    }


def _print_config_hierarchy_overview() -> None:
    print("\n1. Configuration hierarchy levels:")
    print("📋 Configuration hierarchy:")
    print("   Base → Development → Production")


def _print_merged_config(env_name: str, merged: TAnyDict) -> None:
    print(f"\n2. Merging configuration hierarchy: {env_name}")
    print(f"✅ {env_name} configuration:")
    app_config = merged.get("app", {}) if isinstance(merged, dict) else {}
    debug_value = app_config.get("debug") if isinstance(app_config, dict) else "N/A"
    print(f"   Debug: {debug_value}")

    db_config = merged.get("database", {}) if isinstance(merged, dict) else {}
    db_url = db_config.get("url") if isinstance(db_config, dict) else "N/A"
    print(f"   Database: {db_url}")

    log_config = merged.get("logging", {}) if isinstance(merged, dict) else {}
    log_level = log_config.get("level") if isinstance(log_config, dict) else "N/A"
    pool_size = db_config.get("pool_size") if isinstance(db_config, dict) else "N/A"
    if pool_size != "N/A":
        print(f"   Pool size: {pool_size}")
    print(f"   Log level: {log_level}")


def _print_feature_hierarchy_demo() -> None:
    print("\n3. Configuration inheritance patterns:")
    feature_config: TAnyDict = {
        "features": {
            "new_ui": {"enabled": True, "beta": True},
            "analytics": {"enabled": True, "tracking_id": "UA-123456"},
            "caching": {"enabled": False, "ttl": 3600},
        },
    }
    dev_features: TAnyDict = {
        "features": {
            "new_ui": {"enabled": True, "beta": True},
            "analytics": {"enabled": False},
            "caching": {"enabled": True, "ttl": 60},
        },
    }
    feature_merged = merge_configs(feature_config, dev_features)
    print("✅ Feature configuration:")
    features = feature_merged.get("features", {})
    items = features.items() if isinstance(features, dict) else []
    for feature_name, feature_details in items:
        enabled = feature_details.get("enabled", False)
        status = "✅ Enabled" if enabled else "❌ Disabled"
        print(f"   {feature_name}: {status}")


def demonstrate_advanced_configuration_patterns() -> None:
    """Demonstrate advanced configuration patterns.

    Uses railway-oriented programming to compose configuration steps.
    """
    _print_config_section_header("🚀 ADVANCED CONFIGURATION PATTERNS")

    # Chain all configuration pattern demonstrations using single-responsibility methods
    (
        _demonstrate_configuration_validation()
        .flat_map(lambda _: _demonstrate_configuration_transformation())
        .flat_map(lambda _: _demonstrate_configuration_composition())
    )


def _print_config_section_header(title: str) -> None:
    """Print formatted configuration section header."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print(title)
    print("=" * 80)


def _demonstrate_configuration_validation() -> FlextResult[None]:
    """Demonstrate configuration validation patterns."""
    print("\n1. Configuration validation patterns:")

    # Create complex configuration
    complex_config = _create_complex_configuration()

    return _validate_configuration_structure(complex_config).flat_map(
        _display_validation_results,
    )


def _create_complex_configuration() -> TAnyDict:
    """Create complex configuration for validation demonstration."""
    return {
        "server": {
            "host": FlextConstants.Platform.DEFAULT_HOST,
            "port": FlextConstants.Platform.FLEXCORE_PORT,
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


def _validate_configuration_structure(
    config: TAnyDict,
) -> FlextResult[tuple[bool, list[TErrorMessage]]]:
    """Validate configuration structure and return validation results."""
    print("🔍 Validating configuration structure:")

    required_fields = [
        ("server.host", str),
        ("server.port", int),
        ("server.workers", int),
        ("security.ssl_enabled", bool),
        ("monitoring.enabled", bool),
    ]

    validation_errors: list[TErrorMessage] = []

    for field_path, expected_type in required_fields:
        validation_result = _validate_config_field(config, field_path, expected_type)
        if validation_result.is_failure:
            validation_errors.append(
                validation_result.error or f"Validation failed for {field_path}",
            )

    is_valid = len(validation_errors) == 0
    return FlextResult.ok((is_valid, validation_errors))


def _validate_config_field(
    config: TAnyDict,
    field_path: str,
    expected_type: type,
) -> FlextResult[None]:
    """Validate a single configuration field."""
    # Navigate to nested field
    value: object = config
    for key in field_path.split("."):
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return FlextResult.fail(
                f"Field path '{field_path}' not found in configuration",
            )

    # Validate type
    if not isinstance(value, expected_type):
        error_message = (
            f"Field '{field_path}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
        return FlextResult.fail(error_message)

    return FlextResult.ok(None)


def _display_validation_results(
    validation_data: tuple[bool, list[TErrorMessage]],
) -> FlextResult[None]:
    """Display configuration validation results."""
    _is_valid, validation_errors = validation_data

    if validation_errors:
        print("❌ Configuration validation failed:")
        for error in validation_errors:
            print(f"   - {error}")
    else:
        print("✅ Configuration validation passed")

    return FlextResult.ok(None)


def _demonstrate_configuration_transformation() -> FlextResult[None]:
    """Demonstrate configuration transformation patterns."""
    print("\n2. Configuration transformation patterns:")

    # Create transformation configuration
    transform_config = _create_transformation_configuration()

    return _transform_to_connection_strings(transform_config).flat_map(
        lambda transformed: _display_transformed_configuration(
            transformed,
            transform_config,
        ),
    )


def _create_transformation_configuration() -> TAnyDict:
    """Create configuration for transformation demonstration."""
    return {
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


def _transform_to_connection_strings(config: TAnyDict) -> FlextResult[TAnyDict]:
    """Transform configuration to connection strings."""
    transformed_config: TAnyDict = {}

    # Transform database configuration
    db_config = config.get("database", {})
    if isinstance(db_config, dict):
        db_result = _transform_database_config(db_config)
        if db_result.is_success and db_result.data:
            transformed_config["database_url"] = db_result.data

    # Transform Redis configuration
    redis_config = config.get("redis", {})
    if isinstance(redis_config, dict):
        redis_result = _transform_redis_config(redis_config)
        if redis_result.is_success and redis_result.data:
            transformed_config["redis_url"] = redis_result.data

    return FlextResult.ok(transformed_config)


def _transform_database_config(db_config: TAnyDict) -> FlextResult[str]:
    """Transform database configuration to connection string."""
    required_keys = ["host", "port", "name", "user", "password"]
    if not all(key in db_config for key in required_keys):
        return FlextResult.fail("Missing required database configuration keys")

    db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
    return FlextResult.ok(db_url)


def _transform_redis_config(redis_config: TAnyDict) -> FlextResult[str]:
    """Transform Redis configuration to connection string."""
    required_keys = ["host", "port", "db"]
    if not all(key in redis_config for key in required_keys):
        return FlextResult.fail("Missing required Redis configuration keys")

    redis_url = (
        f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
    )
    return FlextResult.ok(redis_url)


def _display_transformed_configuration(
    transformed_config: TAnyDict,
    original_config: TAnyDict,
) -> FlextResult[None]:
    """Display transformed configuration with masked sensitive data."""
    print("🔄 Transformed configuration:")

    for key, value in transformed_config.items():
        # Mask sensitive information
        masked_value = _mask_sensitive_data(str(value), original_config)
        print(f"   {key}: {masked_value}")

    return FlextResult.ok(None)


def _mask_sensitive_data(value: str, original_config: TAnyDict) -> str:
    """Mask sensitive data in configuration values."""
    if "password" in value:
        db_config_raw = original_config.get("database", {})
        if isinstance(db_config_raw, dict):
            password = db_config_raw.get("password", "")
        else:
            password = ""
        if password:
            return value.replace(password, "***")
    return value


def _demonstrate_configuration_composition() -> FlextResult[None]:
    """Demonstrate configuration composition patterns."""
    print("\n3. Configuration composition patterns:")

    config_sources = _create_configuration_sources()

    return _compose_configuration_from_sources(config_sources).flat_map(
        _display_composed_configuration,
    )


def _create_configuration_sources() -> list[tuple[str, TAnyDict]]:
    """Create configuration sources for composition demonstration."""
    return [
        ("defaults", {"timeout": 30, "retries": 3, "debug": False}),
        ("environment", {"timeout": 60, "debug": True}),
        ("user_preferences", {"retries": 5}),
    ]


def _compose_configuration_from_sources(
    config_sources: list[tuple[str, TAnyDict]],
) -> FlextResult[TAnyDict]:
    """Compose configuration from multiple sources."""
    composed_config: TAnyDict = {}

    for source_name, source_config in config_sources:
        print(f"📋 Applying {source_name} configuration:")
        for key, value in source_config.items():
            composed_config[key] = value
            print(f"   {key}: {value}")

    return FlextResult.ok(composed_config)


def _display_composed_configuration(composed_config: TAnyDict) -> FlextResult[None]:
    """Display final composed configuration."""
    print("✅ Final composed configuration:")
    for key, value in composed_config.items():
        print(f"   {key}: {value}")

    return FlextResult.ok(None)


def demonstrate_domain_configuration_integration() -> None:
    """Demonstrate configuration integration with domain models.

    Shows integration with shared domain models using railway-oriented programming.
    """
    _print_domain_config_section_header("🏢 DOMAIN MODEL CONFIGURATION INTEGRATION")

    # Chain all domain configuration demonstrations using single-responsibility methods
    (
        _demonstrate_domain_model_validation()
        .flat_map(_demonstrate_REDACTED_LDAP_BIND_PASSWORD_user_validation)
        .flat_map(_demonstrate_user_creation_from_config)
        .flat_map(_demonstrate_domain_rule_validation)
        .flat_map(_demonstrate_feature_flag_configuration)
    )


def _print_domain_config_section_header(title: str) -> None:
    """Print formatted domain configuration section header."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print(title)
    print("=" * 80)


def _demonstrate_domain_model_validation() -> FlextResult[TAnyDict]:
    """Demonstrate configuration with domain model validation."""
    print("\n1. Configuration with domain model validation:")

    # Configuration for user management service
    user_service_config: TAnyDict = {
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

    print("📋 User service configuration loaded")
    return FlextResult.ok(user_service_config)


def _demonstrate_REDACTED_LDAP_BIND_PASSWORD_user_validation(
    config: TAnyDict,
) -> FlextResult[tuple[TAnyDict, list[object]]]:
    """Demonstrate REDACTED_LDAP_BIND_PASSWORD user validation using shared domain models."""
    print("\n2. Validating REDACTED_LDAP_BIND_PASSWORD users with shared domain models:")

    REDACTED_LDAP_BIND_PASSWORD_users = config.get("REDACTED_LDAP_BIND_PASSWORD_users", [])
    validated_users = []

    if isinstance(REDACTED_LDAP_BIND_PASSWORD_users, list):
        for user_data in REDACTED_LDAP_BIND_PASSWORD_users:
            validation_result = _validate_single_REDACTED_LDAP_BIND_PASSWORD_user(user_data)
            if validation_result.is_success and validation_result.data:
                validated_users.append(validation_result.data)

    REDACTED_LDAP_BIND_PASSWORD_count = len(REDACTED_LDAP_BIND_PASSWORD_users) if isinstance(REDACTED_LDAP_BIND_PASSWORD_users, list) else 0
    print(f"📊 Validated {len(validated_users)}/{REDACTED_LDAP_BIND_PASSWORD_count} REDACTED_LDAP_BIND_PASSWORD users")
    return FlextResult.ok((config, validated_users))


def _validate_single_REDACTED_LDAP_BIND_PASSWORD_user(user_data: object) -> FlextResult[object]:
    """Validate a single REDACTED_LDAP_BIND_PASSWORD user using domain factory."""
    if not isinstance(user_data, dict):
        return FlextResult.fail("Invalid user data format")

    name = user_data.get("name", "")
    email = user_data.get("email", "")
    age = user_data.get("age", 0)

    # Use SharedDomainFactory for validation
    user_result = SharedDomainFactory.create_user(name, email, age)
    if user_result.success:
        user = user_result.data
        if user is not None:
            print(f"✅ Admin user validated: {user.name} ({user.email_address.email})")

            # Log domain operation
            log_domain_operation(
                "REDACTED_LDAP_BIND_PASSWORD_user_configured",
                "SharedUser",
                user.id,
                service="user_management",
                config_role="REDACTED_LDAP_BIND_PASSWORD",
            )
            return FlextResult.ok(user)

    print(f"❌ Invalid REDACTED_LDAP_BIND_PASSWORD user config: {user_result.error}")
    return FlextResult.fail(f"Validation failed: {user_result.error}")


def _demonstrate_user_creation_from_config(
    validated_data: tuple[TAnyDict, list[object]],
) -> FlextResult[tuple[TAnyDict, list[object], list[object]]]:
    """Demonstrate configuration-driven user creation."""
    print("\n3. Configuration-driven user creation:")

    config, validated_users = validated_data

    # Test configuration
    test_user_configs = [
        {"name": "test_user_1", "email": "test1@example.com", "age": 25},
        {"name": "test_user_2", "email": "test2@example.com", "age": 35},
        {"name": "", "email": "invalid", "age": 15},  # Invalid user
    ]

    created_users = []

    for user_config in test_user_configs:
        creation_result = _create_user_from_config(user_config)
        if creation_result.is_success and creation_result.data:
            created_users.append(creation_result.data)

    return FlextResult.ok((config, validated_users, created_users))


def _create_user_from_config(config: object) -> FlextResult[object]:
    """Create a user from configuration data."""
    if not isinstance(config, dict):
        return FlextResult.fail("Invalid config format")

    user_result = SharedDomainFactory.create_user(
        config.get("name", ""),
        config.get("email", ""),
        config.get("age", 0),
    )

    if user_result.success:
        user = user_result.data
        if user is not None:
            print(f"✅ User created from config: {user.name}")
            return FlextResult.ok(user)

    print(f"❌ Failed to create user from config: {user_result.error}")
    return FlextResult.fail(f"Creation failed: {user_result.error}")


def _demonstrate_domain_rule_validation(
    user_data: tuple[TAnyDict, list[object], list[object]],
) -> FlextResult[tuple[TAnyDict, list[object], list[object]]]:
    """Demonstrate configuration validation using domain rules."""
    print("\n4. Configuration validation using domain rules:")

    config, validated_users, created_users = user_data

    validation_config_raw = config.get("user_validation", {})
    if isinstance(validation_config_raw, dict):
        validation_config: dict[str, object] = cast(
            "dict[str, object]",
            validation_config_raw,
        )
        min_age = validation_config.get("min_age", 18)
        max_age = validation_config.get("max_age", 120)
    else:
        min_age = 18
        max_age = 120

    print("📋 Domain validation rules from config:")
    print(f"   Min age: {min_age}")
    print(f"   Max age: {max_age}")

    # Test configuration against domain models
    test_ages = [17, 25, 130]  # Below min, valid, above max

    for test_age in test_ages:
        _validate_age_against_domain_rules(test_age)

    return FlextResult.ok((config, validated_users, created_users))


def _validate_age_against_domain_rules(test_age: int) -> FlextResult[None]:
    """Validate a specific age against domain rules."""
    try:
        user_result = SharedDomainFactory.create_user(
            "test",
            "test@example.com",
            test_age,
        )
        if user_result.success:
            print(f"✅ Age {test_age}: Valid according to domain rules")
        else:
            print(f"❌ Age {test_age}: Invalid - {user_result.error}")
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"❌ Age {test_age}: Validation error - {e}")

    return FlextResult.ok(None)


def _demonstrate_feature_flag_configuration(
    validation_data: tuple[TAnyDict, list[object], list[object]],
) -> FlextResult[None]:
    """Demonstrate configuration-based feature flags with domain context."""
    print("\n5. Configuration-based feature flags with domain context:")

    config, _validated_users, created_users = validation_data
    features_raw = config.get("features", {})

    if isinstance(features_raw, dict):
        features: dict[str, object] = cast("dict[str, object]", features_raw)
        for feature_name, enabled in features.items():
            _configure_single_feature_flag(feature_name, enabled)
    else:
        print("   No valid features configuration found")

    print(
        f"📊 Successfully validated configuration with "
        f"{len(created_users)} domain objects",
    )

    return FlextResult.ok(None)


def _configure_single_feature_flag(
    feature_name: str,
    enabled: object,
) -> FlextResult[None]:
    """Configure a single feature flag with domain logging."""
    if isinstance(enabled, bool):
        status = "✅ Enabled" if enabled else "❌ Disabled"
        print(f"   {feature_name}: {status}")

        # Log feature configuration as domain operation
        log_domain_operation(
            "feature_configured",
            "FeatureFlag",
            feature_name,
            enabled=enabled,
            service="user_management",
        )

    return FlextResult.ok(None)


def main() -> None:
    """Run comprehensive FlextConfig demonstration using shared pattern."""
    # DRY PRINCIPLE: Use SharedDemonstrationPattern to eliminate duplication
    SharedDemonstrationPattern.run_demonstration(
        "FLEXT CONFIG - ENTERPRISE CONFIGURATION DEMONSTRATION",
        [
            demonstrate_basic_configuration,
            demonstrate_environment_integration,
            demonstrate_configuration_merging,
            demonstrate_file_configuration,
            demonstrate_configuration_hierarchies,
            demonstrate_advanced_configuration_patterns,
            demonstrate_domain_configuration_integration,
        ],
    )


if __name__ == "__main__":
    main()
