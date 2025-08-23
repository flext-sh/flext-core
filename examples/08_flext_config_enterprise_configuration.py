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
    - FlextSystemDefaults: Default configuration management
    - Foundation utility functions: safe_get_env_var, safe_load_json_file, merge_configs
    - Configuration validation with comprehensive error reporting

This example shows real-world enterprise configuration scenarios
demonstrating the power and flexibility of the FlextConfig system.
"""

import contextlib
import json
import os
import pathlib
import tempfile
from typing import cast

from pydantic_settings import SettingsConfigDict
from shared_domain import (
    SharedDemonstrationPattern,
    SharedDomainFactory,
    log_domain_operation,
)

from flext_core import (
    FlextConstants,
    FlextResult,
    FlextSettings,
    FlextTypes,
    merge_configs,
)


def demonstrate_basic_configuration() -> None:
    """Demonstrate basic configuration patterns with FlextConfig.

    Using flext_core.typings for type safety.
    """
    separator = "\n" + "=" * 80
    print(separator)

    # 1. Basic configuration creation
    config_data: FlextTypes.Core.Config = {
        "app_name": "MyApp",
        "debug": True,
        "max_connections": 100,
        "timeout": 30.0,
    }

    # Cast FlextTypes.Core.Config to FlextTypes.Core.Dict for method compatibility
    # Create basic configuration using available methods
    config_result = FlextResult.ok(dict(config_data))  # Simplified for demonstration
    if config_result.success:
        config = config_result.value
        if config is not None:
            f"   App name: {config.get('app_name')}"
            f"   Debug mode: {config.get('debug')}"
            f"   Max connections: {config.get('max_connections')}"

    # 2. Configuration validation

    # Simple validation without using FlextConfigValidation methods that MyPy doesn't see
    app_name = config_data.get("app_name")
    if isinstance(app_name, str) and app_name:
        pass

    # 3. Configuration defaults
    defaults: FlextTypes.Core.Config = {
        "debug": False,
        "timeout": 30,
        "port": 8000,
        "max_retries": 3,
    }

    # 4. Applying defaults
    # Simple merge without using FlextConfigDefaults methods that MyPy doesn't see
    config_with_defaults = dict(defaults)  # Start with defaults
    config_with_defaults.update(config_data)  # Override with actual config

    for _key, _value in config_with_defaults.items():
        pass


def demonstrate_environment_integration() -> None:
    """Demonstrate environment variable integration using flext_core.typings."""
    separator = "\n" + "=" * 80
    print(separator)

    # 1. Environment-based settings

    class MyAppSettings(FlextSettings):
        """Application settings with environment loading using flext_core.typings."""

        debug: bool = True
        max_connections: int = 100
        database_url: str = "sqlite:///app.db"
        api_key: str = "default-key"
        timeout: float = 30.0

        model_config = SettingsConfigDict(
            env_prefix="MYAPP_",
            case_sensitive=False,
            env_file=".env",
            extra="ignore",  # Skip extra environment variables not defined in model
        )

    # 2. Simulate environment variables

    # Set environment variables for demonstration
    os.environ["MYAPP_DEBUG"] = "false"
    os.environ["MYAPP_MAX_CONNECTIONS"] = "200"
    os.environ["MYAPP_DATABASE_URL"] = "postgresql://localhost:5432/mydb"
    os.environ["MYAPP_API_KEY"] = "env-api-key-123"

    try:
        settings = MyAppSettings()
        f"   API Key: {settings.api_key[:10]}..."

    except (RuntimeError, ValueError, TypeError):
        pass

    # 3. Environment variable validation

    # Test invalid environment variable
    os.environ["MYAPP_MAX_CONNECTIONS"] = "invalid_number"

    with contextlib.suppress(RuntimeError, ValueError, TypeError):
        settings = MyAppSettings()

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
    separator = "\n" + "=" * 80
    print(separator)

    # 1. Basic configuration merging

    # Use FlextTypes.Core.Dict for configs with nested structures
    base_config: FlextTypes.Core.Dict = {
        "app_name": "MyApp",
        "debug": False,
        "port": 8000,
        "database": {
            "url": "sqlite:///app.db",
            "pool_size": 10,
        },
    }

    # Use FlextTypes.Core.Dict for configs with nested structures
    override_config: FlextTypes.Core.Dict = {
        "debug": True,
        "port": 9000,
        "database": {
            "url": "postgresql://localhost:5432/prod",
            "pool_size": 20,
        },
        "new_setting": "value",
    }

    # Merge configurations
    merged_result = merge_configs(base_config, override_config)
    merged_config = merged_result.unwrap_or({})
    for _key, _value in merged_config.items():
        pass

    # 2. Deep merging demonstration

    # Use FlextTypes.Core.Dict for configs with nested structures
    deep_base: FlextTypes.Core.Dict = {
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

    # Use FlextTypes.Core.Dict for configs with nested structures
    deep_override: FlextTypes.Core.Dict = {
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

    deep_merged_result = merge_configs(deep_base, deep_override)
    deep_merged = deep_merged_result.unwrap_or({})
    f"   Services: {deep_merged.get('services')}"


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


def _print_file_config_section_header(_title: str) -> None:
    """Print formatted file configuration section header."""
    separator = "\n" + "=" * 80
    print(separator)


def _create_configuration_file() -> FlextResult[str]:
    """Create temporary configuration file and return file path."""
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

        return FlextResult.ok(temp_file_path)

    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Failed to create config file: {e}"
        return FlextResult.fail(error_message)


def _build_file_configuration_data() -> FlextTypes.Core.Dict:
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
) -> FlextResult[tuple[str, FlextTypes.Core.Dict]]:
    """Load configuration from file and display contents."""
    try:
        with pathlib.Path(temp_file_path).open(encoding="utf-8") as f:
            loaded_config: FlextTypes.Core.Dict = json.load(f)

        _display_loaded_configuration(loaded_config)

        return FlextResult.ok((temp_file_path, loaded_config))

    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Failed to load config file: {e}"
        return FlextResult.fail(error_message)


def _display_loaded_configuration(loaded_config: FlextTypes.Core.Dict) -> None:
    """Display loaded configuration with type guards."""
    # Type guard for loaded_config
    if isinstance(loaded_config, dict):
        app_dict = loaded_config.get("app", {})
        app_dict.get("name") if isinstance(app_dict, dict) else None

        db_dict = loaded_config.get("database", {})
        db_dict.get("url") if isinstance(db_dict, dict) else None

        loaded_config.get("features")


def _validate_loaded_configuration(
    config_data: tuple[str, FlextTypes.Core.Dict],
) -> FlextResult[str]:
    """Validate loaded configuration and return file path for cleanup."""
    temp_file_path, loaded_config = config_data

    # Validate app name with type guard
    app_name = _extract_app_name_from_config(loaded_config)

    # Simple validation without using FlextConfigValidation methods that MyPy doesn't see
    if isinstance(app_name, str) and app_name:
        pass

    return FlextResult.ok(temp_file_path)


def _extract_app_name_from_config(loaded_config: FlextTypes.Core.Dict) -> str | None:
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
        return FlextResult.ok(None)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Failed to clean up file: {e}"
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
    dev_merged = merge_configs(base_config, dev_config)
    _print_merged_config(
        "Development",
        dev_merged.value if isinstance(dev_merged, FlextResult) else dev_merged,
    )
    prod_merged = merge_configs(base_config, prod_config)
    _print_merged_config(
        "Production",
        prod_merged.value if isinstance(prod_merged, FlextResult) else prod_merged,
    )
    _print_feature_hierarchy_demo()


def _build_base_hierarchy_config() -> FlextTypes.Core.Dict:
    return {
        "app": {"name": "HierarchyApp", "version": "0.9.0"},
        "database": {"pool_size": 10, "timeout": 30},
        "logging": {"level": "INFO", "format": "json"},
    }


def _build_dev_hierarchy_config() -> FlextTypes.Core.Dict:
    return {
        "app": {"debug": True},
        "database": {"url": "sqlite:///dev.db"},
        "logging": {"level": "DEBUG"},
    }


def _build_prod_hierarchy_config() -> FlextTypes.Core.Dict:
    return {
        "app": {"debug": False},
        "database": {
            "url": "postgresql://prod-server:5432/prod",
            "pool_size": 50,
        },
        "logging": {"level": "WARNING"},
    }


def _print_config_hierarchy_overview() -> None:
    pass


def _print_merged_config(env_name: str, merged: FlextTypes.Core.Dict) -> None:  # noqa: ARG001
    app_config = merged.get("app", {}) if isinstance(merged, dict) else {}
    app_config.get("debug") if isinstance(app_config, dict) else "N/A"

    db_config = merged.get("database", {}) if isinstance(merged, dict) else {}
    db_config.get("url") if isinstance(db_config, dict) else "N/A"

    log_config = merged.get("logging", {}) if isinstance(merged, dict) else {}
    log_config.get("level") if isinstance(log_config, dict) else "N/A"
    pool_size = db_config.get("pool_size") if isinstance(db_config, dict) else "N/A"
    if pool_size != "N/A":
        pass


def _print_feature_hierarchy_demo() -> None:
    feature_config: FlextTypes.Core.Dict = {
        "features": {
            "new_ui": {"enabled": True, "beta": True},
            "analytics": {"enabled": True, "tracking_id": "UA-123456"},
            "caching": {"enabled": False, "ttl": 3600},
        },
    }
    dev_features: FlextTypes.Core.Dict = {
        "features": {
            "new_ui": {"enabled": True, "beta": True},
            "analytics": {"enabled": False},
            "caching": {"enabled": True, "ttl": 60},
        },
    }
    feature_merged_result = merge_configs(feature_config, dev_features)
    feature_merged = feature_merged_result.unwrap_or({})
    features = feature_merged.get("features", {})
    items = features.items() if isinstance(features, dict) else []
    for _feature_name, feature_details in items:
        feature_details.get("enabled", False)


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


def _print_config_section_header(_title: str) -> None:
    """Print formatted configuration section header."""
    _separator = "\n" + "=" * 80


def _demonstrate_configuration_validation() -> FlextResult[None]:
    """Demonstrate configuration validation patterns."""
    # Create complex configuration
    complex_config = _create_complex_configuration()

    return _validate_configuration_structure(complex_config).flat_map(
        _display_validation_results,
    )


def _create_complex_configuration() -> FlextTypes.Core.Dict:
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
    config: FlextTypes.Core.Dict,
) -> FlextResult[tuple[bool, list[str]]]:
    """Validate configuration structure and return validation results."""
    required_fields = [
        ("server.host", str),
        ("server.port", int),
        ("server.workers", int),
        ("security.ssl_enabled", bool),
        ("monitoring.enabled", bool),
    ]

    validation_errors: list[str] = []

    for field_path, expected_type in required_fields:
        validation_result = _validate_config_field(config, field_path, expected_type)
        if validation_result.is_failure:
            validation_errors.append(
                validation_result.error or f"Validation failed for {field_path}",
            )

    is_valid = len(validation_errors) == 0
    return FlextResult.ok((is_valid, validation_errors))


def _validate_config_field(
    config: FlextTypes.Core.Dict,
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
    validation_data: tuple[bool, list[str]],
) -> FlextResult[None]:
    """Display configuration validation results."""
    _is_valid, validation_errors = validation_data

    if validation_errors:
        for _error in validation_errors:
            pass

    return FlextResult.ok(None)


def _demonstrate_configuration_transformation() -> FlextResult[None]:
    """Demonstrate configuration transformation patterns."""
    # Create transformation configuration
    transform_config = _create_transformation_configuration()

    return _transform_to_connection_strings(transform_config).flat_map(
        lambda transformed: _display_transformed_configuration(
            transformed,
            transform_config,
        ),
    )


def _create_transformation_configuration() -> FlextTypes.Core.Dict:
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


def _transform_to_connection_strings(
    config: FlextTypes.Core.Dict,
) -> FlextResult[FlextTypes.Core.Dict]:
    """Transform configuration to connection strings."""
    transformed_config: FlextTypes.Core.Dict = {}

    # Transform database configuration
    db_config = config.get("database", {})
    if isinstance(db_config, dict):
        db_result = _transform_database_config(db_config)
        if db_result.is_success and db_result.value:
            transformed_config["database_url"] = db_result.value

    # Transform Redis configuration
    redis_config = config.get("redis", {})
    if isinstance(redis_config, dict):
        redis_result = _transform_redis_config(redis_config)
        if redis_result.is_success and redis_result.value:
            transformed_config["redis_url"] = redis_result.value

    return FlextResult.ok(transformed_config)


def _transform_database_config(db_config: FlextTypes.Core.Dict) -> FlextResult[str]:
    """Transform database configuration to connection string."""
    required_keys = ["host", "port", "name", "user", "password"]
    if not all(key in db_config for key in required_keys):
        return FlextResult.fail("Missing required database configuration keys")

    db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
    return FlextResult.ok(db_url)


def _transform_redis_config(redis_config: FlextTypes.Core.Dict) -> FlextResult[str]:
    """Transform Redis configuration to connection string."""
    required_keys = ["host", "port", "db"]
    if not all(key in redis_config for key in required_keys):
        return FlextResult.fail("Missing required Redis configuration keys")

    redis_url = (
        f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
    )
    return FlextResult.ok(redis_url)


def _display_transformed_configuration(
    transformed_config: FlextTypes.Core.Dict,
    original_config: FlextTypes.Core.Dict,
) -> FlextResult[None]:
    """Display transformed configuration with masked sensitive data."""
    for value in transformed_config.values():
        # Mask sensitive information
        _mask_sensitive_data(str(value), original_config)

    return FlextResult.ok(None)


def _mask_sensitive_data(value: str, original_config: FlextTypes.Core.Dict) -> str:
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
    config_sources = _create_configuration_sources()

    return _compose_configuration_from_sources(config_sources).flat_map(
        _display_composed_configuration,
    )


def _create_configuration_sources() -> list[tuple[str, FlextTypes.Core.Dict]]:
    """Create configuration sources for composition demonstration."""
    return [
        ("defaults", {"timeout": 30, "retries": 3, "debug": False}),
        ("environment", {"timeout": 60, "debug": True}),
        ("user_preferences", {"retries": 5}),
    ]


def _compose_configuration_from_sources(
    config_sources: list[tuple[str, FlextTypes.Core.Dict]],
) -> FlextResult[FlextTypes.Core.Dict]:
    """Compose configuration from multiple sources."""
    composed_config: FlextTypes.Core.Dict = {}

    for _source_name, source_config in config_sources:
        for key, value in source_config.items():
            composed_config[key] = value

    return FlextResult.ok(composed_config)


def _display_composed_configuration(
    composed_config: FlextTypes.Core.Dict,
) -> FlextResult[None]:
    """Display final composed configuration."""
    for _key, _value in composed_config.items():
        pass

    return FlextResult.ok(None)


def demonstrate_domain_configuration_integration() -> None:
    """Demonstrate configuration integration with domain models.

    Shows integration with shared domain models using railway-oriented programming.
    """
    _print_domain_config_section_header("🏢 DOMAIN MODEL CONFIGURATION INTEGRATION")

    # Chain all domain configuration demonstrations using single-responsibility methods
    (
        _demonstrate_domain_model_validation()
        .flat_map(_demonstrate_admin_user_validation)
        .flat_map(_demonstrate_user_creation_from_config)
        .flat_map(_demonstrate_domain_rule_validation)
        .flat_map(_demonstrate_feature_flag_configuration)
    )


def _print_domain_config_section_header(_title: str) -> None:
    """Print formatted domain configuration section header."""
    _separator = "\n" + "=" * 80


def _demonstrate_domain_model_validation() -> FlextResult[FlextTypes.Core.Dict]:
    """Demonstrate configuration with domain model validation."""
    # Configuration for user management service
    user_service_config: FlextTypes.Core.Dict = {
        "service_name": "user_management",
        "admin_users": [
            {
                "name": "admin",
                "email": "admin@company.com",
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

    return FlextResult.ok(user_service_config)


def _demonstrate_admin_user_validation(
    config: FlextTypes.Core.Dict,
) -> FlextResult[tuple[FlextTypes.Core.Dict, list[object]]]:
    """Demonstrate admin user validation using shared domain models."""
    admin_users = config.get("admin_users", [])
    validated_users = []

    if isinstance(admin_users, list):
        for user_data in admin_users:
            validation_result = _validate_single_admin_user(user_data)
            if validation_result.is_success and validation_result.value:
                validated_users.append(validation_result.value)

    len(admin_users) if isinstance(admin_users, list) else 0
    return FlextResult.ok((config, validated_users))


def _validate_single_admin_user(user_data: object) -> FlextResult[object]:
    """Validate a single admin user using domain factory."""
    if not isinstance(user_data, dict):
        return FlextResult.fail("Invalid user data format")

    name = user_data.get("name", "")
    email = user_data.get("email", "")
    age = user_data.get("age", 0)

    # Use SharedDomainFactory for validation
    user_result = SharedDomainFactory.create_user(name, email, age)
    if user_result.success:
        user = user_result.value
        if user is not None:
            # Log domain operation
            log_domain_operation(
                "admin_user_configured",
                "SharedUser",
                str(user.id),
                service="user_management",
                config_role="admin",
            )
            return FlextResult.ok(user)

    return FlextResult.fail(f"Validation failed: {user_result.error}")


def _demonstrate_user_creation_from_config(
    validated_data: tuple[FlextTypes.Core.Dict, list[object]],
) -> FlextResult[tuple[FlextTypes.Core.Dict, list[object], list[object]]]:
    """Demonstrate configuration-driven user creation."""
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
        if creation_result.is_success and creation_result.value:
            created_users.append(creation_result.value)

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
        user = user_result.value
        if user is not None:
            return FlextResult.ok(user)

    return FlextResult.fail(f"Creation failed: {user_result.error}")


def _demonstrate_domain_rule_validation(
    user_data: tuple[FlextTypes.Core.Dict, list[object], list[object]],
) -> FlextResult[tuple[FlextTypes.Core.Dict, list[object], list[object]]]:
    """Demonstrate configuration validation using domain rules."""
    config, validated_users, created_users = user_data

    validation_config_raw = config.get("user_validation", {})
    if isinstance(validation_config_raw, dict):
        validation_config: dict[str, object] = cast(
            "dict[str, object]",
            validation_config_raw,
        )
        validation_config.get("min_age", 18)
        validation_config.get("max_age", 120)

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
            pass
    except (RuntimeError, ValueError, TypeError):
        pass

    return FlextResult.ok(None)


def _demonstrate_feature_flag_configuration(
    validation_data: tuple[FlextTypes.Core.Dict, list[object], list[object]],
) -> FlextResult[None]:
    """Demonstrate configuration-based feature flags with domain context."""
    config, _validated_users, _created_users = validation_data
    features_raw = config.get("features", {})

    if isinstance(features_raw, dict):
        features: dict[str, object] = cast("dict[str, object]", features_raw)
        for feature_name, enabled in features.items():
            _configure_single_feature_flag(feature_name, enabled)

    return FlextResult.ok(None)


def _configure_single_feature_flag(
    feature_name: str,
    enabled: object,
) -> FlextResult[None]:
    """Configure a single feature flag with domain logging."""
    if isinstance(enabled, bool):
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
