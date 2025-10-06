#!/usr/bin/env python3
"""04 - FlextConfig Fundamentals: Complete Configuration Management.

This example demonstrates the COMPLETE FlextConfig API - the foundation
for configuration management across the entire FLEXT ecosystem. FlextConfig provides
environment-aware, type-safe configuration with Pydantic Settings integration.

Key Concepts Demonstrated:
- Environment Configuration: Development, Testing, Staging, Production
- Settings Management: Database, cache, API, performance settings
- Environment Variables: Auto-loading from .env files
- Configuration Validation: Type-safe with Pydantic
- Logging Configuration: JSON output, verbosity, source inclusion
- Performance Settings: Workers, timeouts, retry configuration
- Metadata Access: Service information and environment checks
- Global Singleton: Thread-safe configuration access

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from typing import cast

from flext_core import Flext

from .example_scenarios import ExampleScenarios


class ComprehensiveConfigService(Flext.Service[Flext.Types.Dict]):
    """Service demonstrating ALL FlextConfig patterns and methods."""

    def __init__(self) -> None:
        """Initialize with automatic Flext infrastructure."""
        super().__init__()
        self._scenarios = ExampleScenarios()
        self._reference_config = self._scenarios.config()
        self._production_config = self._scenarios.config(production=True)
        self._metadata = self._scenarios.metadata(tags=["config", "demo"])

    def execute(self) -> Flext.Result[Flext.Types.Dict]:
        """Execute method required by FlextService."""
        config = Flext.Config()
        return Flext.Result[Flext.Types.Dict].ok({
            "environment": config.environment,
            "debug": config.debug,
            "log_level": config.log_level,
            "reference_database": self._reference_config.get("database_url"),
        })

    # ========== GLOBAL SINGLETON ACCESS ==========

    def demonstrate_global_singleton(self) -> None:
        """Show global singleton pattern."""
        print("\n=== Global Singleton Configuration ===")

        # Get global instance (thread-safe singleton)
        config = Flext.Config()
        print(f"✅ Global config instance: {type(config).__name__}")

        # Access basic settings
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.debug}")
        print(f"Log level: {config.log_level}")
        print(f"App name: {config.app_name}")

        # Same instance everywhere
        config2 = Flext.Config()
        print(f"Same instance: {config is config2}")

    # ========== ENVIRONMENT CONFIGURATION ==========

    def demonstrate_environment_config(self) -> None:
        """Show environment-specific configuration."""
        print("\n=== Environment Configuration ===")

        config = Flext.Config()
        target_env = self._production_config.get("environment", "production")

        print(f"Current environment: {config.environment}")
        print(f"Reference production environment: {target_env}")

        print(f"Is development: {config.is_development()}")
        print(f"Is production: {config.is_production()}")

        if config.is_development():
            print("Development mode features:")
            print("  - Debug logging available")
            print("  - Detailed error messages")
            print("  - Performance profiling enabled")
        elif config.is_production():
            print("Production mode features:")
            print("  - Optimized performance")
            print("  - Secure defaults")
            print("  - Minimal logging")

    # ========== LOGGING CONFIGURATION ==========

    def demonstrate_logging_config(self) -> None:
        """Show logging configuration."""
        print("\n=== Logging Configuration ===")

        config = Flext.Config()

        print(f"Log level: {config.log_level}")
        print(f"JSON output: {config.json_output}")
        print(f"Include source: {config.include_source}")
        print(f"Log verbosity: {config.log_verbosity}")
        print(f"Include context: {config.include_context}")
        print(f"Include correlation ID: {config.include_correlation_id}")

        print(f"Scenario metadata tags: {self._metadata['tags']}")

        print(f"Log file: {config.log_file or 'Console only'}")
        if config.log_file:
            print(f"Log file max size: {config.log_file_max_size}")
            print(f"Log file backup count: {config.log_file_backup_count}")

        print(f"Console enabled: {config.console_enabled}")
        print(f"Console colors: {config.console_color_enabled}")

        # Access logging configuration directly from config attributes
        log_config = {
            "log_level": config.log_level,
            "json_output": config.json_output,
            "include_source": config.include_source,
            "log_verbosity": config.log_verbosity,
        }
        print(f"Full logging config: {log_config}")

    # ========== DATABASE CONFIGURATION ==========

    def demonstrate_database_config(self) -> None:
        """Show database configuration."""
        print("\n=== Database Configuration ===")

        config = Flext.Config()
        reference_url = self._reference_config.get("database_url", "Not configured")

        print(f"Database URL: {config.database_url or reference_url}")
        print(f"Reference URL: {reference_url}")
        print(f"Database pool size: {config.database_pool_size}")

        # Direct attribute access (get_database_config removed)
        db_config = {
            "url": config.database_url,
            "pool_size": config.database_pool_size,
        }
        print(f"Database config: {db_config}")

    # ========== CACHE CONFIGURATION ==========

    def demonstrate_cache_config(self) -> None:
        """Show cache configuration."""
        print("\n=== Cache Configuration ===")

        config = Flext.Config()
        reference_ttl = self._reference_config.get("api_timeout", 30)

        print(f"Enable caching: {config.enable_caching}")
        print(f"Cache TTL: {config.cache_ttl}s (reference {reference_ttl}s)")

    # ========== PERFORMANCE CONFIGURATION ==========

    def demonstrate_performance_config(self) -> None:
        """Show performance configuration."""
        print("\n=== Performance Configuration ===")

        config = Flext.Config()

        print(f"Max workers: {config.max_workers}")
        print(f"Timeout seconds: {config.timeout_seconds}s")
        print(f"Max retry attempts: {config.max_retry_attempts}")

        print(
            "Reference retry strategy:",
            {
                "workers": self._reference_config.get("max_connections", 10),
                "timeout": self._reference_config.get("api_timeout", 30),
            },
        )

    # ========== CQRS/EVENT BUS CONFIGURATION ==========

    def demonstrate_cqrs_config(self) -> None:
        """Show CQRS and event bus configuration."""
        print("\n=== CQRS/Event Bus Configuration ===")

        config = Flext.Config()

        # Direct attribute access (get_cqrs_bus_config removed)
        cqrs_config = {
            "auto_context": config.dispatcher_auto_context,
            "timeout_seconds": config.dispatcher_timeout_seconds,
            "enable_metrics": config.dispatcher_enable_metrics,
            "enable_logging": config.dispatcher_enable_logging,
            "log_verbosity": config.log_verbosity,
        }
        print(f"CQRS bus config: {cqrs_config}")
        print(f"CQRS log verbosity: {cqrs_config.get('log_verbosity')}")

        # Dispatcher settings relevant to CQRS
        print(f"Auto context propagation: {config.dispatcher_auto_context}")
        print(f"Enable logging: {config.dispatcher_enable_logging}")
        print(f"Enable metrics: {config.dispatcher_enable_metrics}")

    # ========== API/SECURITY CONFIGURATION ==========

    def demonstrate_api_security_config(self) -> None:
        """Show API and security configuration."""
        print("\n=== API/Security Configuration ===")

        config = Flext.Config()

        # API key for authentication
        print(f"API key configured: {'Yes' if config.api_key else 'No'}")
        if config.api_key:
            print(f"API key (masked): {'*' * 10}")

        # Security settings
        print(f"Mask sensitive data: {config.mask_sensitive_data}")

        # Environment-based security
        if config.is_production():
            print("Production security enabled:")
            print("  - Debug disabled")
            print("  - Sensitive data masked")
            print("  - Secure logging")
        elif config.is_development():
            print("Development mode - relaxed security for debugging")

    # ========== METADATA ACCESS ==========

    def demonstrate_metadata(self) -> None:
        """Show metadata access."""
        print("\n=== Metadata Access ===")

        config = Flext.Config()

        # Direct attribute access (get_metadata removed)
        metadata = {
            "app_name": config.app_name,
            "version": config.version,
            "environment": config.environment,
            "debug": config.debug,
            "trace": config.trace,
        }
        print(f"Service metadata: {metadata}")
        print(f"Scenario metadata: {self._metadata}")

        print(f"App name: {config.app_name}")
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.debug}")

    # ========== ENVIRONMENT VARIABLES ==========

    def demonstrate_env_variables(self) -> None:
        """Show environment variable loading."""
        print("\n=== Environment Variables ===")

        base_env = {
            "FLEXT_ENVIRONMENT": self._reference_config.get(
                "environment",
                "development",
            ),
            "FLEXT_DEBUG": str(self._reference_config.get("debug", False)).lower(),
            "FLEXT_LOG_LEVEL": self._reference_config.get(
                "log_level",
                Flext.Constants.Logging.INFO,
            ),
            "FLEXT_DATABASE_URL": self._reference_config.get(
                "database_url",
                "postgresql://localhost/db",
            ),
            "FLEXT_MAX_WORKERS": str(self._production_config.get("max_connections", 4)),
        }

        for key, value in base_env.items():
            print(f"{key}={value}")

    # ========== CONFIGURATION VALIDATION ==========

    def demonstrate_validation(self) -> None:
        """Show configuration validation."""
        print("\n=== Configuration Validation ===")

        config = Flext.Config()

        validations = [
            (
                "environment",
                config.environment
                in {"development", "test", "staging", "production", "local"},
            ),
            (
                "log_level",
                config.log_level in Flext.Constants.Logging.VALID_LEVELS,
            ),
            ("timeout_seconds", config.timeout_seconds > 0),
            ("max_workers", config.max_workers > 0),
            ("cache_ttl", config.cache_ttl >= 0),
            (
                "reference_database",
                bool(self._reference_config.get("database_url")),
            ),
        ]

        for field, is_valid in validations:
            status = "✅" if is_valid else "❌"
            print(
                f"{status} {field}: Valid"
                if is_valid
                else f"{status} {field}: Invalid",
            )

    # ========== DYNAMIC CONFIGURATION ==========

    def demonstrate_dynamic_config(self) -> None:
        """Show dynamic configuration patterns."""
        print("\n=== Dynamic Configuration ===")

        Flext.Config()

        print("Note: FlextConfig is immutable after initialization")
        print("To change config, create new instance with environment variables")

        print("\nCreating config for different environment:")
        test_config = Flext.Config.create_for_environment("test")
        print(f"Test environment config: {test_config.environment}")
        print(f"Test debug mode: {test_config.debug}")
        print(
            "Production baseline:",
            {
                "environment": self._production_config.get("environment", "production"),
                "debug": self._production_config.get("debug", False),
            },
        )

    # ========== CONFIGURATION EXPORT ==========

    def demonstrate_export(self) -> None:
        """Show configuration export patterns."""
        print("\n=== Configuration Export ===")

        config = Flext.Config()

        # Export as dictionary (safe, no secrets)
        config_dict = config.model_dump(exclude={"api_key"})
        print(f"Exported keys: {list(config_dict.keys())[:10]}...")

        # Export formats
        print("\nExport formats:")
        print("  - JSON: config.model_dump_json()")
        print("  - Dict: config.model_dump()")
        print("  - Pydantic: config.model_copy()")

        # Safe export (no sensitive data)
        safe_config = {
            k: v
            for k, v in config_dict.items()
            if not any(
                sensitive in k
                for sensitive in ["password", "secret", "key", "token", "api"]
            )
        }
        print(f"Safe export: {len(safe_config)} fields")

    # ========== CONFIGURATION METHODS ==========

    def demonstrate_config_methods(self) -> None:
        """Show configuration helper methods."""
        print("\n=== Configuration Methods ===")

        config = Flext.Config()

        # Helper methods
        print("Available helper methods:")
        print(f"  - is_development(): {config.is_development()}")
        print(f"  - is_production(): {config.is_production()}")
        print(
            "  - Config accessed via direct attributes (app_name, database_url, etc.)"
        )
        print(
            "  - create_for_environment(env): Creates config for specific environment",
        )

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated configuration patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Hard-coded configuration (DEPRECATED)
        warnings.warn(
            "Hard-coded configuration is DEPRECATED! Use Flext.Config.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (hard-coded):")
        print("DATABASE_HOST = 'localhost'")
        print("API_PORT = 8000")
        print("DEBUG = True")

        print("\n✅ CORRECT WAY (FlextConfig):")
        print("config = Flext.Config()")
        print("database_url = config.database_url")

        # OLD: Global variables (DEPRECATED)
        warnings.warn(
            "Global configuration variables are DEPRECATED! Use FlextConfig singleton.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (global variables):")
        print("import config")
        print("host = config.DB_HOST")

        print("\n✅ CORRECT WAY (singleton):")
        print("config = Flext.Config()")
        print("url = config.database_url")

        # OLD: Dictionary configuration (DEPRECATED)
        warnings.warn(
            "Dictionary-based config is DEPRECATED! Use typed Flext.Config.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (dictionary):")
        print("config = {'host': 'localhost', 'port': 8000}")
        print("host = config.get('host', 'default')")

        print("\n✅ CORRECT WAY (typed config):")
        print("config = Flext.Config()")
        print("url = config.database_url  # Type-safe!")


def demonstrate_flextcore_config_access() -> None:
    """Demonstrate Flext unified access to configuration.

    Shows how Flext provides convenient access to configuration
    alongside other flext-core components.
    """
    print("\n" + "=" * 60)
    print("FLEXTCORE UNIFIED CONFIG ACCESS")
    print("Modern pattern for configuration with Flext")
    print("=" * 60)

    # 1. Access config through Flext instance
    print("\n=== 1. Config Access Through Flext ===")
    core = Flext()
    config = core.config
    print(f"  ✅ Config accessed: {type(config).__name__}")
    print(f"  ✅ Environment: {config.environment}")
    print(f"  ✅ Log level: {config.log_level}")

    # 2. Direct class access for configuration
    print("\n=== 2. Direct Config Class Access ===")
    direct_config = Flext.Config()
    print(f"  ✅ Direct instantiation: {type(direct_config).__name__}")
    print(f"  ✅ Debug mode: {direct_config.debug}")

    # 3. Factory method for configuration
    print("\n=== 3. Config Factory Method ===")
    factory_config = Flext.get_config()
    print(f"  ✅ Factory config: {type(factory_config).__name__}")
    print(f"  ✅ Timeout: {factory_config.timeout_seconds}s")

    # 4. Combined config with other components
    print("\n=== 4. Config with Integrated Components ===")
    logger = core.logger
    container = core.container

    logger.info("Configuration loaded", extra={"env": config.environment})
    container.register("config", config)

    print(f"  ✅ Logger integrated: {type(logger).__name__}")
    print("  ✅ Config registered in container")
    services_result = container.list_services()
    service_count = len(services_result.unwrap()) if services_result.is_success else 0
    print(f"  ✅ Services: {service_count}")

    # 5. Infrastructure setup with configuration
    print("\n=== 5. Infrastructure with Custom Config ===")
    custom_config = Flext.Config()
    setup_result = Flext.setup_service_infrastructure(
        "config-demo-service", config=custom_config
    )

    if setup_result.is_success:
        infra = cast("Flext.Types.Dict", setup_result.unwrap())
        infra_config = cast("Flext.Config", infra["config"])

        print("  ✅ Infrastructure initialized with custom config:")
        print(f"     - Config type: {type(infra_config).__name__}")
        print(f"     - Environment: {infra_config.environment}")
        print(f"     - Same instance: {infra_config is custom_config}")

    print("\n" + "=" * 60)
    print("✅ Flext config demonstration complete!")
    print("Benefits: Unified access, lazy loading, integrated patterns")
    print("=" * 60)


def main() -> None:
    """Main entry point demonstrating all FlextConfig capabilities."""
    service = ComprehensiveConfigService()

    print("=" * 60)
    print("FLEXTCONFIG COMPLETE API DEMONSTRATION")
    print("Foundation for Configuration Management in FLEXT Ecosystem")
    print("=" * 60)

    # Core patterns
    service.demonstrate_global_singleton()
    service.demonstrate_environment_config()

    # Configuration domains
    service.demonstrate_logging_config()
    service.demonstrate_database_config()
    service.demonstrate_cache_config()
    service.demonstrate_performance_config()
    service.demonstrate_cqrs_config()
    service.demonstrate_api_security_config()

    # Advanced patterns
    service.demonstrate_metadata()
    service.demonstrate_env_variables()
    service.demonstrate_validation()
    service.demonstrate_dynamic_config()

    # Professional patterns
    service.demonstrate_export()
    service.demonstrate_config_methods()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    # Modern Flext pattern demonstration
    demonstrate_flextcore_config_access()

    print("\n" + "=" * 60)
    print("✅ ALL FlextConfig methods demonstrated!")
    print("🎯 Next: See 05_logging_basics.py for FlextLogger patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
