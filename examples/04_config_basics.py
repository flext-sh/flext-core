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

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextDomainService,
    FlextLogger,
    FlextResult,
)

# ========== CONFIGURATION SERVICE ==========


class ComprehensiveConfigService(FlextDomainService[dict[str, object]]):
    """Service demonstrating ALL FlextConfig patterns and methods."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute method required by FlextDomainService."""
        # This is a demonstration service, returns current config
        config = FlextConfig.get_global_instance()
        return FlextResult[dict[str, object]].ok({
            "environment": config.environment,
            "debug": config.debug,
            "log_level": config.log_level,
        })

    # ========== GLOBAL SINGLETON ACCESS ==========

    def demonstrate_global_singleton(self) -> None:
        """Show global singleton pattern."""
        print("\n=== Global Singleton Configuration ===")

        # Get global instance (thread-safe singleton)
        config = FlextConfig.get_global_instance()
        print(f"✅ Global config instance: {type(config).__name__}")

        # Access basic settings
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.debug}")
        print(f"Log level: {config.log_level}")
        print(f"App name: {config.app_name}")

        # Same instance everywhere
        config2 = FlextConfig.get_global_instance()
        print(f"Same instance: {config is config2}")

    # ========== ENVIRONMENT CONFIGURATION ==========

    def demonstrate_environment_config(self) -> None:
        """Show environment-specific configuration."""
        print("\n=== Environment Configuration ===")

        config = FlextConfig.get_global_instance()

        # Environment determines behavior
        print(f"Current environment: {config.environment}")

        # Environment checks
        print(f"Is development: {config.is_development()}")
        print(f"Is production: {config.is_production()}")

        # Environment-specific behavior
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

        config = FlextConfig.get_global_instance()

        # Logging settings
        print(f"Log level: {config.log_level}")
        print(f"JSON output: {config.json_output}")
        print(f"Include source: {config.include_source}")
        print(f"Include context: {config.include_context}")
        print(f"Include correlation ID: {config.include_correlation_id}")

        # Log file settings
        print(f"Log file: {config.log_file or 'Console only'}")
        if config.log_file:
            print(f"Log file max size: {config.log_file_max_size}")
            print(f"Log file backup count: {config.log_file_backup_count}")

        # Console settings
        print(f"Console enabled: {config.console_enabled}")
        print(f"Console colors: {config.console_color_enabled}")

        # Get logging configuration
        log_config = config.get_logging_config()
        print(f"Full logging config: {log_config}")

    # ========== DATABASE CONFIGURATION ==========

    def demonstrate_database_config(self) -> None:
        """Show database configuration."""
        print("\n=== Database Configuration ===")

        config = FlextConfig.get_global_instance()

        # Database settings
        print(f"Database URL: {config.database_url or 'Not configured'}")
        print(f"Database pool size: {config.database_pool_size}")

        # Get database configuration
        db_config = config.get_database_config()
        print(f"Database config: {db_config}")

    # ========== CACHE CONFIGURATION ==========

    def demonstrate_cache_config(self) -> None:
        """Show cache configuration."""
        print("\n=== Cache Configuration ===")

        config = FlextConfig.get_global_instance()

        # Cache settings
        print(f"Enable caching: {config.enable_caching}")
        print(f"Cache TTL: {config.cache_ttl}s")
        print(f"Cache max size: {config.cache_max_size}")

        # Get cache configuration
        cache_config = config.get_cache_config()
        print(f"Cache config: {cache_config}")

    # ========== PERFORMANCE CONFIGURATION ==========

    def demonstrate_performance_config(self) -> None:
        """Show performance configuration."""
        print("\n=== Performance Configuration ===")

        config = FlextConfig.get_global_instance()

        # Performance settings
        print(f"Max workers: {config.max_workers}")
        print(f"Max retry attempts: {config.max_retry_attempts}")
        print(f"Timeout seconds: {config.timeout_seconds}")

        # Circuit breaker
        print(f"Enable circuit breaker: {config.enable_circuit_breaker}")

        # Metrics and tracing
        print(f"Enable metrics: {config.enable_metrics}")
        print(f"Enable tracing: {config.enable_tracing}")

        # Dispatcher settings
        print(f"Dispatcher timeout: {config.dispatcher_timeout_seconds}s")
        print(f"Dispatcher logging: {config.dispatcher_enable_logging}")
        print(f"Dispatcher metrics: {config.dispatcher_enable_metrics}")
        print(f"Dispatcher auto context: {config.dispatcher_auto_context}")

    # ========== CQRS/EVENT BUS CONFIGURATION ==========

    def demonstrate_cqrs_config(self) -> None:
        """Show CQRS and event bus configuration."""
        print("\n=== CQRS/Event Bus Configuration ===")

        config = FlextConfig.get_global_instance()

        # Get CQRS bus configuration
        cqrs_config = config.get_cqrs_bus_config()
        print(f"CQRS bus config: {cqrs_config}")

        # Dispatcher settings relevant to CQRS
        print(f"Auto context propagation: {config.dispatcher_auto_context}")
        print(f"Enable logging: {config.dispatcher_enable_logging}")
        print(f"Enable metrics: {config.dispatcher_enable_metrics}")

    # ========== API/SECURITY CONFIGURATION ==========

    def demonstrate_api_security_config(self) -> None:
        """Show API and security configuration."""
        print("\n=== API/Security Configuration ===")

        config = FlextConfig.get_global_instance()

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

        config = FlextConfig.get_global_instance()

        # Get service metadata
        metadata = config.get_metadata()
        print(f"Service metadata: {metadata}")

        # Individual metadata fields
        print(f"App name: {config.app_name}")
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.debug}")

    # ========== ENVIRONMENT VARIABLES ==========

    def demonstrate_env_variables(self) -> None:
        """Show environment variable loading."""
        print("\n=== Environment Variables ===")

        # Simulate environment variables
        test_vars = {
            "FLEXT_ENVIRONMENT": "test",
            "FLEXT_DEBUG": "true",
            "FLEXT_LOG_LEVEL": "INFO",
            "FLEXT_DATABASE_URL": "postgresql://localhost/db",
            "FLEXT_MAX_WORKERS": "4",
        }

        # Show how environment variables map to config
        print("Environment variable mapping:")
        for key, value in test_vars.items():
            config_key = key.replace("FLEXT_", "").lower()
            print(f"  {key} -> config.{config_key} = {value}")

        # .env file support
        print("\n.env file support:")
        print("  - .env (default)")
        print("  - .env.development")
        print("  - .env.production")
        print("  - .env.local (gitignored)")

    # ========== CONFIGURATION VALIDATION ==========

    def demonstrate_validation(self) -> None:
        """Show configuration validation."""
        print("\n=== Configuration Validation ===")

        config = FlextConfig.get_global_instance()

        # Type validation (Pydantic handles this)
        validations = [
            (
                "environment",
                config.environment
                in {"development", "test", "staging", "production", "local"},
            ),
            (
                "log_level",
                config.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
            ),
            ("timeout_seconds", config.timeout_seconds > 0),
            ("max_workers", config.max_workers > 0),
            ("cache_ttl", config.cache_ttl >= 0),
        ]

        for field, is_valid in validations:
            status = "✅" if is_valid else "❌"
            print(
                f"{status} {field}: Valid" if is_valid else f"{status} {field}: Invalid"
            )

    # ========== DYNAMIC CONFIGURATION ==========

    def demonstrate_dynamic_config(self) -> None:
        """Show dynamic configuration patterns."""
        print("\n=== Dynamic Configuration ===")

        FlextConfig.get_global_instance()

        # FlextConfig is based on Pydantic Settings, which is immutable
        print("Note: FlextConfig is immutable after initialization")
        print("To change config, create new instance with environment variables")

        # You can create new instances for different environments
        print("\nCreating config for different environment:")
        test_config = FlextConfig.create_for_environment("test")
        print(f"Test environment config: {test_config.environment}")
        print(f"Test debug mode: {test_config.debug}")

    # ========== CONFIGURATION EXPORT ==========

    def demonstrate_export(self) -> None:
        """Show configuration export patterns."""
        print("\n=== Configuration Export ===")

        config = FlextConfig.get_global_instance()

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

        config = FlextConfig.get_global_instance()

        # Helper methods
        print("Available helper methods:")
        print(f"  - is_development(): {config.is_development()}")
        print(f"  - is_production(): {config.is_production()}")
        print("  - get_metadata(): Returns service metadata")
        print("  - get_logging_config(): Returns logging configuration")
        print("  - get_database_config(): Returns database configuration")
        print("  - get_cache_config(): Returns cache configuration")
        print("  - get_cqrs_bus_config(): Returns CQRS bus configuration")
        print(
            "  - create_for_environment(env): Creates config for specific environment"
        )

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated configuration patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Hard-coded configuration (DEPRECATED)
        warnings.warn(
            "Hard-coded configuration is DEPRECATED! Use FlextConfig.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (hard-coded):")
        print("DATABASE_HOST = 'localhost'")
        print("API_PORT = 8000")
        print("DEBUG = True")

        print("\n✅ CORRECT WAY (FlextConfig):")
        print("config = FlextConfig.get_global_instance()")
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
        print("config = FlextConfig.get_global_instance()")
        print("url = config.database_url")

        # OLD: Dictionary configuration (DEPRECATED)
        warnings.warn(
            "Dictionary-based config is DEPRECATED! Use typed FlextConfig.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (dictionary):")
        print("config = {'host': 'localhost', 'port': 8000}")
        print("host = config.get('host', 'default')")

        print("\n✅ CORRECT WAY (typed config):")
        print("config = FlextConfig.get_global_instance()")
        print("url = config.database_url  # Type-safe!")


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

    print("\n" + "=" * 60)
    print("✅ ALL FlextConfig methods demonstrated!")
    print("🎯 Next: See 05_logging_basics.py for FlextLogger patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
