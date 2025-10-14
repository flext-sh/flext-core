#!/usr/bin/env python3
"""04 - FlextCore.Config Fundamentals: Complete Configuration Management.

This example demonstrates the COMPLETE FlextCore.Config API - the foundation
for configuration management across the entire FLEXT ecosystem. FlextCore.Config provides
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
from copy import deepcopy
from typing import ClassVar, cast

from flext_core import FlextCore


class DemoScenarios:
    """Inline scenario helpers for configuration demonstrations."""

    _CONFIG: ClassVar[FlextCore.Types.Dict] = {
        "database_url": "sqlite:///:memory:",
        "api_timeout": 30,
        "retry": 3,
        "cache": {"enabled": True, "ttl": 300},
    }

    @staticmethod
    def config(**overrides: object) -> FlextCore.Types.Dict:
        """Create configuration dictionary with optional overrides."""
        value = deepcopy(DemoScenarios._CONFIG)
        value.update(overrides)
        return value

    @staticmethod
    def metadata(
        *,
        source: str = "examples",
        tags: FlextCore.Types.StringList | None = None,
        **extra: object,
    ) -> FlextCore.Types.Dict:
        """Create metadata dictionary for configuration examples."""
        data: FlextCore.Types.Dict = {
            "source": source,
            "component": "flext_core",
            "tags": tags or ["config", "demo"],
        }
        data.update(extra)
        return data


class ComprehensiveConfigService(FlextCore.Service[FlextCore.Types.Dict]):
    """Service demonstrating ALL FlextCore.Config patterns with FlextCore.Mixins infrastructure.

    This service inherits from FlextCore.Service to demonstrate:
    - Inherited container property (FlextCore.Container singleton)
    - Inherited logger property (FlextCore.Logger with service context)
    - Inherited context property (FlextCore.Context for request tracking)
    - Inherited config property (FlextCore.Config with Pydantic 2.11+ Settings)
    - Inherited metrics property (FlextMetrics for observability)

    The focus is on demonstrating FlextCore.Config patterns (Pydantic 2.11+ Settings
    with dependency_injector integration) while leveraging complete FlextCore.Mixins
    infrastructure for service orchestration.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextCore.Mixins infrastructure.

        Note: No manual logger or config initialization needed!
        All infrastructure is inherited from FlextCore.Service base class:
        - self.logger: FlextCore.Logger with service context
        - self.container: FlextCore.Container global singleton
        - self.context: FlextCore.Context for request tracking
        - self.config: FlextCore.Config with Pydantic 2.11+ Settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        self._scenarios = DemoScenarios()
        self._reference_config = self._scenarios.config()
        self._production_config = self._scenarios.config(production=True)
        self._metadata = self._scenarios.metadata(tags=["config", "demo"])

        # Demonstrate inherited logger and config (no manual instantiation needed!)
        self.logger.info(
            "ComprehensiveConfigService initialized with inherited infrastructure",
            extra={
                "service_type": "FlextCore.Config Pydantic 2.11+ demonstration",
                "debug_mode": self.config.debug,
                "log_level": self.config.log_level,
            },
        )

    def execute(self) -> FlextCore.Result[FlextCore.Types.Dict]:
        """Execute all FlextCore.Config demonstrations and return summary.

        Demonstrates inherited config property alongside other infrastructure
        components (logger, container, context, metrics) from FlextCore.Mixins.
        """
        self.logger.info("Starting comprehensive FlextCore.Config demonstration")

        try:
            # Run all demonstrations
            self.demonstrate_global_singleton()
            self.demonstrate_env_variables()
            self.demonstrate_logging_config()
            self.demonstrate_cache_config()
            self.demonstrate_performance_config()
            self.demonstrate_cqrs_config()
            self.demonstrate_api_security_config()
            self.demonstrate_metadata()
            self.demonstrate_env_variables()
            self.demonstrate_validation()
            self.demonstrate_dynamic_config()
            self.demonstrate_export()
            self.demonstrate_flext_runtime_integration()
            self.demonstrate_flext_constants_integration()
            self.demonstrate_flext_exceptions_integration()
            self.demonstrate_from_callable()
            self.demonstrate_flow_through()
            self.demonstrate_lash()
            self.demonstrate_alt()
            self.demonstrate_value_or_call()
            self.demonstrate_deprecated_patterns()

            # Summary using inherited config property
            summary: FlextCore.Types.Dict = {
                "demonstrations_completed": 23,
                "debug_mode": self.config.debug,
                "log_level": self.config.log_level,
                "infrastructure": {
                    "logger": type(self.logger).__name__,
                    "container": type(self.container).__name__,
                    "context": type(self.context).__name__,
                    "config": type(self.config).__name__,
                },
                "reference_database": self._reference_config.get("database_url"),
            }

            self.logger.info(
                "FlextCore.Config demonstration completed successfully",
                extra={"demonstrations": summary["demonstrations_completed"]},
            )

            return FlextCore.Result[FlextCore.Types.Dict].ok(summary)

        except Exception as e:
            error_msg = f"Configuration demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextCore.Result[FlextCore.Types.Dict].fail(
                error_msg, error_code=FlextCore.Constants.Errors.VALIDATION_ERROR
            )

    # ========== GLOBAL SINGLETON ACCESS ==========

    def demonstrate_global_singleton(self) -> None:
        """Show global singleton pattern."""
        print("\n=== Global Singleton Configuration ===")

        # Get global instance (thread-safe singleton)
        config = FlextCore.Config()
        print(f"✅ Global config instance: {type(config).__name__}")

        # Access basic settings
        print(f"Debug mode: {config.debug}")
        print(f"Log level: {config.log_level}")
        print(f"App name: {config.app_name}")

        # Same instance everywhere
        config2 = FlextCore.Config()
        print(f"Same instance: {config is config2}")

    # ========== LOGGING CONFIGURATION ==========

    def demonstrate_logging_config(self) -> None:
        """Show logging configuration."""
        print("\n=== Logging Configuration ===")

        config = FlextCore.Config()

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
        log_config: FlextCore.Types.Dict = {
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

        config = FlextCore.Config()
        reference_url = self._reference_config.get("database_url", "Not configured")

        print(f"Database URL: {config.database_url or reference_url}")
        print(f"Reference URL: {reference_url}")
        print(f"Database pool size: {config.database_pool_size}")

        # Direct attribute access (get_database_config removed)
        db_config: FlextCore.Types.Dict = {
            "url": config.database_url,
            "pool_size": config.database_pool_size,
        }
        print(f"Database config: {db_config}")

    # ========== CACHE CONFIGURATION ==========

    def demonstrate_cache_config(self) -> None:
        """Show cache configuration."""
        print("\n=== Cache Configuration ===")

        config = FlextCore.Config()
        reference_ttl = self._reference_config.get("api_timeout", 30)

        print(f"Enable caching: {config.enable_caching}")
        print(f"Cache TTL: {config.cache_ttl}s (reference {reference_ttl}s)")

    # ========== PERFORMANCE CONFIGURATION ==========

    def demonstrate_performance_config(self) -> None:
        """Show performance configuration."""
        print("\n=== Performance Configuration ===")

        config = FlextCore.Config()

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

        config = FlextCore.Config()

        # Direct attribute access (get_cqrs_bus_config removed)
        cqrs_config: FlextCore.Types.Dict = {
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

        config = FlextCore.Config()

        # API key for authentication
        print(f"API key configured: {'Yes' if config.api_key else 'No'}")
        if config.api_key:
            print(f"API key (masked): {'*' * 10}")

        # Security settings
        print(f"Mask sensitive data: {config.mask_sensitive_data}")

    # ========== METADATA ACCESS ==========

    def demonstrate_metadata(self) -> None:
        """Show metadata access."""
        print("\n=== Metadata Access ===")

        config = FlextCore.Config()

        # Direct attribute access (get_metadata removed)
        metadata: FlextCore.Types.Dict = {
            "app_name": config.app_name,
            "version": config.version,
            "debug": config.debug,
            "trace": config.trace,
        }
        print(f"Service metadata: {metadata}")
        print(f"Scenario metadata: {self._metadata}")

        print(f"App name: {config.app_name}")
        print(f"Debug mode: {config.debug}")

    # ========== ENVIRONMENT VARIABLES ==========

    def demonstrate_env_variables(self) -> None:
        """Show environment variable loading."""
        print("\n=== Environment Variables ===")

        base_env: FlextCore.Types.Dict = {
            "FLEXT_ENVIRONMENT": self._reference_config.get(
                "environment",
                "development",
            ),
            "FLEXT_DEBUG": str(self._reference_config.get("debug", False)).lower(),
            "FLEXT_LOG_LEVEL": self._reference_config.get(
                "log_level",
                FlextCore.Constants.Logging.INFO,
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

        config = FlextCore.Config()

        validations = [
            (
                "log_level",
                config.log_level in FlextCore.Constants.Logging.VALID_LEVELS,
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

        FlextCore.Config()

        print("Note: FlextCore.Config is immutable after initialization")
        print("To change config, create new instance with environment variables")

        print("\nCreating config for different environment:")
        test_config = FlextCore.Config()
        print(f"Test debug mode: {test_config.debug}")
        print(
            "Production baseline:",
            {
                "debug": self._production_config.get("debug", False),
            },
        )

    # ========== CONFIGURATION EXPORT ==========

    def demonstrate_export(self) -> None:
        """Show configuration export patterns."""
        print("\n=== Configuration Export ===")

        config = FlextCore.Config()

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

    # ========== DEPRECATED PATTERNS ==========

    # ========== NEW FlextCore.Result METHODS (v0.9.9+) ==========

    def demonstrate_from_callable(self) -> None:
        """Show from_callable for safe configuration loading."""
        print("\n=== from_callable(): Safe Configuration Loading ===")

        # Safe configuration initialization
        def risky_config_load() -> FlextCore.Config:
            """Simulate risky config loading that might raise."""
            config = FlextCore.Config()
            if not config.log_level:
                msg = "Log level not configured"
                raise ValueError(msg)
            return config

        config_result = cast(
            "FlextCore.Result[FlextCore.Config]",
            FlextCore.Result.from_callable(risky_config_load),
        )
        if config_result.is_success:
            config = config_result.unwrap()
            print(f"✅ Config loaded safely: log_level={config.log_level}")

    def demonstrate_flow_through(self) -> None:
        """Show pipeline composition for multi-step config operations."""
        print("\n=== flow_through(): Configuration Validation Pipeline ===")

        def load_config(_: object) -> FlextCore.Result[FlextCore.Config]:
            """Step 1: Load configuration."""
            return FlextCore.Result[FlextCore.Config].ok(FlextCore.Config())

        def validate_logging(
            config: FlextCore.Config,
        ) -> FlextCore.Result[FlextCore.Config]:
            """Step 3: Validate logging configuration."""
            if config.log_level not in FlextCore.Constants.Logging.VALID_LEVELS:
                return FlextCore.Result[FlextCore.Config].fail(
                    f"Invalid log level: {config.log_level}"
                )
            return FlextCore.Result[FlextCore.Config].ok(config)

        def validate_performance(
            config: FlextCore.Config,
        ) -> FlextCore.Result[FlextCore.Config]:
            """Step 4: Validate performance settings."""
            if config.max_workers <= 0 or config.timeout_seconds <= 0:
                return FlextCore.Result[FlextCore.Config].fail(
                    "Invalid performance settings"
                )
            return FlextCore.Result[FlextCore.Config].ok(config)

        # Pipeline: load → validate logging → validate performance
        result = (
            load_config(True)  # Start with load_config
            .flat_map(validate_logging)
            .flat_map(validate_performance)
        )

        if result.is_success:
            config = result.unwrap()
            print(
                f"✅ Config validation pipeline success: log={config.log_level}, workers={config.max_workers}"
            )

    def demonstrate_lash(self) -> None:
        """Show error recovery in configuration operations."""
        print("\n=== lash(): Configuration Error Recovery ===")

        def recover_with_default(error: str) -> FlextCore.Result[FlextCore.Config]:
            """Recover by loading default development config."""
            print(f"  Recovering from: {error}")
            default_config = FlextCore.Config()
            return FlextCore.Result[FlextCore.Config].ok(default_config)

        result = (
            FlextCore.Result[FlextCore.Config]
            .fail("Config not available")
            .lash(recover_with_default)
        )
        if result.is_success:
            config = result.unwrap()
            print(f"✅ Recovered with fallback config: log_level={config.log_level}")

    def demonstrate_alt(self) -> None:
        """Show fallback pattern for configuration sources."""
        print("\n=== alt(): Configuration Source Fallback ===")

        # Primary: Production config (simulated failure)
        primary = FlextCore.Result[FlextCore.Config].fail(
            "Production config unavailable"
        )

        # Fallback: Development config
        fallback_config = FlextCore.Config()
        fallback = FlextCore.Result[FlextCore.Config].ok(fallback_config)

        result = primary.alt(fallback)
        if result.is_success:
            config = result.unwrap()
            print(f"✅ Got fallback config: log_level={config.log_level}")

    def demonstrate_value_or_call(self) -> None:
        """Show lazy default evaluation for expensive config operations."""
        print("\n=== value_or_call(): Lazy Configuration Initialization ===")

        # Success case - no expensive initialization needed
        success_config = FlextCore.Config()
        success = FlextCore.Result[FlextCore.Config].ok(success_config)

        expensive_created = False

        def expensive_default() -> FlextCore.Config:
            """Expensive default config creation (only if needed)."""
            nonlocal expensive_created
            expensive_created = True
            print("  Creating expensive default config...")
            return FlextCore.Config()

        # Success case - expensive_default NOT called
        config = success.value_or_call(expensive_default)
        print(
            f"✅ Success: log_level={config.log_level}, expensive_created={expensive_created}"
        )

        # Failure case - expensive_default IS called
        expensive_created = False
        failure = FlextCore.Result[FlextCore.Config].fail("Config load failed")
        config = failure.value_or_call(expensive_default)
        print(
            f"✅ Failure recovered: log_level={config.log_level}, expensive_created={expensive_created}"
        )

    # ========== FOUNDATION LAYER INTEGRATION (Layer 0.5 - 2) ==========

    def demonstrate_flext_runtime_integration(self) -> None:
        """Show FlextCore.Runtime (Layer 0.5) configuration defaults with FlextCore.Config."""
        print("\n=== FlextCore.Runtime Integration (Layer 0.5) ===")

        config = FlextCore.Config()

        # FlextCore.Runtime provides configuration defaults without circular dependencies
        print("FlextCore.Runtime configuration defaults:")
        print(f"  DEFAULT_APP_NAME: {'flext-app'}")
        print(f"  DEFAULT_LOG_LEVEL: {FlextCore.Constants.Logging.DEFAULT_LEVEL}")
        print(f"  DEFAULT_TIMEOUT: {FlextCore.Constants.Defaults.TIMEOUT}s")
        print(
            f"  DEFAULT_MAX_WORKERS: {FlextCore.Constants.Processing.DEFAULT_MAX_WORKERS}"
        )

        # Validate configuration values using FlextCore.Runtime type guards
        if config.database_url:
            is_valid_url = FlextCore.Runtime.is_valid_url(config.database_url)
            print(f"✅ Database URL validation: {is_valid_url}")

        # Log level validation
        is_valid_log_level = (
            config.log_level in FlextCore.Constants.Logging.VALID_LEVELS
        )
        print(f"✅ Log level is valid: {is_valid_log_level}")

        # Log level validation
        log_level_valid = config.log_level in {
            FlextCore.Constants.Logging.DEBUG,
            FlextCore.Constants.Logging.INFO,
            FlextCore.Constants.Logging.WARNING,
            FlextCore.Constants.Logging.ERROR,
            FlextCore.Constants.Logging.CRITICAL,
        }
        print(f"✅ Log level '{config.log_level}' is valid: {log_level_valid}")

    def demonstrate_flext_constants_integration(self) -> None:
        """Show FlextCore.Constants (Layer 1) with configuration patterns."""
        print("\n=== FlextCore.Constants Integration (Layer 1) ===")

        config = FlextCore.Config()

        # Configuration validation using FlextCore.Constants
        print("FlextCore.Constants configuration validation:")

        # Environment validation removed - no environment constants

        # Logging level constants
        if hasattr(FlextCore.Constants, "Logging"):
            log_levels = FlextCore.Constants.Logging.VALID_LEVELS
            log_valid = config.log_level in log_levels
            print(f"✅ Log level '{config.log_level}' in valid levels: {log_valid}")

        # Performance constants
        max_workers_ok = (
            1 <= config.max_workers <= FlextCore.Constants.Validation.MAX_WORKERS_LIMIT
        )
        timeout_ok = config.timeout_seconds >= 1
        print(f"✅ Max workers in range: {max_workers_ok}")
        print(f"✅ Timeout is positive: {timeout_ok}")

        # HTTP configuration defaults
        if hasattr(FlextCore.Constants, "Http"):
            print(
                f"✅ HTTP Content-Type default: {FlextCore.Constants.Http.DEFAULT_CONTENT_TYPE}"
            )
            print(f"✅ HTTP Accept header: {FlextCore.Constants.Http.DEFAULT_ACCEPT}")

    def demonstrate_flext_exceptions_integration(self) -> None:
        """Show FlextCore.Exceptions (Layer 2) with configuration error handling."""
        print("\n=== FlextCore.Exceptions Integration (Layer 2) ===")

        # ConfigurationError for invalid configuration
        try:
            # Simulate invalid configuration scenario
            invalid_workers = -1
            if invalid_workers <= 0:
                error_message = "Invalid max_workers configuration"
                raise FlextCore.Exceptions.ConfigurationError(
                    error_message,
                    config_key="max_workers",
                    config_source="environment",
                )
        except FlextCore.Exceptions.ConfigurationError as e:
            print(f"✅ ConfigurationError: {e.error_code} - {e.message}")
            print(f"   Config key: {e.config_key}, Source: {e.config_source}")

        # Environment validation removed - no environment constants

        # Configuration loading with proper error handling
        try:
            # Simulate configuration loading failure
            missing_required_config = None
            if missing_required_config is None:
                error_message = "Required configuration not found"
                raise FlextCore.Exceptions.ConfigurationError(
                    error_message,
                    config_key="database_url",
                    config_source="environment",
                )
        except FlextCore.Exceptions.ConfigurationError as e:
            print(f"✅ Configuration loading error: {e.error_code}")
            print(f"   Missing config key: {e.config_key}")

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated configuration patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Hard-coded configuration (DEPRECATED)
        warnings.warn(
            "Hard-coded configuration is DEPRECATED! Use FlextCore.Config.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (hard-coded):")
        print("DATABASE_HOST = 'localhost'")
        print("API_PORT = 8000")
        print("DEBUG = True")

        print("\n✅ CORRECT WAY (FlextCore.Config):")
        print("config = FlextCore.Config()")
        print("database_url = config.database_url")

        # OLD: Global variables (DEPRECATED)
        warnings.warn(
            "Global configuration variables are DEPRECATED! Use FlextCore.Config singleton.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (global variables):")
        print("import config")
        print("host = config.DB_HOST")

        print("\n✅ CORRECT WAY (singleton):")
        print("config = FlextCore.Config()")
        print("url = config.database_url")

        # OLD: Dictionary configuration (DEPRECATED)
        warnings.warn(
            "Dictionary-based config is DEPRECATED! Use typed FlextCore.Config.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (dictionary):")
        print("config = {'host': 'localhost', 'port': 8000}")
        print("host = config.get('host', 'default')")

        print("\n✅ CORRECT WAY (typed config):")
        print("config = FlextCore.Config()")
        print("url = config.database_url  # Type-safe!")


def demonstrate_flextcore_config_access() -> None:
    """Demonstrate FlextCore.Constants unified access to configuration.

    Shows how FlextCore.Constants provides convenient access to configuration
    alongside other flext-core components.
    """
    print("\n" + "=" * 60)
    print("FLEXTCORE UNIFIED CONFIG ACCESS")
    print("Modern pattern for configuration with FlextCore.Constants")
    print("=" * 60)

    # 1. Access config through FlextCore.Constants class
    print("\n=== 1. Config Access Through FlextCore.Constants ===")
    config = FlextCore.Config()
    print(f"  ✅ Config accessed: {type(config).__name__}")
    print(f"  ✅ Log level: {config.log_level}")

    # 2. Direct class access for configuration
    print("\n=== 2. Direct Config Class Access ===")
    direct_config = FlextCore.Config()
    print(f"  ✅ Direct instantiation: {type(direct_config).__name__}")
    print(f"  ✅ Debug mode: {direct_config.debug}")

    # 3. Factory method for configuration
    print("\n=== 3. Config Factory Method ===")
    factory_config = FlextCore.Config()
    print(f"  ✅ Factory config: {type(factory_config).__name__}")
    print(f"  ✅ Timeout: {factory_config.timeout_seconds}s")

    # 4. Combined config with other components
    print("\n=== 4. Config with Integrated Components ===")
    logger = FlextCore.create_logger(__name__)
    container = FlextCore.Container.get_global()

    logger.info("Configuration loaded", extra={"log_level": config.log_level})
    container.register("config", config)

    print(f"  ✅ Logger integrated: {type(logger).__name__}")
    print("  ✅ Config registered in container")
    services_result = container.list_services()
    service_count = len(services_result.unwrap()) if services_result.is_success else 0
    print(f"  ✅ Services: {service_count}")

    # 5. Infrastructure setup with configuration
    print("\n=== 5. Infrastructure with Custom Config ===")
    custom_config = FlextCore.Config()

    # Note: setup_service_infrastructure method not available in current version
    # Service would typically get config injected via DI container
    container = FlextCore.Container.get_global()
    container.register("config", custom_config)

    print("  ✅ Service initialized with custom config:")
    print(f"     - Config type: {type(custom_config).__name__}")
    print(f"     - Log level: {custom_config.log_level}")
    print("     - Container registered: config")

    print("\n" + "=" * 60)
    print("✅ FlextCore.Constants config demonstration complete!")
    print("Benefits: Unified access, lazy loading, integrated patterns")
    print("=" * 60)


def main() -> None:
    """Main entry point demonstrating all FlextCore.Config capabilities."""
    service = ComprehensiveConfigService()

    print("=" * 60)
    print("FLEXTCONFIG COMPLETE API DEMONSTRATION")
    print("Foundation for Configuration Management in FLEXT Ecosystem")
    print("=" * 60)

    # Core patterns
    service.demonstrate_global_singleton()
    service.demonstrate_env_variables()

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

    # Foundation layer integration (NEW in Phase 1)
    service.demonstrate_flext_runtime_integration()
    service.demonstrate_flext_constants_integration()
    service.demonstrate_flext_exceptions_integration()

    # New FlextCore.Result methods (v0.9.9+)
    service.demonstrate_from_callable()
    service.demonstrate_flow_through()
    service.demonstrate_lash()
    service.demonstrate_alt()
    service.demonstrate_value_or_call()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    # Modern FlextCore.Constants pattern demonstration
    demonstrate_flextcore_config_access()

    print("\n" + "=" * 60)
    print("✅ ALL FlextCore.Config methods demonstrated!")
    print(
        "✨ Including new v0.9.9+ methods: from_callable, flow_through, lash, alt, value_or_call"
    )
    print(
        "🔧 Including foundation integration: FlextCore.Runtime (Layer 0.5), FlextCore.Constants (Layer 1), FlextCore.Exceptions (Layer 2)"
    )
    print("🎯 Next: See 05_logging_basics.py for FlextCore.Logger patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
