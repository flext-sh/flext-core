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
from copy import deepcopy
from typing import ClassVar, cast

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextExceptions,
    FlextLogger,
    FlextResult,
    FlextRuntime,
    FlextService,
)


class DemoScenarios:
    """Inline scenario helpers for configuration demonstrations."""

    _CONFIG: ClassVar[dict[str, object]] = {
        "database_url": "sqlite:///:memory:",
        "api_timeout": 30,
        "retry": 3,
        "cache": {"enabled": True, "ttl": 300},
    }

    @staticmethod
    def config(**overrides: object) -> dict[str, object]:
        """Create configuration dictionary with optional overrides."""
        value = deepcopy(DemoScenarios._CONFIG)
        value.update(overrides)
        return value

    @staticmethod
    def metadata(
        *,
        source: str = "examples",
        tags: list[str] | None = None,
        **extra: object,
    ) -> dict[str, object]:
        """Create metadata dictionary for configuration examples."""
        data: dict[str, object] = {
            "source": source,
            "component": "flext_core",
            "tags": tags or ["config", "demo"],
        }
        data.update(extra)
        return data


class ComprehensiveConfigService(FlextService[dict[str, object]]):
    """Service demonstrating ALL FlextConfig patterns with FlextMixins infrastructure.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextContainer singleton)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextContext for request tracking)
    - Inherited config property (FlextConfig with Pydantic 2.11+ Settings)
    - Inherited metrics property (FlextMetrics for observability)

    The focus is on demonstrating FlextConfig patterns (Pydantic 2.11+ Settings
    with dependency_injector integration) while leveraging complete FlextMixins
    infrastructure for service orchestration.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins infrastructure.

        Note: No manual logger or config initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context
        - self.container: FlextContainer global singleton
        - self.context: FlextContext for request tracking
        - self.config: FlextConfig with Pydantic 2.11+ Settings
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
                "service_type": "FlextConfig Pydantic 2.11+ demonstration",
                "debug_mode": self.config.debug,
                "log_level": self.config.log_level,
            },
        )

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute all FlextConfig demonstrations and return summary.

        Demonstrates inherited config property alongside other infrastructure
        components (logger, container, context, metrics) from FlextMixins.
        """
        self.logger.info("Starting comprehensive FlextConfig demonstration")

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
            summary: dict[str, object] = {
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
                "FlextConfig demonstration completed successfully",
                extra={"demonstrations": summary["demonstrations_completed"]},
            )

            return FlextResult[dict[str, object]].ok(summary)

        except Exception as e:
            error_msg = f"Configuration demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextResult[dict[str, object]].fail(
                error_msg, error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

    # ========== GLOBAL SINGLETON ACCESS ==========

    def demonstrate_global_singleton(self) -> None:
        """Show global singleton pattern."""
        print("\n=== Global Singleton Configuration ===")

        # Get global instance (thread-safe singleton)
        config = FlextConfig()
        print(f"✅ Global config instance: {type(config).__name__}")

        # Access basic settings
        print(f"Debug mode: {config.debug}")
        print(f"Log level: {config.log_level}")
        print(f"App name: {config.app_name}")

        # Same instance everywhere
        config2 = FlextConfig()
        print(f"Same instance: {config is config2}")

    # ========== LOGGING CONFIGURATION ==========

    def demonstrate_logging_config(self) -> None:
        """Show logging configuration."""
        print("\n=== Logging Configuration ===")

        config = FlextConfig()

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
        log_config: dict[str, object] = {
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

        config = FlextConfig()
        reference_url = self._reference_config.get("database_url", "Not configured")

        print(f"Database URL: {config.database_url or reference_url}")
        print(f"Reference URL: {reference_url}")
        print(f"Database pool size: {config.database_pool_size}")

        # Direct attribute access (get_database_config removed)
        db_config: dict[str, object] = {
            "url": config.database_url,
            "pool_size": config.database_pool_size,
        }
        print(f"Database config: {db_config}")

    # ========== CACHE CONFIGURATION ==========

    def demonstrate_cache_config(self) -> None:
        """Show cache configuration."""
        print("\n=== Cache Configuration ===")

        config = FlextConfig()
        reference_ttl = self._reference_config.get("api_timeout", 30)

        print(f"Enable caching: {config.enable_caching}")
        print(f"Cache TTL: {config.cache_ttl}s (reference {reference_ttl}s)")

    # ========== PERFORMANCE CONFIGURATION ==========

    def demonstrate_performance_config(self) -> None:
        """Show performance configuration."""
        print("\n=== Performance Configuration ===")

        config = FlextConfig()

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

        config = FlextConfig()

        # Direct attribute access (get_cqrs_bus_config removed)
        cqrs_config: dict[str, object] = {
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

        config = FlextConfig()

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

        config = FlextConfig()

        # Direct attribute access (get_metadata removed)
        metadata: dict[str, object] = {
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

        base_env: dict[str, object] = {
            "FLEXT_ENVIRONMENT": self._reference_config.get(
                "environment",
                "development",
            ),
            "FLEXT_DEBUG": str(self._reference_config.get("debug", False)).lower(),
            "FLEXT_LOG_LEVEL": self._reference_config.get(
                "log_level",
                FlextConstants.Logging.INFO,
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

        config = FlextConfig()

        validations = [
            (
                "log_level",
                config.log_level in FlextConstants.Logging.VALID_LEVELS,
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

        FlextConfig()

        print("Note: FlextConfig is immutable after initialization")
        print("To change config, create new instance with environment variables")

        print("\nCreating config for different environment:")
        test_config = FlextConfig()
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

        config = FlextConfig()

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

    # ========== NEW FlextResult METHODS (v0.9.9+) ==========

    def demonstrate_from_callable(self) -> None:
        """Show from_callable for safe configuration loading."""
        print("\n=== from_callable(): Safe Configuration Loading ===")

        # Safe configuration initialization
        def risky_config_load() -> FlextConfig:
            """Simulate risky config loading that might raise."""
            config = FlextConfig()
            if not config.log_level:
                msg = "Log level not configured"
                raise ValueError(msg)
            return config

        config_result = cast(
            "FlextResult[FlextConfig]",
            FlextResult.from_callable(risky_config_load),
        )
        if config_result.is_success:
            config = config_result.unwrap()
            print(f"✅ Config loaded safely: log_level={config.log_level}")

    def demonstrate_flow_through(self) -> None:
        """Show pipeline composition for multi-step config operations."""
        print("\n=== flow_through(): Configuration Validation Pipeline ===")

        def load_config(_: object) -> FlextResult[FlextConfig]:
            """Step 1: Load configuration."""
            return FlextResult[FlextConfig].ok(FlextConfig())

        def validate_logging(
            config: FlextConfig,
        ) -> FlextResult[FlextConfig]:
            """Step 3: Validate logging configuration."""
            if config.log_level not in FlextConstants.Logging.VALID_LEVELS:
                return FlextResult[FlextConfig].fail(
                    f"Invalid log level: {config.log_level}"
                )
            return FlextResult[FlextConfig].ok(config)

        def validate_performance(
            config: FlextConfig,
        ) -> FlextResult[FlextConfig]:
            """Step 4: Validate performance settings."""
            if config.max_workers <= 0 or config.timeout_seconds <= 0:
                return FlextResult[FlextConfig].fail("Invalid performance settings")
            return FlextResult[FlextConfig].ok(config)

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

        def recover_with_default(error: str) -> FlextResult[FlextConfig]:
            """Recover by loading default development config."""
            print(f"  Recovering from: {error}")
            default_config = FlextConfig()
            return FlextResult[FlextConfig].ok(default_config)

        result = (
            FlextResult[FlextConfig]
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
        primary = FlextResult[FlextConfig].fail("Production config unavailable")

        # Fallback: Development config
        fallback_config = FlextConfig()
        fallback = FlextResult[FlextConfig].ok(fallback_config)

        result = primary.alt(fallback)
        if result.is_success:
            config = result.unwrap()
            print(f"✅ Got fallback config: log_level={config.log_level}")

    def demonstrate_value_or_call(self) -> None:
        """Show lazy default evaluation for expensive config operations."""
        print("\n=== value_or_call(): Lazy Configuration Initialization ===")

        # Success case - no expensive initialization needed
        success_config = FlextConfig()
        success = FlextResult[FlextConfig].ok(success_config)

        expensive_created = False

        def expensive_default() -> FlextConfig:
            """Expensive default config creation (only if needed)."""
            nonlocal expensive_created
            expensive_created = True
            print("  Creating expensive default config...")
            return FlextConfig()

        # Success case - expensive_default NOT called
        config = success.value_or_call(expensive_default)
        print(
            f"✅ Success: log_level={config.log_level}, expensive_created={expensive_created}"
        )

        # Failure case - expensive_default IS called
        expensive_created = False
        failure = FlextResult[FlextConfig].fail("Config load failed")
        config = failure.value_or_call(expensive_default)
        print(
            f"✅ Failure recovered: log_level={config.log_level}, expensive_created={expensive_created}"
        )

    # ========== FOUNDATION LAYER INTEGRATION (Layer 0.5 - 2) ==========

    def demonstrate_flext_runtime_integration(self) -> None:
        """Show FlextRuntime (Layer 0.5) configuration defaults with FlextConfig."""
        print("\n=== FlextRuntime Integration (Layer 0.5) ===")

        config = FlextConfig()

        # FlextRuntime provides configuration defaults without circular dependencies
        print("FlextRuntime configuration defaults:")
        print(f"  DEFAULT_APP_NAME: {'flext-app'}")
        print(f"  DEFAULT_LOG_LEVEL: {FlextConstants.Logging.DEFAULT_LEVEL}")
        print(f"  DEFAULT_TIMEOUT: {FlextConstants.Defaults.TIMEOUT}s")
        print(f"  DEFAULT_MAX_WORKERS: {FlextConstants.Processing.DEFAULT_MAX_WORKERS}")

        # Validate configuration values using FlextRuntime type guards
        if config.database_url:
            is_valid_url = FlextRuntime.is_valid_url(config.database_url)
            print(f"✅ Database URL validation: {is_valid_url}")

        # Log level validation
        is_valid_log_level = config.log_level in FlextConstants.Logging.VALID_LEVELS
        print(f"✅ Log level is valid: {is_valid_log_level}")

        # Log level validation
        log_level_valid = config.log_level in {
            FlextConstants.Logging.DEBUG,
            FlextConstants.Logging.INFO,
            FlextConstants.Logging.WARNING,
            FlextConstants.Logging.ERROR,
            FlextConstants.Logging.CRITICAL,
        }
        print(f"✅ Log level '{config.log_level}' is valid: {log_level_valid}")

    def demonstrate_flext_constants_integration(self) -> None:
        """Show FlextConstants (Layer 1) with configuration patterns."""
        print("\n=== FlextConstants Integration (Layer 1) ===")

        config = FlextConfig()

        # Configuration validation using FlextConstants
        print("FlextConstants configuration validation:")

        # Environment validation removed - no environment constants

        # Logging level constants
        if hasattr(FlextConstants, "Logging"):
            log_levels = FlextConstants.Logging.VALID_LEVELS
            log_valid = config.log_level in log_levels
            print(f"✅ Log level '{config.log_level}' in valid levels: {log_valid}")

        # Performance constants
        max_workers_ok = (
            1 <= config.max_workers <= FlextConstants.Validation.MAX_WORKERS_LIMIT
        )
        timeout_ok = config.timeout_seconds >= 1
        print(f"✅ Max workers in range: {max_workers_ok}")
        print(f"✅ Timeout is positive: {timeout_ok}")

        # HTTP configuration defaults
        if hasattr(FlextConstants, "FlextWeb"):
            print(f"✅ HTTP Status Min: {FlextConstants.FlextWeb.HTTP_STATUS_MIN}")
            print(f"✅ HTTP Status Max: {FlextConstants.FlextWeb.HTTP_STATUS_MAX}")

    def demonstrate_flext_exceptions_integration(self) -> None:
        """Show FlextExceptions (Layer 2) with configuration error handling."""
        print("\n=== FlextExceptions Integration (Layer 2) ===")

        # ConfigurationError for invalid configuration
        try:
            # Simulate invalid configuration scenario
            invalid_workers = -1
            if invalid_workers <= 0:
                error_message = "Invalid max_workers configuration"
                raise FlextExceptions.ConfigurationError(
                    error_message,
                    config_key="max_workers",
                    config_source="environment",
                )
        except FlextExceptions.ConfigurationError as e:
            print(f"✅ ConfigurationError: {e.error_code} - {e.message}")
            print(f"   Config key: {e.config_key}, Source: {e.config_source}")

        # Environment validation removed - no environment constants

        # Configuration loading with proper error handling
        try:
            # Simulate configuration loading failure
            missing_required_config = None
            if missing_required_config is None:
                error_message = "Required configuration not found"
                raise FlextExceptions.ConfigurationError(
                    error_message,
                    config_key="database_url",
                    config_source="environment",
                )
        except FlextExceptions.ConfigurationError as e:
            print(f"✅ Configuration loading error: {e.error_code}")
            print(f"   Missing config key: {e.config_key}")

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
        print("config = FlextConfig()")
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
        print("config = FlextConfig()")
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
        print("config = FlextConfig()")
        print("url = config.database_url  # Type-safe!")


def demonstrate_flextcore_config_access() -> None:
    """Demonstrate FlextConstants unified access to configuration.

    Shows how FlextConstants provides convenient access to configuration
    alongside other flext-core components.
    """
    print("\n" + "=" * 60)
    print("FLEXTCORE UNIFIED CONFIG ACCESS")
    print("Modern pattern for configuration with FlextConstants")
    print("=" * 60)

    # 1. Access config through FlextConstants class
    print("\n=== 1. Config Access Through FlextConstants ===")
    config = FlextConfig()
    print(f"  ✅ Config accessed: {type(config).__name__}")
    print(f"  ✅ Log level: {config.log_level}")

    # 2. Direct class access for configuration
    print("\n=== 2. Direct Config Class Access ===")
    direct_config = FlextConfig()
    print(f"  ✅ Direct instantiation: {type(direct_config).__name__}")
    print(f"  ✅ Debug mode: {direct_config.debug}")

    # 3. Factory method for configuration
    print("\n=== 3. Config Factory Method ===")
    factory_config = FlextConfig()
    print(f"  ✅ Factory config: {type(factory_config).__name__}")
    print(f"  ✅ Timeout: {factory_config.timeout_seconds}s")

    # 4. Combined config with other components
    print("\n=== 4. Config with Integrated Components ===")
    logger = FlextLogger.create_module_logger(__name__)
    container = FlextContainer.get_global()

    logger.info("Configuration loaded", extra={"log_level": config.log_level})
    container.register("config", config)

    print(f"  ✅ Logger integrated: {type(logger).__name__}")
    print("  ✅ Config registered in container")
    services_result = container.list_services()
    service_count = len(services_result.unwrap()) if services_result.is_success else 0
    print(f"  ✅ Services: {service_count}")

    # 5. Infrastructure setup with configuration
    print("\n=== 5. Infrastructure with Custom Config ===")
    custom_config = FlextConfig()

    # Note: setup_service_infrastructure method not available in current version
    # Service would typically get config injected via DI container
    container = FlextContainer.get_global()
    container.register("config", custom_config)

    print("  ✅ Service initialized with custom config:")
    print(f"     - Config type: {type(custom_config).__name__}")
    print(f"     - Log level: {custom_config.log_level}")
    print("     - Container registered: config")

    print("\n" + "=" * 60)
    print("✅ FlextConstants config demonstration complete!")
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

    # New FlextResult methods (v0.9.9+)
    service.demonstrate_from_callable()
    service.demonstrate_flow_through()
    service.demonstrate_lash()
    service.demonstrate_alt()
    service.demonstrate_value_or_call()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    # Modern FlextConstants pattern demonstration
    demonstrate_flextcore_config_access()

    print("\n" + "=" * 60)
    print("✅ ALL FlextConfig methods demonstrated!")
    print(
        "✨ Including new v0.9.9+ methods: from_callable, flow_through, lash, alt, value_or_call"
    )
    print(
        "🔧 Including foundation integration: FlextRuntime (Layer 0.5), FlextConstants (Layer 1), FlextExceptions (Layer 2)"
    )
    print("🎯 Next: See 05_logging_basics.py for FlextLogger patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
