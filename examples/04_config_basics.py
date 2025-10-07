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

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextExceptions,
    FlextResult,
    FlextRuntime,
    FlextService,
    FlextTypes,
)

from .example_scenarios import ExampleScenarios


class ComprehensiveConfigService(FlextService[FlextTypes.Dict]):
    """Service demonstrating ALL FlextConfig patterns with FlextMixins.Service infrastructure.

    This service inherits from FlextCore.Service to demonstrate:
    - Inherited container property (FlextContainer singleton)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextContext for request tracking)
    - Inherited config property (FlextConfig with Pydantic 2.11+ Settings)
    - Inherited metrics property (FlextMetrics for observability)

    The focus is on demonstrating FlextConfig patterns (Pydantic 2.11+ Settings
    with dependency_injector integration) while leveraging complete FlextMixins.Service
    infrastructure for service orchestration.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins.Service infrastructure.

        Note: No manual logger or config initialization needed!
        All infrastructure is inherited from FlextCore.Service base class:
        - self.logger: FlextLogger with service context
        - self.container: FlextContainer global singleton
        - self.context: FlextContext for request tracking
        - self.config: FlextConfig with Pydantic 2.11+ Settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        self._scenarios = ExampleScenarios()
        self._reference_config = self._scenarios.config()
        self._production_config = self._scenarios.config(production=True)
        self._metadata = self._scenarios.metadata(tags=["config", "demo"])

        # Demonstrate inherited logger and config (no manual instantiation needed!)
        self.logger.info(
            "ComprehensiveConfigService initialized with inherited infrastructure",
            extra={
                "service_type": "FlextConfig Pydantic 2.11+ demonstration",
                "environment": self.config.environment,
                "debug_mode": self.config.debug,
                "log_level": self.config.log_level,
            },
        )

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute all FlextConfig demonstrations and return summary.

        Demonstrates inherited config property alongside other infrastructure
        components (logger, container, context, metrics) from FlextMixins.Service.
        """
        self.logger.info("Starting comprehensive FlextConfig demonstration")

        try:
            # Run all demonstrations
            self.demonstrate_global_singleton()
            self.demonstrate_environment_config()
            self.demonstrate_logging_config()
            self.demonstrate_database_config()
            self.demonstrate_cache_config()
            self.demonstrate_performance_config()
            self.demonstrate_cqrs_config()
            self.demonstrate_api_security_config()
            self.demonstrate_metadata()
            self.demonstrate_env_variables()
            self.demonstrate_validation()
            self.demonstrate_dynamic_config()
            self.demonstrate_export()
            self.demonstrate_config_methods()
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
            summary = {
                "demonstrations_completed": 23,
                "environment": self.config.environment,
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

            return FlextResult[FlextTypes.Dict].ok(summary)

        except Exception as e:
            error_msg = f"Configuration demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextResult[FlextTypes.Dict].fail(
                error_msg, error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

    # ========== GLOBAL SINGLETON ACCESS ==========

    def demonstrate_global_singleton(self) -> None:
        """Show global singleton pattern."""
        print("\n=== Global Singleton Configuration ===")

        # Get global instance (thread-safe singleton)
        config = FlextCore.Config()
        print(f"✅ Global config instance: {type(config).__name__}")

        # Access basic settings
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.debug}")
        print(f"Log level: {config.log_level}")
        print(f"App name: {config.app_name}")

        # Same instance everywhere
        config2 = FlextCore.Config()
        print(f"Same instance: {config is config2}")

    # ========== ENVIRONMENT CONFIGURATION ==========

    def demonstrate_environment_config(self) -> None:
        """Show environment-specific configuration."""
        print("\n=== Environment Configuration ===")

        config = FlextCore.Config()
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

        config = FlextCore.Config()
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

        config = FlextCore.Config()

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

        config = FlextCore.Config()

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
                "environment",
                config.environment
                in {"development", "test", "staging", "production", "local"},
            ),
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

        print("Note: FlextConfig is immutable after initialization")
        print("To change config, create new instance with environment variables")

        print("\nCreating config for different environment:")
        test_config = FlextCore.Config.create_for_environment("test")
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

        config = FlextCore.Config()

        # Export as dictionary (safe, no secrets)
        config_dict = config.model_dump(exclude={"api_key"})
        print(f"Exported keys: {list[object](config_dict.keys())[:10]}...")

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

        config = FlextCore.Config()

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

    # ========== NEW FLEXTRESULT METHODS (v0.9.9+) ==========

    def demonstrate_from_callable(self) -> None:
        """Show from_callable for safe configuration loading."""
        print("\n=== from_callable(): Safe Configuration Loading ===")

        # Safe configuration initialization
        def risky_config_load() -> FlextCore.Config:
            """Simulate risky config loading that might raise."""
            config = FlextCore.Config()
            if not config.environment:
                msg = "Environment not configured"
                raise ValueError(msg)
            return config

        config_result = FlextResult.from_callable(risky_config_load)
        if config_result.is_success:
            config = config_result.unwrap()
            print(f"✅ Config loaded safely: environment={config.environment}")

        # Safe environment validation
        def validate_production() -> str:
            """Validate production environment setup."""
            config = FlextCore.Config()
            if config.is_production() and config.debug:
                msg = "Debug mode cannot be enabled in production"
                raise ValueError(msg)
            return f"Production config valid: debug={config.debug}"

        validation_result = FlextResult.from_callable(validate_production)
        if validation_result.is_success:
            print(f"✅ {validation_result.unwrap()}")
        else:
            print(f"✅ Caught validation error safely: {validation_result.error}")

    def demonstrate_flow_through(self) -> None:
        """Show pipeline composition for multi-step config operations."""
        print("\n=== flow_through(): Configuration Validation Pipeline ===")

        def load_config(_: object) -> FlextResult[FlextCore.Config]:
            """Step 1: Load configuration."""
            return FlextResult[FlextCore.Config].ok(FlextCore.Config())

        def validate_environment(
            config: FlextCore.Config,
        ) -> FlextResult[FlextCore.Config]:
            """Step 2: Validate environment settings."""
            if config.environment not in {
                "development",
                "test",
                "staging",
                "production",
            }:
                return FlextResult[FlextCore.Config].fail(
                    f"Invalid environment: {config.environment}"
                )
            return FlextResult[FlextCore.Config].ok(config)

        def validate_logging(config: FlextCore.Config) -> FlextResult[FlextCore.Config]:
            """Step 3: Validate logging configuration."""
            if config.log_level not in FlextCore.Constants.Logging.VALID_LEVELS:
                return FlextResult[FlextCore.Config].fail(
                    f"Invalid log level: {config.log_level}"
                )
            return FlextResult[FlextCore.Config].ok(config)

        def validate_performance(
            config: FlextCore.Config,
        ) -> FlextResult[FlextCore.Config]:
            """Step 4: Validate performance settings."""
            if config.max_workers <= 0 or config.timeout_seconds <= 0:
                return FlextResult[FlextCore.Config].fail(
                    "Invalid performance settings"
                )
            return FlextResult[FlextCore.Config].ok(config)

        # Pipeline: load → validate env → validate logging → validate performance
        result = (
            load_config(True)  # Start with load_config
            .flat_map(validate_environment)
            .flat_map(validate_logging)
            .flat_map(validate_performance)
        )

        if result.is_success:
            config = result.unwrap()
            print(
                f"✅ Config validation pipeline success: "
                f"env={config.environment}, "
                f"log={config.log_level}, "
                f"workers={config.max_workers}"
            )

    def demonstrate_lash(self) -> None:
        """Show error recovery in configuration operations."""
        print("\n=== lash(): Configuration Error Recovery ===")

        def try_load_production_config() -> FlextResult[FlextCore.Config]:
            """Attempt to load production config (might fail)."""
            return FlextResult[FlextCore.Config].fail("Production config not available")

        def recover_with_default(error: str) -> FlextResult[FlextCore.Config]:
            """Recover by loading default development config."""
            print(f"  Recovering from: {error}")
            default_config = FlextCore.Config.create_for_environment("development")
            return FlextResult[FlextCore.Config].ok(default_config)

        result = try_load_production_config().lash(recover_with_default)
        if result.is_success:
            config = result.unwrap()
            print(
                f"✅ Recovered with fallback config: environment={config.environment}"
            )

    def demonstrate_alt(self) -> None:
        """Show fallback pattern for configuration sources."""
        print("\n=== alt(): Configuration Source Fallback ===")

        # Primary: Production config (simulated failure)
        primary = FlextResult[FlextCore.Config].fail("Production config unavailable")

        # Fallback: Development config
        fallback_config = FlextCore.Config.create_for_environment("development")
        fallback = FlextResult[FlextCore.Config].ok(fallback_config)

        result = primary.alt(fallback)
        if result.is_success:
            config = result.unwrap()
            print(f"✅ Got fallback config: environment={config.environment}")

    def demonstrate_value_or_call(self) -> None:
        """Show lazy default evaluation for expensive config operations."""
        print("\n=== value_or_call(): Lazy Configuration Initialization ===")

        # Success case - no expensive initialization needed
        success_config = FlextCore.Config()
        success = FlextResult[FlextCore.Config].ok(success_config)

        expensive_created = False

        def expensive_default() -> FlextCore.Config:
            """Expensive default config creation (only if needed)."""
            nonlocal expensive_created
            expensive_created = True
            print("  Creating expensive default config...")
            return FlextCore.Config.create_for_environment("development")

        # Success case - expensive_default NOT called
        config = success.value_or_call(expensive_default)
        print(
            f"✅ Success: environment={config.environment}, "
            f"expensive_created={expensive_created}"
        )

        # Failure case - expensive_default IS called
        expensive_created = False
        failure = FlextResult[FlextCore.Config].fail("Config load failed")
        config = failure.value_or_call(expensive_default)
        print(
            f"✅ Failure recovered: environment={config.environment}, "
            f"expensive_created={expensive_created}"
        )

    # ========== FOUNDATION LAYER INTEGRATION (Layer 0.5 - 2) ==========

    def demonstrate_flext_runtime_integration(self) -> None:
        """Show FlextRuntime (Layer 0.5) configuration defaults with FlextConfig."""
        print("\n=== FlextRuntime Integration (Layer 0.5) ===")

        config = FlextCore.Config()

        # FlextRuntime provides configuration defaults without circular dependencies
        print("FlextRuntime configuration defaults:")
        print(f"  DEFAULT_APP_NAME: {'flext-app'}")
        print(f"  DEFAULT_ENVIRONMENT: {FlextConstants.Config.DEFAULT_ENVIRONMENT}")
        print(f"  DEFAULT_LOG_LEVEL: {FlextConstants.Logging.DEFAULT_LEVEL}")
        print(f"  DEFAULT_TIMEOUT: {FlextConstants.Defaults.TIMEOUT}s")
        print(f"  DEFAULT_MAX_WORKERS: {FlextConstants.Processing.DEFAULT_MAX_WORKERS}")

        # Validate configuration values using FlextRuntime type guards
        if config.database_url:
            is_valid_url = FlextRuntime.is_valid_url(config.database_url)
            print(f"✅ Database URL validation: {is_valid_url}")

        # Environment variable validation
        env_value = config.environment or FlextConstants.Config.DEFAULT_ENVIRONMENT
        is_valid_identifier = FlextRuntime.is_valid_identifier(env_value)
        print(f"✅ Environment name is valid identifier: {is_valid_identifier}")

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

        config = FlextCore.Config()

        # Configuration validation using FlextConstants
        print("FlextConstants configuration validation:")

        # Environment constants
        valid_envs = [
            FlextConstants.Environment.ConfigEnvironment.DEVELOPMENT.value,
            FlextConstants.Environment.ConfigEnvironment.TESTING.value,
            FlextConstants.Environment.ConfigEnvironment.STAGING.value,
            FlextConstants.Environment.ConfigEnvironment.PRODUCTION.value,
        ]
        env_valid = config.environment in valid_envs
        print(f"✅ Environment '{config.environment}' is valid: {env_valid}")

        # Logging level constants
        if hasattr(FlextConstants, "Logging"):
            log_levels = FlextConstants.Logging.VALID_LEVELS
            log_valid = config.log_level in log_levels
            print(f"✅ Log level '{config.log_level}' in valid levels: {log_valid}")

        # Performance constants
        max_workers_ok = 1 <= config.max_workers <= 100
        timeout_ok = config.timeout_seconds >= 1
        print(f"✅ Max workers in range: {max_workers_ok}")
        print(f"✅ Timeout is positive: {timeout_ok}")

        # HTTP configuration defaults
        if hasattr(FlextConstants, "Http"):
            print(
                f"✅ HTTP Content-Type default: {FlextConstants.Http.DEFAULT_CONTENT_TYPE}"
            )
            print(f"✅ HTTP Accept header: {FlextConstants.Http.DEFAULT_ACCEPT}")

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

        # ValidationError for environment validation
        try:
            invalid_env = "invalid_environment"
            valid_envs = [
                FlextConstants.Environment.ConfigEnvironment.DEVELOPMENT.value,
                FlextConstants.Environment.ConfigEnvironment.PRODUCTION.value,
            ]
            if invalid_env not in valid_envs:
                error_message = "Invalid environment configuration"
                raise FlextExceptions.ValidationError(
                    error_message,
                    field="environment",
                    value=invalid_env,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
        except FlextExceptions.ValidationError as e:
            print(f"✅ ValidationError: {e.error_code} - {e.message}")
            print(f"   Field: {e.field}, Value: {e.value}")

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
            "Hard-coded configuration is DEPRECATED! Use FlextCore.Config.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (hard-coded):")
        print("DATABASE_HOST = 'localhost'")
        print("API_PORT = 8000")
        print("DEBUG = True")

        print("\n✅ CORRECT WAY (FlextConfig):")
        print("config = FlextCore.Config()")
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
    core = FlextCore()
    config = core.config
    print(f"  ✅ Config accessed: {type(config).__name__}")
    print(f"  ✅ Environment: {config.environment}")
    print(f"  ✅ Log level: {config.log_level}")

    # 2. Direct class access for configuration
    print("\n=== 2. Direct Config Class Access ===")
    direct_config = FlextCore.Config()
    print(f"  ✅ Direct instantiation: {type(direct_config).__name__}")
    print(f"  ✅ Debug mode: {direct_config.debug}")

    # 3. Factory method for configuration
    print("\n=== 3. Config Factory Method ===")
    factory_config = FlextCore.get_config()
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
    custom_config = FlextCore.Config()
    setup_result = FlextCore.setup_service_infrastructure(
        "config-demo-service", config=custom_config
    )

    if setup_result.is_success:
        infra = setup_result.unwrap()
        infra_config = cast("FlextCore.Config", infra["config"])

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

    # Modern Flext pattern demonstration
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
