"""FlextSettings - Configuration Management Module.

This module provides comprehensive configuration management for the FLEXT ecosystem,
implementing Pydantic v2 BaseSettings with dependency injection, environment variable support,
and runtime validation. Serves as the foundation layer (0.5) controlling all other layers.

Scope: Global configuration management, singleton pattern, DI integration, validation,
environment variable handling, thread-safe operations, and dynamic config updates.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.__version__ import __version__
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import T_Namespace, T_Settings, t
from flext_core.utilities import u


class FlextSettings(p.ProtocolSettings, p.Config, FlextRuntime):
    """Configuration management with Pydantic validation and dependency injection.

    Architecture: Layer 0.5 (Configuration Foundation)
    Provides enterprise-grade configuration management for the FLEXT ecosystem
    through p.ProtocolSettings base class with natural protocol multi-inheritance.

    Protocol Implementation: Inherits from p.ProtocolSettings which uses
    ProtocolModelMeta metaclass to resolve the Pydantic/Protocol metaclass conflict.
    Implements p.Config protocol via direct inheritance (not structural typing).

    Core Features:
    - Pydantic v2 BaseSettings with type-safe configuration
    - Environment variable support with FLEXT_ prefix
    - Thread-safe singleton pattern
    - Dependency injection integration
    - Runtime configuration updates
    - Protocol compliance via inheritance (p.Config, p.ProtocolSettings)
    """

    # Singleton pattern
    # Business Rule: Use dict for mutable ClassVar storage (singleton registry)
    # ClassVar dict is mutable storage needed for singleton pattern - this is correct.
    # Type annotation uses dict (not Mapping) because it's mutable storage that needs
    # to be modified at runtime (adding/removing instances).
    #
    # Audit Implication: This registry tracks all configuration instances for
    # singleton pattern. Thread-safe access via _lock ensures no race conditions.
    # Used for configuration instance management across the FLEXT ecosystem.
    _instances: ClassVar[MutableMapping[type[BaseSettings], BaseSettings]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    # Note: implements_protocol() and _protocol_name() are inherited from
    # p.ProtocolSettings. The metaclass ProtocolModelMeta handles protocol
    # detection and compliance validation at class definition time.

    # =========================================================================
    # p.Config Protocol Implementation (validated at class definition)
    # =========================================================================

    def model_copy(
        self,
        *,
        update: Mapping[str, t.ScalarValue] | None = None,
        deep: bool = False,
    ) -> Self:
        """Clone configuration with optional updates (p.Config protocol)."""
        return super().model_copy(update=update, deep=deep)

    # Configuration fields
    # env_file resolved at module load time via FLEXT_ENV_FILE env var
    model_config = SettingsConfigDict(
        env_prefix=c.Platform.ENV_PREFIX,
        env_nested_delimiter=c.Platform.ENV_NESTED_DELIMITER,
        env_file=u.resolve_env_file(),
        env_file_encoding=c.Utilities.DEFAULT_ENCODING,
        case_sensitive=False,
        extra=c.ModelConfig.EXTRA_IGNORE,
        validate_assignment=True,
    )

    # =========================================================================
    # ENV FILE RESOLUTION - Public API for namespace configs
    # =========================================================================

    @staticmethod
    def resolve_env_file() -> str | None:
        """Resolve .env file path from FLEXT_ENV_FILE environment variable.

        This method is the PUBLIC API for all namespace configs to use.
        It ensures all FLEXT ecosystem configs use the same .env resolution logic.

        Precedence (highest to lowest):
        1. FLEXT_ENV_FILE environment variable (custom path)
        2. Default .env file from current directory

        Returns:
            str | None: Path to .env file or None if not found

        Example:
            # In namespace config classes (e.g., FlextLdapSettings)
            model_config = SettingsConfigDict(
                env_prefix="FLEXT_LDAP_",
                env_file=FlextSettings.resolve_env_file(),
                ...
            )

        """
        return u.resolve_env_file()

    # Core configuration
    app_name: str = Field(default="flext", description="Application name")
    version: str = Field(default=__version__, description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    trace: bool = Field(default=False, description="Enable trace mode")

    # Logging configuration (true application config only)
    # FlextRuntime and FlextLogger handle logging infrastructure
    log_level: c.Settings.LogLevel = Field(
        default=c.Settings.LogLevel.INFO,
        description="Log level",
    )
    async_logging: bool = Field(
        default=True,
        description="Enable asynchronous buffered logging for performance",
    )

    # Cache configuration
    enable_caching: bool = Field(
        default=c.Settings.DEFAULT_ENABLE_CACHING,
        description="Enable caching",
    )
    cache_ttl: int = Field(
        default=c.Defaults.CACHE_TTL,
        description="Cache TTL",
    )

    # Database configuration
    database_url: str = Field(
        default=c.Defaults.DATABASE_URL,
        description="Database URL",
    )
    database_pool_size: int = Field(
        default=c.Performance.DEFAULT_DB_POOL_SIZE,
        description="Database pool size",
    )

    # Reliability configuration
    circuit_breaker_threshold: int = Field(
        default=c.Reliability.DEFAULT_FAILURE_THRESHOLD,
        description="Circuit breaker threshold",
    )
    rate_limit_max_requests: int = Field(
        default=c.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
        description="Rate limit max requests",
    )
    rate_limit_window_seconds: int = Field(
        default=c.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        description="Rate limit window",
    )
    retry_delay: int = Field(
        default=c.Reliability.DEFAULT_RETRY_DELAY_SECONDS,
        description="Retry delay",
    )
    max_retry_attempts: int = Field(
        default=c.Reliability.MAX_RETRY_ATTEMPTS,
        description="Max retry attempts",
    )

    # Dispatcher configuration
    enable_timeout_executor: bool = Field(
        default=True,
        description="Enable timeout executor",
    )
    dispatcher_enable_logging: bool = Field(
        default=c.Dispatcher.DEFAULT_ENABLE_LOGGING,
        description="Enable dispatcher logging",
    )
    dispatcher_auto_context: bool = Field(
        default=c.Dispatcher.DEFAULT_AUTO_CONTEXT,
        description="Auto context in dispatcher",
    )
    dispatcher_timeout_seconds: float = Field(
        default=c.Dispatcher.DEFAULT_TIMEOUT_SECONDS,
        description="Dispatcher timeout",
    )
    dispatcher_enable_metrics: bool = Field(
        default=c.Dispatcher.DEFAULT_ENABLE_METRICS,
        description="Enable dispatcher metrics",
    )
    executor_workers: int = Field(
        default=c.Container.DEFAULT_WORKERS,
        description="Executor workers",
    )

    # Processing configuration
    timeout_seconds: float = Field(
        default=c.Network.DEFAULT_TIMEOUT,
        description="Default timeout",
    )
    max_workers: int = Field(
        default=c.Processing.DEFAULT_MAX_WORKERS,
        description="Max workers",
    )
    max_batch_size: int = Field(
        default=c.Performance.MAX_BATCH_SIZE,
        description="Max batch size",
    )

    # Security configuration
    api_key: str | None = Field(default=None, description="API key")

    # Exception configuration
    # Note: Using FailureLevel StrEnum directly for type safety
    exception_failure_level: c.Exceptions.FailureLevel = Field(
        default=c.Exceptions.FAILURE_LEVEL_DEFAULT,
        description="Exception failure level",
    )

    def __new__(cls, **_kwargs: t.ScalarValue) -> Self:
        """Create singleton instance.

        Note: BaseSettings.__init__ accepts **values internally.
        We override __new__ to implement singleton pattern while allowing
        kwargs to be passed for testing and configuration via model_validate.
        """
        base_class = cls
        if base_class not in cls._instances:
            with cls._lock:
                if base_class not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[base_class] = instance
        raw_instance = cls._instances[base_class]
        raw_type = type(raw_instance)
        if raw_type is not cls and not issubclass(raw_type, cls):
            msg = f"Singleton instance is not of expected type {cls.__name__}"
            raise TypeError(msg)
        return raw_instance

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton instance for testing purposes.

        This method is intended for use in tests only to allow
        clean state between test runs.
        """
        with cls._lock:
            keys_to_remove = [
                instance_cls
                for instance_cls in cls._instances
                if instance_cls is cls
                or issubclass(instance_cls, cls)
                or issubclass(cls, instance_cls)
            ]
            for instance_cls in keys_to_remove:
                del cls._instances[instance_cls]

    def __init__(self, **kwargs: t.ScalarValue) -> None:
        """Initialize config with data.

        Note: BaseSettings handles initialization from environment variables,
        .env files, and other sources automatically. Kwargs can be passed for
        testing and explicit configuration (used by model_validate).
        """
        # Check if already initialized (singleton pattern)
        if hasattr(self, "_di_provider"):
            # Instance already initialized - update fields atomically to avoid
            # triggering model validators after each field change.
            if kwargs:
                for key, value in kwargs.items():
                    if key in self.__class__.model_fields:
                        self.__dict__[key] = value
            return

        # First initialization - pass kwargs to BaseSettings.__init__() so that
        # Pydantic field validators run during initialization. BaseSettings will load
        # values from environment variables and .env files, then apply explicit kwargs.
        # Field validators (e.g., validate_ldif_encoding) will run during initialization.
        # Call BaseSettings.__init__ directly to avoid mypy type mismatch with
        # ProtocolSettings intermediate __init__ signature.
        # BaseSettings.__init__ expects fixed keyword types and accepts runtime values.

        BaseSettings.__init__(self, **kwargs)

        # Use runtime bridge for dependency-injector providers (L0.5 pattern)
        self._di_provider: t.ScalarValue | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        """Validate configuration.

        Business Rule: Validates configuration consistency after model initialization.
        Checks database URL scheme validity and ensures trace mode requires debug mode.
        Raises ValueError if configuration is invalid, preventing invalid configurations
        from being used in production systems.

        Audit Implication: Configuration validation ensures audit trail completeness by
        preventing invalid configurations from being used. All configurations are validated
        before being used in production systems. Used by Pydantic v2 model_validator for
        cross-field validation.

        Returns:
            Self: Validated configuration instance

        Raises:
            ValueError: If configuration is invalid

        """
        # Check database URL scheme if provided
        if self.database_url and not self.database_url.startswith((
            "postgresql://",
            "mysql://",
            "sqlite://",
        )):
            msg = "Invalid database URL scheme"
            raise ValueError(msg)

        # Check that trace mode requires debug mode
        if self.trace and not self.debug:
            msg = "Trace mode requires debug mode"
            raise ValueError(msg)

        return self

    @property
    def effective_log_level(self) -> c.Settings.LogLevel:
        """Get effective log level based on debug/trace flags."""
        if self.trace:
            # LogLevel.DEBUG is already compatible with LogLevelLiteral
            return c.Settings.LogLevel.DEBUG
        if self.debug:
            # LogLevel.INFO is already compatible with LogLevelLiteral
            return c.Settings.LogLevel.INFO
        # self.log_level is already LogLevelLiteral (from field_validator)
        return self.log_level

    @classmethod
    def get_global_instance(cls) -> Self:
        """Get the global singleton instance."""
        return cls()

    @classmethod
    def materialize(
        cls,
        *,
        config_overrides: Mapping[str, t.ScalarValue] | None = None,
    ) -> Self:
        """Factory method to create a config instance with optional overrides.

        This is the preferred way to create a configuration instance for
        services that need to apply runtime overrides. It respects Clean
        Architecture principles where each class owns its own instantiation.

        For FlextSettings (base class): clones from the global instance to
        preserve environment-derived values while allowing overrides.
        For subclasses: creates a new instance directly.

        Args:
            config_overrides: Optional mapping of field names to override values.

        Returns:
            New configuration instance with applied overrides.

        Example:
            >>> config = FlextSettings.materialize(
            ...     config_overrides={"app_name": "myapp"}
            ... )
            >>> config.app_name
            'myapp'

        """
        # For FlextSettings itself, clone from global instance
        if cls is FlextSettings:
            global_config = cls.get_global_instance()
            # Use model_copy to clone the instance properly
            # This ensures Pydantic internal state (__pydantic_fields_set__, etc.) is properly initialized
            instance = global_config.model_copy(deep=True)
        else:
            # For subclasses, create directly
            instance = cls()

        # Apply overrides if provided
        if config_overrides:
            instance = instance.model_copy(update=config_overrides, deep=True)

        return instance

    def get_di_config_provider(self) -> t.ScalarValue:
        """Get dependency injection provider for this config.

        Returns a providers.Singleton instance via the runtime bridge.
        Type annotation stays framework-level to avoid DI imports in this module.
        """
        if self._di_provider is None:
            providers_module = FlextRuntime.dependency_providers()
            self._di_provider = providers_module.Singleton(lambda: self)
        return self._di_provider

    def update_from_env(self) -> None:
        """Update configuration from current environment variables."""
        # Implementation would reload from env

    def validate_override(
        self,
        key: str,
        _value: t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue],
    ) -> bool:
        """Validate if an override is acceptable."""
        # Basic validation - could be extended
        return key in self.__class__.model_fields

    def apply_override(
        self,
        key: str,
        value: t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue],
    ) -> None:
        """Apply a validated configuration override."""
        if self.validate_override(key, value):
            setattr(self, key, value)

    class AutoConfig(BaseModel):
        """Auto-configuration model for dynamic config creation."""

        model_config = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            extra="forbid",
        )

        config_class: type[BaseSettings]
        env_prefix: str = Field(default=c.Platform.ENV_PREFIX)
        env_file: str | None = None

        def create_config(self) -> BaseSettings:
            """Create configuration instance."""
            return self.config_class()

    # Registry for namespaced configurations
    # Business Rule: Use dict for mutable ClassVar storage (namespace registry)
    # ClassVar dict is mutable storage needed for namespace registration - this is correct.
    # Type annotation uses dict (not Mapping) because it's mutable storage that needs
    # to be modified at runtime (registering namespace config classes).
    #
    # Audit Implication: This registry tracks namespace configuration classes for
    # auto-registration pattern. Used by @auto_register decorator to map namespace
    # strings to configuration classes.
    _namespace_registry: ClassVar[MutableMapping[str, type[BaseSettings]]] = {}
    _context_overrides: ClassVar[MutableMapping[str, MutableMapping[str, t.ScalarValue]]] = {}

    @staticmethod
    def auto_register(
        namespace: str,
    ) -> Callable[[type[T_Settings]], type[T_Settings]]:
        """Decorator for auto-registering configuration classes.

        Business Rule: Decorator pattern for auto-registering configuration classes
        in namespace registry. Uses TypeVar to preserve the original class type through
        the decorator, ensuring type checkers (pyright/mypy) see the specific class type,
        not BaseSettings. Registers class in _namespace_registry for dynamic namespace
        resolution.

        WARNING: Registered classes should use FlextSettings.resolve_env_file()
        in their model_config.env_file to respect FLEXT_ENV_FILE environment variable.
        Hardcoded '.env' values will trigger a deprecation warning.

        Audit Implication: Auto-registration enables dynamic namespace configuration
        resolution, ensuring audit trail completeness for namespace-based configurations.
        All namespace configurations are registered before being used in production systems.

        Args:
            namespace: Namespace identifier for the configuration

        Returns:
            Decorator function that registers the class while preserving its type

        """

        def decorator(cls: type[T_Settings]) -> type[T_Settings]:
            """Register the configuration class while preserving type."""
            # Note: Previous validation for env_file=".env" was removed because
            # it cannot distinguish between:
            # 1. Hardcoded ".env" (incorrect)
            # 2. FlextSettings.resolve_env_file() returning ".env" (correct)
            # The documentation warns about proper usage of resolve_env_file().
            # Register in namespace registry (namespace stored in registry key, not on class)
            FlextSettings._namespace_registry[namespace] = cls
            return cls

        return decorator

    @classmethod
    def register_namespace(
        cls,
        namespace: str,
        config_class: type[BaseSettings],
    ) -> None:
        """Register a configuration class for a namespace.

        Args:
            namespace: Namespace identifier
            config_class: Configuration class to register

        """
        cls._namespace_registry[namespace] = config_class

    @classmethod
    def get_namespace_config(cls, namespace: str) -> type[BaseSettings] | None:
        """Get configuration class for a namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Configuration class or None if not found

        """
        return cls._namespace_registry.get(namespace)

    def get_namespace(
        self,
        namespace: str,
        config_type: type[T_Namespace],
    ) -> T_Namespace:
        """Get configuration instance for a namespace.

        Business Rule: Resolves namespace configuration class from registry and
        instantiates it. Validates namespace exists and config class is subclass of
        expected type. Raises ValueError if namespace not found, TypeError if type
        mismatch. Used for dynamic namespace configuration resolution.

        Audit Implication: Namespace resolution ensures audit trail completeness by
        validating namespace configurations before use. All namespace configurations
        are validated before being used in production systems.

        Args:
            namespace: Namespace identifier
            config_type: Expected configuration type

        Returns:
            Configuration instance

        Raises:
            ValueError: If namespace not found
            TypeError: If type mismatch

        """
        config_class_raw = self.get_namespace_config(namespace)
        if config_class_raw is None:
            msg = f"Namespace '{namespace}' not registered"
            raise ValueError(msg)
        if not issubclass(config_class_raw, config_type):
            msg = f"Namespace '{namespace}' config class {config_class_raw} is not subclass of {config_type}"
            raise TypeError(msg)
        # Instantiate the config class properly - Pydantic models need regular instantiation
        # config_class is already validated as subclass of config_type, safe to instantiate
        config_class: type[T_Namespace] = config_class_raw
        return config_class()

    # __getattr__ removed - use get_namespace() method explicitly
    # Example: config.get_namespace("ldif", FlextLdifSettings) instead of config.ldif

    @classmethod
    def for_context(
        cls,
        context_id: str,
        **overrides: t.ScalarValue,
    ) -> Self:
        """Get configuration instance with context-specific overrides.

        Creates a configuration instance with overrides specific to the given
        context. Context overrides are applied on top of the base configuration.

        Args:
            context_id: Unique identifier for the execution context.
            **overrides: Configuration field overrides for this context.

        Returns:
            Self: Configuration instance with context overrides applied.

        Example:
            >>> config = FlextSettings.for_context(
            ...     "worker_1", log_level="DEBUG", timeout=60
            ... )

        """
        # Get base instance
        base = cls.get_global_instance()
        # Apply context overrides
        context_overrides = cls._context_overrides.get(context_id, {})
        all_overrides = {**context_overrides, **overrides}
        if all_overrides:
            return base.model_copy(update=all_overrides)
        return base

    @classmethod
    def register_context_overrides(
        cls,
        context_id: str,
        **overrides: t.ScalarValue,
    ) -> None:
        """Register context-specific configuration overrides.

        Registers overrides that will be automatically applied when using
        `for_context()` with the same context_id.

        Args:
            context_id: Unique identifier for the execution context.
            **overrides: Configuration field overrides to register.

        Example:
            >>> FlextSettings.register_context_overrides(
            ...     "worker_1", log_level="DEBUG", timeout=60
            ... )
            >>> config = FlextSettings.for_context("worker_1")

        """
        if context_id not in cls._context_overrides:
            cls._context_overrides[context_id] = {}
        cls._context_overrides[context_id].update(overrides)

    @classmethod
    def reset_global_instance(cls) -> None:
        """Reset the global singleton instance for testing."""
        cls._instances.clear()
        cls._context_overrides.clear()


__all__ = ["FlextSettings"]
