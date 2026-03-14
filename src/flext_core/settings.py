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
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core import FlextRuntime, T_Settings, __version__, c, t, u
from flext_core.typings import T_Namespace


class FlextSettings(BaseSettings, FlextRuntime):
    """Configuration management with Pydantic validation and dependency injection.

    Architecture: Layer 0.5 (Configuration Foundation)
    Provides enterprise-grade configuration management for the FLEXT ecosystem
    through p.ProtocolSettings base class with natural protocol multi-inheritance.

    Core Features:
    - Pydantic v2 BaseSettings with type-safe configuration
    - Environment variable support with FLEXT_ prefix
    - Thread-safe singleton pattern
    - Dependency injection integration
    - Runtime configuration updates
    - Protocol compliance via inheritance (p.Config, p.ProtocolSettings)
    """

    _instances: ClassVar[dict[type[Self], Self]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    model_config = SettingsConfigDict(
        env_prefix=c.Platform.ENV_PREFIX,
        env_nested_delimiter=c.Platform.ENV_NESTED_DELIMITER,
        env_file=u.resolve_env_file(),
        env_file_encoding=c.Utilities.DEFAULT_ENCODING,
        case_sensitive=False,
        extra=c.ModelConfig.EXTRA_IGNORE,
        validate_assignment=True,
    )

    app_name: Annotated[str, Field(default="flext", description="Application name")]
    version: Annotated[
        str, Field(default=__version__, description="Application version")
    ]
    debug: Annotated[bool, Field(default=False, description="Enable debug mode")]
    trace: Annotated[bool, Field(default=False, description="Enable trace mode")]
    log_level: Annotated[
        c.Settings.LogLevel,
        Field(default=c.Settings.LogLevel.INFO, description="Log level"),
    ]
    async_logging: Annotated[
        bool,
        Field(
            default=True,
            description="Enable asynchronous buffered logging for performance",
        ),
    ]
    enable_caching: Annotated[
        bool,
        Field(default=c.Settings.DEFAULT_ENABLE_CACHING, description="Enable caching"),
    ]
    cache_ttl: Annotated[
        int, Field(default=c.Defaults.CACHE_TTL, description="Cache TTL")
    ]
    database_url: Annotated[
        str, Field(default=c.Defaults.DATABASE_URL, description="Database URL")
    ]
    database_pool_size: Annotated[
        int,
        Field(
            default=c.Performance.DEFAULT_DB_POOL_SIZE, description="Database pool size"
        ),
    ]
    circuit_breaker_threshold: Annotated[
        int,
        Field(
            default=c.Reliability.DEFAULT_FAILURE_THRESHOLD,
            description="Circuit breaker threshold",
        ),
    ]
    rate_limit_max_requests: Annotated[
        int,
        Field(
            default=c.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
            description="Rate limit max requests",
        ),
    ]
    rate_limit_window_seconds: Annotated[
        int,
        Field(
            default=c.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
            description="Rate limit window",
        ),
    ]
    retry_delay: Annotated[
        int,
        Field(
            default=c.Reliability.DEFAULT_RETRY_DELAY_SECONDS, description="Retry delay"
        ),
    ]
    max_retry_attempts: Annotated[
        int,
        Field(
            default=c.Reliability.MAX_RETRY_ATTEMPTS, description="Max retry attempts"
        ),
    ]
    enable_timeout_executor: Annotated[
        bool, Field(default=True, description="Enable timeout executor")
    ]
    dispatcher_enable_logging: Annotated[
        bool,
        Field(
            default=c.Dispatcher.DEFAULT_ENABLE_LOGGING,
            description="Enable dispatcher logging",
        ),
    ]
    dispatcher_auto_context: Annotated[
        bool,
        Field(
            default=c.Dispatcher.DEFAULT_AUTO_CONTEXT,
            description="Auto context in dispatcher",
        ),
    ]
    dispatcher_timeout_seconds: Annotated[
        float,
        Field(
            default=c.Dispatcher.DEFAULT_TIMEOUT_SECONDS,
            description="Dispatcher timeout",
        ),
    ]
    dispatcher_enable_metrics: Annotated[
        bool,
        Field(
            default=c.Dispatcher.DEFAULT_ENABLE_METRICS,
            description="Enable dispatcher metrics",
        ),
    ]
    executor_workers: Annotated[
        int, Field(default=c.Container.DEFAULT_WORKERS, description="Executor workers")
    ]
    timeout_seconds: Annotated[
        float, Field(default=c.Network.DEFAULT_TIMEOUT, description="Default timeout")
    ]
    max_workers: Annotated[
        int, Field(default=c.Processing.DEFAULT_MAX_WORKERS, description="Max workers")
    ]
    max_batch_size: Annotated[
        int, Field(default=c.Performance.MAX_BATCH_SIZE, description="Max batch size")
    ]
    api_key: Annotated[str | None, Field(default=None, description="API key")]
    exception_failure_level: Annotated[
        c.Exceptions.FailureLevel,
        Field(
            default=c.Exceptions.FAILURE_LEVEL_DEFAULT,
            description="Exception failure level",
        ),
    ]
    _di_provider: t.Scalar | None = PrivateAttr(default=None)

    def __new__(cls, **_kwargs: t.NormalizedValue) -> Self:
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
        if not isinstance(raw_instance, cls):
            cls_name = getattr(cls, "__name__", type(cls).__name__)
            msg = f"Singleton instance is not of expected type {cls_name}"
            raise TypeError(msg)
        return raw_instance

    def __init__(self, **kwargs: t.NormalizedValue) -> None:
        """Initialize config with data.

        Kwargs are applied as field overrides after base env/config loading
        to avoid type conflicts with BaseSettings internal parameters.
        """
        model_fields = self.__class__.model_fields
        if hasattr(self, "_di_provider"):
            if kwargs:
                for key, value in kwargs.items():
                    if key in model_fields:
                        setattr(self, key, value)
            return

        BaseSettings.__init__(self)
        if kwargs:
            for key, value in kwargs.items():
                if key in model_fields:
                    setattr(self, key, value)

    @computed_field
    @property
    def effective_log_level(self) -> c.Settings.LogLevel:
        """Get effective log level based on debug/trace flags."""
        return u.resolve_effective_log_level(
            trace=self.trace,
            debug=self.debug,
            log_level=self.log_level,
        )

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton instance for testing purposes.

        This method is intended for use in tests only to allow
        clean state between test runs.
        """
        with cls._lock:
            keys_to_remove = [
                instance_cls
                for instance_cls, instance in cls._instances.items()
                if isinstance(instance, cls)
            ]
            for instance_cls in keys_to_remove:
                del cls._instances[instance_cls]

    @classmethod
    def get_global(cls, *, overrides: Mapping[str, t.Scalar] | None = None) -> Self:
        """Get global settings, optionally materialized with overrides."""
        u.normalize_env_log_level()
        if overrides is None:
            return cls()
        if cls is FlextSettings:
            global_config = cls.get_global()
            instance = global_config.model_copy(deep=True)
        else:
            instance = cls()
        if overrides:
            update_data = dict(overrides.items())
            instance = instance.model_copy(update=update_data, deep=True)
        return instance

    def apply_override(
        self, key: str, value: t.Scalar | Sequence[t.Scalar] | Mapping[str, t.Scalar]
    ) -> bool:
        """Validate and apply a configuration override.

        Checks field existence in model_fields before applying via setattr.

        Args:
            key: Configuration key to override
            value: New value to set

        Returns:
            True if override was valid and applied, False otherwise.

        """
        if key not in self.__class__.model_fields:
            return False
        setattr(self, key, value)
        return True

    def get_di_config_provider(self) -> t.Scalar:
        """Get dependency injection provider for this config.

        Returns a providers.Singleton instance via the runtime bridge.
        Type annotation stays framework-level to avoid DI imports in this module.
        """
        if self._di_provider is None:
            providers_module = FlextRuntime.dependency_providers()
            self._di_provider = providers_module.Singleton(lambda: self)
        provider = self._di_provider
        if provider is None:
            msg = "DI provider not initialized"
            raise RuntimeError(msg)
        return provider

    @model_validator(mode="after")
    def _validate_configuration(self) -> Self:
        """Validate configuration.

        Business Rule: Validates configuration consistency after model initialization.
        Delegates to ``u`` validation utilities for database URL scheme
        and trace/debug consistency checks.

        Returns:
            Self: Validated configuration instance

        Raises:
            ValueError: If configuration is invalid

        """
        u.validate_database_url_scheme(self.database_url)
        u.validate_trace_requires_debug(
            trace=self.trace,
            debug=self.debug,
        )
        return self

    class AutoConfig(BaseModel):
        """Auto-configuration model for dynamic config creation."""

        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

        config_class: Annotated[
            type[BaseSettings],
            Field(description="Settings class to instantiate"),
        ]
        env_prefix: Annotated[
            str,
            Field(
                default=c.Platform.ENV_PREFIX,
                description="Environment variable prefix for settings resolution",
            ),
        ]
        env_file: Annotated[
            str | None,
            Field(
                default=None,
                description="Path to .env file for environment variable loading",
            ),
        ]

        def create_config(self) -> BaseSettings:
            """Create configuration instance."""
            return self.config_class()

    _namespace_registry: ClassVar[dict[str, type[BaseSettings]]] = {}
    _context_overrides: ClassVar[dict[str, dict[str, t.Scalar]]] = {}

    def __getattr__(self, name: str) -> BaseSettings:
        """Resolve namespace-style attribute access to registered settings."""
        pydantic_private = object.__getattribute__(self, "__pydantic_private__")
        if pydantic_private is not None and name in pydantic_private:
            return pydantic_private[name]
        namespace = name.lower()
        if namespace in {"core", "root", "settings"}:
            return FlextSettings.get_global()
        namespace_key = namespace
        config_class = self._namespace_registry.get(namespace_key)
        if config_class is None:
            normalized = u.normalize_alnum(namespace)
            if normalized:
                for key, value in self._namespace_registry.items():
                    key_normalized = u.normalize_alnum(key)
                    if normalized == key_normalized or normalized.startswith(
                        key_normalized
                    ):
                        namespace_key = key
                        config_class = value
                        break
        if config_class is None:
            msg = f"Namespace '{name}' not registered"
            raise AttributeError(msg)
        return self.get_namespace(namespace_key, config_class)

    @classmethod
    def for_context(cls, context_id: str, **overrides: t.Scalar) -> Self:
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
        base = cls.get_global()
        context_overrides = cls._context_overrides.get(context_id, {})
        all_overrides = {**context_overrides, **overrides}
        if all_overrides:
            return base.model_copy(update=all_overrides)
        return base

    @classmethod
    def get_namespace_config(cls, namespace: str) -> type[BaseSettings] | None:
        """Internal namespace registry lookup."""
        return cls._namespace_registry.get(namespace)

    @classmethod
    def register_context_overrides(cls, context_id: str, **overrides: t.Scalar) -> None:
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
    def register_namespace(
        cls,
        namespace: str,
        config_class: type[BaseSettings] | None = None,
        *,
        decorator: bool = False,
    ) -> Callable[[type[T_Settings]], type[T_Settings]] | None:
        """Register a configuration class for a namespace.

        When ``decorator=True``, returns a decorator that registers the class.

        Args:
            namespace: Namespace identifier
            config_class: Configuration class to register
            decorator: If True, return a decorator-style registrar

        """
        if decorator:

            def namespace_decorator(
                class_to_register: type[T_Settings],
            ) -> type[T_Settings]:
                """Register the configuration class while preserving type."""
                cls._namespace_registry[namespace] = class_to_register
                return class_to_register

            return namespace_decorator
        if config_class is None:
            msg = "config_class is required when decorator=False"
            raise ValueError(msg)
        cls._namespace_registry[namespace] = config_class
        return None

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset the global singleton instance for testing."""
        cls._reset_instance()
        cls._context_overrides.clear()

    @staticmethod
    def auto_register(namespace: str) -> Callable[[type[T_Settings]], type[T_Settings]]:
        """Build a decorator that registers a settings class by namespace."""

        def decorator(cls: type[T_Settings]) -> type[T_Settings]:
            FlextSettings._namespace_registry[namespace] = cls
            return cls

        return decorator

    def get_namespace(
        self, namespace: str, config_type: type[T_Namespace]
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
        config_class_raw = self._namespace_registry.get(namespace)
        if config_class_raw is None:
            msg = f"Namespace '{namespace}' not registered"
            raise ValueError(msg)
        config_instance = config_class_raw()
        if u.is_instance_of(config_instance, config_type):
            return config_instance
        msg = f"Namespace '{namespace}' config instance {config_instance.__class__.__name__} is not instance of {config_type.__name__}"
        raise TypeError(msg)


__all__ = ["FlextSettings"]
