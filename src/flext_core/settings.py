"""FlextSettings - Settings Management Module.

This module provides comprehensive settings management for the FLEXT ecosystem,
implementing Pydantic v2 BaseSettings with dependency injection, environment variable support,
and runtime validation. Serves as the foundation layer (0.5) controlling all other layers.

Scope: Global settings management, singleton pattern, DI integration, validation,
environment variable handling, thread-safe operations, and dynamic settings updates.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import threading
from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
)
from pathlib import Path
from typing import Annotated, ClassVar, Self, override

from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from flext_core import (
    FlextModelsExceptionParams,
    FlextModelsSettings,
    __version__,
    c,
    e,
    p,
    t,
    u,
)


def _resolve_env_file_bootstrap() -> str:
    """Resolve .env file path from FLEXT_ENV_FILE env var.

    Bootstrap-layer inline function (AGENTS.md §2.6 exception).
    Avoids circular import of FlextUtilitiesSettings during settings initialization.
    """
    custom_env_file = os.environ.get(c.ENV_FILE_ENV_VAR)
    if custom_env_file:
        custom_path = Path(custom_env_file)
        if custom_path.exists():
            return str(custom_path.resolve())
        return custom_env_file
    default_path = Path.cwd() / c.ENV_FILE_DEFAULT
    if default_path.exists():
        return str(default_path.resolve())
    return c.ENV_FILE_DEFAULT


def _validate_database_url_scheme_bootstrap(url: str) -> None:
    """Validate database URL scheme is postgresql://, mysql://, or sqlite://.

    Bootstrap-layer inline validator (AGENTS.md §2.6 exception).
    """
    if url and not url.startswith(("postgresql://", "mysql://", "sqlite://")):
        raise ValueError(c.ERR_CONFIG_INVALID_DB_URL_SCHEME)


def _validate_trace_requires_debug_bootstrap(*, trace: bool, debug: bool) -> None:
    """Validate that trace mode requires debug mode.

    Bootstrap-layer inline validator (AGENTS.md §2.6 exception).
    """
    if trace and not debug:
        raise ValueError(c.ERR_CONFIG_TRACE_REQUIRES_DEBUG)


class FlextSettings(BaseSettings):
    """Settings management with Pydantic validation and dependency injection.

    Architecture: Layer 0.5 (Settings Foundation)
    Provides enterprise-grade settings management for the FLEXT ecosystem
    through Pydantic BaseSettings with natural protocol compliance.

    Core Features:
    - Pydantic v2 BaseSettings with type-safe settings
    - Environment variable support with FLEXT_ prefix
    - Thread-safe singleton pattern
    - Dependency injection integration
    - Runtime settings updates
    - Protocol compliance via inheritance (p.Settings)
    """

    _instances: ClassVar[MutableMapping[type[Self], Self]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _singleton_enabled: ClassVar[bool] = True

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix=c.ENV_PREFIX,
        env_nested_delimiter=c.ENV_NESTED_DELIMITER,
        env_file=_resolve_env_file_bootstrap(),
        env_file_encoding=c.DEFAULT_ENCODING,
        case_sensitive=False,
        extra=c.EXTRA_CONFIG_IGNORE,
        validate_assignment=True,
    )

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Auto-discover parent env prefixes from MRO for fallback resolution.

        Uses Pydantic's built-in env_settings for the leaf class, then adds
        parent env prefixes as fallback sources in MRO order.
        Priority: init > leaf env_prefix > parent env_prefixes (MRO order) > dotenv > secrets.
        """
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        leaf_prefix = cls.model_config.get("env_prefix", "")
        for parent in cls.__mro__:
            cfg: Mapping[str, t.Container] | None = getattr(
                parent,
                "model_config",
                None,
            )
            if not isinstance(cfg, Mapping):
                continue
            raw_prefix = cfg.get("env_prefix", "")
            parent_prefix = str(raw_prefix) if raw_prefix else ""
            if parent_prefix and parent_prefix != leaf_prefix:
                sources.append(
                    EnvSettingsSource(settings_cls, env_prefix=parent_prefix),
                )
        sources.extend([dotenv_settings, file_secret_settings])
        return tuple(sources)

    app_name: Annotated[str, Field(description="Application name")] = c.DEFAULT_APP_NAME
    version: Annotated[str, Field(description="Application version")] = __version__
    debug: Annotated[bool, Field(description="Enable debug mode")] = False
    trace: Annotated[bool, Field(description="Enable trace mode")] = False
    log_level: Annotated[c.LogLevel, Field(description="Log level")] = c.LogLevel.INFO
    async_logging: Annotated[
        bool,
        Field(
            description="Enable asynchronous buffered logging for performance",
        ),
    ] = True
    enable_caching: Annotated[bool, Field(description="Enable caching")] = (
        c.ASYNC_ENABLED
    )
    cache_ttl: Annotated[t.PositiveInt, Field(description="Cache TTL")] = c.CACHE_TTL
    database_url: Annotated[str, Field(description="Database URL")] = c.DATABASE_URL
    database_pool_size: Annotated[
        t.PositiveInt,
        Field(
            description="Database pool size",
        ),
    ] = c.DEFAULT_PAGE_SIZE
    circuit_breaker_threshold: Annotated[
        t.PositiveInt,
        Field(
            description="Circuit breaker threshold",
        ),
    ] = c.BACKUP_COUNT
    rate_limit_max_requests: Annotated[
        t.PositiveInt,
        Field(
            description="Rate limit max requests",
        ),
    ] = c.HTTP_STATUS_MIN
    rate_limit_window_seconds: Annotated[
        t.PositiveInt,
        Field(
            description="Rate limit window",
        ),
    ] = c.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
    retry_delay: Annotated[
        t.PositiveInt,
        Field(
            description="Retry delay",
        ),
    ] = c.DEFAULT_RETRY_DELAY_SECONDS
    max_retry_attempts: Annotated[
        t.RetryCount,
        Field(
            description="Max retry attempts",
        ),
    ] = c.MAX_RETRY_ATTEMPTS
    enable_timeout_executor: Annotated[
        bool, Field(description="Enable timeout executor")
    ] = True
    dispatcher_enable_logging: Annotated[
        bool,
        Field(
            description="Enable dispatcher logging",
        ),
    ] = c.ASYNC_ENABLED
    dispatcher_auto_context: Annotated[
        bool,
        Field(
            description="Auto context in dispatcher",
        ),
    ] = c.ASYNC_ENABLED
    dispatcher_timeout_seconds: Annotated[
        t.PositiveTimeout,
        Field(
            description="Dispatcher timeout",
        ),
    ] = c.DEFAULT_TIMEOUT_SECONDS
    dispatcher_enable_metrics: Annotated[
        bool,
        Field(
            description="Enable dispatcher metrics",
        ),
    ] = c.ASYNC_ENABLED
    executor_workers: Annotated[
        t.WorkerCount, Field(description="Executor workers")
    ] = c.DEFAULT_MAX_WORKERS
    timeout_seconds: Annotated[
        t.PositiveTimeout, Field(description="Default timeout")
    ] = c.DEFAULT_TIMEOUT_SECONDS
    max_workers: Annotated[t.WorkerCount, Field(description="Max workers")] = (
        c.DEFAULT_MAX_WORKERS
    )
    max_batch_size: Annotated[t.BatchSize, Field(description="Max batch size")] = (
        c.MAX_ITEMS
    )
    api_key: Annotated[str | None, Field(description="API key")] = None
    exception_failure_level: Annotated[
        c.FailureLevel,
        Field(
            description="Exception failure level",
        ),
    ] = c.FAILURE_LEVEL_DEFAULT
    _di_provider: t.Scalar | None = PrivateAttr(default=None)

    def __new__(cls, **_kwargs: t.SettingsValue | None) -> Self:
        """Create singleton instance.

        Note: BaseSettings.__init__ accepts **values internally.
        We override __new__ to implement singleton pattern while allowing
        kwargs to be passed for testing and settings via model_validate.
        """
        if not cls._singleton_enabled:
            return super().__new__(cls)
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

    def __init__(self, **kwargs: t.SettingsValue | None) -> None:
        """Initialize settings with data.

        Kwargs are applied as field overrides after base env/settings loading
        to avoid type conflicts with BaseSettings internal parameters.
        Uses __dict__ update + single revalidation to avoid triggering
        validate_assignment validators on each individual field set.
        """
        model_fields = self.__class__.model_fields
        if hasattr(self, "_di_provider"):
            if kwargs:
                valid_kwargs = {k: v for k, v in kwargs.items() if k in model_fields}
                vars(self).update(valid_kwargs)
                self.__pydantic_validator__.validate_python(
                    self.__dict__,
                    self_instance=self,
                )
            return

        BaseSettings.__init__(self)
        if kwargs:
            valid_kwargs = {k: v for k, v in kwargs.items() if k in model_fields}
            vars(self).update(valid_kwargs)
            self.__pydantic_validator__.validate_python(
                self.__dict__,
                self_instance=self,
            )

    @computed_field
    @property
    def effective_log_level(self) -> c.LogLevel:
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
    def fetch_global(cls, *, overrides: t.ScalarMapping | None = None) -> Self:
        """Get global settings, optionally materialized with overrides."""
        if overrides is None:
            return cls()
        if cls is FlextSettings:
            global_config = cls.fetch_global()
            instance = global_config.model_copy(deep=True)
        else:
            instance = cls()
        if overrides:
            update_data = dict(overrides)
            instance = instance.model_copy(update=update_data, deep=True)
        return instance

    def apply_override(
        self,
        key: str,
        value: t.Scalar | t.ScalarList | t.ScalarMapping,
    ) -> bool:
        """Validate and apply a settings override.

        Checks field existence in model_fields before applying via setattr.

        Args:
            key: Settings key to override
            value: New value to set

        Returns:
            True if override was valid and applied, False otherwise.

        """
        if key not in self.__class__.model_fields:
            return False
        setattr(self, key, value)
        return True

    def resolve_di_settings_provider(self) -> t.Scalar:
        """Get dependency injection provider for this settings.

        Returns a providers.Singleton instance via the runtime bridge.
        Type annotation stays framework-level to avoid DI imports in this module.
        """
        if self._di_provider is None:
            providers_module = u.dependency_providers()
            self._di_provider = providers_module.Singleton(lambda: self)
        provider = self._di_provider
        if provider is None:
            msg = c.ERR_SETTINGS_DI_PROVIDER_NOT_INITIALIZED
            raise RuntimeError(msg)
        return provider

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        """Normalize log level to uppercase before enum validation."""
        return v.upper()

    @model_validator(mode="after")
    def _validate_settings(self) -> Self:
        """Validate settings.

        Business Rule: Validates settings consistency after model initialization.
        Delegates to bootstrap validation utilities for database URL scheme
        and trace/debug consistency checks.

        Returns:
            Self: Validated settings instance

        Raises:
            ValueError: If settings are invalid

        """
        _validate_database_url_scheme_bootstrap(self.database_url)
        _validate_trace_requires_debug_bootstrap(
            trace=self.trace,
            debug=self.debug,
        )
        return self

    AutoSettings: ClassVar[type[FlextModelsSettings.AutoSettings]] = (
        FlextModelsSettings.AutoSettings
    )

    _namespace_registry: ClassVar[MutableMapping[str, t.SettingsClass]] = {}
    _context_overrides: ClassVar[t.ScopedScalarRegistry] = {}

    def __getattr__(self, name: str) -> p.Settings:
        """Resolve namespace-style attribute access to registered settings."""
        pydantic_private = object.__getattribute__(self, "__pydantic_private__")
        if pydantic_private is not None and name in pydantic_private:
            return pydantic_private[name]
        namespace = name.lower()
        if namespace in {"core", "root", "settings"}:
            return FlextSettings.fetch_global()
        namespace_key = namespace
        settings_class = self._namespace_registry.get(namespace_key)
        if settings_class is None:
            normalized = u.normalize_alnum(namespace)
            if normalized:
                for key, value in self._namespace_registry.items():
                    key_normalized = u.normalize_alnum(key)
                    if normalized == key_normalized or normalized.startswith(
                        key_normalized,
                    ):
                        namespace_key = key
                        settings_class = value
                        break
        if settings_class is None:
            msg = f"Namespace '{name}' not registered"
            raise AttributeError(msg)
        return self.fetch_namespace(namespace_key, settings_class)

    @classmethod
    def for_context(cls, context_id: str, **overrides: t.Scalar) -> Self:
        """Get settings instance with context-specific overrides.

        Creates a settings instance with overrides specific to the given
        context. Context overrides are applied on top of the base settings.

        Args:
            context_id: Unique identifier for the execution context.
            **overrides: Settings field overrides for this context.

        Returns:
            Self: Settings instance with context overrides applied.

        Example:
            >>> settings = FlextSettings.for_context(
            ...     "worker_1", log_level="DEBUG", timeout=60
            ... )

        """
        base = cls.fetch_global()
        context_overrides = cls._context_overrides.get(context_id, {})
        all_overrides = {**context_overrides, **overrides}
        if all_overrides:
            return base.model_copy(update=all_overrides)
        return base

    @classmethod
    def resolve_namespace_settings(cls, namespace: str) -> t.SettingsClass | None:
        """Internal namespace registry lookup."""
        return cls._namespace_registry.get(namespace)

    @classmethod
    def registered_namespaces(cls) -> t.StrSequence:
        """Return the currently registered settings namespaces."""
        return tuple(cls._namespace_registry.keys())

    @classmethod
    def register_context_overrides(cls, context_id: str, **overrides: t.Scalar) -> None:
        """Register context-specific settings overrides.

        Registers overrides that will be automatically applied when using
        `for_context()` with the same context_id.

        Args:
            context_id: Unique identifier for the execution context.
            **overrides: Settings field overrides to register.

        Example:
            >>> FlextSettings.register_context_overrides(
            ...     "worker_1", log_level="DEBUG", timeout=60
            ... )
            >>> settings = FlextSettings.for_context("worker_1")

        """
        cls._context_overrides.setdefault(context_id, {}).update(overrides)

    @classmethod
    def register_namespace[TSettings: p.Settings](
        cls,
        namespace: str,
        settings_class: type[TSettings] | None = None,
        *,
        decorator: bool = False,
    ) -> Callable[[type[TSettings]], type[TSettings]] | None:
        """Register a settings class for a namespace.

        When ``decorator=True``, returns a decorator that registers the class.

        Args:
            namespace: Namespace identifier
            settings_class: Settings class to register
            decorator: If True, return a decorator-style registrar

        """
        if decorator:

            def namespace_decorator(
                class_to_register: type[TSettings],
            ) -> type[TSettings]:
                """Register the settings class while preserving type."""
                cls._namespace_registry[namespace] = class_to_register
                return class_to_register

            return namespace_decorator
        if settings_class is None:
            msg = c.ERR_SETTINGS_CLASS_REQUIRED_FOR_NON_DECORATOR
            raise ValueError(msg)
        cls._namespace_registry[namespace] = settings_class
        return None

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset the global singleton instance for testing."""
        cls._reset_instance()
        cls._context_overrides.clear()

    @staticmethod
    def auto_register[TSettings: p.Settings](
        namespace: str,
    ) -> Callable[[type[TSettings]], type[TSettings]]:
        """Build a decorator that registers a settings class by namespace."""

        def decorator(cls: type[TSettings]) -> type[TSettings]:
            FlextSettings._namespace_registry[namespace] = cls
            return cls

        return decorator

    def fetch_namespace[TNamespace: p.Settings](
        self,
        namespace: str,
        settings_type: type[TNamespace],
    ) -> TNamespace:
        """Get settings instance for a namespace.

        Business Rule: Resolves namespace settings class from registry and
        instantiates it. Validates namespace exists and settings class is subclass of
        expected type. Raises ValueError if namespace not found, TypeError if type
        mismatch. Used for dynamic namespace settings resolution.

        Audit Implication: Namespace resolution ensures audit trail completeness by
        validating namespace settings before use. All namespace settings are
        validated before being used in production systems.

        Args:
            namespace: Namespace identifier
            settings_type: Expected settings type

        Returns:
            Settings instance

        Raises:
            ValueError: If namespace not found
            TypeError: If type mismatch

        """
        settings_class_raw = self._namespace_registry.get(namespace)
        if settings_class_raw is None:
            params = FlextModelsExceptionParams.ConfigurationErrorParams(
                config_key=namespace,
                config_source="namespace_registry",
            )
            raise ValueError(
                e.render_template(
                    "Namespace '{namespace}' not registered",
                    namespace=namespace,
                    params=params,
                ),
            )
        settings_instance = settings_class_raw()
        if isinstance(settings_instance, settings_type):
            return settings_instance
        params = FlextModelsExceptionParams.TypeErrorParams(
            expected_type=settings_type.__name__,
            actual_type=settings_instance.__class__.__name__,
        )
        raise TypeError(
            e.render_template(
                "Namespace '{namespace}' settings instance {instance_class} is not instance of {expected_type}",
                namespace=namespace,
                instance_class=settings_instance.__class__.__name__,
                expected_type=settings_type.__name__,
                params=params,
            ),
        )


__all__: list[str] = ["FlextSettings"]
