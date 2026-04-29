"""FlextSettings — Settings Management Facade.

This module composes all settings concerns via MRO, matching the namespace
pattern used by ``c``, ``m``, ``p``, ``t`` and ``u``.  The concrete
implementation lives in the ``_settings`` subpackage; this file is only a
thin façade that inherits ``BaseSettings`` plus the relevant mixins.

Singleton storage is per-class (``cls._instance``) rather than a central
``_instances`` dictionary, so every concrete settings class owns its own
lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Literal, Self, override

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.sources.base import (
    DefaultSettingsSource,
    InitSettingsSource,
)
from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource
from pydantic_settings.sources.providers.secrets import SecretsSettingsSource

from flext_core import c, e, p, t, u
from flext_core._models.settings import FlextModelsSettings
from flext_core._settings.base import FlextSettingsBase
from flext_core._settings.context import FlextSettingsContext
from flext_core._settings.core import FlextSettingsCore
from flext_core._settings.database import FlextSettingsDatabase
from flext_core._settings.di import FlextSettingsDI
from flext_core._settings.dispatcher import FlextSettingsDispatcher
from flext_core._settings.infrastructure import FlextSettingsInfrastructure
from flext_core._settings.registry import FlextSettingsRegistry


def _resolve_env_file_bootstrap() -> str:
    """Resolve .env file path from FLEXT_ENV_FILE env var."""
    custom_env_file = os.environ.get(c.ENV_FILE_ENV_VAR)
    if custom_env_file:
        custom_path = Path(custom_env_file)
        return str(custom_path.resolve()) if custom_path.exists() else custom_env_file
    default_path = Path.cwd() / c.ENV_FILE_DEFAULT
    return str(default_path.resolve()) if default_path.exists() else c.ENV_FILE_DEFAULT


class FlextSettings(
    BaseSettings,
    FlextSettingsBase,
    FlextSettingsCore,
    FlextSettingsDatabase,
    FlextSettingsDispatcher,
    FlextSettingsInfrastructure,
    FlextSettingsDI,
    FlextSettingsRegistry,
    FlextSettingsContext,
):
    """Enterprise-grade settings management composed via MRO.

    Architecture: Layer 0.5 (Settings Foundation)
    Provides type-safe settings through Pydantic BaseSettings with environment
    variable support, per-class singleton lifecycle, and namespace registry.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix=c.ENV_PREFIX,
        env_nested_delimiter=c.ENV_NESTED_DELIMITER,
        env_file=_resolve_env_file_bootstrap(),
        env_file_encoding=c.DEFAULT_ENCODING,
        case_sensitive=False,
        extra=c.EXTRA_CONFIG_IGNORE,
        validate_assignment=True,
    )

    def __init__(self, **kwargs: t.SettingsInput) -> None:
        """Initialize settings with data.

        First initialization delegates directly to BaseSettings so incoming
        payloads participate in the native settings build pipeline. Repeated
        construction of the singleton revalidates one packed payload update.
        """
        if hasattr(self, "_di_provider"):
            if kwargs:
                init_payload = {
                    **self.model_dump(exclude_computed_fields=True),
                    **dict(kwargs),
                }
                sources, init_kwargs = self.__class__._runtime_settings_sources(
                    init_payload,
                )
                built_values = self.__class__._settings_build_values(
                    sources,
                    init_kwargs,
                )
                self.__pydantic_validator__.validate_python(
                    built_values,
                    self_instance=self,
                )
            return

        if kwargs:
            sources, init_kwargs = self.__class__._runtime_settings_sources(
                dict(kwargs),
            )
            super().__init__(_build_sources=(sources, init_kwargs))
            return

        super().__init__()

    @classmethod
    def _runtime_settings_sources(
        cls,
        init_payload: dict[str, t.SettingsInput],
    ) -> tuple[tuple[PydanticBaseSettingsSource, ...], dict[str, t.SettingsInput]]:
        """Build the native settings sources tuple for an init payload."""
        default_settings = DefaultSettingsSource(cls)
        init_settings = InitSettingsSource(cls, init_kwargs=init_payload)
        env_settings = EnvSettingsSource(cls)
        dotenv_settings = DotEnvSettingsSource(cls)
        file_secret_settings = SecretsSettingsSource(cls)
        sources = cls.settings_customise_sources(
            cls,
            init_settings=init_settings,
            env_settings=env_settings,
            dotenv_settings=dotenv_settings,
            file_secret_settings=file_secret_settings,
        ) + (default_settings,)
        return sources, init_payload

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
            cfg: t.JsonMapping | None = getattr(
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

    @classmethod
    @override
    def model_validate(
        cls,
        obj: t.ConfigModelInput,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: t.MetadataInput | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate settings payloads through the constructor for mappings.

        ``BaseSettings`` subclasses need constructor-based validation so init
        payloads are combined with env-backed defaults instead of being reduced
        to the pre-built settings source snapshot.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Mapping):
            payload = dict(obj)
            if not any(
                option is not None
                for option in (
                    strict,
                    extra,
                    from_attributes,
                    context,
                    by_alias,
                    by_name,
                )
            ):
                return cls(**payload)
        resolved_extra: Literal["allow", "forbid", "ignore"] | None
        match extra:
            case "allow" | "forbid" | "ignore":
                resolved_extra = extra
            case _:
                resolved_extra = None
        return super().model_validate(
            obj,
            strict=strict,
            extra=resolved_extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def fetch_global(cls, *, overrides: t.ScalarMapping | None = None) -> Self:
        """Get global settings, optionally materialized with overrides."""
        if overrides is None:
            return cls()
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

        Returns:
            True if override was valid and applied, False otherwise.

        """
        if key not in self.__class__.model_fields:
            return False
        setattr(self, key, value)
        return True

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset the global singleton instance for testing."""
        cls._reset_instance()
        cls._context_overrides.clear()

    def __getattr__(self, name: str) -> p.Settings:
        """Resolve namespace-style attribute access to registered settings."""
        try:
            pydantic_private: dict[str, p.Settings] | None = (
                object.__getattribute__(self, "__pydantic_private__")
            )
        except AttributeError:
            pydantic_private = None
        if pydantic_private is not None and name in pydantic_private:
            return pydantic_private[name]
        namespace = name.lower()
        if namespace in {"core", "root", "settings"}:
            return self.__class__.fetch_global()
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
        """
        base = cls.fetch_global()
        context_overrides = cls._context_overrides.get(context_id, {})
        all_overrides = {**context_overrides, **overrides}
        if not all_overrides:
            return base
        copied = base.model_copy(update=all_overrides)
        # ``model_copy(update=...)`` bypasses validators by design; re-run them
        # so field-level validators (e.g., ``log_level`` enum coercion) apply
        copied.__pydantic_validator__.validate_python(
            copied.__dict__, self_instance=copied
        )
        return copied

    def fetch_namespace[TNamespace: p.Settings](
        self,
        namespace: str,
        settings_type: type[TNamespace],
    ) -> TNamespace:
        """Get settings instance for a namespace.

        Raises:
            ValueError: If namespace not found
            TypeError: If type mismatch

        """
        settings_class_raw = self._namespace_registry.get(namespace)
        if settings_class_raw is None:
            raise ValueError(
                e.render_template(
                    "Namespace '{namespace}' not registered",
                    namespace=namespace,
                ),
            )
        settings_instance = settings_class_raw()
        if isinstance(settings_instance, settings_type):
            return settings_instance
        raise TypeError(
            e.render_template(
                "Namespace '{namespace}' settings instance {instance_class} is not instance of {expected_type}",
                namespace=namespace,
                instance_class=settings_instance.__class__.__name__,
                expected_type=settings_type.__name__,
            ),
        )

    @model_validator(mode="after")
    def _validate_settings(self) -> Self:
        """Validate settings consistency after model initialization."""
        if self.database_url and not self.database_url.startswith((
            "postgresql://",
            "mysql://",
            "sqlite://",
        )):
            raise ValueError(c.ERR_CONFIG_INVALID_DB_URL_SCHEME)
        if self.trace and not self.debug:
            raise ValueError(c.ERR_CONFIG_TRACE_REQUIRES_DEBUG)
        return self

    AutoSettings: ClassVar[type[FlextModelsSettings.AutoSettings]] = (
        FlextModelsSettings.AutoSettings
    )


# Expose only FlextSettings as public API
__all__: list[str] = ["FlextSettings"]
