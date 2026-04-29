"""FlextSettings — Settings Management Facade.

Composes all settings concerns via MRO, matching the namespace pattern used
by ``c``, ``m``, ``p``, ``t`` and ``u``.  The concrete implementation lives in
``_settings/``; this file is a thin façade inheriting ``BaseSettings`` plus the
relevant mixins.

Singleton storage is per-class (``cls._instance``) so every concrete settings
class owns its own lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextProtocols as p,
    FlextTypes as t,
    FlextUtilities as u,
)
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

    # ``__init__`` is intentionally omitted.  Pydantic v2 ``BaseSettings``
    # handles both first-time construction and re-initialisation of an
    # existing instance correctly when ``extra='ignore'``.  The singleton
    # logic lives exclusively in ``__new__`` (see ``FlextSettingsBase``).

    @classmethod
    def fetch_global(cls, *, overrides: t.ScalarMapping | None = None) -> Self:
        """Get global settings, optionally materialized with overrides."""
        instance = cls()
        if not overrides:
            return instance
        copied = instance.model_copy(update=dict(overrides), deep=True)
        copied.__pydantic_validator__.validate_python(
            copied.__dict__, self_instance=copied
        )
        return copied

    def clone(self, **overrides: t.JsonPayload | None) -> Self:
        """Create a deep copy with optional field overrides and re-validation.

        This is the canonical way for containers and services to obtain an
        isolated settings snapshot without mutating the global singleton.
        Nested models (e.g. ``Meltano`` inner classes) are deep-copied by
        default so mutations in the clone do not leak back to the source.

        Args:
            **overrides: Keyword arguments mapped to model field names.

        Returns:
            A new settings instance with overrides applied.

        """
        if not overrides:
            return self.model_copy(deep=True)
        copied = self.model_copy(update=dict(overrides), deep=True)
        # ``model_copy(update=...)`` bypasses validators by design; re-run them
        # so field-level validators (e.g., ``log_level`` enum coercion) apply.
        copied.__pydantic_validator__.validate_python(
            copied.__dict__, self_instance=copied
        )
        return copied

    def apply_override(
        self,
        key: str,
        value: t.Scalar | t.ScalarList | t.ScalarMapping,
    ) -> bool:
        """Validate and apply a settings override.

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
        if (
            name in self.__class__.model_fields
            or name in self.__class__.model_computed_fields
        ):
            msg = f"{name!r} is a model field/computed_field — not a namespace"
            raise AttributeError(msg)
        try:
            pydantic_private: dict[str, p.Settings] | None = object.__getattribute__(
                self, "__pydantic_private__"
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
        """Get settings instance with context-specific overrides."""
        base = cls.fetch_global()
        context_overrides = cls._context_overrides.get(context_id, {})
        all_overrides = {**context_overrides, **overrides}
        if not all_overrides:
            return base
        return base.clone(**all_overrides)

    def fetch_namespace[TNamespace: p.Settings](
        self,
        namespace: str,
        settings_type: type[TNamespace],
    ) -> TNamespace:
        """Get settings instance for a namespace.

        Raises:
            ValueError: If namespace not found.
            TypeError: If type mismatch.

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


__all__: list[str] = ["FlextSettings"]
