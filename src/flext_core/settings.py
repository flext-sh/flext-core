"""FlextSettings — root settings facade.

Composes the per-class singleton + canonical helper API provided by
:class:`FlextSettingsBase` with the root field mixins
(``Core``/``Database``/``Dispatcher``/``Infrastructure``) plus DI/Registry/Context.

Helper methods (``fetch_global``, ``clone``, ``update_global``,
``validate_overrides``, ``clone_for_injection``, ``reset_for_testing``,
``resolve_env_file``) live on :class:`FlextSettingsBase` so project-specific
settings classes can inherit them without contaminating their fields with the
root concrete fields.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar, Self

from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from flext_core._runtime._base import FlextRuntimeBase as FlextRuntime

from ._constants.environment import FlextConstantsEnvironment
from ._constants.errors import FlextConstantsErrors
from ._constants.serialization import FlextConstantsSerialization
from ._constants.settings import FlextConstantsSettings
from ._protocols.settings import FlextProtocolsSettings as p
from ._settings.base import FlextSettingsBase
from ._settings.context import FlextSettingsContext
from ._settings.core import FlextSettingsCore
from ._settings.database import FlextSettingsDatabase
from ._settings.di import FlextSettingsDI
from ._settings.dispatcher import FlextSettingsDispatcher
from ._settings.infrastructure import FlextSettingsInfrastructure
from ._settings.registry import FlextSettingsRegistry
from ._typings.base import FlextTypingBase as t


class FlextSettings(
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
    Provides type-safe settings through Pydantic ``BaseSettings`` with environment
    variable support, per-class singleton lifecycle (via ``FlextSettingsBase``),
    and namespace registry (via ``FlextSettingsRegistry``).
    """

    Base: ClassVar[type[FlextSettingsBase]] = FlextSettingsBase

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix=FlextConstantsEnvironment.ENV_PREFIX,
        env_nested_delimiter=FlextConstantsEnvironment.ENV_NESTED_DELIMITER,
        env_file=FlextSettingsBase.resolve_env_file(),
        env_file_encoding=FlextConstantsSerialization.DEFAULT_ENCODING,
        case_sensitive=False,
        extra=FlextConstantsSettings.EXTRA_CONFIG_IGNORE,
        validate_assignment=True,
    )

    def apply_override(
        self,
        key: str,
        value: t.Scalar | t.ScalarList | t.ScalarMapping,
    ) -> bool:
        """DEPRECATED — replaced by ``update_global(**overrides)`` (Pydantic-2).

        Kept until Phase 4.2 sweep. New code MUST use ``update_global``.

        Returns:
            True if override was valid and applied, False otherwise.

        """
        if key not in self.__class__.model_fields:
            return False
        setattr(self, key, value)
        return True

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
                self, "__pydantic_private__",
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
            normalized = FlextRuntime.normalize_alnum(namespace)
            if normalized:
                for key, value in self._namespace_registry.items():
                    key_normalized = FlextRuntime.normalize_alnum(key)
                    if normalized == key_normalized or normalized.startswith(
                        key_normalized,
                    ):
                        namespace_key = key
                        settings_class = value
                        break
        if settings_class is None:
            msg = f"Namespace '{name}' not registered"
            raise AttributeError(msg)
        namespace_settings: p.Settings = self.fetch_namespace(
            namespace_key, settings_class,
        )
        return namespace_settings

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
            error_message = f"Namespace '{namespace}' not registered"
            raise ValueError(error_message)
        raw_instance = getattr(settings_class_raw, "_instance", None)
        settings_instance = (
            raw_instance
            if isinstance(raw_instance, p.Settings)
            else settings_class_raw()
        )
        if isinstance(settings_instance, settings_type):
            return settings_instance
        error_message = (
            f"Namespace '{namespace}' settings instance "
            f"{settings_instance.__class__.__name__} is not instance of "
            f"{settings_type.__name__}"
        )
        raise TypeError(error_message)

    @model_validator(mode="after")
    def _validate_settings(self) -> Self:
        """Validate settings consistency after model initialization."""
        if self.database_url and not self.database_url.startswith((
            "postgresql://",
            "mysql://",
            "sqlite://",
        )):
            raise ValueError(FlextConstantsErrors.ERR_CONFIG_INVALID_DB_URL_SCHEME)
        if self.trace and not self.debug:
            raise ValueError(FlextConstantsErrors.ERR_CONFIG_TRACE_REQUIRES_DEBUG)
        return self


__all__: list[str] = ["FlextSettings", "FlextSettingsBase"]
