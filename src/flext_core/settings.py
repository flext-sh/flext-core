"""FlextSettings — root settings facade.

Composes ``BaseSettings`` with the per-class singleton + canonical helper API
provided by :class:`FlextSettingsBase` and the root field mixins
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
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core import (
    FlextModelsSettings,
    FlextSettingsBase,
    FlextSettingsContext,
    FlextSettingsCore,
    FlextSettingsDatabase,
    FlextSettingsDI,
    FlextSettingsDispatcher,
    FlextSettingsInfrastructure,
    FlextSettingsRegistry,
    c,
    e,
    p,
    t,
    u,
)


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
    Provides type-safe settings through Pydantic ``BaseSettings`` with environment
    variable support, per-class singleton lifecycle (via ``FlextSettingsBase``),
    and namespace registry (via ``FlextSettingsRegistry``).
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix=c.ENV_PREFIX,
        env_nested_delimiter=c.ENV_NESTED_DELIMITER,
        env_file=FlextSettingsBase.resolve_env_file(),
        env_file_encoding=c.DEFAULT_ENCODING,
        case_sensitive=False,
        extra=c.EXTRA_CONFIG_IGNORE,
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
        raw_instance: FlextSettingsBase | None = getattr(
            settings_class_raw,
            "_instance",
            None,
        )
        settings_instance = (
            raw_instance if raw_instance is not None else settings_class_raw()
        )
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
