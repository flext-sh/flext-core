"""FlextSettingsRegistry — namespace registry.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import ClassVar

from flext_core import c, p, t


class FlextSettingsRegistry:
    """Mixin that adds a project-wide namespace registry.

    Each concrete settings class can be registered under a string namespace
    (e.g. ``"dbt_oracle"``) via ``@FlextSettings.auto_register("dbt_oracle")``.
    The registry itself is stored as a ClassVar on *this* mixin, but because it
    is declared here every subclass in the MRO shares the same dict object
    (which is the desired global registry behaviour).

    Dynamic attribute resolution (``__getattr__``) and ``fetch_namespace``
    live in the façade since they need ``BaseSettings`` methods.
    """

    _namespace_registry: ClassVar[MutableMapping[str, t.SettingsClass]] = {}

    @classmethod
    def resolve_namespace_settings(cls, namespace: str) -> t.SettingsClass | None:
        """Internal namespace registry lookup."""
        return cls._namespace_registry.get(namespace)

    @classmethod
    def registered_namespaces(cls) -> t.StrSequence:
        """Return the currently registered settings namespaces."""
        return tuple(cls._namespace_registry.keys())

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
        """
        if decorator:
            return cls.auto_register(namespace)
        if settings_class is None:
            msg = c.ERR_SETTINGS_CLASS_REQUIRED_FOR_NON_DECORATOR
            raise ValueError(msg)
        cls._namespace_registry[namespace] = settings_class
        return None

    @staticmethod
    def auto_register[TSettings: p.Settings](
        namespace: str,
    ) -> Callable[[type[TSettings]], type[TSettings]]:
        """Build a decorator that registers a settings class by namespace."""

        def decorator(cls: type[TSettings]) -> type[TSettings]:
            FlextSettingsRegistry._namespace_registry[namespace] = cls
            return cls

        return decorator


__all__: list[str] = ["FlextSettingsRegistry"]
