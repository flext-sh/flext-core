"""FlextProtocolsSettings - settings protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from .base import FlextProtocolsBase as p
from .result import FlextProtocolsResult as pr

if TYPE_CHECKING:
    from flext_core import FlextTypes as t


class FlextProtocolsSettings:
    """Protocols for configurable components and settings."""

    @runtime_checkable
    class Configurable(p.Base, Protocol):
        """Protocol for component configuration."""

        def apply(self, settings: t.UserOverridesMapping | None = None) -> Self:
            """Apply configuration overrides."""
            ...

    @runtime_checkable
    class Settings(pr.HasModelDump, p.Base, Protocol):
        """Minimal Pydantic-2 settings contract and universal log level.

        Declares the operation surface (``fetch_global``, ``clone``,
        ``update_global``, ``model_copy``, ``model_dump``) and ``log_level``,
        which ``FlextSettings`` guarantees for every project subclass.
        Concrete root fields (``app_name``, ``version``, ``timeout_seconds``,
        ``dispatcher_*``, …) are declared on ``FlextSettings`` itself, not on
        the protocol — project subclasses must NOT inherit them
        (workspace rule 3 isolation).
        """

        @property
        def log_level(self) -> str:
            """Universal runtime log level."""
            ...

        @classmethod
        def fetch_global(cls, *, overrides: t.ScalarMapping | None = None) -> Self:
            """Return the global singleton settings instance, optionally with overrides."""
            ...

        def model_copy(
            self,
            *,
            update: t.SettingsOverridesMapping | None = None,
            deep: bool = False,
        ) -> Self:
            """Create a copy of the model, optionally updating fields or deep copying.

            Args:
                update: Dictionary of values to update in the copied model.
                deep: If True, perform a deep copy of the model fields.

            Returns:
                A new instance of the model.

            """
            ...

        def clone(self, **overrides: t.SettingsOverride | None) -> Self:
            """Create a deep copy with optional field overrides.

            This is the canonical way for containers and services to obtain an
            isolated settings snapshot without mutating the global singleton.

            Args:
                **overrides: Keyword arguments mapped to model field names.

            Returns:
                A new settings instance with overrides applied.

            """
            ...

        @classmethod
        def update_global(cls, **overrides: t.SettingsOverride | None) -> Self:
            """Replace the per-class singleton via Pydantic-2 ``model_copy(update=…)``.

            Pure Pydantic-2 mutation — no ``setattr``, no ``__setattr__``
            override, no ``apply_override`` ad-hoc method. Subsequent
            ``fetch_global()`` calls return the new instance — propagates
            globally per the singleton-unification rule.

            Args:
                **overrides: Keyword arguments mapped to model field names.

            Returns:
                The new settings instance now installed as the singleton.

            """
            ...

    @runtime_checkable
    class SettingsType(Protocol):
        """Protocol for concrete settings classes with singleton access."""

        @classmethod
        def fetch_global(
            cls, *, overrides: t.ScalarMapping | None = None
        ) -> FlextProtocolsSettings.Settings:
            """Return the global singleton settings instance."""
            ...


__all__: list[str] = ["FlextProtocolsSettings"]
