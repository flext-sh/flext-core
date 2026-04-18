"""FlextProtocolsSettings - settings protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core import FlextProtocolsBase as p, FlextProtocolsResult

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsSettings:
    """Protocols for configurable components and settings."""

    @runtime_checkable
    class Configurable(p.Base, Protocol):
        """Protocol for component configuration."""

        def configure(
            self,
            settings: t.FlatContainerMapping | None = None,
        ) -> Self:
            """Configure component with settings."""
            ...

    @runtime_checkable
    class Settings(
        FlextProtocolsResult.HasModelDump,
        p.Base,
        Protocol,
    ):
        """Configuration protocol based on Pydantic BaseSettings pattern.

        Reflects real implementations like FlextSettings which uses Pydantic BaseSettings.
        Configuration items use direct field access (Pydantic standard) rather than
        explicit get/set methods. Supports cloning via model_copy() and optional
        override methods.
        """

        app_name: str
        "Application name bound to the configuration."
        version: str
        "Semantic version of the running application."
        enable_caching: bool
        "Enable caching for query operations."
        timeout_seconds: float
        "Default timeout in seconds for operations."
        dispatcher_auto_context: bool
        "Enable automatic context management in dispatcher."
        dispatcher_enable_logging: bool
        "Enable logging in dispatcher operations."

        @classmethod
        def fetch_global(
            cls,
            *,
            overrides: t.ScalarMapping | None = None,
        ) -> Self:
            """Return the global singleton settings instance, optionally with overrides."""
            ...

        def model_copy(
            self,
            *,
            update: t.FlatContainerMapping | None = None,
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

    @runtime_checkable
    class SettingsType(Protocol):
        """Protocol for concrete settings classes with singleton access."""

        @classmethod
        def fetch_global(
            cls,
            *,
            overrides: t.ScalarMapping | None = None,
        ) -> FlextProtocolsSettings.Settings:
            """Return the global singleton settings instance."""
            ...


__all__: list[str] = ["FlextProtocolsSettings"]
