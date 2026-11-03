# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/flext_mixins
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsContext, FlextProtocolsSettings
    from flext_core._typings import FlextTypingBase


class FlextProtocolsFlextMixins:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ServiceState(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings_type(self) -> FlextProtocolsSettings.SettingsType | None:
            """FlextSettings class used to initialize the service runtime."""
            ...

        @property
        def runtime_settings(self) -> FlextProtocolsSettings.Settings | None:
            """Pre-built settings instance used directly for the runtime."""
            ...

        @property
        def settings_overrides(self) -> FlextTypingBase.JsonMapping | None:
            """Settings overrides applied at instantiation."""
            ...

        @property
        def initial_context(self) -> FlextProtocolsContext.Context | None:
            """Initial context for the service scope."""
            ...


__all__: list[str] = [
    "FlextProtocolsFlextMixins",
]
