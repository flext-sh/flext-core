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
    from flext_core import t
    from flext_core._protocols import FlextProtocolsContext, FlextProtocolsSettings


class FlextProtocolsFlextMixins:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ServiceState(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        settings_type: FlextProtocolsSettings.SettingsType | None
        runtime_settings: FlextProtocolsSettings.Settings | None
        settings_overrides: t.JsonMapping | None
        initial_context: FlextProtocolsContext.Context | None


__all__: list[str] = [
    "FlextProtocolsFlextMixins",
]
