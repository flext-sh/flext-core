"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from flext_core import p, t
from .flextprotocolscontext_part_02 import (
    FlextProtocolsContext as FlextProtocolsContextPart02,
)


class FlextProtocolsContext(FlextProtocolsContextPart02):
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options a service base declares for its runtime."""

        settings: p.Settings | None
        settings_type: t.SettingsClass | None
        settings_overrides: t.ScalarMapping | None
        context: p.Context | None
        dispatcher: p.Dispatcher | None


__all__: list[str] = ["FlextProtocolsContext"]
