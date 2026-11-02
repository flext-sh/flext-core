# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/export
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core import p


class FlextProtocolsContextExport:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextExport(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def data(self) -> Mapping[str, t.JsonPayload]: ...
        @property
        def metadata(self) -> p.Metadata | p.Dict | None: ...
        @property
        def statistics(self) -> t.JsonMapping: ...


__all__: list[str] = [
    "FlextProtocolsContextExport",
]
