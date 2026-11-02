# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/containers
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Protocol, runtime_checkable

from flext_core import t


class FlextProtocolsContainers:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ValidatorCallable(Protocol):
        """Structural data contract projected from the owning model."""

        root: t.ValidatorCallable

    @runtime_checkable
    class _MappingRootBase(Protocol):
        """Structural data contract projected from the owning model."""

        root: MutableMapping[str, t.JsonPayload]

    @runtime_checkable
    class Dict(Protocol):
        """Structural data contract projected from the owning model."""

        root: MutableMapping[str, t.JsonPayload]

    @runtime_checkable
    class ConfigMap(Protocol):
        """Structural data contract projected from the owning model."""

        root: MutableMapping[str, t.JsonPayload]

    @runtime_checkable
    class ObjectList(Protocol):
        """Structural data contract projected from the owning model."""

        root: list[t.JsonPayload]


__all__: list[str] = [
    "FlextProtocolsContainers",
]
