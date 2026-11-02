# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/containers
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, Mapping, MutableMapping, ValuesView
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._typings import FlextTypesServices


class FlextProtocolsContainers:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ValidatorCallable(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def root(self) -> FlextTypesServices.ValidatorCallable:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class _MappingRootBase(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def root(self) -> MutableMapping[str, FlextTypesServices.JsonPayload]:
            """Structural member projected from the owning model."""
            ...

        def keys(self) -> KeysView[str]:
            """Return a view over the mapping's string keys."""

        def values(self) -> ValuesView[FlextTypesServices.JsonPayload]:
            """Return a view over the stored JSON payloads."""

        def items(self) -> ItemsView[str, FlextTypesServices.JsonPayload]:
            """Return a view over the key-payload pairs."""

        def get(
            self, key: str, default: FlextTypesServices.JsonPayload = None
        ) -> FlextTypesServices.JsonPayload:
            """Return the payload stored under ``key``."""

        def update(self, other: Mapping[str, FlextTypesServices.JsonPayload]) -> None:
            """Merge ``other`` into the mapping, overwriting existing keys."""

        def clear(self) -> None:
            """Remove every key-payload pair from the mapping."""

        def pop(
            self, key: str, *args: FlextTypesServices.JsonPayload
        ) -> FlextTypesServices.JsonPayload:
            """Remove ``key`` and return its payload."""

        def popitem(self) -> tuple[str, FlextTypesServices.JsonPayload]:
            """Remove the most recently inserted key-payload pair."""

        def setdefault(
            self, key: str, default: FlextTypesServices.JsonPayload
        ) -> FlextTypesServices.JsonPayload:
            """Return the payload under ``key``, storing ``default`` if absent."""

    @runtime_checkable
    class Dict(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def root(self) -> MutableMapping[str, FlextTypesServices.JsonPayload]:
            """Validated runtime key-value mapping."""
            ...

        def keys(self) -> KeysView[str]:
            """Return a view over the mapping's string keys."""

        def values(self) -> ValuesView[FlextTypesServices.JsonPayload]:
            """Return a view over the stored JSON payloads."""

        def items(self) -> ItemsView[str, FlextTypesServices.JsonPayload]:
            """Return a view over the key-payload pairs."""

        def get(
            self, key: str, default: FlextTypesServices.JsonPayload = None
        ) -> FlextTypesServices.JsonPayload:
            """Return the payload stored under ``key``."""

        def update(self, other: Mapping[str, FlextTypesServices.JsonPayload]) -> None:
            """Merge ``other`` into the mapping, overwriting existing keys."""

        def clear(self) -> None:
            """Remove every key-payload pair from the mapping."""

        def pop(
            self, key: str, *args: FlextTypesServices.JsonPayload
        ) -> FlextTypesServices.JsonPayload:
            """Remove ``key`` and return its payload."""

        def popitem(self) -> tuple[str, FlextTypesServices.JsonPayload]:
            """Remove the most recently inserted key-payload pair."""

        def setdefault(
            self, key: str, default: FlextTypesServices.JsonPayload
        ) -> FlextTypesServices.JsonPayload:
            """Return the payload under ``key``, storing ``default`` if absent."""

    @runtime_checkable
    class ConfigMap(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def root(self) -> MutableMapping[str, FlextTypesServices.JsonPayload]:
            """Validated runtime configuration mapping."""
            ...

        def keys(self) -> KeysView[str]:
            """Return a view over the mapping's string keys."""

        def values(self) -> ValuesView[FlextTypesServices.JsonPayload]:
            """Return a view over the stored JSON payloads."""

        def items(self) -> ItemsView[str, FlextTypesServices.JsonPayload]:
            """Return a view over the key-payload pairs."""

        def get(
            self, key: str, default: FlextTypesServices.JsonPayload = None
        ) -> FlextTypesServices.JsonPayload:
            """Return the payload stored under ``key``."""

        def update(self, other: Mapping[str, FlextTypesServices.JsonPayload]) -> None:
            """Merge ``other`` into the mapping, overwriting existing keys."""

        def clear(self) -> None:
            """Remove every key-payload pair from the mapping."""

        def pop(
            self, key: str, *args: FlextTypesServices.JsonPayload
        ) -> FlextTypesServices.JsonPayload:
            """Remove ``key`` and return its payload."""

        def popitem(self) -> tuple[str, FlextTypesServices.JsonPayload]:
            """Remove the most recently inserted key-payload pair."""

        def setdefault(
            self, key: str, default: FlextTypesServices.JsonPayload
        ) -> FlextTypesServices.JsonPayload:
            """Return the payload under ``key``, storing ``default`` if absent."""

    @runtime_checkable
    class ObjectList(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def root(self) -> list[FlextTypesServices.JsonPayload]:
            """Structural member projected from the owning model."""
            ...


__all__: list[str] = [
    "FlextProtocolsContainers",
]
