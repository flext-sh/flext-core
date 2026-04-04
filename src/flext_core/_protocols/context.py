"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable

from pydantic_settings import BaseSettings

from flext_core import FlextProtocolsResult, t

if TYPE_CHECKING:
    from flext_core._models._context._export import FlextModelsContextExport


class FlextProtocolsContext:
    """Protocols for context and runtime bootstrap options."""

    @runtime_checkable
    class ContextRead(Protocol):
        """Read-only context operations."""

        def get(
            self,
            key: str,
            scope: str = ...,
        ) -> FlextProtocolsResult.Result[t.RuntimeAtomic]:
            """Get a context value by key and scope."""
            ...

        def has(self, key: str, scope: str = ...) -> bool:
            """Check if a key exists in the given scope."""
            ...

        def keys(self) -> t.StrSequence:
            """Return all keys across all scopes."""
            ...

        def values(self) -> t.ContainerList:
            """Return all values across all scopes."""
            ...

        def items(self) -> Sequence[tuple[str, t.RecursiveContainer]]:
            """Return all key-value pairs across all scopes."""
            ...

    @runtime_checkable
    class ContextWrite(Protocol):
        """Write context operations."""

        @overload
        def set(
            self,
            key_or_data: str,
            value: t.RuntimeAtomic,
            *,
            scope: str = ...,
        ) -> FlextProtocolsResult.Result[bool]: ...

        @overload
        def set(
            self,
            key_or_data: t.ConfigMap,
            value: None = ...,
            *,
            scope: str = ...,
        ) -> FlextProtocolsResult.Result[bool]: ...

        def set(
            self,
            key_or_data: str | t.ConfigMap,
            value: t.RuntimeAtomic | None = ...,
            *,
            scope: str = ...,
        ) -> FlextProtocolsResult.Result[bool]:
            """Set a context value or bulk-set from ConfigMap."""
            ...

        def remove(self, key: str, scope: str = ...) -> None:
            """Remove a key from the given scope."""
            ...

        def clear(self) -> None:
            """Clear all context data across all scopes."""
            ...

    @runtime_checkable
    class ContextLifecycle(Protocol):
        """Context lifecycle operations."""

        def clone(self) -> Self:
            """Clone context for isolated execution."""
            ...

        def merge(
            self,
            other: Self | t.ConfigMap | t.ContainerMapping,
        ) -> Self:
            """Merge another context or mapping into this one."""
            ...

        def validate_context(self) -> FlextProtocolsResult.Result[bool]:
            """Validate context state consistency."""
            ...

    @runtime_checkable
    class ContextExport(Protocol):
        """Context export/serialization operations."""

        def export(
            self,
            *,
            include_statistics: bool = ...,
            include_metadata: bool = ...,
            as_dict: bool = ...,
        ) -> FlextModelsContextExport.ContextExport | t.ContainerMapping:
            """Export context state as the canonical context export model or dict."""
            ...

    @runtime_checkable
    class ContextMetadataAccess(Protocol):
        """Context metadata read/write operations."""

        def get_metadata(
            self,
            key: str,
        ) -> FlextProtocolsResult.Result[t.RuntimeAtomic]:
            """Get a metadata value by key."""
            ...

        def set_metadata(self, key: str, value: t.MetadataValue) -> None:
            """Set a metadata value by key."""
            ...

    @runtime_checkable
    class Context(
        ContextRead,
        ContextWrite,
        ContextLifecycle,
        ContextExport,
        ContextMetadataAccess,
        Protocol,
    ):
        """Full context protocol — composed from capability sub-protocols.

        Consumers should depend on the NARROWEST sub-protocol they need:
        - ContextRead: read-only access (get/has/keys/values/items)
        - ContextWrite: mutation (set/remove/clear)
        - ContextLifecycle: clone/merge/validate
        - ContextExport: serialization
        - ContextMetadataAccess: metadata operations
        """

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options for service initialization."""

        config_type: type[BaseSettings] | None
        config_overrides: t.ScalarMapping | None
        context: FlextProtocolsContext.Context | None
        subproject: str | None
        services: Mapping[str, t.RegisterableService] | None
        factories: Mapping[str, t.FactoryCallable] | None
        resources: Mapping[str, t.ResourceCallable] | None
        container_overrides: t.ScalarMapping | None
        wire_modules: Sequence[ModuleType | str] | None
        wire_packages: t.StrSequence | None
        wire_classes: Sequence[type] | None


__all__ = ["FlextProtocolsContext"]
