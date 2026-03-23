"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from flext_core import t

if TYPE_CHECKING:
    from flext_core import r


class FlextProtocolsContext:
    """Protocols for context and runtime bootstrap options."""

    @runtime_checkable
    class ContextRead(Protocol):
        """Read-only context operations."""

        def get(self, key: str, scope: str = ...) -> r[t.RuntimeAtomic]:
            """Get a context value by key and scope."""
            ...

        def has(self, key: str, scope: str = ...) -> bool:
            """Check if a key exists in the given scope."""
            ...

        def keys(self) -> Sequence[str]:
            """Return all keys across all scopes."""
            ...

        def values(self) -> Sequence[t.NormalizedValue]:
            """Return all values across all scopes."""
            ...

        def items(self) -> Sequence[tuple[str, t.NormalizedValue]]:
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
        ) -> r[bool]: ...

        @overload
        def set(
            self,
            key_or_data: t.ConfigMap,
            value: None = ...,
            *,
            scope: str = ...,
        ) -> r[bool]: ...

        def set(
            self,
            key_or_data: str | t.ConfigMap,
            value: t.RuntimeAtomic | None = ...,
            *,
            scope: str = ...,
        ) -> r[bool]:
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
            other: Self | t.ConfigMap | Mapping[str, t.NormalizedValue],
        ) -> Self:
            """Merge another context or mapping into this one."""
            ...

        def validate_context(self) -> r[bool]:
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
        ) -> BaseModel | Mapping[str, t.NormalizedValue]:
            """Export context state as serializable ConfigMap or dict."""
            ...

    @runtime_checkable
    class ContextMetadataAccess(Protocol):
        """Context metadata read/write operations."""

        def get_metadata(self, key: str) -> r[t.RuntimeAtomic]:
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
        config_overrides: Mapping[str, t.Scalar] | None
        context: FlextProtocolsContext.Context | None
        subproject: str | None
        services: Mapping[str, t.RegisterableService] | None
        factories: Mapping[str, t.FactoryCallable] | None
        resources: Mapping[str, t.ResourceCallable] | None
        container_overrides: Mapping[str, t.Scalar] | None
        wire_modules: Sequence[ModuleType | str] | None
        wire_packages: Sequence[str] | None
        wire_classes: Sequence[type] | None


__all__ = ["FlextProtocolsContext"]
