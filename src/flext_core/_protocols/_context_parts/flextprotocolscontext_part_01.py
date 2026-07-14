"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from contextlib import AbstractContextManager

if TYPE_CHECKING:
    # NOTE (multi-agent, mro-wkii.17.26): this protocol loads during p/t
    # composition; the root facades are needed only by postponed annotations.
    from flext_core import p, t


class FlextProtocolsContext:
    """Protocols for context and runtime bootstrap options."""

    @runtime_checkable
    class ContextRead(Protocol):
        """Read-only context operations."""

        def get(self, key: str) -> p.Result[t.JsonPayload]:
            """Get a context value by key."""
            ...

        def has(self, key: str) -> bool:
            """Check if a key exists in the context scope."""
            ...

        def keys(self) -> t.StrSequence:
            """Return all keys in the context scope."""
            ...

        def values(self) -> list[t.JsonPayload]:
            """Return all values in the context scope."""
            ...

        def items(self) -> list[tuple[str, t.JsonPayload]]:
            """Return all key-value pairs in the context scope."""
            ...

    @runtime_checkable
    class ContextWrite(Protocol):
        """Write context operations."""

        def set(self, key: str, value: t.JsonPayload) -> p.Result[bool]:
            """Set a context value in the context scope."""
            ...

        def remove(self, key: str) -> None:
            """Remove a key from the context scope."""
            ...

        def clear(self) -> None:
            """Clear all context data in the context scope."""
            ...

    @runtime_checkable
    class ContextLifecycle(Protocol):
        """Context lifecycle operations."""

        def clone(self) -> Self:
            """Clone context for isolated execution."""
            ...

        def merge(self, other: Self | t.MappingKV[str, t.JsonPayload]) -> Self:
            """Merge another context or mapping into this one."""
            ...

    @runtime_checkable
    class ContextExport(Protocol):
        """Context export/serialization operations."""

        def export(self, *, as_dict: bool = ...) -> dict[str, t.JsonPayload] | Self:
            """Export context state as a dict or the context instance itself."""
            ...

    @runtime_checkable
    class ContextMetadataAccess(Protocol):
        """Context metadata read/write operations."""

        def resolve_metadata(self, key: str) -> p.Result[t.JsonPayload]:
            """Get a metadata value by key."""
            ...

        def apply_metadata(self, key: str, value: t.JsonValue) -> None:
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
    class ContextServiceNamespace(Protocol):
        """Protocol for the service-scoped context namespace."""

        @staticmethod
        def fetch_service(service_name: str) -> p.Result[t.RegisterableService]:
            """Resolve a service from the configured container."""
            ...

        @staticmethod
        def register_service(
            service_name: str, service: t.RegisterableService
        ) -> p.Result[bool]:
            """Register a service through the configured container."""
            ...

        @staticmethod
        def service_context(
            service_name: str, version: str | None = None
        ) -> AbstractContextManager[None]:
            """Create a service context scope manager."""
            ...

    @runtime_checkable
    class ContextCorrelationNamespace(Protocol):
        """Protocol for correlation-id helpers on the context class."""

        @staticmethod
        def resolve_correlation_id() -> str | None:
            """Resolve the current correlation id."""
            ...

        @staticmethod
        def apply_correlation_id(correlation_id: str | None) -> None:
            """Apply or clear the current correlation id."""
            ...


__all__: list[str] = ["FlextProtocolsContext"]
