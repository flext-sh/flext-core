"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from contextlib import AbstractContextManager
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from flext_core import (
        FlextProtocolsContainer,
        FlextProtocolsHandler,
        FlextProtocolsRegistry,
        FlextProtocolsResult,
        FlextProtocolsSettings,
        t,
    )


class FlextProtocolsContext:
    """Protocols for context and runtime bootstrap options."""

    @runtime_checkable
    class ContextRead(Protocol):
        """Read-only context operations."""

        def get(
            self,
            key: str,
        ) -> FlextProtocolsResult.Result[t.JsonPayload]:
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

        def set(
            self,
            key: str,
            value: t.JsonPayload,
        ) -> FlextProtocolsResult.Result[bool]:
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

        def merge(
            self,
            other: Self | Mapping[str, t.JsonPayload],
        ) -> Self:
            """Merge another context or mapping into this one."""
            ...

    @runtime_checkable
    class ContextExport(Protocol):
        """Context export/serialization operations."""

        def export(
            self,
            *,
            as_dict: bool = ...,
        ) -> dict[str, t.JsonPayload] | Self:
            """Export context state as a dict or the context instance itself."""
            ...

    @runtime_checkable
    class ContextMetadataAccess(Protocol):
        """Context metadata read/write operations."""

        def resolve_metadata(
            self,
            key: str,
        ) -> FlextProtocolsResult.Result[t.JsonPayload]:
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
        def fetch_service(
            service_name: str,
        ) -> FlextProtocolsResult.Result[t.RegisterableService]:
            """Resolve a service from the configured container."""
            ...

        @staticmethod
        def register_service(
            service_name: str,
            service: t.RegisterableService,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a service through the configured container."""
            ...

        @staticmethod
        def service_context(
            service_name: str,
            version: str | None = None,
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

    @runtime_checkable
    class ContextRequestNamespace(Protocol):
        """Protocol for request-level helpers on the context class."""

        @staticmethod
        def resolve_operation_name() -> str | None:
            """Resolve the current operation name."""
            ...

        @staticmethod
        def apply_operation_name(operation_name: str) -> None:
            """Apply the current operation name."""
            ...

    @runtime_checkable
    class ContextPerformanceNamespace(Protocol):
        """Protocol for performance-scoped context helpers."""

        @staticmethod
        def timed_operation(
            operation_name: str | None = None,
        ) -> AbstractContextManager[t.JsonMapping]:
            """Create a timed operation scope."""
            ...

    @runtime_checkable
    class ContextSerializationNamespace(Protocol):
        """Protocol for context serialization helpers."""

        @staticmethod
        def export_full_context() -> t.JsonMapping:
            """Export the active global context variables."""
            ...

    @runtime_checkable
    class ContextUtilitiesNamespace(Protocol):
        """Protocol for class-level context utilities."""

        @staticmethod
        def clear_context() -> None:
            """Clear the active context variables."""
            ...

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure and return the active correlation id."""
            ...

    @runtime_checkable
    class ContextType(Protocol):
        """Protocol for context classes exposing the canonical factory surface."""

        Service: FlextProtocolsContext.ContextServiceNamespace
        Correlation: FlextProtocolsContext.ContextCorrelationNamespace
        Request: FlextProtocolsContext.ContextRequestNamespace
        Performance: FlextProtocolsContext.ContextPerformanceNamespace
        Serialization: FlextProtocolsContext.ContextSerializationNamespace
        Utilities: FlextProtocolsContext.ContextUtilitiesNamespace

        @classmethod
        def create(
            cls,
            initial_data: t.JsonMapping | None = None,
            *,
            operation_id: str | None = None,
            user_id: str | None = None,
            metadata: t.JsonMapping | None = None,
            auto_correlation_id: bool = True,
        ) -> FlextProtocolsContext.Context:
            """Create a new context instance."""
            ...

        @classmethod
        def resolve_container(cls) -> FlextProtocolsContainer.Container:
            """Resolve the configured container."""
            ...

        @classmethod
        def configure_container(
            cls,
            container: FlextProtocolsContainer.Container,
        ) -> None:
            """Configure the container used by the context service namespace."""
            ...

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options for service initialization."""

        settings: FlextProtocolsSettings.Settings | None
        settings_type: type | None
        settings_overrides: t.ScalarMapping | None
        context: FlextProtocolsContext.Context | None
        dispatcher: FlextProtocolsHandler.Dispatcher | None
        registry: FlextProtocolsRegistry.Registry | None
        subproject: str | None
        services: Mapping[str, t.RegisterableService] | None
        factories: Mapping[str, t.FactoryCallable] | None
        resources: Mapping[str, t.ResourceCallable] | None
        container_overrides: t.ScalarMapping | None
        wire_modules: Sequence[ModuleType | str] | None
        wire_packages: t.StrSequence | None
        wire_classes: Sequence[type] | None


__all__: list[str] = ["FlextProtocolsContext"]
