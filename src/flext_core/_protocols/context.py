"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core import m, p


class FlextProtocolsContext:
    """Protocols for context and runtime bootstrap options."""

    @runtime_checkable
    class ContextRead(Protocol):
        """Read-only context operations."""

        def get(
            self,
            key: str,
            scope: str = ...,
        ) -> p.Result[t.RuntimeAtomic]:
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
        ) -> p.Result[bool]: ...

        @overload
        def set(
            self,
            key_or_data: t.ConfigMap,
            value: None = ...,
            *,
            scope: str = ...,
        ) -> p.Result[bool]: ...

        def set(
            self,
            key_or_data: str | t.ConfigMap,
            value: t.RuntimeAtomic | None = ...,
            *,
            scope: str = ...,
        ) -> p.Result[bool]:
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

        def validate_context(self) -> p.Result[bool]:
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
        ) -> m.ContextExport | t.ContainerMapping:
            """Export context state as the canonical context export model or dict."""
            ...

    @runtime_checkable
    class ContextMetadataAccess(Protocol):
        """Context metadata read/write operations."""

        def resolve_metadata(
            self,
            key: str,
        ) -> p.Result[t.RuntimeAtomic]:
            """Get a metadata value by key."""
            ...

        def apply_metadata(self, key: str, value: t.MetadataValue) -> None:
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
        def fetch_service(service_name: str) -> p.Result[t.Scalar]:
            """Resolve a service from the configured container."""
            ...

        @staticmethod
        def register_service(
            service_name: str,
            service: t.RegisterableService,
        ) -> p.Result[bool]:
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
        ) -> AbstractContextManager[t.ConfigMap]:
            """Create a timed operation scope."""
            ...

    @runtime_checkable
    class ContextSerializationNamespace(Protocol):
        """Protocol for context serialization helpers."""

        @staticmethod
        def export_full_context() -> t.ContainerMapping:
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
            initial_data: t.ConfigMap | None = None,
            *,
            operation_id: str | None = None,
            user_id: str | None = None,
            metadata: t.ConfigMap | None = None,
            auto_correlation_id: bool = True,
        ) -> FlextProtocolsContext.Context:
            """Create a new context instance."""
            ...

        @classmethod
        def resolve_container(cls) -> p.Container:
            """Resolve the configured container."""
            ...

        @classmethod
        def configure_container(cls, container: p.Container) -> None:
            """Configure the container used by the context service namespace."""
            ...

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options for service initialization."""

        settings: p.Settings | None
        settings_type: type | None
        settings_overrides: t.ScalarMapping | None
        context: FlextProtocolsContext.Context | None
        dispatcher: p.Dispatcher | None
        registry: p.Registry | None
        subproject: str | None
        services: Mapping[str, t.RegisterableService] | None
        factories: Mapping[str, t.FactoryCallable] | None
        resources: Mapping[str, t.ResourceCallable] | None
        container_overrides: t.ScalarMapping | None
        wire_modules: Sequence[ModuleType | str] | None
        wire_packages: t.StrSequence | None
        wire_classes: Sequence[type] | None


__all__ = ["FlextProtocolsContext"]
