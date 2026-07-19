"""Context operation protocols composed by the context facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase

if TYPE_CHECKING:
    # mro-wkii.17.26 (codex): reverse p/t edges are annotation-only while the
    # public protocol facade is still being composed.
    import contextvars
    from contextlib import AbstractContextManager
    from flext_core import p, t


class FlextProtocolsContextOperations:
    """Instance, service, and correlation context contracts."""

    # mro-wkii.17.26.25 (kimi): expose context state/data protocols consumed by
    # _utilities/context_*.py so Pyrefly can resolve p.ContextRuntimeState and
    # p.ContextData through the canonical protocols facade.
    @runtime_checkable
    class ContextRuntimeState(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural contract for the mutable context runtime state model."""

        @property
        def metadata(self) -> p.Metadata: ...

        @property
        def hooks(self) -> t.ContextHookMap: ...

        @property
        def statistics(self) -> p.HasModelDump: ...

        @property
        def active(self) -> bool: ...

        @property
        def suspended(self) -> bool: ...

        @property
        def scope_vars(
            self,
        ) -> t.MappingKV[str, contextvars.ContextVar[p.ConfigMap | None]]: ...

        def with_operation_update(self, operation: str) -> Self: ...

        def resolve_scope_var(
            self, scope: str
        ) -> tuple[Self, contextvars.ContextVar[p.ConfigMap | None]]: ...

    @runtime_checkable
    class ContextData(FlextProtocolsBase.BaseModel, Protocol):
        """Structural contract for context initialization data."""

        @property
        def data(self) -> t.MappingKV[str, t.Scalar]: ...

        @property
        def metadata(self) -> p.Metadata | t.MappingKV[str, t.Scalar] | None: ...

    @runtime_checkable
    class ContextContainerState(Protocol):
        """Immutable binding state between a context and its container."""

        @property
        def container(self) -> p.Container | None: ...

        def with_container(self, container: p.Container | None) -> Self: ...

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

        def export(
            self, *, as_dict: bool = ...
        ) -> t.MappingKV[str, t.JsonPayload] | Self:
            """Export the current context as a snapshot or mapping."""
            ...

    @runtime_checkable
    class ContextExport(FlextProtocolsBase.BaseModel, Protocol):
        """Validated context snapshot returned by export operations."""

        @property
        def data(self) -> t.MappingKV[str, t.JsonPayload]: ...

        @property
        def metadata(self) -> p.Metadata | p.Dict | None: ...

        @property
        def statistics(self) -> t.JsonMapping: ...

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
        ContextRead, ContextWrite, ContextLifecycle, ContextMetadataAccess, Protocol
    ):
        """Full context protocol composed from capability sub-protocols."""

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


__all__: tuple[str, ...] = ("FlextProtocolsContextOperations",)
