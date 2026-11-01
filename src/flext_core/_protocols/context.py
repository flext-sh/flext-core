"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from flext_core import t
    from flext_core._protocols import FlextProtocolsContainer, FlextProtocolsResult
from typing import TYPE_CHECKING

from flext_core._protocols import FlextProtocolsResult

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from flext_core import t
    from flext_core._protocols import FlextProtocolsContainer
from contextvars import ContextVar
from datetime import datetime

from flext_core import t
from flext_core._protocols import FlextProtocolsHandler, FlextProtocolsSettings

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsContainer


class FlextProtocolsContext:
    """Public facade for FlextProtocolsContext."""

    @runtime_checkable
    class ContextRead(Protocol):
        """Read-only context operations."""

        def get(self, key: str) -> FlextProtocolsResult.Result[t.JsonPayload]:
            """Get a context value by key."""
            ...

        def has(self, key: str) -> bool:
            """Check if a key exists in the context scope."""
            ...

        def keys(self) -> Sequence[str]:
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
            self, key: str, value: t.JsonPayload
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

        def merge(self, other: Self | Mapping[str, t.JsonPayload]) -> Self:
            """Merge another context or mapping into this one."""
            ...

    @runtime_checkable
    class ContextExport(Protocol):
        """Context export/serialization operations."""

        def export(
            self,
            *,
            as_dict: bool = ...,
        ) -> Mapping[str, t.JsonPayload] | Self:
            """Export context state as a dict or the context instance itself."""
            ...

    @runtime_checkable
    class ContextMetadataAccess(Protocol):
        """Context metadata read/write operations."""

        def resolve_metadata(
            self, key: str
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
        ) -> FlextProtocolsResult.Result[FlextProtocolsContainer.RegisterableService]:
            """Resolve a service from the configured container."""
            ...

        @staticmethod
        def register_service(
            service_name: str,
            service: FlextProtocolsContainer.RegisterableService,
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
        ) -> AbstractContextManager[
            FlextProtocolsContainer.MutableRootDict[t.JsonPayload]
        ]:
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
        """Protocol for flat context classes exposing the canonical class API."""

        @classmethod
        def create(cls, **initial_data: t.JsonPayload) -> FlextProtocolsContext.Context:
            """Create a new context instance."""
            ...

        @classmethod
        def resolve_container(cls) -> FlextProtocolsContainer.Container:
            """Resolve the configured container."""
            ...

        @classmethod
        def configure_container(
            cls, container: FlextProtocolsContainer.Container
        ) -> None:
            """Configure the container used by the context service namespace."""
            ...

        @staticmethod
        def fetch_service(
            service_name: str,
        ) -> FlextProtocolsResult.Result[FlextProtocolsContainer.RegisterableService]:
            """Resolve a named service from the configured container."""
            ...

        @staticmethod
        def register_service(
            service_name: str,
            service: FlextProtocolsContainer.RegisterableService,
        ) -> FlextProtocolsResult.Result[bool]:
            """Register a named service through the configured container."""
            ...

        @staticmethod
        def resolve_correlation_id() -> str | None:
            """Resolve the active correlation id."""
            ...

        @staticmethod
        def new_correlation(
            correlation_id: str | None = None,
            parent_id: str | None = None,
        ) -> AbstractContextManager[str]:
            """Create a scoped correlation-id context manager."""
            ...

        @staticmethod
        def apply_correlation_id(correlation_id: str | None) -> None:
            """Apply or clear the active correlation id."""
            ...

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure and return the active correlation id."""
            ...

        @staticmethod
        def service_context(
            service_name: str,
            version: str | None = None,
        ) -> AbstractContextManager[None]:
            """Create a service-scoped context manager."""
            ...

        @staticmethod
        def resolve_operation_name() -> str | None:
            """Resolve the active operation name."""
            ...

        @staticmethod
        def apply_operation_name(operation_name: str) -> None:
            """Apply the current operation name."""
            ...

        @staticmethod
        def timed_operation(
            operation_name: str | None = None,
        ) -> AbstractContextManager[
            FlextProtocolsContainer.MutableRootDict[t.JsonPayload]
        ]:
            """Create a timed operation scope."""
            ...

        @staticmethod
        def export_full_context() -> Mapping[str, t.Scalar]:
            """Export the active process context state."""
            ...

        @staticmethod
        def clear_context() -> None:
            """Clear all active process context state."""
            ...

    @runtime_checkable
    class ContextProxyToken(Protocol):
        """Stored structlog value needed to restore a context variable."""

        @property
        def key(self) -> str: ...

        @property
        def previous_value(self) -> t.JsonPayload | datetime | None: ...

    @runtime_checkable
    class ContextProxy[ValueT: t.JsonPayload | datetime](Protocol):
        """Context-variable operations sharing structlog's backing storage."""

        def get(self) -> t.JsonPayload | datetime | None: ...

        def set(
            self, value: ValueT | None
        ) -> FlextProtocolsContext.ContextProxyToken: ...

        @staticmethod
        def reset(token: FlextProtocolsContext.ContextProxyToken) -> None: ...

    @runtime_checkable
    class ContextMetadata(Protocol):
        """Metadata snapshot copied when creating a context runtime."""

        @property
        def attributes(self) -> t.JsonMapping: ...

        def model_copy(self, *, deep: bool = False) -> Self: ...

    @runtime_checkable
    class ContextStatistics(Protocol):
        """Validated counters and operation metrics."""

        @property
        def sets(self) -> int: ...

        @property
        def gets(self) -> int: ...

        @property
        def removes(self) -> int: ...

        @property
        def clears(self) -> int: ...

        @property
        def operations(self) -> t.JsonMapping: ...

        def model_copy(
            self,
            *,
            update: Mapping[str, t.JsonPayload] | None = None,
            deep: bool = False,
        ) -> Self: ...

    @runtime_checkable
    class ContextRuntimeState(Protocol):
        """Data capabilities consumed by context runtime helpers."""

        @property
        def metadata(self) -> FlextProtocolsContext.ContextMetadata: ...

        @property
        def hooks(self) -> t.ContextHookMap: ...

        @property
        def statistics(self) -> FlextProtocolsContext.ContextStatistics: ...

        @property
        def active(self) -> bool: ...

        @property
        def suspended(self) -> bool: ...

        @property
        def scope_vars(
            self,
        ) -> Mapping[
            str,
            ContextVar[FlextProtocolsContainer.MutableRootDict[t.JsonPayload] | None],
        ]: ...

        def model_copy(
            self,
            *,
            update: Mapping[
                str,
                FlextProtocolsContext.ContextStatistics
                | Mapping[
                    str,
                    ContextVar[
                        FlextProtocolsContainer.MutableRootDict[t.JsonPayload] | None
                    ],
                ],
            ]
            | None = None,
            deep: bool = False,
        ) -> Self: ...

    @runtime_checkable
    class ServiceRuntime(Protocol):
        """Actual scoped runtime selected by a component's composition root."""

        @property
        def settings(self) -> FlextProtocolsSettings.Settings: ...

        @property
        def context(self) -> FlextProtocolsContext.Context: ...

        @property
        def container(self) -> FlextProtocolsContainer.Container: ...

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher: ...

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options a service base declares for its runtime."""

        @property
        def settings(self) -> FlextProtocolsSettings.Settings | None: ...

        @property
        def settings_type(self) -> FlextProtocolsSettings.SettingsType | None: ...

        @property
        def settings_overrides(self) -> Mapping[str, t.JsonValue] | None: ...

        @property
        def context(self) -> FlextProtocolsContext.Context | None: ...

        @property
        def container(self) -> FlextProtocolsContainer.Container | None: ...

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher | None: ...


__all__: list[str] = ["FlextProtocolsContext"]
