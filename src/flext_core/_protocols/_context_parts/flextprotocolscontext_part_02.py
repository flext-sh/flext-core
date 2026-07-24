"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from flext_core import m, p, t
from .flextprotocolscontext_part_01 import (
    FlextProtocolsContext as FlextProtocolsContextPart01,
)


class FlextProtocolsContext(FlextProtocolsContextPart01):
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

    class ContextPerformanceNamespace(Protocol):
        """Protocol for performance-scoped context helpers."""

        @staticmethod
        def timed_operation(
            operation_name: str | None = None,
        ) -> AbstractContextManager[t.JsonMapping]:
            """Create a timed operation scope."""
            ...

    class ContextSerializationNamespace(Protocol):
        """Protocol for context serialization helpers."""

        @staticmethod
        def export_full_context() -> t.JsonMapping:
            """Export the active global context variables."""
            ...

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

    class ContextType(Protocol):
        """Protocol for flat context classes exposing the canonical class API."""

        @classmethod
        def create(cls, **initial_data: t.JsonPayload) -> p.Context:
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

        @staticmethod
        def fetch_service(service_name: str) -> p.Result[t.RegisterableService]:
            """Resolve a named service from the configured container."""
            ...

        @staticmethod
        def register_service(
            service_name: str, service: t.RegisterableService
        ) -> p.Result[bool]:
            """Register a named service through the configured container."""
            ...

        @staticmethod
        def resolve_correlation_id() -> str | None:
            """Resolve the active correlation id."""
            ...

        @staticmethod
        def new_correlation(
            correlation_id: str | None = None, parent_id: str | None = None
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
            service_name: str, version: str | None = None
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
        ) -> AbstractContextManager[m.ConfigMap]:
            """Create a timed operation scope."""
            ...

        @staticmethod
        def export_full_context() -> t.MappingKV[str, t.Scalar]:
            """Export the active process context state."""
            ...

        @staticmethod
        def clear_context() -> None:
            """Clear all active process context state."""
            ...


__all__: list[str] = ["FlextProtocolsContext"]
