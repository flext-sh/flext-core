"""Context and correlation utilities enabling the context-first pillar.

The helpers correspond to the observability commitments in ``README.md`` and
``docs/architecture.md`` for the FLEXT 1.0.0 release: correlation inheritance,
request metadata, and latency tracking that integrate directly with
``FlextDispatcher`` and ``FlextLogger``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Final

from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextContext:
    """Hierarchical context manager for request-, service-, and perf-scopes.

    It is the single entry point referenced across the modernization plan: all
    dispatcher, container, and logging surfaces depend on these context vars to
    propagate correlation IDs and structured metadata.
    """

    # =========================================================================
    # Variables Domain - Context variables organized by functionality
    # =========================================================================

    class Variables:
        """Context variables organized by domain."""

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[ContextVar[str | None]] = ContextVar(
                "correlation_id",
                default=None,
            )
            PARENT_CORRELATION_ID: Final[ContextVar[str | None]] = ContextVar(
                "parent_correlation_id",
                default=None,
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[ContextVar[str | None]] = ContextVar(
                "service_name",
                default=None,
            )
            SERVICE_VERSION: Final[ContextVar[str | None]] = ContextVar(
                "service_version",
                default=None,
            )
            ENVIRONMENT: Final[ContextVar[str | None]] = ContextVar(
                "environment",
                default=None,
            )

        class Request:
            """Request context variables for metadata."""

            USER_ID: Final[ContextVar[str | None]] = ContextVar("user_id", default=None)
            REQUEST_ID: Final[ContextVar[str | None]] = ContextVar(
                "request_id",
                default=None,
            )
            REQUEST_TIMESTAMP: Final[ContextVar[datetime | None]] = ContextVar(
                "request_timestamp",
                default=None,
            )

        class Performance:
            """Performance context variables for timing."""

            OPERATION_NAME: Final[ContextVar[str | None]] = ContextVar(
                "operation_name",
                default=None,
            )
            OPERATION_START_TIME: Final[ContextVar[datetime | None]] = ContextVar(
                "operation_start_time",
                default=None,
            )
            OPERATION_METADATA: Final[ContextVar[FlextTypes.Core.Dict | None]] = (
                ContextVar("operation_metadata", default=None)
            )

    # =========================================================================
    # Correlation Domain - Distributed tracing and correlation ID management
    # =========================================================================

    class Correlation:
        """Distributed tracing and correlation ID management."""

        @staticmethod
        def get_correlation_id() -> str | None:
            """Get current correlation ID."""
            return FlextContext.Variables.Correlation.CORRELATION_ID.get()

        @staticmethod
        def set_correlation_id(correlation_id: str) -> None:
            """Set correlation ID."""
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate unique correlation ID."""
            correlation_id = FlextUtilities.Generators.generate_correlation_id()
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)
            return correlation_id

        @staticmethod
        def get_parent_correlation_id() -> str | None:
            """Get parent correlation ID."""
            return FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.get()

        @staticmethod
        def set_parent_correlation_id(parent_id: str) -> None:
            """Set parent correlation ID."""
            FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(parent_id)

        @staticmethod
        @contextmanager
        def new_correlation(
            correlation_id: str | None = None,
            parent_id: str | None = None,
        ) -> Generator[str]:
            """Create correlation context scope."""
            # Generate correlation ID if not provided
            if correlation_id is None:
                correlation_id = FlextUtilities.Generators.generate_correlation_id()

            # Save current context
            current_correlation = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get()
            )

            # Set new context
            correlation_token = FlextContext.Variables.Correlation.CORRELATION_ID.set(
                correlation_id,
            )

            # Set parent context
            parent_token: Token[str | None] | None = None
            if parent_id:
                parent_token = (
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(
                        parent_id,
                    )
                )
            elif current_correlation:
                # Current correlation becomes parent
                parent_token = (
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(
                        current_correlation,
                    )
                )

            try:
                yield correlation_id
            finally:
                # Restore previous context
                FlextContext.Variables.Correlation.CORRELATION_ID.reset(
                    correlation_token,
                )
                if parent_token:
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.reset(
                        parent_token,
                    )

        @staticmethod
        @contextmanager
        def inherit_correlation() -> Generator[str | None]:
            """Inherit or create correlation ID."""
            existing_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if existing_id:
                # Use existing correlation
                yield existing_id
            else:
                # Create new correlation context
                with FlextContext.Correlation.new_correlation() as new_id:
                    yield new_id

    # =========================================================================
    # Service Domain - Service identification and lifecycle context
    # =========================================================================

    class Service:
        """Service identification and lifecycle context management."""

        @staticmethod
        def get_service_name() -> str | None:
            """Get current service name."""
            return FlextContext.Variables.Service.SERVICE_NAME.get()

        @staticmethod
        def set_service_name(service_name: str) -> None:
            """Set service name."""
            FlextContext.Variables.Service.SERVICE_NAME.set(service_name)

        @staticmethod
        def get_service_version() -> str | None:
            """Get current service version."""
            return FlextContext.Variables.Service.SERVICE_VERSION.get()

        @staticmethod
        def set_service_version(version: str) -> None:
            """Set service version."""
            FlextContext.Variables.Service.SERVICE_VERSION.set(version)

        @staticmethod
        @contextmanager
        def service_context(
            service_name: str,
            version: str | None = None,
        ) -> Generator[None]:
            """Create service context scope."""
            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.Service.SERVICE_NAME.get()
            _ = FlextContext.Variables.Service.SERVICE_VERSION.get()

            # Set new context
            name_token = FlextContext.Variables.Service.SERVICE_NAME.set(service_name)
            version_token = None
            if version:
                version_token = FlextContext.Variables.Service.SERVICE_VERSION.set(
                    version,
                )

            try:
                yield
            finally:
                # Restore previous context
                FlextContext.Variables.Service.SERVICE_NAME.reset(name_token)
                if version_token:
                    FlextContext.Variables.Service.SERVICE_VERSION.reset(version_token)

    # =========================================================================
    # Request Domain - User and request metadata management
    # =========================================================================

    class Request:
        """Request-level context management for user and operation metadata."""

        @staticmethod
        def get_user_id() -> str | None:
            """Get current user ID."""
            return FlextContext.Variables.Request.USER_ID.get()

        @staticmethod
        def set_user_id(user_id: str) -> None:
            """Set user ID in context."""
            FlextContext.Variables.Request.USER_ID.set(user_id)

        @staticmethod
        def get_operation_name() -> str | None:
            """Get the current operation name from context."""
            return FlextContext.Variables.Performance.OPERATION_NAME.get()

        @staticmethod
        def set_operation_name(operation_name: str) -> None:
            """Set operation name in context."""
            FlextContext.Variables.Performance.OPERATION_NAME.set(operation_name)

        @staticmethod
        def get_request_id() -> str | None:
            """Get current request ID from context."""
            return FlextContext.Variables.Request.REQUEST_ID.get()

        @staticmethod
        def set_request_id(request_id: str) -> None:
            """Set request ID in context."""
            FlextContext.Variables.Request.REQUEST_ID.set(request_id)

        @staticmethod
        @contextmanager
        def request_context(
            *,
            user_id: str | None = None,
            operation_name: str | None = None,
            request_id: str | None = None,
            metadata: FlextTypes.Core.Dict | None = None,
        ) -> Generator[None]:
            """Create request metadata context scope with automatic cleanup."""
            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.Request.USER_ID.get()
            _ = FlextContext.Variables.Performance.OPERATION_NAME.get()
            _ = FlextContext.Variables.Request.REQUEST_ID.get()
            _ = FlextContext.Variables.Performance.OPERATION_METADATA.get()

            # Set new context
            user_token = (
                FlextContext.Variables.Request.USER_ID.set(user_id) if user_id else None
            )
            operation_token = (
                FlextContext.Variables.Performance.OPERATION_NAME.set(operation_name)
                if operation_name
                else None
            )
            request_token = (
                FlextContext.Variables.Request.REQUEST_ID.set(request_id)
                if request_id
                else None
            )
            metadata_token = (
                FlextContext.Variables.Performance.OPERATION_METADATA.set(metadata)
                if metadata
                else None
            )

            try:
                yield
            finally:
                # Restore previous context
                if user_token is not None:
                    FlextContext.Variables.Request.USER_ID.reset(user_token)
                if operation_token is not None:
                    FlextContext.Variables.Performance.OPERATION_NAME.reset(
                        operation_token,
                    )
                if request_token is not None:
                    FlextContext.Variables.Request.REQUEST_ID.reset(request_token)
                if metadata_token is not None:
                    FlextContext.Variables.Performance.OPERATION_METADATA.reset(
                        metadata_token,
                    )

    # =========================================================================
    # Performance Domain - Operation timing and performance tracking
    # =========================================================================

    class Performance:
        """Performance monitoring and timing context management for operations."""

        @staticmethod
        def get_operation_start_time() -> datetime | None:
            """Get operation start time from context."""
            return FlextContext.Variables.Performance.OPERATION_START_TIME.get()

        @staticmethod
        def set_operation_start_time(start_time: datetime | None = None) -> None:
            """Set operation start time in context."""
            if start_time is None:
                start_time = datetime.now(UTC)
            FlextContext.Variables.Performance.OPERATION_START_TIME.set(start_time)

        @staticmethod
        def get_operation_metadata() -> FlextTypes.Core.Dict | None:
            """Get operation metadata from context."""
            return FlextContext.Variables.Performance.OPERATION_METADATA.get()

        @staticmethod
        def set_operation_metadata(metadata: FlextTypes.Core.Dict) -> None:
            """Set operation metadata in context."""
            FlextContext.Variables.Performance.OPERATION_METADATA.set(metadata)

        @staticmethod
        def add_operation_metadata(key: str, value: object) -> None:
            """Add single metadata entry to operation context."""
            current_metadata = (
                FlextContext.Variables.Performance.OPERATION_METADATA.get() or {}
            )
            current_metadata[key] = value
            FlextContext.Variables.Performance.OPERATION_METADATA.set(current_metadata)

        @staticmethod
        @contextmanager
        def timed_operation(
            operation_name: str | None = None,
        ) -> Generator[FlextTypes.Core.Dict]:
            """Create timed operation context with performance tracking."""
            start_time = datetime.now(UTC)
            operation_metadata: FlextTypes.Core.Dict = {
                "start_time": start_time,
                "operation_name": operation_name,
            }

            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.Performance.OPERATION_START_TIME.get()
            _ = FlextContext.Variables.Performance.OPERATION_METADATA.get()
            _ = FlextContext.Variables.Performance.OPERATION_NAME.get()

            # Set new context
            start_token = FlextContext.Variables.Performance.OPERATION_START_TIME.set(
                start_time,
            )
            metadata_token = FlextContext.Variables.Performance.OPERATION_METADATA.set(
                operation_metadata,
            )
            operation_token = None
            if operation_name:
                operation_token = FlextContext.Variables.Performance.OPERATION_NAME.set(
                    operation_name,
                )

            try:
                yield operation_metadata
            finally:
                # Calculate duration
                end_time = datetime.now(UTC)
                duration = (end_time - start_time).total_seconds()
                operation_metadata.update(
                    {
                        "end_time": end_time,
                        "duration_seconds": duration,
                    },
                )

                # Restore previous context
                FlextContext.Variables.Performance.OPERATION_START_TIME.reset(
                    start_token,
                )
                FlextContext.Variables.Performance.OPERATION_METADATA.reset(
                    metadata_token,
                )
                if operation_token:
                    FlextContext.Variables.Performance.OPERATION_NAME.reset(
                        operation_token,
                    )

    # =========================================================================
    # Serialization Domain - Context serialization for cross-service communication
    # =========================================================================

    class Serialization:
        """Context serialization and deserialization for cross-service communication."""

        @staticmethod
        def get_full_context() -> FlextTypes.Core.Dict:
            """Get current context as dictionary."""
            context_vars = FlextContext.Variables
            return {
                "correlation_id": context_vars.Correlation.CORRELATION_ID.get(),
                "parent_correlation_id": context_vars.Correlation.PARENT_CORRELATION_ID.get(),
                "service_name": context_vars.Service.SERVICE_NAME.get(),
                "service_version": context_vars.Service.SERVICE_VERSION.get(),
                "user_id": context_vars.Request.USER_ID.get(),
                "operation_name": context_vars.Performance.OPERATION_NAME.get(),
                "request_id": context_vars.Request.REQUEST_ID.get(),
                "operation_start_time": context_vars.Performance.OPERATION_START_TIME.get(),
                "operation_metadata": context_vars.Performance.OPERATION_METADATA.get(),
            }

        @staticmethod
        def get_correlation_context() -> FlextTypes.Core.Headers:
            """Get correlation context for cross-service propagation."""
            context: FlextTypes.Core.Headers = {}

            correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if correlation_id:
                context["X-Correlation-Id"] = str(correlation_id)

            parent_id = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.get()
            if parent_id:
                context["X-Parent-Correlation-Id"] = str(parent_id)

            service_name = FlextContext.Variables.Service.SERVICE_NAME.get()
            if service_name:
                context["X-Service-Name"] = str(service_name)

            return context

        @staticmethod
        def set_from_context(context: Mapping[str, object]) -> None:
            """Set context from dictionary (e.g., from HTTP headers)."""
            correlation_id = context.get("X-Correlation-Id") or context.get(
                "correlation_id",
            )
            if correlation_id and isinstance(correlation_id, str):
                FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

            parent_id = context.get("X-Parent-Correlation-Id") or context.get(
                "parent_correlation_id",
            )
            if parent_id and isinstance(parent_id, str):
                FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(parent_id)

            service_name = context.get("X-Service-Name") or context.get("service_name")
            if service_name and isinstance(service_name, str):
                FlextContext.Variables.Service.SERVICE_NAME.set(service_name)

            user_id = context.get("X-User-Id") or context.get("user_id")
            if user_id and isinstance(user_id, str):
                FlextContext.Variables.Request.USER_ID.set(user_id)

    # =========================================================================
    # Utilities Domain - Context utility methods and helpers
    # =========================================================================

    class Utilities:
        """Utility methods for context management and helper operations."""

        @staticmethod
        def clear_context() -> None:
            """Clear all context variables."""
            # Clear string context variables
            for context_var in [
                FlextContext.Variables.Correlation.CORRELATION_ID,
                FlextContext.Variables.Correlation.PARENT_CORRELATION_ID,
                FlextContext.Variables.Service.SERVICE_NAME,
                FlextContext.Variables.Service.SERVICE_VERSION,
                FlextContext.Variables.Request.USER_ID,
                FlextContext.Variables.Request.REQUEST_ID,
                FlextContext.Variables.Performance.OPERATION_NAME,
            ]:
                with contextlib.suppress(LookupError):
                    context_var.set(None)

            # Clear typed context variables
            with contextlib.suppress(LookupError):
                FlextContext.Variables.Performance.OPERATION_START_TIME.set(None)
            with contextlib.suppress(LookupError):
                FlextContext.Variables.Performance.OPERATION_METADATA.set(None)
            with contextlib.suppress(LookupError):
                FlextContext.Variables.Request.REQUEST_TIMESTAMP.set(None)

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure correlation ID exists, creating one if needed."""
            correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if not correlation_id:
                correlation_id = FlextContext.Correlation.generate_correlation_id()
            return correlation_id

        @staticmethod
        def has_correlation_id() -> bool:
            """Check if correlation ID is set in context."""
            return FlextContext.Variables.Correlation.CORRELATION_ID.get() is not None

        @staticmethod
        def get_context_summary() -> str:
            """Get a human-readable context summary for debugging."""
            context = FlextContext.Serialization.get_full_context()
            parts: FlextTypes.Core.StringList = []

            correlation_id = context["correlation_id"]
            if isinstance(correlation_id, str) and correlation_id:
                parts.append(f"correlation={correlation_id[:8]}...")

            service_name = context["service_name"]
            if isinstance(service_name, str) and service_name:
                parts.append(f"service={service_name}")

            operation_name = context["operation_name"]
            if isinstance(operation_name, str) and operation_name:
                parts.append(f"operation={operation_name}")

            user_id = context["user_id"]
            if isinstance(user_id, str) and user_id:
                parts.append(f"user={user_id}")

            return (
                f"FlextContext({', '.join(parts)})" if parts else "FlextContext(empty)"
            )


__all__: FlextTypes.Core.StringList = [
    "FlextContext",
]
