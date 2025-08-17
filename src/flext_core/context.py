"""Context management for correlation IDs and request tracking."""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

# =============================================================================
# CONTEXT VARIABLES
# =============================================================================

# Request correlation context
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_parent_correlation_id: ContextVar[str | None] = ContextVar(
    "parent_correlation_id",
    default=None,
)

# Service identification context
_service_name: ContextVar[str | None] = ContextVar("service_name", default=None)
_service_version: ContextVar[str | None] = ContextVar("service_version", default=None)

# Request metadata context
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_operation_name: ContextVar[str | None] = ContextVar("operation_name", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Performance context
_operation_start_time: ContextVar[datetime | None] = ContextVar(
    "operation_start_time",
    default=None,
)
_operation_metadata: ContextVar[dict[str, object] | None] = ContextVar(
    "operation_metadata",
    default=None,
)


# =============================================================================
# CONTEXT MANAGEMENT CLASS
# =============================================================================


class FlextContext:
    """Centralized context management for FLEXT operations.

    Provides thread-safe context management using Python contextvars
    for correlation IDs, service identification, and request metadata.
    """

    # =============================================================================
    # CORRELATION ID MANAGEMENT
    # =============================================================================

    @staticmethod
    def get_correlation_id() -> str | None:
        """Get a current correlation ID from context."""
        return _correlation_id.get()

    @staticmethod
    def set_correlation_id(correlation_id: str) -> None:
        """Set correlation ID in context."""
        _correlation_id.set(correlation_id)

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate new correlation ID and set in context."""
        correlation_id = str(uuid.uuid4())
        _correlation_id.set(correlation_id)
        return correlation_id

    @staticmethod
    def get_parent_correlation_id() -> str | None:
        """Get parent correlation ID from context."""
        return _parent_correlation_id.get()

    @staticmethod
    def set_parent_correlation_id(parent_id: str) -> None:
        """Set parent correlation ID in context."""
        _parent_correlation_id.set(parent_id)

    @staticmethod
    @contextmanager
    def new_correlation(
        correlation_id: str | None = None,
        parent_id: str | None = None,
    ) -> Generator[str]:
        """Create a new correlation context scope.

        Args:
            correlation_id: Specific correlation ID to use (generates if None)
            parent_id: Parent correlation ID for hierarchical tracing

        Yields:
            str: The correlation ID for this context

        """
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        # Save current context
        current_correlation = _correlation_id.get()
        _parent_correlation_id.get()

        # Set new context
        correlation_token = _correlation_id.set(correlation_id)

        # Set parent context
        parent_token = None
        if parent_id:
            parent_token = _parent_correlation_id.set(parent_id)
        elif current_correlation:
            # Current correlation becomes parent
            parent_token = _parent_correlation_id.set(current_correlation)

        try:
            yield correlation_id
        finally:
            # Restore previous context
            _correlation_id.reset(correlation_token)
            if parent_token:
                _parent_correlation_id.reset(parent_token)

    @staticmethod
    @contextmanager
    def inherit_correlation() -> Generator[str | None]:
        """Inherit existing correlation or create new one.

        Yields:
            str | None: The correlation ID (existing or new)

        """
        existing_id = _correlation_id.get()
        if existing_id:
            # Use existing correlation
            yield existing_id
        else:
            # Create new correlation context
            with FlextContext.new_correlation() as new_id:
                yield new_id

    # =============================================================================
    # SERVICE IDENTIFICATION
    # =============================================================================

    @staticmethod
    def get_service_name() -> str | None:
        """Get current service name from context."""
        return _service_name.get()

    @staticmethod
    def set_service_name(service_name: str) -> None:
        """Set the service name in context."""
        _service_name.set(service_name)

    @staticmethod
    def get_service_version() -> str | None:
        """Get the current service version from context."""
        return _service_version.get()

    @staticmethod
    def set_service_version(version: str) -> None:
        """Set service version in context."""
        _service_version.set(version)

    @staticmethod
    @contextmanager
    def service_context(
        service_name: str,
        version: str | None = None,
    ) -> Generator[None]:
        """Create service identification context scope.

        Args:
            service_name: Name of the service
            version: Version of the service

        """
        # Save current context
        _service_name.get()
        _service_version.get()

        # Set new context
        name_token = _service_name.set(service_name)
        version_token = None
        if version:
            version_token = _service_version.set(version)

        try:
            yield
        finally:
            # Restore previous context
            _service_name.reset(name_token)
            if version_token:
                _service_version.reset(version_token)

    # =============================================================================
    # REQUEST METADATA
    # =============================================================================

    @staticmethod
    def get_user_id() -> str | None:
        """Get the current user ID from context."""
        return _user_id.get()

    @staticmethod
    def set_user_id(user_id: str) -> None:
        """Set user ID in context."""
        _user_id.set(user_id)

    @staticmethod
    def get_operation_name() -> str | None:
        """Get the current operation name from context."""
        return _operation_name.get()

    @staticmethod
    def set_operation_name(operation_name: str) -> None:
        """Set operation name in context."""
        _operation_name.set(operation_name)

    @staticmethod
    def get_request_id() -> str | None:
        """Get current request ID from context."""
        return _request_id.get()

    @staticmethod
    def set_request_id(request_id: str) -> None:
        """Set request ID in context."""
        _request_id.set(request_id)

    @staticmethod
    @contextmanager
    def request_context(
        *,
        user_id: str | None = None,
        operation_name: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Generator[None]:
        """Create request metadata context scope.

        Args:
            user_id: User identifier for the request
            operation_name: Name of the operation being performed
            request_id: External request identifier
            metadata: Additional request metadata

        """
        # Save current context
        _user_id.get()
        _operation_name.get()
        _request_id.get()
        _operation_metadata.get()

        # Set a new context with proper token typing
        user_token = _user_id.set(user_id) if user_id else None
        operation_token = (
            _operation_name.set(operation_name) if operation_name else None
        )
        request_token = _request_id.set(request_id) if request_id else None
        metadata_token = _operation_metadata.set(metadata) if metadata else None

        try:
            yield
        finally:
            # Restore previous context with proper typing
            if user_token is not None:
                _user_id.reset(user_token)
            if operation_token is not None:
                _operation_name.reset(operation_token)
            if request_token is not None:
                _request_id.reset(request_token)
            if metadata_token is not None:
                _operation_metadata.reset(metadata_token)

    # =============================================================================
    # PERFORMANCE CONTEXT
    # =============================================================================

    @staticmethod
    def get_operation_start_time() -> datetime | None:
        """Get operation start time from context."""
        return _operation_start_time.get()

    @staticmethod
    def set_operation_start_time(start_time: datetime | None = None) -> None:
        """Set operation start time in context."""
        if start_time is None:
            start_time = datetime.now(UTC)
        _operation_start_time.set(start_time)

    @staticmethod
    def get_operation_metadata() -> dict[str, object] | None:
        """Get operation metadata from context."""
        return _operation_metadata.get()

    @staticmethod
    def set_operation_metadata(metadata: dict[str, object]) -> None:
        """Set operation metadata in context."""
        _operation_metadata.set(metadata)

    @staticmethod
    def add_operation_metadata(key: str, value: object) -> None:
        """Add single metadata entry to operation context."""
        current_metadata = _operation_metadata.get() or {}
        current_metadata[key] = value
        _operation_metadata.set(current_metadata)

    @staticmethod
    @contextmanager
    def timed_operation(
        operation_name: str | None = None,
    ) -> Generator[dict[str, object]]:
        """Create timed operation context with performance tracking.

        Args:
            operation_name: Name of the operation being timed

        Yields:
            Dict[str, object]: Operation metadata dictionary

        """
        start_time = datetime.now(UTC)
        # Type-safe operation metadata
        operation_metadata: dict[str, object] = {
            "start_time": start_time,
            "operation_name": operation_name,
        }

        # Save current context
        _operation_start_time.get()
        _operation_metadata.get()
        _operation_name.get()

        # Set new context
        start_token = _operation_start_time.set(start_time)
        metadata_token = _operation_metadata.set(operation_metadata)
        operation_token = None
        if operation_name:
            operation_token = _operation_name.set(operation_name)

        try:
            yield operation_metadata
        finally:
            # Calculate duration
            end_time = datetime.now(UTC)
            duration = (end_time - start_time).total_seconds()
            # Type-safe update
            operation_metadata.update(
                {"end_time": end_time, "duration_seconds": duration},
            )

            # Restore previous context
            _operation_start_time.reset(start_token)
            _operation_metadata.reset(metadata_token)
            if operation_token:
                _operation_name.reset(operation_token)

    # =============================================================================
    # CONTEXT SERIALIZATION
    # =============================================================================

    @staticmethod
    def get_full_context() -> dict[str, object]:
        """Get complete current context as dictionary.

        Returns:
            Dict[str, object]: All context variables with current values

        """
        return {
            "correlation_id": _correlation_id.get(),
            "parent_correlation_id": _parent_correlation_id.get(),
            "service_name": _service_name.get(),
            "service_version": _service_version.get(),
            "user_id": _user_id.get(),
            "operation_name": _operation_name.get(),
            "request_id": _request_id.get(),
            "operation_start_time": _operation_start_time.get(),
            "operation_metadata": _operation_metadata.get(),
        }

    @staticmethod
    def get_correlation_context() -> dict[str, str]:
        """Get correlation context for cross-service propagation.

        Returns:
            Dict[str, str]: Correlation context for HTTP headers/bridge calls

        """
        context = {}

        correlation_id = _correlation_id.get()
        if correlation_id:
            context["X-Correlation-Id"] = correlation_id

        parent_id = _parent_correlation_id.get()
        if parent_id:
            context["X-Parent-Correlation-Id"] = parent_id

        service_name = _service_name.get()
        if service_name:
            context["X-Service-Name"] = service_name

        return context

    @staticmethod
    def set_from_context(context: Mapping[str, object]) -> None:
        """Set context from dictionary (e.g., from HTTP headers).

        Args:
            context: Context dictionary with values to set

        """
        correlation_id = context.get("X-Correlation-Id") or context.get(
            "correlation_id",
        )
        if correlation_id and isinstance(correlation_id, str):
            _correlation_id.set(correlation_id)

        parent_id = context.get("X-Parent-Correlation-Id") or context.get(
            "parent_correlation_id",
        )
        if parent_id and isinstance(parent_id, str):
            _parent_correlation_id.set(parent_id)

        service_name = context.get("X-Service-Name") or context.get("service_name")
        if service_name and isinstance(service_name, str):
            _service_name.set(service_name)

        user_id = context.get("X-User-Id") or context.get("user_id")
        if user_id and isinstance(user_id, str):
            _user_id.set(user_id)

    @staticmethod
    def clear_context() -> None:
        """Clear all context variables."""
        # Clear string context variables
        for context_var in [
            _correlation_id,
            _parent_correlation_id,
            _service_name,
            _service_version,
            _user_id,
            _operation_name,
            _request_id,
        ]:
            with contextlib.suppress(LookupError):
                context_var.set(None)

        # Clear typed context variables
        with contextlib.suppress(LookupError):
            _operation_start_time.set(None)
        with contextlib.suppress(LookupError):
            _operation_metadata.set(None)

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    @staticmethod
    def ensure_correlation_id() -> str:
        """Ensure correlation ID exists, creating one if needed.

        Returns:
            str: Existing or newly created correlation ID

        """
        correlation_id = _correlation_id.get()
        if not correlation_id:
            correlation_id = FlextContext.generate_correlation_id()
        return correlation_id

    @staticmethod
    def has_correlation_id() -> bool:
        """Check if correlation ID is set in context.

        Returns:
            bool: True if correlation ID exists

        """
        return _correlation_id.get() is not None

    @staticmethod
    def get_context_summary() -> str:
        """Get a human-readable context summary for debugging.

        Returns:
            str: Context summary string

        """
        context = FlextContext.get_full_context()
        parts = []

        correlation_id = context["correlation_id"]
        if correlation_id and isinstance(correlation_id, str):
            parts.append(f"correlation={correlation_id[:8]}...")

        service_name = context["service_name"]
        if service_name and isinstance(service_name, str):
            parts.append(f"service={service_name}")

        operation_name = context["operation_name"]
        if operation_name and isinstance(operation_name, str):
            parts.append(f"operation={operation_name}")

        user_id = context["user_id"]
        if user_id and isinstance(user_id, str):
            parts.append(f"user={user_id}")

        return f"FlextContext({', '.join(parts)})" if parts else "FlextContext(empty)"


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [
    "FlextContext",
]
