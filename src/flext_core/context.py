"""Context management and distributed tracing for the FLEXT core library.

Provides hierarchical context management system for the FLEXT ecosystem following
Clean Architecture principles with thread-safe context variables, distributed tracing,
and cross-service correlation ID propagation.

This module implements a comprehensive, hierarchical context management system
organized into logical domains following the established flext-core patterns:
    - Variables: Context variables by domain (Correlation, Service, etc.)
    - Correlation: Distributed tracing and correlation ID management
    - Service: Service identification and lifecycle context
    - Request: User and request metadata management
    - Performance: Operation timing and performance tracking
    - Serialization: Cross-service context serialization
    - Utilities: Context utility methods and helpers

Architecture Features:
    - Thread-safe context management using Python contextvars
    - Hierarchical organization following unified flext-core patterns
    - Type safety with Python 3.13+ Final[type] annotations
    - SOLID principles implementation throughout
    - Clean Architecture domain separation
    - Cross-service context propagation support

Examples:
    Basic context operations::

        from flext_core.context import FlextContext

        # Generate correlation ID
        correlation_id = FlextContext.Correlation.generate_correlation_id()

        # Service context with automatic cleanup
        with FlextContext.Service.service_context("user-service", "1.2.0"):
            # Nested correlation context
            with FlextContext.Correlation.new_correlation() as corr_id:
                # Performance tracking
                with FlextContext.Performance.timed_operation("user_creation"):
                    # All context automatically managed and restored
                    pass

Note:
    All context variables are thread-safe and support nested scopes with automatic
    cleanup. Context propagation is supported for cross-service communication.

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

# =============================================================================
# FlextContext - Hierarchical Context Management System
# =============================================================================


class FlextContext:
    """Hierarchical context management system following Clean Architecture principles.

    This class implements a comprehensive, hierarchical context management system
    for the FLEXT ecosystem, organizing context functionality by domain and following
    Clean Architecture principles with SOLID design patterns.

    The context system is organized into distinct domains:
        - Variables: Context variables organized by domain
        - Correlation: Distributed tracing and correlation ID management
        - Service: Service identification and lifecycle context
        - Request: User and request metadata management
        - Performance: Operation timing and performance tracking
        - Serialization: Cross-service context serialization
        - Utilities: Context utility methods and helpers

    Architecture Features:
        - Thread-safe context management using Python contextvars
        - Hierarchical organization following unified flext-core patterns
        - Type safety with Python 3.13+ Final[type] annotations
        - SOLID principles implementation throughout
        - Clean Architecture domain separation
        - Cross-service context propagation support

    Architecture Principles:
        - Single Responsibility: Each nested class has a single domain focus
        - Open/Closed: Easy to extend with new context operation categories
        - Liskov Substitution: Consistent interface across all context managers
        - Interface Segregation: Clients depend only on context operations they use
        - Dependency Inversion: High-level operations independent of details

    Examples:
        Basic context operations::

            from flext_core.context import FlextContext

            # Generate correlation ID
            correlation_id = FlextContext.Correlation.generate_correlation_id()

            # Service context with automatic cleanup
            with FlextContext.Service.service_context("user-service", "1.2.0"):
                # Nested correlation context
                with FlextContext.Correlation.new_correlation() as corr_id:
                    # Performance tracking
                    with FlextContext.Performance.timed_operation("user_creation"):
                        # All context automatically managed and restored
                        pass

    """

    # =========================================================================
    # Variables Domain - Context variables organized by functionality
    # =========================================================================

    class Variables:
        """Context variables organized by domain for Clean Architecture.

        This class provides a structured organization of all context variables used
        throughout the FLEXT ecosystem, grouped by domain and functionality for better
        maintainability, discoverability, and adherence to SOLID principles.
        """

        class Correlation:
            """Correlation variables for distributed tracing/request tracking."""

            CORRELATION_ID: Final[ContextVar[str | None]] = ContextVar(
                "correlation_id", default=None
            )
            PARENT_CORRELATION_ID: Final[ContextVar[str | None]] = ContextVar(
                "parent_correlation_id", default=None
            )

        class Service:
            """Service context variables for service identification and versioning."""

            SERVICE_NAME: Final[ContextVar[str | None]] = ContextVar(
                "service_name", default=None
            )
            SERVICE_VERSION: Final[ContextVar[str | None]] = ContextVar(
                "service_version", default=None
            )
            ENVIRONMENT: Final[ContextVar[str | None]] = ContextVar(
                "environment", default=None
            )

        class Request:
            """Request context variables for request metadata and user tracking."""

            USER_ID: Final[ContextVar[str | None]] = ContextVar("user_id", default=None)
            REQUEST_ID: Final[ContextVar[str | None]] = ContextVar(
                "request_id", default=None
            )
            REQUEST_TIMESTAMP: Final[ContextVar[datetime | None]] = ContextVar(
                "request_timestamp", default=None
            )

        class Performance:
            """Performance context variables for operation timing and monitoring."""

            OPERATION_NAME: Final[ContextVar[str | None]] = ContextVar(
                "operation_name", default=None
            )
            OPERATION_START_TIME: Final[ContextVar[datetime | None]] = ContextVar(
                "operation_start_time", default=None
            )
            OPERATION_METADATA: Final[ContextVar[FlextTypes.Core.Dict | None]] = (
                ContextVar("operation_metadata", default=None)
            )

    # =========================================================================
    # Correlation Domain - Distributed tracing and correlation ID management
    # =========================================================================

    class Correlation:
        """Distributed tracing and correlation ID management.

        Provides comprehensive correlation ID management functionality for the FLEXT
        ecosystem, implementing distributed tracing patterns, parent-child relationship
        tracking, and cross-service correlation for observability.

        Features:
            - Thread-safe correlation ID generation and management
            - Parent-child relationship tracking for nested operations
            - Context-aware scope management with automatic cleanup
            - Inheritance patterns for existing correlation propagation
            - Type-safe operations with comprehensive error handling

        Examples:
            Basic correlation management::

                # Generate correlation ID
                correlation_id = FlextContext.Correlation.generate_correlation_id()

                # Context manager with automatic cleanup
                with FlextContext.Correlation.new_correlation() as corr_id:
                    # Nested correlation with parent tracking
                    with FlextContext.Correlation.new_correlation() as nested_id:
                        parent_id = FlextContext.Correlation.get_parent_correlation_id()

            Correlation inheritance::

                # Inherit existing correlation or create new one
                with FlextContext.Correlation.inherit_correlation() as corr_id:
                    # Use existing or new correlation
                    pass

        """

        @staticmethod
        def get_correlation_id() -> str | None:
            """Get current correlation ID from context.

            Returns:
                str | None: Current correlation ID or None if not set

            """
            return FlextContext.Variables.Correlation.CORRELATION_ID.get()

        @staticmethod
        def set_correlation_id(correlation_id: str) -> None:
            """Set correlation ID in current context.

            Args:
                correlation_id: Correlation ID to set in context

            """
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate a new unique correlation ID.

            Returns:
                str: The generated correlation ID

            """
            correlation_id = FlextUtilities.Generators.generate_correlation_id()
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)
            return correlation_id

        @staticmethod
        def get_parent_correlation_id() -> str | None:
            """Get parent correlation ID from context.

            Returns:
                str | None: Parent correlation ID or None if not set

            """
            return FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.get()

        @staticmethod
        def set_parent_correlation_id(parent_id: str) -> None:
            """Set parent correlation ID in current context.

            Args:
                parent_id: Parent correlation ID to set in context

            """
            FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(parent_id)

        @staticmethod
        @contextmanager
        def new_correlation(
            correlation_id: str | None = None,
            parent_id: str | None = None,
        ) -> Generator[str]:
            """Create a new correlation context scope with automatic cleanup.

            Args:
                correlation_id: Specific correlation ID to use (generates if None)
                parent_id: Parent correlation ID for hierarchical tracing

            Yields:
                str: The correlation ID for this context

            """
            # Generate correlation ID if not provided
            if correlation_id is None:
                from flext_core.utilities import FlextUtilities  # noqa: PLC0415

                correlation_id = FlextUtilities.Generators.generate_correlation_id()

            # Save current context
            current_correlation = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get()
            )

            # Set new context
            correlation_token = FlextContext.Variables.Correlation.CORRELATION_ID.set(
                correlation_id
            )

            # Set parent context
            parent_token: Token[str | None] | None = None
            if parent_id:
                parent_token = (
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(
                        parent_id
                    )
                )
            elif current_correlation:
                # Current correlation becomes parent
                parent_token = (
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(
                        current_correlation
                    )
                )

            try:
                yield correlation_id
            finally:
                # Restore previous context
                FlextContext.Variables.Correlation.CORRELATION_ID.reset(
                    correlation_token
                )
                if parent_token:
                    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.reset(
                        parent_token
                    )

        @staticmethod
        @contextmanager
        def inherit_correlation() -> Generator[str | None]:
            """Inherit existing correlation or create new one.

            Yields:
                str | None: The correlation ID (existing or new)

            """
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
        """Service identification and lifecycle context management.

        Provides comprehensive service identification functionality for the FLEXT
        ecosystem, implementing service-to-service communication patterns, versioning
        support, and service mesh integration for microservices.

        Key Features:
            - Service name and version identification
            - Thread-safe service context management
            - Context managers for scoped service identification
            - Service mesh integration support
            - Cross-service context propagation

        Examples:
            Basic service identification::

                # Set service information
                FlextContext.Service.set_service_name("user-service")
                FlextContext.Service.set_service_version("1.2.0")

                # Get current service information
                service_name = FlextContext.Service.get_service_name()
                version = FlextContext.Service.get_service_version()

            Context manager usage::

                # Service context with automatic cleanup
                with FlextContext.Service.service_context("user-service", "1.2.0"):
                    print(f"Service: {FlextContext.Service.get_service_name()}")
                    # Context automatically restored

        """

        @staticmethod
        def get_service_name() -> str | None:
            """Get current service name from context.

            Returns:
                str | None: Current service name or None if not set

            """
            return FlextContext.Variables.Service.SERVICE_NAME.get()

        @staticmethod
        def set_service_name(service_name: str) -> None:
            """Set the service name in context.

            Args:
                service_name: Name of the service to set in context

            """
            FlextContext.Variables.Service.SERVICE_NAME.set(service_name)

        @staticmethod
        def get_service_version() -> str | None:
            """Get the current service version from context.

            Returns:
                str | None: Current service version or None if not set

            """
            return FlextContext.Variables.Service.SERVICE_VERSION.get()

        @staticmethod
        def set_service_version(version: str) -> None:
            """Set service version in context.

            Args:
                version: Version string to set in context

            """
            FlextContext.Variables.Service.SERVICE_VERSION.set(version)

        @staticmethod
        @contextmanager
        def service_context(
            service_name: str,
            version: str | None = None,
        ) -> Generator[None]:
            """Create service identification context scope with automatic cleanup.

            Args:
                service_name: Name of the service
                version: Version of the service (optional)

            Yields:
                None: Context manager yields nothing

            """
            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.Service.SERVICE_NAME.get()
            _ = FlextContext.Variables.Service.SERVICE_VERSION.get()

            # Set new context
            name_token = FlextContext.Variables.Service.SERVICE_NAME.set(service_name)
            version_token = None
            if version:
                version_token = FlextContext.Variables.Service.SERVICE_VERSION.set(
                    version
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
        """Request-level context management for user and operation metadata.

        Provides comprehensive request metadata management functionality for the FLEXT
        ecosystem, implementing user identification, operation tracking, and request
        lifecycle management following security and audit best practices.

        Key Features:
            - User identification and authorization context
            - Operation name tracking for audit and monitoring
            - Request ID management for correlation
            - Thread-safe request context management
            - Context managers for scoped request metadata

        Examples:
            Basic request metadata management::

                # Set request information
                FlextContext.Request.set_user_id("user123")
                FlextContext.Request.set_operation_name("create_user")
                FlextContext.Request.set_request_id("req456")

                # Get current request information
                user_id = FlextContext.Request.get_user_id()
                operation = FlextContext.Request.get_operation_name()

            Context manager usage::

                # Request context with automatic cleanup
                with FlextContext.Request.request_context(
                    user_id="user123",
                    operation_name="update_profile",
                    request_id="req789",
                ):
                    print(f"User: {FlextContext.Request.get_user_id()}")
                    # Context automatically restored

        """

        @staticmethod
        def get_user_id() -> str | None:
            """Get the current user ID from context.

            Returns:
                str | None: Current user ID or None if not set

            """
            return FlextContext.Variables.Request.USER_ID.get()

        @staticmethod
        def set_user_id(user_id: str) -> None:
            """Set user ID in context.

            Args:
                user_id: User identifier to set in context

            """
            FlextContext.Variables.Request.USER_ID.set(user_id)

        @staticmethod
        def get_operation_name() -> str | None:
            """Get the current operation name from context.

            Returns:
                str | None: Current operation name or None if not set

            """
            return FlextContext.Variables.Performance.OPERATION_NAME.get()

        @staticmethod
        def set_operation_name(operation_name: str) -> None:
            """Set operation name in context.

            Args:
                operation_name: Name of the operation to set in context

            """
            FlextContext.Variables.Performance.OPERATION_NAME.set(operation_name)

        @staticmethod
        def get_request_id() -> str | None:
            """Get current request ID from context.

            Returns:
                str | None: Current request ID or None if not set

            """
            return FlextContext.Variables.Request.REQUEST_ID.get()

        @staticmethod
        def set_request_id(request_id: str) -> None:
            """Set request ID in context.

            Args:
                request_id: Request identifier to set in context

            """
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
            """Create request metadata context scope with automatic cleanup.

            Args:
                user_id: User identifier for the request
                operation_name: Name of the operation being performed
                request_id: External request identifier
                metadata: Additional request metadata

            Yields:
                None: Context manager yields nothing

            """
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
                        operation_token
                    )
                if request_token is not None:
                    FlextContext.Variables.Request.REQUEST_ID.reset(request_token)
                if metadata_token is not None:
                    FlextContext.Variables.Performance.OPERATION_METADATA.reset(
                        metadata_token
                    )

    # =========================================================================
    # Performance Domain - Operation timing and performance tracking
    # =========================================================================

    class Performance:
        """Performance monitoring and timing context management for operations.

        Provides comprehensive performance tracking functionality for the FLEXT
        ecosystem, implementing operation timing, metadata collection, and resource
        monitoring following observability and performance engineering patterns.

        Key Features:
            - Operation timing with automatic duration calculation
            - Custom metadata collection during operations
            - Thread-safe performance context management
            - Context managers for scoped performance tracking
            - Integration with correlation and request context

        Examples:
            Basic performance tracking::

                # Set operation start time
                FlextContext.Performance.set_operation_start_time()

                # Add custom metadata
                FlextContext.Performance.add_operation_metadata("user_id", "123")

                # Get operation metadata
                metadata = FlextContext.Performance.get_operation_metadata()

            Timed operation context::

                # Automatic timing with metadata
                with FlextContext.Performance.timed_operation(
                    "user_creation"
                ) as metadata:
                    # Perform operation
                    user = create_user(data)

                    # Add custom metadata
                    FlextContext.Performance.add_operation_metadata("user_id", user.id)

                    # Duration automatically calculated
                    print(f"Duration: {metadata['duration_seconds']}s")

        """

        @staticmethod
        def get_operation_start_time() -> datetime | None:
            """Get operation start time from context.

            Returns:
                datetime | None: Operation start time or None if not set

            """
            return FlextContext.Variables.Performance.OPERATION_START_TIME.get()

        @staticmethod
        def set_operation_start_time(start_time: datetime | None = None) -> None:
            """Set operation start time in context.

            Args:
                start_time: Start time to set (current time if None)

            """
            if start_time is None:
                start_time = datetime.now(UTC)
            FlextContext.Variables.Performance.OPERATION_START_TIME.set(start_time)

        @staticmethod
        def get_operation_metadata() -> FlextTypes.Core.Dict | None:
            """Get operation metadata from context.

            Returns:
                Dict[str, object] | None: Operation metadata or None if not set

            """
            return FlextContext.Variables.Performance.OPERATION_METADATA.get()

        @staticmethod
        def set_operation_metadata(metadata: FlextTypes.Core.Dict) -> None:
            """Set operation metadata in context.

            Args:
                metadata: Metadata dictionary to set in context

            """
            FlextContext.Variables.Performance.OPERATION_METADATA.set(metadata)

        @staticmethod
        def add_operation_metadata(key: str, value: object) -> None:
            """Add single metadata entry to operation context.

            Args:
                key: Metadata key
                value: Metadata value

            """
            current_metadata = (
                FlextContext.Variables.Performance.OPERATION_METADATA.get() or {}
            )
            current_metadata[key] = value
            FlextContext.Variables.Performance.OPERATION_METADATA.set(current_metadata)

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
            operation_metadata: dict[str, object] = {
                "start_time": start_time,
                "operation_name": operation_name,
            }

            # Save current context (for potential future use in logging/debugging)
            _ = FlextContext.Variables.Performance.OPERATION_START_TIME.get()
            _ = FlextContext.Variables.Performance.OPERATION_METADATA.get()
            _ = FlextContext.Variables.Performance.OPERATION_NAME.get()

            # Set new context
            start_token = FlextContext.Variables.Performance.OPERATION_START_TIME.set(
                start_time
            )
            metadata_token = FlextContext.Variables.Performance.OPERATION_METADATA.set(
                operation_metadata
            )
            operation_token = None
            if operation_name:
                operation_token = FlextContext.Variables.Performance.OPERATION_NAME.set(
                    operation_name
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
                    }
                )

                # Restore previous context
                FlextContext.Variables.Performance.OPERATION_START_TIME.reset(
                    start_token
                )
                FlextContext.Variables.Performance.OPERATION_METADATA.reset(
                    metadata_token
                )
                if operation_token:
                    FlextContext.Variables.Performance.OPERATION_NAME.reset(
                        operation_token
                    )

    # =========================================================================
    # Serialization Domain - Context serialization for cross-service communication
    # =========================================================================

    class Serialization:
        """Context serialization and deserialization for cross-service communication.

        Provides functionality for serializing context state for HTTP headers,
        message queues, and other cross-service communication mechanisms following
        distributed systems and observability patterns.

        Key Features:
            - Full context serialization to dictionary
            - Correlation context extraction for HTTP headers
            - Context deserialization from external sources
            - Type-safe serialization with proper typing
            - Cross-service context propagation support

        Examples:
            Context serialization::

                # Get complete current context
                full_context = FlextContext.Serialization.get_full_context()

                # Get correlation context for HTTP headers
                headers = FlextContext.Serialization.get_correlation_context()
                response = httpx.get("/api/endpoint", headers=headers)

            Context deserialization::

                # Set context from incoming headers
                FlextContext.Serialization.set_from_context(request.headers)

                # Set context from message metadata
                FlextContext.Serialization.set_from_context(message.metadata)

        """

        @staticmethod
        def get_full_context() -> FlextTypes.Core.Dict:
            """Get complete current context as dictionary.

            Returns:
                Dict[str, object]: All context variables with current values

            """
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
        def get_correlation_context() -> dict[str, str]:
            """Get correlation context for cross-service propagation.

            Returns:
                Dict[str, str]: Correlation context for HTTP headers/bridge calls

            """
            context: dict[str, str] = {}

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
            """Set context from dictionary (e.g., from HTTP headers).

            Args:
                context: Context dictionary with values to set

            """
            correlation_id = context.get("X-Correlation-Id") or context.get(
                "correlation_id"
            )
            if correlation_id and isinstance(correlation_id, str):
                FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

            parent_id = context.get("X-Parent-Correlation-Id") or context.get(
                "parent_correlation_id"
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
        """Utility methods for context management and helper operations.

        Provides utility functionality for context management, debugging,
        and maintenance operations following utility and helper patterns.

        Key Features:
            - Context clearing and reset operations
            - Correlation ID ensuring and validation
            - Context summary generation for debugging
            - Context state inspection utilities
            - Type-safe utility operations

        Examples:
            Context utilities::

                # Ensure correlation ID exists
                correlation_id = FlextContext.Utilities.ensure_correlation_id()

                # Check if correlation ID is set
                has_correlation = FlextContext.Utilities.has_correlation_id()

                # Get human-readable context summary
                summary = FlextContext.Utilities.get_context_summary()

                # Clear all context variables
                FlextContext.Utilities.clear_context()

        """

        @staticmethod
        def clear_context() -> None:
            """Clear all context variables.

            Resets all context variables to their default values (None).
            """
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
            """Ensure correlation ID exists, creating one if needed.

            Returns:
                str: Existing or newly created correlation ID

            """
            correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if not correlation_id:
                correlation_id = FlextContext.Correlation.generate_correlation_id()
            return correlation_id

        @staticmethod
        def has_correlation_id() -> bool:
            """Check if correlation ID is set in context.

            Returns:
                bool: True if correlation ID exists

            """
            return FlextContext.Variables.Correlation.CORRELATION_ID.get() is not None

        @staticmethod
        def get_context_summary() -> str:
            """Get a human-readable context summary for debugging.

            Returns:
                str: Context summary string

            """
            context = FlextContext.Serialization.get_full_context()
            parts: list[str] = []

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


# =============================================================================
# Backward Compatibility Layer
# =============================================================================

# Alias for backward compatibility
FlextContexts = FlextContext

# Context variable aliases for backward compatibility
_correlation_id: Final[ContextVar[str | None]] = (
    FlextContext.Variables.Correlation.CORRELATION_ID
)
_parent_correlation_id: Final[ContextVar[str | None]] = (
    FlextContext.Variables.Correlation.PARENT_CORRELATION_ID
)
_service_name: Final[ContextVar[str | None]] = (
    FlextContext.Variables.Service.SERVICE_NAME
)
_service_version: Final[ContextVar[str | None]] = (
    FlextContext.Variables.Service.SERVICE_VERSION
)
_user_id: Final[ContextVar[str | None]] = FlextContext.Variables.Request.USER_ID
_operation_name: Final[ContextVar[str | None]] = (
    FlextContext.Variables.Performance.OPERATION_NAME
)
_request_id: Final[ContextVar[str | None]] = FlextContext.Variables.Request.REQUEST_ID
_operation_start_time: Final[ContextVar[datetime | None]] = (
    FlextContext.Variables.Performance.OPERATION_START_TIME
)
_operation_metadata: Final[ContextVar[FlextTypes.Core.Dict | None]] = (
    FlextContext.Variables.Performance.OPERATION_METADATA
)


# =============================================================================
# Exports
# =============================================================================

__all__: list[str] = [
    "FlextContext",  # ONLY main class exported
]
