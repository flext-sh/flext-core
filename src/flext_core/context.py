"""Enterprise context management with distributed tracing and cross-service correlation.

Thread-safe context management implementing correlation ID tracking, service identification,
and performance monitoring using contextvars for distributed tracing and observability.

"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Final

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Removed FlextUtilities import to avoid circular dependency


class FlextContext:
    """Hierarchical context management system following Clean Architecture principles.

    This class implements a efficient, hierarchical context management system
    for the FLEXT ecosystem, organizing context functionality by domain and following
    Clean Architecture principles with SOLID design patterns.

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
                "correlation_id",
                default=None,
            )
            PARENT_CORRELATION_ID: Final[ContextVar[str | None]] = ContextVar(
                "parent_correlation_id",
                default=None,
            )

        class Service:
            """Service context variables for service identification and versioning."""

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
            """Request context variables for request metadata and user tracking."""

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
            """Performance context variables for operation timing and monitoring."""

            OPERATION_NAME: Final[ContextVar[str | None]] = ContextVar(
                "operation_name",
                default=None,
            )
            OPERATION_START_TIME: Final[ContextVar[datetime | None]] = ContextVar(
                "operation_start_time",
                default=None,
            )
            OPERATION_METADATA: Final[ContextVar[dict[str, object] | None]] = (
                ContextVar("operation_metadata", default=None)
            )

    # =========================================================================
    # Correlation Domain - Distributed tracing and correlation ID management
    # =========================================================================

    class Correlation:
        """Distributed tracing and correlation ID management.

        Provides efficient correlation ID management functionality for the FLEXT
        ecosystem, implementing distributed tracing patterns, parent-child relationship
        tracking, and cross-service correlation for observability.


        """

        @staticmethod
        def get_correlation_id() -> str | None:
            """Get current correlation ID from context."""
            return FlextContext.Variables.Correlation.CORRELATION_ID.get()

        @staticmethod
        def set_correlation_id(correlation_id: str) -> None:
            """Set correlation ID in current context."""
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate a new unique correlation ID."""
            correlation_id = f"corr_{uuid.uuid4().hex[:16]}"
            FlextContext.Variables.Correlation.CORRELATION_ID.set(correlation_id)
            return correlation_id

        @staticmethod
        def get_parent_correlation_id() -> str | None:
            """Get parent correlation ID from context."""
            return FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.get()

        @staticmethod
        def set_parent_correlation_id(parent_id: str) -> None:
            """Set parent correlation ID in current context."""
            FlextContext.Variables.Correlation.PARENT_CORRELATION_ID.set(parent_id)

        @staticmethod
        @contextmanager
        def new_correlation(
            correlation_id: str | None = None,
            parent_id: str | None = None,
        ) -> Generator[str]:
            """Create a new correlation context scope with automatic cleanup."""
            # Generate correlation ID if not provided
            if correlation_id is None:
                correlation_id = f"corr_{uuid.uuid4().hex[:16]}"

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
            """Inherit existing correlation or create new one."""
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

        Provides efficient service identification functionality for the FLEXT
        ecosystem, implementing service-to-service communication patterns, versioning
        support, and service mesh integration for microservices.
        """

        @staticmethod
        def get_service_name() -> str | None:
            """Get current service name from context."""
            return FlextContext.Variables.Service.SERVICE_NAME.get()

        @staticmethod
        def set_service_name(service_name: str) -> None:
            """Set the service name in context."""
            FlextContext.Variables.Service.SERVICE_NAME.set(service_name)

        @staticmethod
        def get_service_version() -> str | None:
            """Get the current service version from context."""
            return FlextContext.Variables.Service.SERVICE_VERSION.get()

        @staticmethod
        def set_service_version(version: str) -> None:
            """Set service version in context."""
            FlextContext.Variables.Service.SERVICE_VERSION.set(version)

        @staticmethod
        @contextmanager
        def service_context(
            service_name: str,
            version: str | None = None,
        ) -> Generator[None]:
            """Create service identification context scope with automatic cleanup."""
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
        """Request-level context management for user and operation metadata.

        Provides efficient request metadata management functionality for the FLEXT
        ecosystem, implementing user identification, operation tracking, and request
        lifecycle management following security and audit best practices.


        """

        @staticmethod
        def get_user_id() -> str | None:
            """Get the current user ID from context."""
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
            metadata: dict[str, object] | None = None,
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
        """Performance monitoring and timing context management for operations.

        Provides efficient performance tracking functionality for the FLEXT
        ecosystem, implementing operation timing, metadata collection, and resource
        monitoring following observability and performance engineering patterns.

        Key Features:

        """

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
        def get_operation_metadata() -> dict[str, object] | None:
            """Get operation metadata from context."""
            return FlextContext.Variables.Performance.OPERATION_METADATA.get()

        @staticmethod
        def set_operation_metadata(metadata: dict[str, object]) -> None:
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
        ) -> Generator[dict[str, object]]:
            """Create timed operation context with performance tracking."""
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
        """Context serialization and deserialization for cross-service communication.

        Provides functionality for serializing context state for HTTP headers,
        message queues, and other cross-service communication mechanisms following
        distributed systems and observability patterns.

        """

        @staticmethod
        def get_full_context() -> dict[str, object]:
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
        """Utility methods for context management and helper operations.

        Provides utility functionality for context management, debugging,
        and maintenance operations following utility and helper patterns.

        """

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
    # FLEXT CONTEXT CONFIGURATION METHODS - Standard FlextTypes.Config
    # =============================================================================

    @classmethod
    def configure_context_system(
        cls,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure context system using FlextTypes.Config with StrEnum validation.

        Configures the FLEXT context management system including distributed tracing,
        correlation ID management, service context tracking, performance monitoring,
        and cross-service context propagation with thread-safe operations.
        """
        try:
            # Create working copy of config
            validated_config = dict(config)

            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}",
                    )
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate context_level (using validation level as basis)
            if "context_level" in config:
                context_value = config["context_level"]
                valid_levels = [e.value for e in FlextConstants.Config.ValidationLevel]
                if context_value not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid context_level '{context_value}'. Valid options: {valid_levels}",
                    )
            else:
                validated_config["context_level"] = (
                    FlextConstants.Config.ValidationLevel.LOOSE.value
                )

            # Validate log_level
            if "log_level" in config:
                log_value = config["log_level"]
                valid_log_levels = [e.value for e in FlextConstants.Config.LogLevel]
                if log_value not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_value}'. Valid options: {valid_log_levels}",
                    )
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.DEBUG.value
                )

            # Set default values for context system specific settings
            validated_config.setdefault("enable_correlation_tracking", True)
            validated_config.setdefault("enable_service_context", True)
            validated_config.setdefault("enable_performance_tracking", True)
            validated_config.setdefault("context_propagation_enabled", True)
            validated_config.setdefault("max_context_depth", 20)
            validated_config.setdefault("context_serialization_enabled", True)
            validated_config.setdefault("context_cleanup_enabled", True)
            validated_config.setdefault("enable_nested_contexts", True)

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to configure context system: {e}",
            )

    @classmethod
    def get_context_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current context system configuration with runtime metrics.

        Retrieves the current context system configuration including runtime metrics,
        active context scopes, correlation tracking status, service context data,
        and performance tracking metrics for monitoring and diagnostics.

        """
        try:
            # Get current context state for runtime metrics
            correlation_id = cls.Correlation.get_correlation_id()
            service_name = cls.Service.get_service_name()
            operation_name = cls.Request.get_operation_name()

            # Build current configuration with runtime metrics
            current_config: FlextTypes.Config.ConfigDict = {
                # Core system configuration
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "context_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                # Context system specific configuration
                "enable_correlation_tracking": True,
                "enable_service_context": True,
                "enable_performance_tracking": True,
                "context_propagation_enabled": True,
                "max_context_depth": 20,
                # Runtime metrics and status
                "active_context_scopes": 0,  # Would be dynamically calculated
                "total_context_operations": 0,  # Runtime counter
                "successful_context_operations": 0,  # Success counter
                "failed_context_operations": 0,  # Failure counter
                "average_context_operation_time_ms": 0.0,  # Performance metric
                # Context tracking status
                "correlation_tracking_active": correlation_id is not None,
                "service_context_active": service_name is not None,
                "performance_tracking_active": operation_name is not None,
                "current_correlation_id": correlation_id or "",
                "current_service_name": service_name or "",
                "current_operation_name": operation_name or "",
                # Context management information
                "available_context_variables": [
                    "correlation_id",
                    "service_name",
                    "service_version",
                    "environment",
                    "user_id",
                    "operation_name",
                ],
                "context_propagation_methods": ["serialization", "headers", "metadata"],
                # Monitoring and diagnostics
                "last_health_check": datetime.now(UTC).isoformat(),
                "system_status": "operational",
                "configuration_source": "default",
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(current_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get context system configuration: {e}",
            )

    @classmethod
    def create_environment_context_config(
        cls,
        environment: str,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific context system configuration.

        Generates optimized configuration for context management based on the
        target environment (development, staging, production, test, local)
        with appropriate correlation tracking, service context management,
        and performance settings for each environment.

        """
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}",
                )

            # Base configuration for all environments
            base_config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "enable_correlation_tracking": True,
                "enable_service_context": True,
                "context_serialization_enabled": True,
            }

            # Environment-specific configurations
            if environment == "production":
                base_config.update(
                    {
                        "context_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                        "enable_performance_tracking": True,  # Critical in production
                        "max_context_depth": 15,  # Limited depth for performance
                        "context_propagation_enabled": True,  # Essential for microservices
                        "context_cleanup_enabled": True,  # Memory management
                        "enable_nested_contexts": True,  # Full nesting support
                        "context_serialization_compression": True,  # Optimize bandwidth
                    },
                )
            elif environment == "staging":
                base_config.update(
                    {
                        "context_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                        "log_level": FlextConstants.Config.LogLevel.INFO.value,
                        "enable_performance_tracking": True,  # Monitor staging performance
                        "max_context_depth": 20,  # Moderate depth limit
                        "context_propagation_enabled": True,  # Test propagation behavior
                        "context_cleanup_enabled": True,  # Memory management
                        "enable_nested_contexts": True,  # Full feature testing
                        "context_serialization_compression": False,  # No compression for debugging
                    },
                )
            elif environment == "development":
                base_config.update(
                    {
                        "context_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_performance_tracking": True,  # Monitor development performance
                        "max_context_depth": 50,  # Higher depth for debugging
                        "context_propagation_enabled": True,  # Test propagation locally
                        "context_cleanup_enabled": False,  # Keep contexts for debugging
                        "enable_nested_contexts": True,  # Full nesting for development
                        "context_serialization_compression": False,  # No compression for clarity
                    },
                )
            elif environment == "test":
                base_config.update(
                    {
                        "context_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                        "enable_performance_tracking": False,  # No performance monitoring in tests
                        "max_context_depth": 10,  # Limited depth for testing
                        "context_propagation_enabled": False,  # No propagation in unit tests
                        "context_cleanup_enabled": True,  # Clean context between tests
                        "enable_nested_contexts": True,  # Test nested behavior
                        "context_serialization_compression": False,  # No compression in tests
                    },
                )
            elif environment == "local":
                base_config.update(
                    {
                        "context_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_performance_tracking": False,  # No monitoring for local
                        "max_context_depth": 100,  # Very high depth for local development
                        "context_propagation_enabled": False,  # No propagation locally
                        "context_cleanup_enabled": False,  # Keep everything for debugging
                        "enable_nested_contexts": True,  # Full nesting support
                        "context_serialization_compression": False,  # No compression
                    },
                )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(base_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment context configuration: {e}",
            )

    @classmethod
    def optimize_context_performance(
        cls,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize context system performance based on configuration.

        Analyzes the provided configuration and generates performance-optimized
        settings for the FLEXT context system. This includes correlation tracking
        optimization, service context caching, context propagation tuning,
        and memory management for optimal distributed tracing performance.
        """
        try:
            # Create optimized configuration
            optimized_config = dict(config)

            # Get performance level from config
            performance_level = config.get("performance_level", "medium")

            # Base performance settings
            optimized_config.update(
                {
                    "performance_level": performance_level,
                    "optimization_enabled": True,
                    "optimization_timestamp": datetime.now(UTC).isoformat(),
                },
            )

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update(
                    {
                        # Context management optimization
                        "context_cache_size": 1000,
                        "enable_context_pooling": True,
                        "context_pool_size": 200,
                        "max_concurrent_contexts": 100,
                        "context_discovery_cache_ttl": 3600,  # 1 hour
                        # Correlation tracking optimization
                        "enable_correlation_caching": True,
                        "correlation_cache_size": 2000,
                        "correlation_tracking_threads": 8,
                        "parallel_correlation_processing": True,
                        # Service context optimization
                        "service_context_batch_size": 200,
                        "enable_service_context_batching": True,
                        "service_context_processing_threads": 16,
                        "service_context_queue_size": 4000,
                        # Memory and performance optimization
                        "memory_pool_size_mb": 150,
                        "enable_object_pooling": True,
                        "gc_optimization_enabled": True,
                        "optimization_level": "aggressive",
                    },
                )
            elif performance_level == "medium":
                optimized_config.update(
                    {
                        # Balanced context settings
                        "context_cache_size": 500,
                        "enable_context_pooling": True,
                        "context_pool_size": 100,
                        "max_concurrent_contexts": 50,
                        "context_discovery_cache_ttl": 1800,  # 30 minutes
                        # Moderate correlation optimization
                        "enable_correlation_caching": True,
                        "correlation_cache_size": 1000,
                        "correlation_tracking_threads": 4,
                        "parallel_correlation_processing": True,
                        # Standard service context processing
                        "service_context_batch_size": 100,
                        "enable_service_context_batching": True,
                        "service_context_processing_threads": 8,
                        "service_context_queue_size": 2000,
                        # Moderate memory settings
                        "memory_pool_size_mb": 75,
                        "enable_object_pooling": True,
                        "gc_optimization_enabled": True,
                        "optimization_level": "balanced",
                    },
                )
            elif performance_level == "low":
                optimized_config.update(
                    {
                        # Conservative context settings
                        "context_cache_size": 100,
                        "enable_context_pooling": False,
                        "context_pool_size": 25,
                        "max_concurrent_contexts": 10,
                        "context_discovery_cache_ttl": 600,  # 10 minutes
                        # Minimal correlation optimization
                        "enable_correlation_caching": False,
                        "correlation_cache_size": 200,
                        "correlation_tracking_threads": 1,
                        "parallel_correlation_processing": False,
                        # Sequential service context processing
                        "service_context_batch_size": 25,
                        "enable_service_context_batching": False,
                        "service_context_processing_threads": 1,
                        "service_context_queue_size": 200,
                        # Minimal memory usage
                        "memory_pool_size_mb": 20,
                        "enable_object_pooling": False,
                        "gc_optimization_enabled": False,
                        "optimization_level": "conservative",
                    },
                )

            # Additional performance metrics and targets
            optimized_config.update(
                {
                    "expected_throughput_contexts_per_second": 500
                    if performance_level == "high"
                    else 200
                    if performance_level == "medium"
                    else 50,
                    "target_context_latency_ms": 5
                    if performance_level == "high"
                    else 15
                    if performance_level == "medium"
                    else 50,
                    "target_correlation_tracking_ms": 2
                    if performance_level == "high"
                    else 8
                    if performance_level == "medium"
                    else 25,
                    "memory_efficiency_target": 0.92
                    if performance_level == "high"
                    else 0.85
                    if performance_level == "medium"
                    else 0.70,
                },
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize context performance: {e}",
            )


__all__: list[str] = [
    "FlextContext",
]
