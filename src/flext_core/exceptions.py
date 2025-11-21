"""Structured exception hierarchy with error codes and correlation tracking.

This module provides FlextExceptions, a namespace class containing structured
exception types with error codes, correlation IDs, and metadata for comprehensive
error handling throughout the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Callable
from typing import ClassVar, cast

from typing import TYPE_CHECKING

import structlog

from flext_core._models.metadata import Metadata
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.runtime import FlextRuntime

if TYPE_CHECKING:
    from flext_core.protocols import FlextProtocols

# CRITICAL: NO import from utilities - causes circular import (exceptions → utilities → _utilities → result → exceptions)
# CRITICAL: NO import from models - causes circular import (exceptions → models → _models → validation → result → exceptions)
# CRITICAL: Use zero-dependency Metadata from _models/metadata.py - breaks circular import
# CRITICAL: Use FlextProtocols.MetadataProtocol for TYPE HINTS (structural typing)
# CRITICAL: Use Metadata class for INSTANTIATION (zero-dependency module, NO lazy imports)
# CRITICAL: Use FlextRuntime for type guards (is_dict_like) - runtime.py is Layer 0.5, safe to import

# Reserved keyword names used across all exception classes
_EXCEPTION_RESERVED_KEYS: frozenset[str] = frozenset({
    "correlation_id",
    "metadata",
    "auto_log",
    "auto_correlation",
    "config",
})


class FlextExceptions:
    """Structured exception hierarchy with error codes and correlation tracking.

    Architecture: Layer 1 (Foundation - Error Hierarchy)
    ===================================================
    Provides a comprehensive collection of structured exception types with
    error codes, correlation IDs, and metadata for comprehensive error handling
    throughout the FLEXT ecosystem. All exceptions integrate with FlextResult
    error handling and structured logging via structlog.

    **Structural Typing and Protocol Compliance**:
    This class satisfies FlextProtocols.Exception through structural typing
    (duck typing) via the following protocol-compliant interface:
    - 13 exception classes with consistent error handling
    - All extend BaseError with structured metadata
    - Error codes from FlextConstants for consistency
    - Correlation ID generation for distributed tracing
    - Automatic structured logging via structlog
    - Exception chaining with cause preservation
    - Flexible metadata for context propagation

    **Exception Hierarchy** (13 types):
    1. **BaseError**: Base class with error codes, correlation IDs, metadata
       - Automatic structured logging on creation (optional)
       - Correlation ID generation for distributed tracing
       - Exception chaining with cause tracking
       - Methods: with_context(), chain_from(), to_dict()

    2. **ValidationError**: Input validation failures
       - Field-specific validation errors
       - Value tracking for invalid data
       - Integrates with FlextUtilities validation

    3. **ConfigurationError**: Configuration-related errors
       - Config key tracking (which config failed)
       - Config source tracking (file, env, etc.)
       - Integration with FlextConfig validation

    4. **ConnectionError**: Network and connection failures
       - Host and port tracking
       - Timeout duration tracking
       - Connection pool exhaustion handling

    5. **TimeoutError**: Operation timeout errors
       - Timeout duration tracking
       - Operation name for context
       - Integration with @timeout decorator

    6. **AuthenticationError**: Authentication failures
       - Authentication method tracking
       - User ID tracking for audit trails
       - Integration with FlextAuth patterns

    7. **AuthorizationError**: Permission and authorization failures
       - User ID tracking
       - Resource being accessed
       - Permission being checked
       - Integration with RBAC patterns

    8. **NotFoundError**: Resource not found errors
       - Resource type tracking
       - Resource ID tracking
       - REST API 404 responses

    9. **ConflictError**: Resource conflict errors
       - Resource type and ID tracking
       - Conflict reason tracking
       - Handles duplicates and concurrent modifications

    10. **RateLimitError**: Rate limiting violations
        - Limit and window tracking
        - Retry-after duration
        - Integration with rate limiting patterns

    11. **CircuitBreakerError**: Circuit breaker open errors
        - Service name tracking
        - Failure count tracking
        - Reset timeout duration
        - Integration with resilience patterns

    12. **TypeError**: Type-related errors
        - Expected type tracking
        - Actual type tracking
        - Validation error handling

    13. **AttributeAccessError**: Attribute access errors
        - Attribute name tracking
        - Access context information
        - Missing/invalid attribute handling

    14. **OperationError**: Operation-specific errors
        - Operation name tracking
        - Failure reason tracking
        - Business logic violations

    **Core Features**:
    1. **Structured Error Codes**: All use FlextConstants.Errors for consistency
    2. **Correlation ID Tracking**: Unique IDs for distributed tracing
    3. **Automatic Logging**: Optional structured logging via structlog
    4. **Exception Chaining**: Preserve cause exceptions with __cause__
    5. **Flexible Metadata**: Arbitrary key-value context storage
    6. **String Representation**: Human-readable with error code and correlation ID
    7. **Dictionary Serialization**: to_dict() for JSON/logging serialization
    8. **Context Enrichment**: with_context() for adding metadata
    9. **Metrics Tracking**: record_exception(), get_metrics() for monitoring
    10. **Factory Methods**: create() and create_error() for dynamic creation

    **Integration Points**:
    - **FlextResult** (Layer 1): Error handling in railway pattern
    - **FlextConstants** (Layer 0): Error codes for categorization
    - **FlextDecorators** (Layer 3): @retry, @timeout use TimeoutError
    - **FlextLogger** (Layer 4): Structured logging via structlog
    - **FlextConfig** (Layer 4): Configuration validation errors
    - **All layers**: Error handling and exception propagation

    **Defensive Programming Patterns**:
    1. Automatic logging uses try/except to prevent logging failures
    2. Exception chaining preserves cause for debugging
    3. Metadata update via dict.update() for atomicity
    4. Correlation ID generated only when explicitly requested
    5. All exception attributes stored as instance variables

    **Structured Logging Integration**:
    - BaseError._log_exception() uses structlog.get_logger()
    - Exception chaining logged with chain_from()
    - All exception metadata included in structured logs

    **Exception Factory Methods**:
    1. **create_error(error_type, message)**: By type name
    2. **create(message, error_code, **kwargs)**: Smart type detection
    3. FlextExceptions as callable: delegates to create()

    **Metrics and Monitoring**:
    - record_exception(exception_type): Track exception counts
    - get_metrics(): Retrieve exception statistics
    - clear_metrics(): Reset exception counters
    - Integration with monitoring and alerting systems

    **Thread Safety**:
    - All exceptions are thread-safe (no mutable shared state)
    - Metadata dictionaries are instance-specific
    - Correlation ID generation uses uuid.uuid4()
    - Metrics dictionary is ClassVar (protected by GIL in CPython)

    **Performance Characteristics**:
    - O(1) exception creation
    - O(n) structured logging where n = metadata size
    - O(1) context binding via dict.update()
    - O(1) metrics recording

    **Usage Patterns**:

    1. Simple validation error:
        >>> if not email:
        ...     raise FlextExceptions.ValidationError("Email required")

    2. Validation error with field information:
        >>> raise FlextExceptions.ValidationError(
        ...     "Invalid email format", field="email", value=user_input
        ... )

    3. Configuration error with source tracking:
        >>> raise FlextExceptions.ConfigurationError(
        ...     "Missing API key", config_key="API_KEY", config_source="environment"
        ... )

    4. Connection error with host/port tracking:
        >>> raise FlextExceptions.ConnectionError(
        ...     "Failed to connect to database",
        ...     host="db.example.com",
        ...     port=5432,
        ...     timeout=30.0,
        ... )

    5. Exception with automatic logging:
        >>> raise FlextExceptions.ValidationError(
        ...     "Invalid user", auto_log=True, auto_correlation=True
        ... )

    6. Exception chaining:
        >>> try:
        ...     risky_operation()
        ... except ValueError as e:
        ...     raise FlextExceptions.OperationError("Operation failed").chain_from(e)

    7. Context enrichment:
        >>> raise FlextExceptions.OperationError("Cannot process order").with_context(
        ...     order_id="ORD-123", customer_id="CUST-456"
        ... )

    8. Using factory methods:
        >>> error = FlextExceptions.create(
        ...     "Validation failed", field="email", value="invalid"
        ... )

    9. Serialization for logging/API responses:
        >>> try:
        ...     validate_data(data)
        ... except FlextExceptions.ValidationError as e:
        ...     error_dict = e.to_dict()
        ...     # Send to API response or log

    10. Exception tracking:
        >>> try:
        ...     risky_operation()
        ... except FlextExceptions.BaseError as e:
        ...     FlextExceptions.record_exception(type(e).__name__)
        >>>
        >>> stats = FlextExceptions.get_metrics()
        >>> print(f"Total exceptions: {stats['total_exceptions']}")

    **Complete Integration Example**:
        >>> from flext_core import FlextExceptions, FlextResult
        >>>
        >>> def process_user(user_data: dict) -> FlextResult[dict]:
        ...     try:
        ...         # Validation
        ...         if not user_data.get("email"):
        ...             raise FlextExceptions.ValidationError(
        ...                 "Email required", field="email"
        ...             ).with_context(user_id=user_data.get("id"))
        ...
        ...         # Processing
        ...         result = save_user(user_data)
        ...         return FlextResult[dict].ok(result)
        ...
        ...     except FlextExceptions.ValidationError as e:
        ...         # Automatic context and logging
        ...         return FlextResult[dict].fail(
        ...             str(e), error_code=e.error_code, metadata=e.metadata
        ...         )
        ...     except Exception as e:
        ...         # Unknown errors as operation errors
        ...         return FlextResult[dict].fail(str(e), error_code="OPERATION_ERROR")
    """

    # =========================================================================
    # HIERARCHICAL EXCEPTION CONFIGURATION SYSTEM (Nested in FlextExceptions)
    # =========================================================================

    class Configuration:
        """Hierarchical exception handling configuration for FlextExceptions.

        Supports 4-level hierarchy for fine-grained exception handling control:
        1. Call Level (highest priority) - Temporary override for specific operation
        2. Container Level - Inherits from library automatically
        3. Library Level - Per-exception-type configuration
        4. Global Level (lowest priority) - Default from FlextConfig

        This nested class maintains the hierarchical system while keeping
        FlextExceptions as the only public class per module.
        """

        # Module-level state for hierarchical exception configuration
        # Global default (lowest priority) - now loaded from FlextConfig
        _global_failure_level: ClassVar[
            FlextConstants.Exceptions.FailureLevel | None
        ] = None

        # Library-level: library_name -> {exception_type: FailureLevel}
        _library_exception_levels: ClassVar[
            dict[str, dict[type[Exception], FlextConstants.Exceptions.FailureLevel]]
        ] = {}

        # Container-level: container_id -> FailureLevel
        _container_exception_levels: ClassVar[
            dict[str, FlextConstants.Exceptions.FailureLevel]
        ] = {}

        # Call-level: thread-local storage for temporary overrides
        _call_level_context: ClassVar[
            contextvars.ContextVar[FlextConstants.Exceptions.FailureLevel | None]
        ] = contextvars.ContextVar("exception_mode", default=None)

        @classmethod
        def set_global_level(
            cls,
            level: FlextConstants.Exceptions.FailureLevel,
        ) -> None:
            """Set global failure level (lowest priority)."""
            cls._global_failure_level = level

        @classmethod
        def _get_global_failure_level(cls) -> FlextConstants.Exceptions.FailureLevel:
            """Get global failure level from FlextConfig (Pydantic 2 Settings handles env vars)."""
            if cls._global_failure_level is not None:
                return cls._global_failure_level

            try:
                # Get from FlextConfig singleton (Pydantic Settings handles environment variables)
                config = FlextConfig.get_global_instance()
                level_str = config.exception_failure_level
                cls._global_failure_level = FlextConstants.Exceptions.FailureLevel(
                    level_str.lower(),
                )
                return cls._global_failure_level
            except (AttributeError, ValueError, TypeError):
                # Use default STRICT level if config unavailable or invalid
                cls._global_failure_level = (
                    FlextConstants.Exceptions.FailureLevel.STRICT
                )
                return cls._global_failure_level

        @classmethod
        def register_library_exception_level(
            cls,
            library_name: str,
            exception_type: type[Exception],
            level: FlextConstants.Exceptions.FailureLevel,
        ) -> None:
            """Register exception handling level for specific exception type in library."""
            if library_name not in cls._library_exception_levels:
                cls._library_exception_levels[library_name] = {}
            cls._library_exception_levels[library_name][exception_type] = level

        @classmethod
        def set_container_level(
            cls,
            container_id: str,
            level: FlextConstants.Exceptions.FailureLevel,
        ) -> None:
            """Set container-level failure level."""
            cls._container_exception_levels[container_id] = level

        @classmethod
        def get_effective_level(
            cls,
            library_name: str | None = None,
            container_id: str | None = None,
            exception_type: type[Exception] | None = None,
        ) -> FlextConstants.Exceptions.FailureLevel:
            """Resolve effective failure level using hierarchy: Call > Container > Library > Global."""
            # 1. Call level (highest priority)
            call_level = cls._call_level_context.get()
            if call_level is not None:
                return call_level

            # 2. Container level
            if container_id and container_id in cls._container_exception_levels:
                return cls._container_exception_levels[container_id]

            # 3. Library level (per exception type)
            if (
                library_name
                and exception_type
                and library_name in cls._library_exception_levels
                and exception_type in cls._library_exception_levels[library_name]
            ):
                return cls._library_exception_levels[library_name][exception_type]

            # 4. Global default (lowest priority) - from FlextConfig (Pydantic Settings)
            return cls._get_global_failure_level()

    class BaseError(Exception):
        """Base exception class for all FLEXT exceptions with structured logging.

        Provides common functionality including error codes, correlation IDs,
        structured metadata, and automatic exception logging via structlog.

        Advanced Features (Phase 1):
        - Automatic structured logging on exception creation
        - Correlation ID generation and propagation
        - Exception chaining with cause tracking
        - Structured metadata with context

        Attributes:
            error_code: Error code for categorization
            correlation_id: Correlation ID for tracking
            metadata: Additional structured metadata
            timestamp: When the error occurred
            auto_log: Whether to automatically log on creation

        """

        def __init__(
            self,
            message: str,
            *,
            config: object | None = None,
            error_code: str | None = FlextConstants.Errors.UNKNOWN_ERROR,
            correlation_id: str | None = None,
            metadata: FlextProtocols.MetadataProtocol | dict[str, object] | None = None,
            auto_log: bool = False,
            auto_correlation: bool = False,
            **extra_kwargs: object,
        ) -> None:
            """Initialize base error with structured logging.

            Config Pattern (Recommended):
                # Note: ExceptionConfig is available via FlextModels.Config.ExceptionConfig
                # but importing FlextModels here would cause circular import
                # Use direct parameters instead or import ExceptionConfig separately
                raise FlextException(
                    "Error occurred",
                    error_code="ERR001",
                    correlation_id="abc-123",
                    auto_log=True
                )

            Direct Parameters:
                raise FlextException(
                    "Error occurred",
                    error_code="ERR001",
                    correlation_id="abc-123",
                    metadata=Metadata(attributes={"user_id": "123"}),
                    auto_log=True
                )

            Args:
                message: Error message
                config: ExceptionConfig instance (Pydantic v2)
                error_code: Optional error code
                correlation_id: Optional correlation ID
                metadata: MetadataProtocol instance (Pydantic model or compatible)
                auto_log: Whether to automatically log
                auto_correlation: Whether to auto-generate ID
                **extra_kwargs: Additional kwargs merged into metadata

            """
            # Extract config values (config takes priority)
            if config is not None:
                error_code = getattr(config, "error_code", error_code)
                correlation_id = getattr(config, "correlation_id", correlation_id)
                metadata = getattr(config, "metadata", metadata)
                auto_log = getattr(config, "auto_log", auto_log)
                auto_correlation = getattr(config, "auto_correlation", auto_correlation)
                config_extra = getattr(config, "extra_kwargs", {})
                if FlextRuntime.is_dict_like(config_extra):
                    extra_kwargs = {**config_extra, **extra_kwargs}

            super().__init__(message)
            self.message = message
            self.error_code = error_code
            self.correlation_id: str | None
            if auto_correlation and not correlation_id:
                # Generate short correlation ID inline (avoid circular import with utilities)
                self.correlation_id = f"exc_{uuid.uuid4().hex[:8]}"
            else:
                # Type narrowing: correlation_id is str | None, but we assign it directly
                # The type checker will accept this as the attribute allows str | None
                self.correlation_id = correlation_id
            # STRICT mode: Convert dict to Metadata, merge extra_kwargs
            # Always use Metadata directly (satisfies MetadataProtocol via structural typing)
            if metadata is None:
                self.metadata: Metadata = Metadata(
                    attributes=extra_kwargs or {},
                )
            elif FlextRuntime.is_dict_like(metadata):
                # Convert dict to Metadata, merge with extra_kwargs
                # Type narrowing: metadata is dict-like, safe to use directly
                if FlextRuntime.is_dict_like(metadata):
                    metadata_dict = (
                        dict(metadata) if not isinstance(metadata, dict) else metadata
                    )
                else:
                    metadata_dict = (
                        dict(metadata.items()) if hasattr(metadata, "items") else {}
                    )
                merged_attrs = (
                    {**metadata_dict, **extra_kwargs} if extra_kwargs else metadata_dict
                )
                self.metadata = Metadata(attributes=merged_attrs)
            elif extra_kwargs:
                # Merge extra_kwargs into Metadata.attributes
                # Type narrowing: metadata is Metadata (not dict, not None)
                metadata_obj = cast("Metadata", metadata)
                merged_attrs = {**metadata_obj.attributes, **extra_kwargs}
                self.metadata = Metadata(attributes=merged_attrs)
            else:
                # Type narrowing: metadata is already Metadata (satisfies MetadataProtocol structurally)
                # Metadata satisfies MetadataProtocol via structural typing - use cast for mypy
                self.metadata = cast("Metadata", metadata)
            self.timestamp = time.time()
            self.auto_log = auto_log

            # Automatic structured logging
            if auto_log:
                self._log_exception()

        def _log_exception(self) -> None:
            """Log exception with structured context."""
            try:
                logger = structlog.get_logger()
                logger.error(
                    "exception_raised",
                    event_type="exception",
                    error_type=self.__class__.__name__,
                    error_code=self.error_code,
                    message=self.message,
                    correlation_id=self.correlation_id,
                    timestamp=self.timestamp,
                    metadata=self.metadata,
                )
            except Exception as e:
                # Don't fail if logging fails, but log the error using standard logging

                logging.getLogger(__name__).debug(
                    "Logging failed in exception handler: %s",
                    e,
                )

        def __str__(self) -> str:
            """String representation with error code and correlation ID."""
            parts: list[str] = []
            if self.error_code:
                parts.append(f"[{self.error_code}]")
            if self.correlation_id:
                parts.append(f"(ID: {self.correlation_id})")
            parts.append(self.message)
            return " ".join(parts)

        def to_dict(self) -> dict[str, object]:
            """Convert exception to dictionary representation."""
            return {
                "error_type": self.__class__.__name__,
                "message": self.message,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
                "metadata": self.metadata.attributes,  # Return attributes dict only (user-provided metadata)
            }

        def with_context(self, **context: object) -> FlextExceptions.BaseError:
            """Add additional context to exception metadata.

            Args:
                **context: Context key-value pairs to add

            Returns:
                Self for chaining

            Example:
                >>> raise FlextExceptions.ValidationError("Invalid").with_context(
                ...     field="email", user_id=123
                ... )

            """
            # STRICT mode: metadata is Pydantic model, create new instance with merged attributes
            existing_attrs = self.metadata.attributes
            new_attrs = {**existing_attrs, **context}
            self.metadata = Metadata(attributes=new_attrs)
            return self

        def chain_from(self, cause: Exception) -> FlextExceptions.BaseError:
            """Chain this exception from a cause exception.

            Args:
                cause: Original exception that caused this

            Returns:
                Self for chaining

            Example:
                >>> try:
                ...     dangerous_operation()
                ... except ValueError as e:
                ...     raise FlextExceptions.ValidationError(
                ...         "Validation failed"
                ...     ).chain_from(e)

            """
            self.__cause__ = cause
            parent_correlation_id = getattr(cause, "correlation_id", None)
            if parent_correlation_id is not None:
                # STRICT mode: metadata is Pydantic model, update attributes
                existing_attrs = self.metadata.attributes
                new_attrs = {
                    **existing_attrs,
                    "parent_correlation_id": parent_correlation_id,
                }
                self.metadata = Metadata(attributes=new_attrs)

            # Log exception chaining
            try:
                logger = structlog.get_logger()
                logger.warning(
                    "exception_chained",
                    event_type="exception_chain",
                    child_error=self.__class__.__name__,
                    parent_error=cause.__class__.__name__,
                    correlation_id=self.correlation_id,
                )
            except Exception as e:
                # Don't fail if logging fails, but log the error using standard logging
                import logging

                logging.getLogger(__name__).debug(
                    "Logging failed in exception handler: %s",
                    e,
                )

            return self

    class ValidationError(BaseError):
        """Exception raised for validation failures.

        Used when input data fails validation rules, including field-level
        validation errors with specific field information.

        Usage:
            raise FlextExceptions.ValidationError(
                "Email required",
                field="email",
                value=user_input
            )
        """

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: object | None = None,
            error_code: str = FlextConstants.Errors.VALIDATION_ERROR,
            **kwargs: object,
        ) -> None:
            """Initialize validation error.

            Args:
                message: Validation error message
                field: Optional field name that failed validation
                value: Optional invalid value
                error_code: Error code (defaults to VALIDATION_ERROR)
                **kwargs: correlation_id, metadata, config, auto_log, auto_correlation, etc.

            """
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={"field": field, "value": value},
            )
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config,
                **extra_kwargs,
            )
            self.field = field
            self.value = value

    class ConfigurationError(BaseError):
        """Exception raised for configuration-related errors.

        Used when configuration loading, parsing, or validation fails,
        including missing required configuration or invalid values.
        """

        def __init__(
            self,
            message: str,
            *,
            config_key: str | None = None,
            config_source: str | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            **kwargs: object,
        ) -> None:
            """Initialize configuration error.

            Args:
                message: Configuration error message
                config_key: Optional configuration key that caused the error
                config_source: Optional configuration source (file, env, etc.)
                error_code: Optional error code override
                **kwargs: correlation_id, metadata, config, auto_log, auto_correlation, etc.

            """
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "config_key": config_key,
                    "config_source": config_source,
                },
            )
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config,
                **extra_kwargs,
            )
            self.config_key = config_key
            self.config_source = config_source

    class ConnectionError(BaseError):
        """Exception raised for connection-related failures.

        Used when network connections, database connections, or service
        connections fail, including timeouts and unreachable endpoints.
        """

        def __init__(
            self,
            message: str,
            *,
            host: str | None = None,
            port: int | None = None,
            timeout: float | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            **kwargs: object,
        ) -> None:
            """Initialize connection error.

            Args:
                message: Connection error message
                host: Optional host that failed to connect
                port: Optional port that failed to connect
                timeout: Optional timeout value
                error_code: Optional error code override
                **kwargs: correlation_id, metadata, config, auto_log, auto_correlation, etc.

            """
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={"host": host, "port": port, "timeout": timeout},
            )
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config,
                **extra_kwargs,
            )
            self.host = host
            self.port = port
            self.timeout = timeout

    class TimeoutError(BaseError):
        """Exception raised for operation timeouts.

        Used when operations exceed configured timeout limits,
        including network timeouts, processing timeouts, and I/O timeouts.
        """

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            operation: str | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            **kwargs: object,
        ) -> None:
            """Initialize timeout error.

            Args:
                message: Timeout error message
                timeout_seconds: Optional timeout duration in seconds
                operation: Optional operation that timed out
                error_code: Optional error code override
                **kwargs: correlation_id, metadata, config, auto_log, auto_correlation, etc.

            """
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "timeout_seconds": timeout_seconds,
                    "operation": operation,
                },
            )
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config,
                **extra_kwargs,
            )
            self.timeout_seconds = timeout_seconds
            self.operation = operation

    class AuthenticationError(BaseError):
        """Exception raised for authentication failures.

        Used when authentication credentials are invalid, expired,
        or authentication services are unavailable.
        """

        def __init__(
            self,
            message: str,
            *,
            auth_method: str | None = None,
            user_id: str | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            **kwargs: object,
        ) -> None:
            """Initialize authentication error.

            Args:
                message: Authentication error message
                auth_method: Optional authentication method used
                user_id: Optional user ID that failed authentication
                error_code: Optional error code override
                **kwargs: correlation_id, metadata, config, auto_log, auto_correlation, etc.

            """
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={"auth_method": auth_method, "user_id": user_id},
            )
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config,
                **extra_kwargs,
            )
            self.auth_method = auth_method
            self.user_id = user_id

    class AuthorizationError(BaseError):
        """Exception raised for authorization failures.

        Used when authenticated users lack permissions for requested operations,
        including role-based and resource-based authorization failures.
        """

        def __init__(
            self,
            message: str,
            *,
            user_id: str | None = None,
            resource: str | None = None,
            permission: str | None = None,
            error_code: str = FlextConstants.Errors.AUTHORIZATION_ERROR,
            **kwargs: object,
        ) -> None:
            """Initialize authorization error.

            Args:
                message: Authorization error message
                user_id: Optional user ID that was denied access
                resource: Optional resource being accessed
                permission: Optional permission being checked
                error_code: Optional error code override
                **kwargs: correlation_id, metadata, config, auto_log, auto_correlation, etc.

            """
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "user_id": user_id,
                    "resource": resource,
                    "permission": permission,
                },
            )
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config,
                **extra_kwargs,
            )
            self.user_id = user_id
            self.resource = resource
            self.permission = permission

    class NotFoundError(BaseError):
        """Exception raised when requested resources are not found.

        Used when database records, files, endpoints, or other resources
        cannot be located or do not exist.
        """

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            **kwargs: object,
        ) -> None:
            """Initialize not found error.

            Args:
                message: Not found error message
                resource_type: Optional type of resource not found
                resource_id: Optional ID of resource not found
                error_code: Optional error code override
                **kwargs: correlation_id, metadata, config, auto_log, auto_correlation, etc.

            """
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                },
            )
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config,
                **extra_kwargs,
            )
            self.resource_type = resource_type
            self.resource_id = resource_id

    class ConflictError(BaseError):
        """Exception raised for resource conflicts.

        Used when operations conflict with existing state, including
        concurrent modifications, duplicate resources, and constraint violations.
        """

        def __init__(
            self,
            message: str,
            *,
            config: object | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            resource_type: str | None = None,
            resource_id: str | None = None,
            conflict_reason: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize conflict error.

            Args:
                message: Conflict error message
                config: ExceptionConfig instance (Pydantic v2)
                error_code: Optional error code override
                resource_type: Optional type of resource in conflict
                resource_id: Optional ID of resource in conflict
                conflict_reason: Optional reason for the conflict
                **kwargs: Additional context stored in metadata

            """
            if config is not None:
                kwargs["config"] = config
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config_obj,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "conflict_reason": conflict_reason,
                },
            )
            super().__init__(
                message,
                error_code=error_code,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config_obj,
                **extra_kwargs,
            )
            self.resource_type = resource_type
            self.resource_id = resource_id
            self.conflict_reason = conflict_reason

    class RateLimitError(BaseError):
        """Exception raised when rate limits are exceeded.

        Used when operations exceed configured rate limits,
        including API rate limits and resource usage limits.
        """

        def __init__(
            self,
            message: str,
            *,
            config: object | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            limit: int | None = None,
            window_seconds: int | None = None,
            retry_after: int | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize rate limit error.

            Args:
                message: Rate limit error message
                config: ExceptionConfig instance (Pydantic v2)
                error_code: Optional error code override
                limit: Optional rate limit that was exceeded
                window_seconds: Optional time window for rate limiting
                retry_after: Optional seconds to wait before retrying
                **kwargs: Additional context stored in metadata

            """
            if config is not None:
                kwargs["config"] = config
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config_obj,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "limit": limit,
                    "window_seconds": window_seconds,
                    "retry_after": retry_after,
                },
            )
            super().__init__(
                message,
                error_code=error_code or FlextConstants.Errors.OPERATION_ERROR,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config_obj,
                **extra_kwargs,
            )
            self.limit = limit
            self.window_seconds = window_seconds
            self.retry_after = retry_after

    class CircuitBreakerError(BaseError):
        """Exception raised when circuit breaker is open.

        Used when circuit breakers are open due to repeated failures,
        preventing further operations until the circuit resets.
        """

        def __init__(
            self,
            message: str,
            *,
            config: object | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            service_name: str | None = None,
            failure_count: int | None = None,
            reset_timeout: int | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize circuit breaker error.

            Args:
                message: Circuit breaker error message
                config: ExceptionConfig instance (Pydantic v2)
                error_code: Optional error code override
                service_name: Optional name of service with open circuit
                failure_count: Optional number of failures that opened circuit
                reset_timeout: Optional seconds until circuit resets
                **kwargs: Additional context stored in metadata

            """
            if config is not None:
                kwargs["config"] = config
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config_obj,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "service_name": service_name,
                    "failure_count": failure_count,
                    "reset_timeout": reset_timeout,
                },
            )
            super().__init__(
                message,
                error_code=error_code or FlextConstants.Errors.EXTERNAL_SERVICE_ERROR,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config_obj,
                **extra_kwargs,
            )
            self.service_name = service_name
            self.failure_count = failure_count
            self.reset_timeout = reset_timeout

    class TypeError(BaseError):
        """Exception raised for type-related errors.

        Used when type validation, conversion, or type mismatches occur,
        including invalid type arguments and type constraint violations.
        """

        def __init__(
            self,
            message: str,
            *,
            config: object | None = None,
            error_code: str = FlextConstants.Errors.TYPE_ERROR,
            expected_type: str | None = None,
            actual_type: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize type error.

            Args:
                message: Type error message
                config: ExceptionConfig instance (Pydantic v2)
                error_code: Optional error code override
                expected_type: Optional expected type name
                actual_type: Optional actual type name
                **kwargs: Additional context stored in metadata

            """
            if config is not None:
                kwargs["config"] = config
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config_obj,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "expected_type": expected_type,
                    "actual_type": actual_type,
                },
            )
            super().__init__(
                message,
                error_code=error_code or FlextConstants.Errors.TYPE_ERROR,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config_obj,
                **extra_kwargs,
            )
            self.expected_type = expected_type
            self.actual_type = actual_type

    class OperationError(BaseError):
        """Exception raised for operation-related errors.

        Used when operations fail due to business logic, invalid states,
        or operation-specific constraints.
        """

        def __init__(
            self,
            message: str,
            *,
            config: object | None = None,
            error_code: str = FlextConstants.Errors.OPERATION_ERROR,
            operation: str | None = None,
            reason: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize operation error.

            Args:
                message: Operation error message
                config: ExceptionConfig instance (Pydantic v2)
                error_code: Optional error code override
                operation: Optional operation name that failed
                reason: Optional reason for the failure
                **kwargs: Additional context stored in metadata

            """
            if config is not None:
                kwargs["config"] = config
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config_obj,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={"operation": operation, "reason": reason},
            )
            super().__init__(
                message,
                error_code=error_code or FlextConstants.Errors.OPERATION_ERROR,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config_obj,
                **extra_kwargs,
            )
            self.operation = operation
            self.reason = reason

    # =========================================================================
    # UTILITY METHODS - Convenience functions for error creation
    # =========================================================================

    @staticmethod
    def prepare_exception_kwargs(
        kwargs: dict[str, object],
        specific_params: dict[str, object] | None = None,
    ) -> tuple[
        str | None,
        FlextProtocols.MetadataProtocol | dict[str, object] | None,
        bool,
        bool,
        object | None,
        dict[str, object],
    ]:
        """Prepare kwargs for exception initialization (DRY helper).

        Extracts common parameters and filters kwargs for BaseError.__init__().
        Eliminates 30-40 lines of duplicate code from each exception class.

        Args:
            kwargs: Raw kwargs from exception __init__
            specific_params: Dict of specific parameters to add to metadata

        Returns:
            Tuple of (correlation_id, metadata, auto_log, auto_correlation,
                      config, extra_kwargs)

        """
        # Add specific params to kwargs for metadata
        if specific_params:
            for key, value in specific_params.items():
                if value is not None:
                    kwargs.setdefault(key, value)

        # Extract common parameters with proper type casting
        correlation_id = cast("str | None", kwargs.get("correlation_id"))
        # metadata removed - caller must create Metadata if needed
        auto_log = bool(kwargs.get("auto_log"))
        auto_correlation = bool(kwargs.get("auto_correlation"))
        config = kwargs.get("config")

        # Filter out reserved keys (keep metadata in kwargs if passed)
        extra_kwargs = {
            k: v for k, v in kwargs.items() if k not in _EXCEPTION_RESERVED_KEYS
        }

        # Extract metadata from kwargs if present (must be MetadataProtocol)
        metadata = cast(
            "FlextProtocols.MetadataProtocol | dict[str, object] | None",
            kwargs.get("metadata"),
        )

        return (
            correlation_id,
            metadata,
            auto_log,
            auto_correlation,
            config,
            extra_kwargs,
        )

    @staticmethod
    def extract_common_kwargs(
        kwargs: dict[str, object],
    ) -> tuple[str | None, FlextProtocols.MetadataProtocol | dict[str, object] | None]:
        """Extract correlation_id and metadata from kwargs.

        Used by exception factory methods (e.g., create()) to extract
        common parameters before passing to __init__().

        Args:
            kwargs: Raw kwargs containing correlation_id and/or metadata

        Returns:
            Tuple of (correlation_id, metadata)

        """
        correlation_id = cast("str | None", kwargs.get("correlation_id"))
        metadata = cast(
            "FlextProtocols.MetadataProtocol | dict[str, object] | None",
            kwargs.get("metadata"),
        )
        return correlation_id, metadata

    @staticmethod
    def create_error(
        error_type: str,
        message: str,
    ) -> FlextExceptions.BaseError:
        """Create an error instance by type name.

        Args:
            error_type: Name of the error class (e.g., 'ValidationError')
            message: Error message

        Returns:
            Error instance

        Raises:
            ValueError: If error type is not recognized

        """
        error_classes: dict[str, type[FlextExceptions.BaseError]] = {
            "ValidationError": FlextExceptions.ValidationError,
            "ConfigurationError": FlextExceptions.ConfigurationError,
            "ConnectionError": FlextExceptions.ConnectionError,
            "TimeoutError": FlextExceptions.TimeoutError,
            "AuthenticationError": FlextExceptions.AuthenticationError,
            "AuthorizationError": FlextExceptions.AuthorizationError,
            "NotFoundError": FlextExceptions.NotFoundError,
            "ConflictError": FlextExceptions.ConflictError,
            "RateLimitError": FlextExceptions.RateLimitError,
            "CircuitBreakerError": FlextExceptions.CircuitBreakerError,
            "TypeError": FlextExceptions.TypeError,
            "OperationError": FlextExceptions.OperationError,
            "AttributeError": FlextExceptions.AttributeAccessError,
        }

        error_class = error_classes.get(error_type)
        if not error_class:
            msg = f"Unknown error type: {error_type}"
            raise ValueError(msg)

        return error_class(message)

    @staticmethod
    def _create_validation_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.ValidationError:
        """Create ValidationError from kwargs."""
        return FlextExceptions.ValidationError(
            message,
            error_code=error_code or FlextConstants.Errors.VALIDATION_ERROR,
            field=cast("str | None", kwargs.get("field")),
            value=kwargs.get("value"),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _create_configuration_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.ConfigurationError:
        """Create ConfigurationError from kwargs."""
        return FlextExceptions.ConfigurationError(
            message,
            error_code=error_code or FlextConstants.Errors.CONFIGURATION_ERROR,
            config_key=cast("str | None", kwargs.get("config_key")),
            config_source=cast("str | None", kwargs.get("config_source")),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _create_operation_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.OperationError:
        """Create OperationError from kwargs."""
        return FlextExceptions.OperationError(
            message,
            error_code=error_code or FlextConstants.Errors.OPERATION_ERROR,
            operation=cast("str | None", kwargs.get("operation")),
            reason=cast("str | None", kwargs.get("reason")),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _create_connection_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.ConnectionError:
        """Create ConnectionError from kwargs."""
        return FlextExceptions.ConnectionError(
            message,
            error_code=error_code or FlextConstants.Errors.CONNECTION_ERROR,
            host=cast("str | None", kwargs.get("host")),
            port=cast("int | None", kwargs.get("port")),
            timeout=cast("float | None", kwargs.get("timeout")),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _create_timeout_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.TimeoutError:
        """Create TimeoutError from kwargs."""
        return FlextExceptions.TimeoutError(
            message,
            error_code=error_code or FlextConstants.Errors.TIMEOUT_ERROR,
            timeout_seconds=cast("float | None", kwargs.get("timeout_seconds")),
            operation=cast("str | None", kwargs.get("operation")),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _create_authorization_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.AuthorizationError:
        """Create AuthorizationError from kwargs."""
        final_error_code: str = error_code or FlextConstants.Errors.AUTHORIZATION_ERROR
        return FlextExceptions.AuthorizationError(
            message,
            error_code=final_error_code,
            user_id=cast("str | None", kwargs.get("user_id")),
            resource=cast("str | None", kwargs.get("resource")),
            permission=cast("str | None", kwargs.get("permission")),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _create_authentication_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.AuthenticationError:
        """Create AuthenticationError from kwargs."""
        final_error_code: str = error_code or FlextConstants.Errors.AUTHENTICATION_ERROR
        return FlextExceptions.AuthenticationError(
            message,
            error_code=final_error_code,
            auth_method=cast("str | None", kwargs.get("auth_method")),
            user_id=cast("str | None", kwargs.get("user_id")),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _create_not_found_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.NotFoundError:
        """Create NotFoundError from kwargs."""
        final_error_code: str = error_code or FlextConstants.Errors.NOT_FOUND_ERROR
        return FlextExceptions.NotFoundError(
            message,
            error_code=final_error_code,
            resource_type=cast("str | None", kwargs.get("resource_type")),
            resource_id=cast("str | None", kwargs.get("resource_id")),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _create_attribute_access_error(
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | None,
    ) -> FlextExceptions.AttributeAccessError:
        """Create AttributeAccessError from kwargs."""
        final_error_code: str = error_code or FlextConstants.Errors.ATTRIBUTE_ERROR
        return FlextExceptions.AttributeAccessError(
            message,
            error_code=final_error_code,
            attribute_name=cast("str | None", kwargs.get("attribute_name")),
            attribute_context=cast(
                "dict[str, object] | None",
                kwargs.get("attribute_context"),
            ),
            correlation_id=correlation_id,
            metadata=metadata,
        )

    @staticmethod
    def _determine_error_type(kwargs: dict[str, object]) -> str | None:
        """Determine error type from kwargs.

        Args:
            kwargs: Keyword arguments to analyze

        Returns:
            str | None: Error type name or None for default

        """
        if "field" in kwargs or "value" in kwargs:
            return "validation"
        if (
            "config_key" in kwargs
            or "config_file" in kwargs
            or "config_source" in kwargs
        ):
            return "configuration"
        if "operation" in kwargs:
            return "operation"
        if "host" in kwargs or "port" in kwargs:
            return "connection"
        if "timeout_seconds" in kwargs:
            return "timeout"
        if "user_id" in kwargs and "permission" in kwargs:
            return "authorization"
        if "auth_method" in kwargs:
            return "authentication"
        if "resource_id" in kwargs:
            return "not_found"
        if "attribute_name" in kwargs:
            return "attribute_access"
        return None

    @staticmethod
    def _get_error_creator(
        error_type: str | None,
    ) -> (
        Callable[
            [
                str,
                str | None,
                dict[str, object],
                str | None,
                FlextProtocols.MetadataProtocol | None,
            ],
            FlextExceptions.BaseError,
        ]
        | None
    ):
        """Get error creator function by type.

        Args:
            error_type: Type of error

        Returns:
            Callable that creates error or None for default

        """
        creators: dict[
            str,
            Callable[
                [
                    str,
                    str | None,
                    dict[str, object],
                    str | None,
                    FlextProtocols.MetadataProtocol | None,
                ],
                FlextExceptions.BaseError,
            ],
        ] = {
            "validation": FlextExceptions._create_validation_error,
            "configuration": FlextExceptions._create_configuration_error,
            "operation": FlextExceptions._create_operation_error,
            "connection": FlextExceptions._create_connection_error,
            "timeout": FlextExceptions._create_timeout_error,
            "authorization": FlextExceptions._create_authorization_error,
            "authentication": FlextExceptions._create_authentication_error,
            "not_found": FlextExceptions._create_not_found_error,
            "attribute_access": FlextExceptions._create_attribute_access_error,
        }
        return creators.get(error_type) if error_type else None

    @staticmethod
    def _create_error_by_type(
        error_type: str | None,
        message: str,
        error_code: str | None,
        kwargs: dict[str, object],
        correlation_id: str | None,
        metadata: FlextProtocols.MetadataProtocol | dict[str, object] | None,
    ) -> FlextExceptions.BaseError:
        """Create error instance by type.

        Args:
            error_type: Type of error to create
            message: Error message
            error_code: Optional error code
            kwargs: Keyword arguments
            correlation_id: Optional correlation ID
            metadata: Optional metadata

        Returns:
            BaseError: Error instance

        """
        creator = FlextExceptions._get_error_creator(error_type)

        # Convert dict to Metadata for type safety
        # Always use Metadata directly (satisfies MetadataProtocol via structural typing)
        normalized_metadata: Metadata | None = None
        if metadata is not None:
            if FlextRuntime.is_dict_like(metadata):
                normalized_metadata = Metadata(attributes=metadata)
            else:
                # Type narrowing: metadata is already Metadata (satisfies MetadataProtocol structurally)
                # Metadata satisfies MetadataProtocol via structural typing - use cast for mypy
                normalized_metadata = cast("Metadata", metadata)

        # Metadata satisfies MetadataProtocol via structural typing
        metadata_protocol: FlextProtocols.MetadataProtocol | None = cast(
            "FlextProtocols.MetadataProtocol | None",
            normalized_metadata,
        )

        if creator:
            return creator(
                message,
                error_code,
                kwargs,
                correlation_id,
                metadata_protocol,
            )
        # Strict: metadata must be Metadata or None
        return FlextExceptions.BaseError(
            message,
            error_code=error_code,
            metadata=metadata_protocol,
            correlation_id=correlation_id,
        )

    @staticmethod
    def create(
        message: str,
        error_code: str | None = None,
        **kwargs: object,
    ) -> FlextExceptions.BaseError:
        """Create an error instance with flexible parameters.

        This method attempts to determine the appropriate error type based on
        the provided keyword arguments and creates the corresponding error instance.

        Args:
            message: Error message
            error_code: Optional error code
            **kwargs: Additional parameters used to determine error type and initialize it

        Returns:
            Error instance of the appropriate type

        """
        correlation_id, metadata = FlextExceptions.extract_common_kwargs(kwargs)
        error_type = FlextExceptions._determine_error_type(kwargs)
        return FlextExceptions._create_error_by_type(
            error_type,
            message,
            error_code,
            kwargs,
            correlation_id,
            metadata,
        )

    # Metrics tracking for exception monitoring
    _exception_counts: ClassVar[dict[str, int]] = {}

    @classmethod
    def record_exception(cls, exception_type: str) -> None:
        """Record an exception occurrence for metrics tracking.

        Args:
            exception_type: Type/name of the exception

        """
        if exception_type not in cls._exception_counts:
            cls._exception_counts[exception_type] = 0
        cls._exception_counts[exception_type] += 1

    @classmethod
    def get_metrics(cls) -> dict[str, object]:
        """Get exception metrics.

        Returns:
            Dictionary containing exception counts and statistics

        """
        total = sum(cls._exception_counts.values(), 0)
        return {
            "total_exceptions": total,
            "exception_counts": cls._exception_counts.copy(),
            "unique_exception_types": len(cls._exception_counts),
        }

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._exception_counts.clear()

    def __call__(
        self,
        message: str,
        error_code: str | None = None,
        **kwargs: object,
    ) -> FlextExceptions.BaseError:
        """Make FlextExceptions callable - delegates to create method.

        Args:
            message: Error message
            error_code: Optional error code
            **kwargs: Additional parameters

        Returns:
            Error instance

        """
        return self.create(message, error_code, **kwargs)

    class AttributeAccessError(BaseError):
        """Exception raised for attribute access errors.

        Used when trying to access attributes that don't exist or are not accessible,
        including missing methods, properties, or invalid attribute names.
        """

        def __init__(
            self,
            message: str,
            *,
            config: object | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            attribute_name: str | None = None,
            attribute_context: dict[str, object] | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize attribute error.

            Args:
                message: Attribute error message
                config: ExceptionConfig instance (Pydantic v2)
                error_code: Optional error code override
                attribute_name: Optional name of the missing/invalid attribute
                attribute_context: Optional context about the attribute access
                **kwargs: Additional context stored in metadata

            """
            if config is not None:
                kwargs["config"] = config
            (
                correlation_id,
                metadata,
                auto_log,
                auto_correlation,
                config_obj,
                extra_kwargs,
            ) = FlextExceptions.prepare_exception_kwargs(
                kwargs,
                specific_params={
                    "attribute_name": attribute_name,
                    "attribute_context": attribute_context,
                },
            )
            super().__init__(
                message,
                error_code=error_code or FlextConstants.Errors.ATTRIBUTE_ERROR,
                correlation_id=correlation_id,
                metadata=metadata,
                auto_log=auto_log,
                auto_correlation=auto_correlation,
                config=config_obj,
                **extra_kwargs,
            )
            self.attribute_name = attribute_name
            self.attribute_context = attribute_context


__all__ = [
    "FlextExceptions",
]
