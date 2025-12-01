"""FlextExceptions - Structured exception hierarchy for FLEXT ecosystem.

This module provides FlextExceptions, a namespace class containing structured
exception types with error codes, correlation IDs, and metadata for comprehensive
error handling throughout the FLEXT ecosystem.

Architecture: Layer 1 (Foundation - Error Hierarchy)
========================================================
Provides a comprehensive collection of structured exception types with
error codes, correlation IDs, and metadata for comprehensive error handling
throughout the FLEXT ecosystem. All exceptions integrate with FlextResult
error handling and structured logging via structlog.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Callable, Mapping
from typing import ClassVar, Self, cast

from flext_core._models.config import FlextModelsConfig
from flext_core._models.metadata import Metadata, MetadataAttributeValue
from flext_core._utilities.type_guards import FlextUtilitiesTypeGuards
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


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
        ...     dangerous_operation()
        ... except ValueError as e:
        ...     raise FlextExceptions.ValidationError("Validation failed").chain_from(e)

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
        >>> from flext_core import FlextResult
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

    **Thread Safety**:
    All constants are immutable (typing.Final). Access is thread-safe as
    there is no mutable shared state in this module.
    """

    class Configuration:
        """Hierarchical exception handling configuration."""

        _global_failure_level: ClassVar[
            FlextConstants.Exceptions.FailureLevel | None
        ] = None
        _library_exception_levels: ClassVar[
            FlextTypes.Types.NestedExceptionLevelMapping
        ] = {}
        _container_exception_levels: ClassVar[
            FlextTypes.Types.ExceptionLevelMapping
        ] = {}
        _call_level_context: ClassVar[
            contextvars.ContextVar[FlextConstants.Exceptions.FailureLevel | None]
        ] = contextvars.ContextVar("exception_mode", default=None)

        @classmethod
        def set_global_level(
            cls,
            level: FlextConstants.Exceptions.FailureLevel,
        ) -> None:
            """Set the global failure level for exception handling."""
            cls._global_failure_level = level

        @classmethod
        def _get_global_failure_level(cls) -> FlextConstants.Exceptions.FailureLevel:
            """Get the global failure level, initializing from config if needed."""
            if cls._global_failure_level is not None:
                return cls._global_failure_level
            try:
                config = FlextConfig.get_global_instance()
                # exception_failure_level is now a FailureLevel StrEnum directly
                cls._global_failure_level = config.exception_failure_level
                return cls._global_failure_level
            except (AttributeError, ValueError, TypeError):
                cls._global_failure_level = (
                    FlextConstants.Exceptions.FailureLevel.STRICT
                )
                return cls._global_failure_level

        @classmethod
        def register_library_exception_level(
            cls,
            library_name: str,
            exception_type: type,
            level: str,
        ) -> None:
            """Register exception level for a specific library and exception type."""
            if library_name not in cls._library_exception_levels:
                cls._library_exception_levels[library_name] = {}
            cls._library_exception_levels[library_name][exception_type] = level

        @classmethod
        def set_container_level(cls, container_id: str, level: str) -> None:
            """Set exception level for a specific container."""
            cls._container_exception_levels[container_id] = level

        @classmethod
        def get_effective_level(
            cls,
            library_name: str | None = None,
            container_id: str | None = None,
            exception_type: type | None = None,
        ) -> FlextConstants.Exceptions.FailureLevel:
            """Get the effective failure level for the given context."""
            call_level = cls._call_level_context.get()
            if call_level is not None:
                return call_level
            if container_id and container_id in cls._container_exception_levels:
                return FlextConstants.Exceptions.FailureLevel(
                    cls._container_exception_levels[container_id],
                )
            if (
                library_name
                and exception_type
                and library_name in cls._library_exception_levels
                and exception_type in cls._library_exception_levels[library_name]
            ):
                return FlextConstants.Exceptions.FailureLevel(
                    cls._library_exception_levels[library_name][exception_type],
                )
            return cls._get_global_failure_level()

    class BaseError(Exception):
        """Base exception class with structured logging."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.UNKNOWN_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize base exception with structured metadata and correlation.

            Tracks correlation IDs and structured metadata for distributed tracing.

            Args:
                message: Human-readable error message
                error_code: Error code from FlextConstants.Errors
                context: Optional dict with correlation_id, metadata, auto_log, auto_correlation
                **extra_kwargs: Additional metadata attributes

            """
            super().__init__(message)
            self.message = message
            self.error_code = error_code

            # Extract context values
            if context is not None:
                correlation_id = context.get("correlation_id")
                metadata = context.get("metadata")
                auto_log = context.get("auto_log", False)
                auto_correlation = context.get("auto_correlation", False)
                # Merge context extra_kwargs with provided extra_kwargs
                context_kwargs = {
                    k: v
                    for k, v in context.items()
                    if k
                    not in (
                        "correlation_id",
                        "metadata",
                        "auto_log",
                        "auto_correlation",
                    )
                }
                merged_kwargs = {**context_kwargs, **extra_kwargs}
            else:
                correlation_id = None
                metadata = None
                auto_log = False
                auto_correlation = False
                merged_kwargs = extra_kwargs

            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if auto_correlation and not correlation_id
                else correlation_id
            )
            if metadata is None:
                # Normalize attributes to MetadataAttributeValue
                normalized_attrs: dict[str, MetadataAttributeValue] = {}
                for k, v in (merged_kwargs or {}).items():
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        normalized_attrs[k] = (
                            FlextUtilitiesTypeGuards.normalize_to_metadata_value(v)
                        )
                self.metadata = Metadata(attributes=normalized_attrs)
            else:
                self.metadata = cast("Metadata", metadata)
            if FlextRuntime.is_dict_like(metadata):
                # Normalize attributes to MetadataAttributeValue
                merged_attrs: dict[str, MetadataAttributeValue] = {}
                merged_dict = (
                    {**cast("dict", metadata), **merged_kwargs}
                    if merged_kwargs
                    else cast("dict", metadata)
                )
                for k, v in merged_dict.items():
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        merged_attrs[k] = (
                            FlextUtilitiesTypeGuards.normalize_to_metadata_value(v)
                        )
                self.metadata = Metadata(attributes=merged_attrs)
            elif merged_kwargs:
                existing_attrs = self.metadata.attributes
                new_attrs = {
                    k: v
                    for k, v in {**existing_attrs, **merged_kwargs}.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                self.metadata = Metadata(attributes=new_attrs)
            self.timestamp = time.time()
            self.auto_log = auto_log
            if auto_log:
                self._log_exception()

        def _log_exception(self) -> None:
            """Log the exception using structured logging."""
            try:
                # Get logger - BindableLogger protocol has methods at runtime
                logger = FlextRuntime.get_logger()
                # Type narrowing: logger from structlog.get_logger() has error/debug/warning methods
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
                FlextRuntime.get_logger(__name__).debug(
                    "Logging failed in exception handler",
                    exc_info=e,
                )

        def __str__(self) -> str:
            """Return string representation with error code and correlation ID."""
            parts = []
            if self.error_code:
                parts.append(f"[{self.error_code}]")
            if self.correlation_id:
                parts.append(f"(ID: {self.correlation_id})")
            parts.append(self.message)
            return " ".join(parts)

        def to_dict(self) -> FlextTypes.Types.ErrorTypeMapping:
            """Convert exception to dictionary representation."""
            # Convert metadata.attributes to compatible type
            metadata_dict: dict[
                str,
                str
                | int
                | float
                | bool
                | list[str | int | float | bool | None]
                | dict[str, str | int | float | bool | None]
                | None,
            ] = {}
            for k, v in self.metadata.attributes.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    metadata_dict[k] = v
            result: FlextTypes.Types.ErrorTypeMapping = {
                "error_type": self.__class__.__name__,
                "message": self.message,
                "error_code": self.error_code,
                "correlation_id": self.correlation_id or None,
                "timestamp": self.timestamp,
                "metadata": metadata_dict,
            }
            return result

        def with_context(self, **context: FlextTypes.GeneralValueType) -> Self:
            """Add context information to the exception metadata."""
            existing_attrs = self.metadata.attributes
            new_attrs = {
                k: v
                for k, v in {**existing_attrs, **context}.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            self.metadata = Metadata(attributes=new_attrs)
            return self

        def chain_from(self, cause: Exception) -> Self:
            """Chain this exception from another exception."""
            self.__cause__ = cause
            if parent_correlation_id := getattr(cause, "correlation_id", None):
                existing_attrs = self.metadata.attributes
                new_attrs = {
                    **existing_attrs,
                    "parent_correlation_id": parent_correlation_id,
                }
                self.metadata = Metadata(attributes=new_attrs)
            try:
                logger = FlextRuntime.get_logger()
                logger.warning(
                    "exception_chained",
                    event_type="exception_chain",
                    child_error=self.__class__.__name__,
                    parent_error=cause.__class__.__name__,
                    correlation_id=self.correlation_id,
                )
            except Exception as e:
                FlextRuntime.get_logger(__name__).debug(
                    "Logging failed in exception handler",
                    exc_info=e,
                )
            return self

    # Specific exception classes with minimal code
    class ValidationError(BaseError):
        """Exception raised for input validation failures."""

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: FlextTypes.GeneralValueType | None = None,
            error_code: str = FlextConstants.Errors.VALIDATION_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize validation error with field and value information."""
            validation_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                validation_context.update(context)
            validation_context.update({
                "field": field,
                "value": value,
                **extra_kwargs,
            })
            super().__init__(
                message,
                error_code=error_code,
                context=validation_context if validation_context else None,
            )
            self.field = field
            self.value = value

    class ConfigurationError(BaseError):
        """Exception raised for configuration-related errors."""

        def __init__(
            self,
            message: str,
            *,
            config_key: str | None = None,
            config_source: str | None = None,
            error_code: str = FlextConstants.Errors.CONFIGURATION_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize configuration error with config context."""
            config_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                config_context.update(context)
            config_context.update({
                "config_key": config_key,
                "config_source": config_source,
                **extra_kwargs,
            })
            super().__init__(
                message,
                error_code=error_code,
                context=config_context if config_context else None,
            )
            self.config_key = config_key
            self.config_source = config_source

    class ConnectionError(BaseError):
        """Exception raised for network and connection failures."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.CONNECTION_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize connection error with network context."""
            conn_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                conn_context.update(context)
            conn_context.update(extra_kwargs)
            super().__init__(
                message,
                error_code=error_code,
                context=conn_context if conn_context else None,
            )
            self.host = conn_context.get("host")
            self.port = conn_context.get("port")
            self.timeout = conn_context.get("timeout")

    class TimeoutError(BaseError):
        """Exception raised for operation timeout errors."""

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            operation: str | None = None,
            error_code: str = FlextConstants.Errors.TIMEOUT_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize timeout error with timeout context."""
            timeout_context: dict[str, MetadataAttributeValue] = {}
            if context is not None:
                for k, v in context.items():
                    timeout_context[k] = (
                        FlextUtilitiesTypeGuards.normalize_to_metadata_value(v)
                    )
            if timeout_seconds is not None:
                timeout_context["timeout_seconds"] = timeout_seconds
            if operation is not None:
                timeout_context["operation"] = operation
            for k, v in extra_kwargs.items():
                timeout_context[k] = (
                    FlextUtilitiesTypeGuards.normalize_to_metadata_value(v)
                )
            super().__init__(
                message,
                error_code=error_code,
                context=timeout_context if timeout_context else None,
            )
            self.timeout_seconds = timeout_seconds
            self.operation = operation

    class AuthenticationError(BaseError):
        """Exception raised for authentication failures."""

        def __init__(
            self,
            message: str,
            *,
            auth_method: str | None = None,
            user_id: str | None = None,
            error_code: str = FlextConstants.Errors.AUTHENTICATION_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize authentication error with auth context."""
            auth_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                auth_context.update(context)
            auth_context.update({
                "auth_method": auth_method,
                "user_id": user_id,
                **extra_kwargs,
            })
            super().__init__(
                message,
                error_code=error_code,
                context=auth_context if auth_context else None,
            )
            self.auth_method = auth_method
            self.user_id = user_id

    class AuthorizationError(BaseError):
        """Exception raised for permission and authorization failures."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.AUTHORIZATION_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize authorization error with permission context."""
            authz_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                authz_context.update(context)
            authz_context.update(extra_kwargs)
            super().__init__(
                message,
                error_code=error_code,
                context=authz_context if authz_context else None,
            )
            self.user_id = authz_context.get("user_id")
            self.resource = authz_context.get("resource")
            self.permission = authz_context.get("permission")

    class NotFoundError(BaseError):
        """Exception raised when a resource is not found."""

        def __init__(
            self,
            message: str,
            *,
            resource_type: str | None = None,
            resource_id: str | None = None,
            error_code: str = FlextConstants.Errors.NOT_FOUND_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize not found error with resource context."""
            # Extract context values if provided
            correlation_id_val: str | None = None
            metadata_val: Metadata | None = None
            auto_log_val = False
            auto_correlation_val = False
            if context is not None:
                corr_id = context.get("correlation_id")
                if isinstance(corr_id, str):
                    correlation_id_val = corr_id
                metadata_obj = context.get("metadata")
                if isinstance(metadata_obj, Metadata):
                    metadata_val = metadata_obj
                auto_log_obj = context.get("auto_log")
                if isinstance(auto_log_obj, bool):
                    auto_log_val = auto_log_obj
                auto_corr_obj = context.get("auto_correlation")
                if isinstance(auto_corr_obj, bool):
                    auto_correlation_val = auto_corr_obj
            # Build extra_kwargs from notfound-specific fields and context
            notfound_kwargs: dict[str, MetadataAttributeValue] = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }
            # Convert extra_kwargs to MetadataAttributeValue
            for k, v in extra_kwargs.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    notfound_kwargs[k] = cast(MetadataAttributeValue, v)
            if context is not None:
                for k, v in context.items():
                    if k not in (
                        "correlation_id",
                        "metadata",
                        "auto_log",
                        "auto_correlation",
                    ):
                        if isinstance(
                            v, (str, int, float, bool, list, dict, type(None))
                        ):
                            notfound_kwargs[k] = cast(MetadataAttributeValue, v)
            # Build context dict
            notfound_context: dict[str, FlextTypes.GeneralValueType] = {
                "auto_log": auto_log_val,
                "auto_correlation": auto_correlation_val,
            }
            if correlation_id_val is not None:
                notfound_context["correlation_id"] = correlation_id_val
            if metadata_val is not None:
                notfound_context["metadata"] = cast(FlextTypes.GeneralValueType, metadata_val)
            # Convert notfound_kwargs to GeneralValueType for context
            for k, v in notfound_kwargs.items():
                notfound_context[k] = cast(FlextTypes.GeneralValueType, v)
            super().__init__(
                message,
                error_code=error_code,
                context=notfound_context if notfound_context else None,
            )
            self.resource_type = resource_type
            self.resource_id = resource_id

    class ConflictError(BaseError):
        """Exception raised for resource conflicts."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.ALREADY_EXISTS,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize conflict error with resource context."""
            conflict_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                conflict_context.update(context)
            conflict_context.update(extra_kwargs)
            super().__init__(
                message,
                error_code=error_code,
                context=conflict_context if conflict_context else None,
            )
            self.resource_type = conflict_context.get("resource_type")
            self.resource_id = conflict_context.get("resource_id")
            self.conflict_reason = conflict_context.get("conflict_reason")

    class RateLimitError(BaseError):
        """Exception raised when rate limits are exceeded."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.OPERATION_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize rate limit error with limit context."""
            rate_limit_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                rate_limit_context.update(context)
            rate_limit_context.update(extra_kwargs)
            super().__init__(
                message,
                error_code=error_code,
                context=rate_limit_context if rate_limit_context else None,
            )
            self.limit = rate_limit_context.get("limit")
            self.window_seconds = rate_limit_context.get("window_seconds")
            self.retry_after = rate_limit_context.get("retry_after")

    class CircuitBreakerError(BaseError):
        """Exception raised when circuit breaker is open."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str = FlextConstants.Errors.EXTERNAL_SERVICE_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize circuit breaker error with service context."""
            cb_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                cb_context.update(context)
            cb_context.update(extra_kwargs)
            super().__init__(
                message,
                error_code=error_code,
                context=cb_context if cb_context else None,
            )
            self.service_name = service_name
            self.failure_count = failure_count
            self.reset_timeout = reset_timeout

    class TypeError(BaseError):
        """Exception raised for type mismatch errors."""

        def __init__(
            self,
            message: str,
            *,
            expected_type: type | None = None,
            actual_type: type | None = None,
            error_code: str = FlextConstants.Errors.TYPE_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize type error with type information."""
            type_context: dict[str, MetadataAttributeValue] = {}
            if context is not None:
                for k, v in context.items():
                    type_context[k] = (
                        FlextUtilitiesTypeGuards.normalize_to_metadata_value(v)
                    )
            type_context["expected_type"] = (
                expected_type.__qualname__ if expected_type else None
            )
            type_context["actual_type"] = (
                actual_type.__qualname__ if actual_type else None
            )
            for k, v in extra_kwargs.items():
                type_context[k] = (
                    FlextUtilitiesTypeGuards.normalize_to_metadata_value(v)
                )
            super().__init__(
                message,
                error_code=error_code,
                context=type_context if type_context else None,
            )
            self.expected_type = expected_type
            self.actual_type = actual_type

    class OperationError(BaseError):
        """Exception raised for general operation failures."""

        def __init__(
            self,
            message: str,
            *,
            operation: str | None = None,
            reason: str | None = None,
            error_code: str = FlextConstants.Errors.OPERATION_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize operation error with operation context."""
            op_context: dict[str, MetadataAttributeValue] = {}
            if context is not None:
                for k, v in context.items():
                    op_context[k] = (
                        FlextUtilitiesTypeGuards.normalize_to_metadata_value(v)
                    )
            if operation is not None:
                op_context["operation"] = operation
            if reason is not None:
                op_context["reason"] = reason
            for k, v in extra_kwargs.items():
                op_context[k] = (
                    FlextUtilitiesTypeGuards.normalize_to_metadata_value(v)
                )
            super().__init__(
                message,
                error_code=error_code,
                context=op_context if op_context else None,
            )
            self.operation = operation
            self.reason = reason

    class AttributeAccessError(BaseError):
        """Exception raised for attribute access errors."""

        def __init__(
            self,
            message: str,
            *,
            attribute_name: str | None = None,
            attribute_context: str | None = None,
            error_code: str = FlextConstants.Errors.ATTRIBUTE_ERROR,
            context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
            **extra_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Initialize attribute access error with attribute context."""
            attr_context: dict[str, FlextTypes.GeneralValueType] = {}
            if context is not None:
                attr_context.update(context)
            attr_context.update({
                "attribute_name": attribute_name,
                "attribute_context": attribute_context,
                **extra_kwargs,
            })
            super().__init__(
                message,
                error_code=error_code,
                context=attr_context or None,
            )
            self.attribute_name = attribute_name
            self.attribute_context = attribute_context

    @staticmethod
    def prepare_exception_kwargs(
        kwargs: dict[str, FlextTypes.GeneralValueType],
        specific_params: dict[str, FlextTypes.GeneralValueType] | None = None,
    ) -> tuple[
        str | None,
        FlextTypes.GeneralValueType,
        bool,
        bool,
        FlextTypes.GeneralValueType,
        dict[str, FlextTypes.GeneralValueType],
    ]:
        """Prepare exception kwargs by extracting common parameters."""
        if specific_params:
            for key, value in specific_params.items():
                if value is not None:
                    kwargs.setdefault(key, value)
        extra_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "correlation_id",
                "metadata",
                "auto_log",
                "auto_correlation",
                "config",
            }
        }
        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id: str | None = (
            str(correlation_id_raw)
            if correlation_id_raw is not None and isinstance(correlation_id_raw, str)
            else None
        )
        return (
            correlation_id,
            kwargs.get("metadata"),
            bool(kwargs.get("auto_log")),
            bool(kwargs.get("auto_correlation")),
            kwargs.get("config"),
            extra_kwargs,
        )

    @staticmethod
    def extract_common_kwargs(
        kwargs: Mapping[str, FlextTypes.GeneralValueType],
    ) -> tuple[
        FlextTypes.GeneralValueType | None,
        FlextTypes.GeneralValueType | None,
    ]:
        """Extract correlation_id and metadata from kwargs.

        Returns raw values without narrowing types, as they could be any FlextTypes.GeneralValueType.
        Callers should validate types as needed.
        """
        return (kwargs.get("correlation_id"), kwargs.get("metadata"))

    @staticmethod
    def create_error(error_type: str, message: str) -> FlextExceptions.BaseError:
        """Create an exception instance based on error type."""
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
    def _determine_error_type(
        kwargs: Mapping[str, FlextTypes.GeneralValueType],
    ) -> str | None:
        """Determine error type from kwargs using pattern matching.

        Returns:
            Error type string or None if no match

        """
        # Error type detection patterns - ordered by specificity
        # Each pattern checks if any key in the set is present
        error_patterns: list[tuple[list[str], str]] = [
            (["field", "value"], "validation"),
            (["config_key", "config_source"], "configuration"),
            (["operation"], "operation"),
            (["host", "port"], "connection"),
            (["timeout_seconds"], "timeout"),
            (["user_id", "permission"], "authorization"),  # Both required
            (["auth_method"], "authentication"),
            (["resource_id"], "not_found"),
            (["attribute_name"], "attribute_access"),
        ]

        for keys, error_type in error_patterns:
            # Special case: authorization requires both keys
            if error_type == "authorization":
                if "user_id" in kwargs and "permission" in kwargs:
                    return error_type
            elif any(key in kwargs for key in keys):
                return error_type

        return None

    @staticmethod
    def _get_error_creator(
        error_type: str,
    ) -> (
        Callable[
            [
                str,
                str | None,
                Mapping[str, MetadataAttributeValue],
                str | None,
                Metadata | None,
            ],
            BaseError,
        ]
        | None
    ):
        creators = {
            "validation": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.ValidationError(
                msg,
                error_code=code or FlextConstants.Errors.VALIDATION_ERROR,
                field=kwargs.get("field"),
                value=kwargs.get("value"),
                correlation_id=cid,
                metadata=meta,
            ),
            "configuration": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.ConfigurationError(
                msg,
                error_code=code or FlextConstants.Errors.CONFIGURATION_ERROR,
                config_key=kwargs.get("config_key"),
                config_source=kwargs.get("config_source"),
                correlation_id=cid,
                metadata=meta,
            ),
            "operation": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.OperationError(
                msg,
                error_code=code or FlextConstants.Errors.OPERATION_ERROR,
                operation=kwargs.get("operation"),
                reason=kwargs.get("reason"),
                correlation_id=cid,
                metadata=meta,
            ),
            "connection": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.ConnectionError(
                msg,
                error_code=code or FlextConstants.Errors.CONNECTION_ERROR,
                host=kwargs.get("host"),
                port=kwargs.get("port"),
                timeout=kwargs.get("timeout"),
                correlation_id=cid,
                metadata=meta,
            ),
            "timeout": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.TimeoutError(
                msg,
                error_code=code or FlextConstants.Errors.TIMEOUT_ERROR,
                timeout_seconds=kwargs.get("timeout_seconds"),
                operation=kwargs.get("operation"),
                correlation_id=cid,
                metadata=meta,
            ),
            "authorization": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.AuthorizationError(
                msg,
                error_code=code or FlextConstants.Errors.AUTHORIZATION_ERROR,
                user_id=kwargs.get("user_id"),
                resource=kwargs.get("resource"),
                permission=kwargs.get("permission"),
                correlation_id=cid,
                metadata=meta,
            ),
            "authentication": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.AuthenticationError(
                msg,
                error_code=code or FlextConstants.Errors.AUTHENTICATION_ERROR,
                auth_method=kwargs.get("auth_method"),
                user_id=kwargs.get("user_id"),
                correlation_id=cid,
                metadata=meta,
            ),
            "not_found": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.NotFoundError(
                msg,
                error_code=code or FlextConstants.Errors.NOT_FOUND_ERROR,
                resource_type=kwargs.get("resource_type"),
                resource_id=kwargs.get("resource_id"),
                correlation_id=cid,
                metadata=meta,
            ),
            "attribute_access": lambda msg,
            code,
            kwargs,
            cid,
            meta: FlextExceptions.AttributeAccessError(
                msg,
                error_code=code or FlextConstants.Errors.ATTRIBUTE_ERROR,
                attribute_name=kwargs.get("attribute_name"),
                attribute_context=kwargs.get("attribute_context"),
                correlation_id=cid,
                metadata=meta,
            ),
        }
        return creators.get(error_type)

    @staticmethod
    def _create_error_by_type(
        error_type: str | None,
        message: str,
        error_code: str | None,
        context: Mapping[str, FlextTypes.GeneralValueType] | None = None,
    ) -> FlextExceptions.BaseError:
        """Create error by type using context dict."""
        # Build context with error_code
        error_context: dict[str, FlextTypes.GeneralValueType] = {}
        if context is not None:
            error_context.update(context)
        if error_code is not None:
            error_context["error_code"] = error_code

        # Create appropriate error class based on type
        error_classes: dict[str, type[BaseError]] = {
            "validation": FlextExceptions.ValidationError,
            "configuration": FlextExceptions.ConfigurationError,
            "connection": FlextExceptions.ConnectionError,
            "timeout": FlextExceptions.TimeoutError,
            "authentication": FlextExceptions.AuthenticationError,
            "authorization": FlextExceptions.AuthorizationError,
            "not_found": FlextExceptions.NotFoundError,
            "operation": FlextExceptions.OperationError,
            "attribute_access": FlextExceptions.AttributeAccessError,
        }

        error_class = error_classes.get(error_type or "") if error_type else None
        if error_class:
            return error_class(
                message,
                error_code=error_code or FlextConstants.Errors.UNKNOWN_ERROR,
                context=error_context if error_context else None,
            )

        return FlextExceptions.BaseError(
            message,
            error_code=error_code or FlextConstants.Errors.UNKNOWN_ERROR,
            context=error_context if error_context else None,
        )

    @staticmethod
    def create(
        message: str,
        error_code: str | None = None,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> FlextExceptions.BaseError:
        """Create an appropriate exception instance based on kwargs context."""
        correlation_id_obj, metadata_obj = FlextExceptions.extract_common_kwargs(kwargs)
        error_type = FlextExceptions._determine_error_type(kwargs)
        # Convert correlation_id_obj to str | None
        correlation_id: str | None = (
            str(correlation_id_obj)
            if correlation_id_obj is not None and isinstance(correlation_id_obj, str)
            else None
        )
        # Convert metadata_obj to dict | None (only pass dicts to _create_error_by_type)
        metadata_dict: dict[str, FlextTypes.GeneralValueType] | None = None
        if isinstance(metadata_obj, dict):
            metadata_dict = metadata_obj
        # Note: if metadata_obj is dict-like but not a dict (Mapping interface),
        # we don't convert it since Metadata expects concrete dict with FlextTypes.GeneralValueType values
        # Build context dict
        error_context: dict[str, FlextTypes.GeneralValueType] = {}
        if correlation_id is not None:
            error_context["correlation_id"] = correlation_id
        if metadata_dict is not None:
            error_context["metadata"] = metadata_dict
        error_context.update(kwargs)

        return FlextExceptions._create_error_by_type(
            error_type,
            message,
            error_code,
            context=error_context or None,
        )

    _exception_counts: ClassVar[FlextTypes.Types.ExceptionMetricsMapping] = {}

    @classmethod
    def record_exception(cls, exception_type: type) -> None:
        """Record an exception occurrence for metrics tracking."""
        if exception_type not in cls._exception_counts:
            cls._exception_counts[exception_type] = 0
        cls._exception_counts[exception_type] += 1

    @classmethod
    def get_metrics(cls) -> FlextTypes.Types.ErrorTypeMapping:
        """Get exception metrics and statistics."""
        total = sum(cls._exception_counts.values(), 0)
        # Serialize exception counts as a single string for compatibility with ErrorTypeMapping
        exception_counts_list = [
            f"{exc_type.__qualname__ if hasattr(exc_type, '__qualname__') else str(exc_type)}:{count}"
            for exc_type, count in cls._exception_counts.items()
        ]
        exception_counts_str = ";".join(exception_counts_list)
        # Build exception_counts dict for test compatibility
        exception_counts_dict: dict[str, int] = {}
        for exc_type, count in cls._exception_counts.items():
            exc_name = (
                exc_type.__qualname__
                if hasattr(exc_type, "__qualname__")
                else str(exc_type)
            )
            exception_counts_dict[exc_name] = count
        # Build result dict matching ErrorTypeMapping type
        # ErrorTypeMapping allows dict[str, int] for exception_counts
        # Cast to satisfy type checker - dict[str, int] is compatible with dict[str, str | int | ...]
        result: FlextTypes.Types.ErrorTypeMapping = {
            "total_exceptions": total,
            "exception_counts": cast(
                "dict[str, str | int | float | bool | list[str | int | float | bool | None] | dict[str, str | int | float | bool | None] | None]",
                exception_counts_dict,
            ),
            "exception_counts_summary": exception_counts_str,  # String format for summary
            "unique_exception_types": len(cls._exception_counts),
        }
        return result

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._exception_counts.clear()

    def __call__(
        self,
        message: str,
        error_code: str | None = None,
        **kwargs: FlextTypes.Types.ExceptionKwargsType,
    ) -> FlextExceptions.BaseError:
        """Create exception by calling the class instance."""
        return self.create(message, error_code, **kwargs)


__all__ = ["FlextExceptions"]
