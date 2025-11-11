"""Context management patterns extracted from FlextModels.

This module contains the FlextModelsContext class with all context-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Context instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import Annotated, Generic, cast

import structlog.contextvars
from pydantic import BaseModel, Field, computed_field, field_validator

from flext_core._models.entity import FlextModelsEntity
from flext_core.runtime import FlextRuntime
from flext_core.typings import T
from flext_core.utilities import FlextUtilities


class FlextModelsContext:
    """Context management pattern container class.

    This class acts as a namespace container for context management patterns.
    All nested classes are accessed via FlextModels.Context.* in the main models.py.
    """

    class StructlogProxyToken(FlextModelsEntity.Value):
        """Token for resetting structlog context variables.

        Used by StructlogProxyContextVar to track previous values and enable
        rollback to previous context state. Inherits from Value for immutability
        and validation.

        This is a lightweight immutable value object that stores the necessary
        information to restore a context variable to its previous state.

        Attributes:
            key: The context variable key being tracked
            previous_value: The value before the set operation (None if unset)

        Examples:
            >>> token = FlextModelsContext.StructlogProxyToken(
            ...     key="correlation_id", previous_value="abc-123"
            ... )
            >>> token.key
            'correlation_id'
            >>> token.previous_value
            'abc-123'

        """

        key: Annotated[
            str,
            Field(
                min_length=1,
                pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
                description="Unique key for the context variable (alphanumeric + underscore)",
                examples=["correlation_id", "service_name", "user_id"],
            ),
        ]
        previous_value: Annotated[
            object | None,
            Field(
                default=None,
                description="Previous value before set operation",
            ),
        ] = None

    class StructlogProxyContextVar(Generic[T]):
        """ContextVar-like proxy using structlog as backend (single source of truth).

        ARCHITECTURAL NOTE: This proxy delegates ALL operations to structlog's
        contextvar storage. This ensures FlextContext.Variables and FlextLogger
        use THE SAME underlying storage, eliminating dual storage and sync issues.

        Key Principles:
            - Single Source of Truth: structlog's contextvar dict
            - Zero Synchronization: No dual storage, no sync needed
            - Thread Safety: structlog handles all thread safety
            - Performance: Direct delegation, no overhead

        Usage:
            >>> var = FlextModelsContext.StructlogProxyContextVar[str](
            ...     "correlation_id", default=None
            ... )
            >>> var.set("abc-123")
            >>> var.get()  # Returns "abc-123"

        """

        def __init__(
            self,
            key: str,
            default: T | None = None,
        ) -> None:
            """Initialize proxy context variable.

            Args:
                key: Unique key for this context variable
                default: Default value when not set

            """
            super().__init__()
            self._key = key
            self._default: T | None = default

        def get(self) -> T | None:
            """Get current value from structlog context.

            Returns:
                Current value with type T (from Generic[T] contract), or None.

            Architecture:
                structlog stores dict[str, object] (untyped).
                We cast to T | None to honor the Generic[T] contract.
                Type safety is ensured by FlextContext using typed variables.

            """
            structlog_context = structlog.contextvars.get_contextvars()
            if not structlog_context:
                return self._default
            # Cast from structlog's untyped storage to our Generic[T] contract
            return cast("T | None", structlog_context.get(self._key, self._default))

        def set(self, value: T | None) -> FlextModelsContext.StructlogProxyToken:
            """Set value in structlog context.

            Args:
                value: Value to set in structlog's contextvar (can be None to clear)

            Returns:
                Token for potential reset

            """
            # Get current value before setting
            current_value = self.get()

            if value is not None:
                structlog.contextvars.bind_contextvars(**{self._key: value})
            else:
                # Unbind if setting to None
                structlog.contextvars.unbind_contextvars(self._key)

            # Create token for reset functionality
            return FlextModelsContext.StructlogProxyToken(
                key=self._key, previous_value=current_value
            )

        def reset(self, token: FlextModelsContext.StructlogProxyToken) -> None:
            """Reset to previous value using token.

            Args:
                token: Token from previous set() call

            Note:
                structlog.contextvars doesn't support token-based reset.
                Use unbind_contextvars() or clear_contextvars() instead.

            """
            # Simplified implementation - structlog uses bind/unbind, not tokens
            # In practice, context managers handle cleanup via bind/unbind
            if token.previous_value is None:
                structlog.contextvars.unbind_contextvars(token.key)
            else:
                structlog.contextvars.bind_contextvars(**{
                    token.key: token.previous_value
                })

    class Token(FlextModelsEntity.Value):
        """Token for context variable reset operations.

        Used by FlextContext to track context variable changes and enable
        rollback to previous values.

        This immutable value object stores the state needed to restore a
        context variable to its previous value, enabling proper cleanup
        in context managers and error handlers.

        Attributes:
            key: The context variable key being tracked
            old_value: The value before the set operation (None if unset)

        Examples:
            >>> token = FlextModelsContext.Token(key="user_id", old_value="user-123")
            >>> token.key
            'user_id'
            >>> token.old_value
            'user-123'

        """

        key: Annotated[
            str,
            Field(
                min_length=1,
                description="Unique key for the context variable",
                examples=["user_id", "request_id", "session_id"],
            ),
        ]
        old_value: Annotated[
            object | None,
            Field(
                default=None,
                description="Previous value before set operation",
            ),
        ]

    class ContextData(FlextModelsEntity.Value):
        """Lightweight container for initializing context state.

        Used by FlextContext initialization to provide initial data and metadata.

        This immutable value object encapsulates the initial state for a
        FlextContext instance, separating actual context data from metadata
        about the context itself.

        Attributes:
            data: Initial context data (key-value pairs)
            metadata: Context metadata (creation time, source, etc.)

        Examples:
            >>> context_data = FlextModelsContext.ContextData(
            ...     data={"user_id": "123", "correlation_id": "abc-xyz"},
            ...     metadata={"source": "api", "created_at": "2025-01-01T00:00:00Z"},
            ... )
            >>> context_data.data["user_id"]
            '123'
            >>> context_data.metadata["source"]
            'api'

        """

        data: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                description="Initial context data as key-value pairs",
            ),
        ] = Field(default_factory=dict)
        metadata: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = Field(default_factory=dict)

        @field_validator("data", "metadata", mode="before")
        @classmethod
        def validate_dict_serializable(cls, v: object) -> dict[str, object]:
            """Validate that dict[str, object] values are JSON-serializable.

            Uses mode='before' to validate raw input before Pydantic processing.
            Only allows basic JSON-serializable types: str, int, float, bool, list, dict, None.
            """
            if not FlextRuntime.is_dict_like(v):
                msg = f"Value must be a dictionary, got {type(v).__name__}"
                raise TypeError(msg)

            # Recursively check all values are JSON-serializable
            def check_serializable(obj: object, path: str = "") -> None:
                """Recursively check if object is JSON-serializable."""
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return
                if FlextRuntime.is_dict_like(obj):
                    for key, val in obj.items():
                        if not isinstance(key, str):
                            msg = f"Dictionary keys must be strings at {path}.{key}"
                            raise TypeError(msg)
                        check_serializable(val, f"{path}.{key}")
                elif FlextRuntime.is_list_like(obj):
                    for i, item in enumerate(obj):
                        check_serializable(item, f"{path}[{i}]")
                else:
                    msg = f"Non-JSON-serializable type {type(obj).__name__} at {path}"
                    raise TypeError(msg)

            check_serializable(v)
            return v

    class ContextExport(FlextModelsEntity.Value):
        """Typed snapshot returned by export_snapshot.

        Provides a complete serializable snapshot of context state including
        data, metadata, and statistics.

        This immutable value object represents a complete export of a FlextContext
        instance, suitable for persistence, transmission, or debugging. All fields
        are JSON-serializable for easy cross-service communication.

        Attributes:
            data: All context data from all scopes
            metadata: Context metadata (creation info, source, etc.)
            statistics: Usage statistics (set/get/remove counts, etc.)

        Examples:
            >>> export = FlextModelsContext.ContextExport(
            ...     data={"user_id": "123", "correlation_id": "abc-xyz"},
            ...     metadata={"source": "api", "version": "1.0"},
            ...     statistics={"sets": 5, "gets": 10, "removes": 2},
            ... )
            >>> export.data["user_id"]
            '123'
            >>> export.statistics["sets"]
            5

        """

        data: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                description="All context data from all scopes",
            ),
        ] = Field(default_factory=dict)
        metadata: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                description="Context metadata (creation info, source, version)",
            ),
        ] = Field(default_factory=dict)
        statistics: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                description="Usage statistics (operation counts, timing info)",
            ),
        ] = Field(default_factory=dict)

        @field_validator("data", "metadata", "statistics", mode="before")
        @classmethod
        def validate_dict_serializable(cls, v: object) -> dict[str, object]:
            """Validate that dict[str, object] values are JSON-serializable.

            Uses mode='before' to validate raw input before Pydantic processing.
            Only allows basic JSON-serializable types: str, int, float, bool, list, dict, None.
            """
            if not FlextRuntime.is_dict_like(v):
                msg = f"Value must be a dictionary, got {type(v).__name__}"
                raise TypeError(msg)

            # Recursively check all values are JSON-serializable
            def check_serializable(obj: object, path: str = "") -> None:
                """Recursively check if object is JSON-serializable."""
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return
                if FlextRuntime.is_dict_like(obj):
                    for key, val in obj.items():
                        if not isinstance(key, str):
                            msg = f"Dictionary keys must be strings at {path}.{key}"
                            raise TypeError(msg)
                        check_serializable(val, f"{path}.{key}")
                elif FlextRuntime.is_list_like(obj):
                    for i, item in enumerate(obj):
                        check_serializable(item, f"{path}[{i}]")
                else:
                    msg = f"Non-JSON-serializable type {type(obj).__name__} at {path}"
                    raise TypeError(msg)

            check_serializable(v)
            return v

        @computed_field
        def total_data_items(self) -> int:
            """Compute total number of data items across all scopes."""
            return len(self.data)

        @computed_field
        def has_statistics(self) -> bool:
            """Check if statistics are available."""
            return bool(self.statistics)

    class ContextScopeData(BaseModel):
        """Scope-specific data container for context management.

        Enhanced to support backward compatible dict handling while
        maintaining strong typing for scope data and metadata.

        Attributes:
            scope_name: Name of the scope (e.g., 'request', 'operation')
            scope_type: Type/category of scope
            data: Scope-specific data key-value pairs
            metadata: Metadata associated with this scope

        Examples:
            >>> scope = FlextModelsContext.ContextScopeData(
            ...     scope_name="request",
            ...     scope_type="http",
            ...     data={"method": "POST", "path": "/api/orders"},
            ...     metadata={"trace_id": "trace-123"},
            ... )

        """

        scope_name: Annotated[
            str,
            Field(min_length=1, description="Name of the scope"),
        ] = ""
        scope_type: Annotated[
            str,
            Field(default="", description="Type/category of scope"),
        ] = ""
        data: Annotated[
            dict[str, object],
            Field(default_factory=dict, description="Scope data"),
        ] = Field(default_factory=dict)
        metadata: Annotated[
            dict[str, object],
            Field(default_factory=dict, description="Scope metadata"),
        ] = Field(default_factory=dict)

        @field_validator("data", mode="before")
        @classmethod
        def _validate_data(cls, v: object) -> dict[str, object]:
            """Validate scope data (using FlextUtilities.Generators.ensure_dict)."""
            return FlextUtilities.Generators.ensure_dict(v)

        @field_validator("metadata", mode="before")
        @classmethod
        def _validate_metadata(cls, v: object) -> dict[str, object]:
            """Validate scope metadata (using FlextUtilities.Generators.ensure_dict)."""
            return FlextUtilities.Generators.ensure_dict(v)

    class ContextStatistics(BaseModel):
        """Statistics tracking for context operations and metrics.

        Enhanced to replace dict[str, float] metrics storage across
        handlers.py and loggings.py modules for structured metrics
        tracking and performance monitoring.

        Attributes:
            sets: Number of set operations
            gets: Number of get operations
            removes: Number of remove operations
            clears: Number of clear operations
            operations: Extensible operation/metrics counts dictionary

        Examples:
            >>> stats = FlextModelsContext.ContextStatistics(
            ...     sets=10,
            ...     gets=25,
            ...     operations={
            ...         "response_time_ms": 42.5,
            ...         "items_processed": 100,
            ...     },
            ... )
            >>> stats.sets
            10

        """

        sets: Annotated[
            int,
            Field(default=0, ge=0, description="Number of set operations"),
        ] = 0
        gets: Annotated[
            int,
            Field(default=0, ge=0, description="Number of get operations"),
        ] = 0
        removes: Annotated[
            int,
            Field(default=0, ge=0, description="Number of remove operations"),
        ] = 0
        clears: Annotated[
            int,
            Field(default=0, ge=0, description="Number of clear operations"),
        ] = 0
        operations: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                description="Extensible operation/metrics counts",
            ),
        ] = Field(default_factory=dict)

        @field_validator("operations", mode="before")
        @classmethod
        def _validate_operations(cls, v: object) -> dict[str, object]:
            """Validate operations (using FlextUtilities.Generators.ensure_dict)."""
            return FlextUtilities.Generators.ensure_dict(v)

    class ContextMetadata(BaseModel):
        """Metadata storage for context objects with full tracing support.

        Enhanced to replace dict[str, object] across multiple modules
        including context.py, handlers.py, dispatcher.py, and config.py
        for consistent, strongly-typed metadata handling.

        Attributes:
            user_id: Associated user ID
            correlation_id: Primary correlation ID for distributed tracing
            parent_correlation_id: Parent request's correlation ID (for nested calls)
            request_id: HTTP request identifier
            session_id: User session identifier
            tenant_id: Tenant/Organization ID for multi-tenancy
            handler_mode: Handler execution mode (command/query/event)
            message_type: Type of message being processed
            message_id: Unique identifier for the message
            custom_fields: Additional extensible metadata key-value pairs

        Examples:
            >>> # Tracing context
            >>> meta = FlextModelsContext.ContextMetadata(
            ...     correlation_id="corr-123",
            ...     parent_correlation_id="parent-456",
            ...     user_id="user-789",
            ... )

            >>> # Handler execution context
            >>> meta = FlextModelsContext.ContextMetadata(
            ...     handler_mode="command",
            ...     message_type="PaymentCommand",
            ...     message_id="msg-001",
            ... )

        """

        user_id: Annotated[
            str | None,
            Field(default=None, description="Associated user ID"),
        ] = None
        correlation_id: Annotated[
            str | None,
            Field(
                default=None,
                description="Primary correlation ID for distributed tracing",
            ),
        ] = None
        parent_correlation_id: Annotated[
            str | None,
            Field(
                default=None,
                description="Parent request's correlation ID for nested calls",
            ),
        ] = None
        request_id: Annotated[
            str | None,
            Field(default=None, description="HTTP request identifier"),
        ] = None
        session_id: Annotated[
            str | None,
            Field(default=None, description="User session identifier"),
        ] = None
        tenant_id: Annotated[
            str | None,
            Field(default=None, description="Tenant/Organization ID"),
        ] = None
        handler_mode: Annotated[
            str | None,
            Field(
                default=None,
                description="Handler mode (command/query/event)",
            ),
        ] = None
        message_type: Annotated[
            str | None,
            Field(default=None, description="Type of message being processed"),
        ] = None
        message_id: Annotated[
            str | None,
            Field(default=None, description="Unique message identifier"),
        ] = None
        custom_fields: Annotated[
            dict[str, object],
            Field(
                default_factory=dict,
                description="Extensible custom metadata fields",
            ),
        ] = Field(default_factory=dict)

        @field_validator("custom_fields", mode="before")
        @classmethod
        def _validate_custom_fields(cls, v: object) -> dict[str, object]:
            """Validate custom_fields (using FlextUtilities.Generators.ensure_dict)."""
            return FlextUtilities.Generators.ensure_dict(v)

    class ContextDomainData(BaseModel):
        """Domain-specific context data storage."""

        domain_name: Annotated[
            str | None,
            Field(default=None, description="Domain name/identifier"),
        ] = None
        domain_type: Annotated[
            str | None,
            Field(default=None, description="Type of domain"),
        ] = None
        domain_data: Annotated[
            dict[str, object],
            Field(default_factory=dict, description="Domain-specific data"),
        ] = Field(default_factory=dict)
        domain_metadata: Annotated[
            dict[str, object],
            Field(default_factory=dict, description="Domain metadata"),
        ] = Field(default_factory=dict)


__all__ = ["FlextModelsContext"]
