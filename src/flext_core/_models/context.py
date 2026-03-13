"""Context management patterns extracted from FlextModels.

This module contains the FlextModelsContext class with all context-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Context instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Self

import structlog.contextvars
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from flext_core import c, t
from flext_core._models.base import FlextModelFoundation
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.entity import FlextModelsEntity
from flext_core.runtime import FlextRuntime

_V = FlextModelFoundation.Validators


def _normalize_to_mapping(v: object) -> Mapping[str, object]:
    if v is None:
        out: dict[str, object] = {}
        return out
    if isinstance(v, Mapping):
        validated = _V.dict_str_metadata_adapter().validate_python(v)
        return dict(validated)
    if isinstance(v, BaseModel):
        return v.model_dump()
    msg = f"Cannot normalize {type(v)} to Mapping"
    raise ValueError(msg)


def _normalize_metadata_before(v: object | None) -> object | None:
    if v is None:
        return None
    if isinstance(v, FlextModelFoundation.Metadata):
        return v
    if isinstance(v, Mapping):
        return FlextModelFoundation.Metadata.model_validate({
            "attributes": _V.dict_str_metadata_adapter().validate_python(v)
        })
    return v


def _normalize_statistics_before(v: object) -> Mapping[str, object]:
    if v is None:
        out: dict[str, object] = {}
        return out
    return _normalize_to_mapping(v)


class FlextModelsContext:
    """Context management pattern container class.

    This class acts as a namespace container for context management patterns.
    All nested classes can be accessed via FlextModels.* (type aliases) or
    directly via FlextModelsContext.*
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
                min_length=c.Reliability.RETRY_COUNT_MIN,
                pattern=c.Platform.PATTERN_IDENTIFIER_WITH_UNDERSCORE,
                description="Context variable key (alphanumeric, underscore)",
                examples=["correlation_id", "service_name", "user_id"],
            ),
        ]
        previous_value: Annotated[
            object | None,
            Field(default=None, description="Previous value before set operation"),
        ] = None

    class StructlogProxyContextVar[T: object]:
        """ContextVar-like proxy using structlog as backend (single source of truth).

        Type Parameter T is bounded by object - all storable context values.

        ARCHITECTURAL NOTE: This proxy delegates ALL operations to structlog's
        contextvar storage. This ensures FlextContext.Variables and FlextLogger
        use THE SAME underlying storage, eliminating dual storage and sync issues.

        Key Principles:
            - Single Source of Truth: structlog's contextvar dict
            - Zero Synchronization: No dual storage, no sync needed
            - Thread Safety: structlog handles all thread safety
            - Performance: Direct delegation, no overhead

        Type Safety Note:
            structlog.contextvars.get_contextvars() has incomplete type hints:
            it returns mapping[str, object] but we control the values (always
            object). We cast to mapping[str, object]
            to recover proper type information. This is NECESSARY to enable
            type inference of T through the proxy.

        Usage:
            >>> var = FlextModelsContext.StructlogProxyContextVar[str](
            ...     "correlation_id", default=None
            ... )
            >>> var.set("abc-123")
            >>> var.get()  # Returns "abc-123"

        """

        def __init__(self, key: str, default: T | None = None) -> None:
            """Initialize proxy context variable.

            Args:
                key: Unique key for this context variable
                default: Default value when not set

            """
            super().__init__()
            self._key = key
            self._default: T | None = default

        @staticmethod
        def reset(token: FlextModelsContext.StructlogProxyToken) -> None:
            """Reset to previous value using token.

            Args:
                token: Token from previous set() call

            Note:
                structlog.contextvars doesn't support token-based reset.
                Use unbind_contextvars() or clear_contextvars() instead.

            """
            if token.previous_value is None:
                structlog.contextvars.unbind_contextvars(token.key)
            else:
                _ = structlog.contextvars.bind_contextvars(**{
                    token.key: token.previous_value
                })

        def get(self) -> object | None:
            """Get current value from structlog context.

            Returns:
                Current value with type T (from Generic[T] contract), or None.

            Architecture:
                structlog.contextvars.get_contextvars() returns a mapping;
                we control stored values via set() (object). Key presence
                is checked then the value is returned; T is bounded to object.

            """
            contextvars_data = structlog.contextvars.get_contextvars()
            structlog_context: Mapping[str, object] = contextvars_data
            if self._key not in structlog_context:
                return self._default
            value = structlog_context[self._key]
            if value is None:
                return self._default
            return value

        def set(self, value: T | None) -> FlextModelsContext.StructlogProxyToken:
            """Set value in structlog context.

            Args:
                value: Value to set in structlog's contextvar (can be None to clear)

            Returns:
                Token for potential reset

            """
            current_value = self.get()
            if value is not None:
                _ = structlog.contextvars.bind_contextvars(**{self._key: value})
            else:
                structlog.contextvars.unbind_contextvars(self._key)
            prev_value: object | None = current_value
            return FlextModelsContext.StructlogProxyToken(
                key=self._key, previous_value=prev_value
            )

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
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Unique key for the context variable",
                examples=["user_id", "request_id", "session_id"],
            ),
        ]
        old_value: Annotated[
            object | None,
            Field(default=None, description="Previous value before set operation"),
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
            >>> from flext_core import FlextModels
            >>> context_data = FlextModelsContext.ContextData(
                data={"user_id": "123", "correlation_id": "abc-xyz"},
                metadata=FlextModelFoundation.Metadata(
                    attributes={
                        "source": "api",
                        "created_at": "2025-01-01T00:00:00Z",
                    }
                ),
            )
            >>> context_data.data["user_id"]
            '123'
            >>> context_data.metadata.attributes["source"]
            'api'

        """

        data: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Initial context data as key-value pairs",
            ),
        ]
        metadata: Annotated[
            FlextModelFoundation.Metadata | FlextModelsContainers.Dict | None,
            BeforeValidator(_normalize_metadata_before),
            Field(
                default=None,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = None
        model_config = ConfigDict(extra=c.ModelConfig.EXTRA_IGNORE)

        @classmethod
        def check_json_serializable(cls, obj: object, path: str = "") -> None:
            """Recursively check if object is JSON-serializable."""
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return
            if FlextRuntime.is_dict_like(obj):
                for key, val in obj.items():
                    cls.check_json_serializable(val, f"{path}.{key}")
                return
            if FlextRuntime.is_list_like(obj) and (not isinstance(obj, (str, bytes))):
                seq_obj: Sequence[object] = obj
                for i, item in enumerate(seq_obj):
                    cls.check_json_serializable(item, f"{path}[{i}]")
                return
            msg = f"Non-JSON-serializable type {obj.__class__.__name__} at {path}"
            raise TypeError(msg)

        @classmethod
        def normalize_to_serializable_value(cls, val: object) -> object:
            normalized = cls.normalize_to_general_value(val)
            if normalized is None or isinstance(normalized, (str, int, float, bool)):
                return normalized
            if isinstance(normalized, Mapping):
                normalized_map = _V.dict_str_metadata_adapter().validate_python(
                    normalized
                )
                return {
                    str(key): cls.normalize_to_serializable_value(val)
                    for key, val in normalized_map.items()
                }
            if FlextRuntime.is_list_like(normalized):
                return [
                    cls.normalize_to_serializable_value(item) for item in normalized
                ]
            return str(normalized)

        @field_validator("data", mode="before")
        @classmethod
        def validate_dict_serializable(
            cls,
            v: FlextModelsContainers.Dict | Mapping[str, t.Scalar] | BaseModel | None,
        ) -> Mapping[str, object]:
            """Validate that ConfigurationMapping values are JSON-serializable.

            STRICT mode: Also accepts FlextModelFoundation.Metadata and converts to dict.
            Uses mode='before' to validate raw input before Pydantic processing.
            Only allows JSON-serializable types: str, int, float, bool, list, dict,
            None.
            """
            working_value: dict[str, object]
            normalized_mapping: Mapping[str, object]
            if v is None:
                return {}
            if isinstance(v, FlextModelFoundation.Metadata):
                normalized_mapping = FlextModelsContainers.ConfigMap(
                    root=dict(v.attributes.items())
                ).root
            elif isinstance(v, BaseModel):
                dump_result = v.model_dump()
                normalized_mapping = FlextModelsContainers.ConfigMap(dump_result).root
            else:
                normalized_mapping = dict(v)
            working_value = {
                str(k): FlextModelsContext.ContextData.normalize_to_serializable_value(
                    val
                )
                for k, val in normalized_mapping.items()
            }
            FlextModelsContext.ContextData.check_json_serializable(working_value)
            return dict(working_value)

        @staticmethod
        def normalize_to_general_value(val: object) -> object:
            """Normalize to container; raises TypeError for non-normalizable types."""
            if val is None:
                return ""
            if isinstance(val, t.SCALAR_TYPES):
                return val
            if isinstance(val, BaseModel):
                return val
            if FlextRuntime.is_dict_like(val) or FlextRuntime.is_list_like(val):
                return FlextRuntime.normalize_to_container(val)
            if hasattr(val, "__iter__"):
                return str(val)
            msg = f"Non-normalizable type {type(val).__name__}"
            raise TypeError(msg)

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
            >>> from flext_core import FlextModels
            >>> export = FlextModelsContext.ContextExport(
                data={"user_id": "123", "correlation_id": "abc-xyz"},
                metadata=FlextModelFoundation.Metadata(
                    attributes={"source": "api", "version": "1.0"}
                ),
                statistics={"sets": 5, "gets": 10, "removes": 2},
            )
            >>> export.data["user_id"]
            '123'
            >>> export.statistics["sets"]
            5

        """

        data: Annotated[
            Mapping[str, object],
            Field(
                default_factory=dict,
                description="All context data from all scopes",
            ),
        ]
        metadata: Annotated[
            FlextModelFoundation.Metadata | FlextModelsContainers.Dict | None,
            BeforeValidator(_normalize_metadata_before),
            Field(
                default=None,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = None
        statistics: Annotated[
            Mapping[str, object],
            BeforeValidator(_normalize_statistics_before),
            Field(
                default_factory=dict,
                description="Usage statistics (operation counts, timing info)",
            ),
        ]

        @computed_field
        def has_statistics(self) -> bool:
            """Check if statistics are available."""
            return bool(self.statistics)

        @computed_field
        def total_data_items(self) -> int:
            """Compute total number of data items across all scopes."""
            return len(self.data)

        @field_validator("data", mode="before")
        @classmethod
        def validate_dict_serializable(
            cls,
            v: FlextModelsContainers.Dict | Mapping[str, t.Scalar] | BaseModel | None,
        ) -> Mapping[str, object]:
            """Validate that ConfigurationMapping values are JSON-serializable.

            Uses mode='before' to validate raw input before Pydantic processing.
            Accepts Pydantic models (converts via model_dump) or dict.
            Only allows JSON-serializable types: str, int, float, bool, list, dict,
            None.
            """
            working_value: dict[str, object]
            normalized_mapping: Mapping[str, object]
            if v is None:
                return {}
            if isinstance(v, FlextModelFoundation.Metadata):
                normalized_mapping = FlextModelsContainers.ConfigMap(
                    root=dict(v.attributes.items())
                ).root
            elif isinstance(v, BaseModel):
                dump_result = v.model_dump()
                normalized_mapping = FlextModelsContainers.ConfigMap(dump_result).root
            else:
                normalized_mapping = dict(v)
            working_value = {
                str(k): FlextModelsContext.ContextData.normalize_to_serializable_value(
                    val
                )
                for k, val in normalized_mapping.items()
            }
            FlextModelsContext.ContextData.check_json_serializable(working_value)
            return dict(working_value)

    class ContextScopeData(FlextModelFoundation.ArbitraryTypesModel):
        """Scope-specific data container for context management.

        Enhanced to support dict handling while
        maintaining strong typing for scope data and metadata.

        Attributes:
            scope_name: Name of the scope (e.g., 'request', 'operation')
            scope_type: Type/category of scope
            data: Scope-specific data key-value pairs
            metadata: FlextModelFoundation.Metadata associated with this scope

        Examples:
            >>> from flext_core import FlextModels
            >>> scope = FlextModelsContext.ContextScopeData(
                scope_name="request",
                scope_type="http",
                data={"method": "POST", "path": "/api/orders"},
                metadata=FlextModelFoundation.Metadata(
                    attributes={"trace_id": "trace-123"}
                ),
            )

        """

        scope_name: Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Name of the scope",
            ),
        ] = ""
        scope_type: Annotated[
            str, Field(default="", description="Type/category of scope")
        ] = ""
        data: Annotated[
            Mapping[str, object],
            BeforeValidator(_normalize_to_mapping),
            Field(default_factory=dict, description="Scope data"),
        ]
        metadata: Annotated[
            Mapping[str, object],
            BeforeValidator(_normalize_to_mapping),
            Field(default_factory=dict, description="Scope metadata"),
        ]

    class ContextStatistics(FlextModelFoundation.ArbitraryTypesModel):
        """Statistics tracking for context operations and metrics.

        Enhanced to replace dict-based metrics storage across
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
            Field(default=c.ZERO, ge=c.ZERO, description="Number of set operations"),
        ] = c.ZERO
        gets: Annotated[
            int,
            Field(default=c.ZERO, ge=c.ZERO, description="Number of get operations"),
        ] = c.ZERO
        removes: Annotated[
            int,
            Field(default=c.ZERO, ge=c.ZERO, description="Number of remove operations"),
        ] = c.ZERO
        clears: Annotated[
            int,
            Field(default=c.ZERO, ge=c.ZERO, description="Number of clear operations"),
        ] = c.ZERO
        operations: Annotated[
            Mapping[str, object],
            BeforeValidator(_normalize_to_mapping),
            Field(
                default_factory=dict,
                description="Additional metric counters and timing values grouped by metric key.",
            ),
        ]

    class ContextMetadata(BaseModel):
        """Metadata storage for context objects with full tracing support.

        Enhanced to use object and Mapping patterns
        across multiple modules including context.py, handlers.py, dispatcher.py,
        and config.py for consistent, strongly-typed metadata handling.

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
            str | None, Field(default=None, description="Associated user ID")
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
            str | None, Field(default=None, description="HTTP request identifier")
        ] = None
        session_id: Annotated[
            str | None, Field(default=None, description="User session identifier")
        ] = None
        tenant_id: Annotated[
            str | None, Field(default=None, description="Tenant/Organization ID")
        ] = None
        handler_mode: Annotated[
            str | None,
            Field(default=None, description="Handler mode (command/query/event)"),
        ] = None
        message_type: Annotated[
            str | None,
            Field(default=None, description="Type of message being processed"),
        ] = None
        message_id: Annotated[
            str | None, Field(default=None, description="Unique message identifier")
        ] = None
        custom_fields: Annotated[
            Mapping[str, object],
            BeforeValidator(_normalize_to_mapping),
            Field(
                default_factory=dict,
                description="Custom metadata attributes for caller-specific tracing and context.",
            ),
        ]

        @model_validator(mode="after")
        def validate_context_protocol(self) -> Self:
            """Validate context instance has get() and set() methods."""
            context_field = None
            for field_name in self.__class__.model_fields:
                if "context" in field_name.lower():
                    context_field = getattr(self, field_name, None)
                    break
            if context_field is None:
                return self
            if hasattr(context_field, "get") and hasattr(context_field, "set"):
                return self
            msg = "Context must have get() and set() methods"
            raise ValueError(msg)

    class ContextDomainData(BaseModel):
        """Domain-specific context data storage."""

        domain_name: Annotated[
            str | None, Field(default=None, description="Domain name/identifier")
        ] = None
        domain_type: Annotated[
            str | None, Field(default=None, description="Type of domain")
        ] = None
        domain_data: Annotated[
            Mapping[str, object],
            Field(
                default_factory=dict,
                description="Domain payload values scoped to the current business context.",
            ),
        ]
        domain_metadata: Annotated[
            Mapping[str, object],
            Field(
                default_factory=dict,
                description="Domain metadata attributes describing origin and processing state.",
            ),
        ]


__all__ = ["FlextModelsContext"]
