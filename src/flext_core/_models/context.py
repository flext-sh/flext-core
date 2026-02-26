"""Context management patterns extracted from FlextModels.

This module contains the FlextModelsContext class with all context-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Context instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Annotated

import structlog.contextvars
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from flext_core._models.base import FlextModelFoundation
from flext_core._models.entity import FlextModelsEntity
from flext_core.constants import c
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


def _normalize_metadata_before(
    value: FlextModelFoundation.Metadata
    | t.ConfigMap
    | Mapping[str, t.ConfigMapValue]
    | None,
) -> FlextModelFoundation.Metadata:
    """BeforeValidator: normalize metadata to FlextModelFoundation.Metadata."""
    return FlextModelsContext.normalize_metadata(value)


def _normalize_to_mapping(
    v: t.GuardInputValue,
) -> Mapping[str, t.GuardInputValue]:
    """BeforeValidator: normalize dict-like input to a mapping."""
    if isinstance(v, t.Dict):
        return v.root
    if isinstance(v, Mapping):
        return {
            str(k): FlextRuntime.normalize_to_general_value(val) for k, val in v.items()
        }
    if isinstance(v, BaseModel):
        dumped = v.model_dump()
        if not FlextRuntime.is_dict_like(dumped):
            return {}
        return {
            str(k): FlextRuntime.normalize_to_metadata_value(val)
            for k, val in dumped.items()
        }
    if v is None:
        return {}
    msg = f"must be dict or BaseModel, got {v.__class__.__name__}"
    raise TypeError(msg)


def _normalize_statistics_before(
    v: t.GuardInputValue,
) -> Mapping[str, t.GuardInputValue]:
    """BeforeValidator: normalize statistics input."""
    if v is None:
        return {}
    if FlextRuntime.is_dict_like(v):
        return dict(v)
    if isinstance(v, BaseModel):
        return FlextModelsContext.to_general_value_dict(v.model_dump())
    msg = f"statistics must be dict or BaseModel, got {v.__class__.__name__}"
    raise TypeError(msg)


class FlextModelsContext:
    """Context management pattern container class.

    This class acts as a namespace container for context management patterns.
    All nested classes can be accessed via FlextModels.* (type aliases) or
    directly via FlextModelsContext.*
    """

    @staticmethod
    def normalize_metadata(
        value: FlextModelFoundation.Metadata
        | t.ConfigMap
        | Mapping[str, t.ConfigMapValue]
        | None,
    ) -> FlextModelFoundation.Metadata:
        """Normalize metadata input to FlextModelFoundation.Metadata."""
        if value is None:
            return FlextModelFoundation.Metadata(attributes={})
        if isinstance(value, FlextModelFoundation.Metadata):
            return value
        if not FlextRuntime.is_dict_like(value):
            msg = (
                f"metadata must be None, dict, or FlextModelsBase.Metadata, "
                f"got {value.__class__.__name__}"
            )
            raise TypeError(msg)
        attributes: Mapping[str, t.MetadataAttributeValue] = {
            str(k): FlextRuntime.normalize_to_metadata_value(v)
            for k, v in (
                value.root.items() if isinstance(value, t.ConfigMap) else value.items()
            )
        }
        return FlextModelFoundation.Metadata(attributes=attributes)

    @staticmethod
    def to_general_value_dict(
        value: t.ConfigMap | Mapping[str, t.ConfigMapValue],
    ) -> Mapping[str, t.MetadataAttributeValue]:
        """Convert dict-like value to metadata mapping for Metadata."""
        if not FlextRuntime.is_dict_like(value):
            return {}
        return {
            str(k): FlextRuntime.normalize_to_metadata_value(v)
            for k, v in (
                value.root.items() if isinstance(value, t.ConfigMap) else value.items()
            )
        }

    class ArbitraryTypesModel(FlextModelFoundation.ArbitraryTypesModel):
        """Base model with arbitrary types support - real inheritance."""

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
            t.ConfigMapValue | None,
            Field(
                default=None,
                description="Previous value before set operation",
            ),
        ] = None

    class StructlogProxyContextVar[T: t.ConfigMapValue]:
        """ContextVar-like proxy using structlog as backend (single source of truth).

        Type Parameter T is bounded by PayloadValue - all storable context values.

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
            it returns mapping[str, t.GuardInputValue] but we control the values (always
            t.GuardInputValue). We cast to mapping[str, t.GuardInputValue]
            to recover proper type information. This is NECESSARY to enable
            type inference of T through the proxy.

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
                structlog.contextvars.get_contextvars() returns a mapping;
                we control stored values via set() (t.GuardInputValue). Key presence
                is checked then the value is returned; T is bounded to t.GuardInputValue.

            """
            contextvars_data = structlog.contextvars.get_contextvars()
            if not isinstance(contextvars_data, Mapping):
                return self._default
            structlog_context: Mapping[str, t.ConfigMapValue] = contextvars_data
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
            # Get current value before setting
            current_value = self.get()

            if value is not None:
                # T is bounded to PayloadValue in generic contract
                # Store directly - type parameter constraint guarantees compatibility
                _ = structlog.contextvars.bind_contextvars(**{
                    self._key: value,
                })
            else:
                # Unbind if setting to None
                structlog.contextvars.unbind_contextvars(self._key)

            # Create token for reset functionality
            # Normalize current_value to t.GuardInputValue for storage
            # T is bounded to PayloadValue, so current_value is already compatible
            prev_value: t.ConfigMapValue | None = current_value

            return FlextModelsContext.StructlogProxyToken(
                key=self._key,
                previous_value=prev_value,
            )

        @staticmethod
        def reset(token: FlextModelsContext.StructlogProxyToken) -> None:
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
                _ = structlog.contextvars.bind_contextvars(**{
                    token.key: token.previous_value,
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
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Unique key for the context variable",
                examples=["user_id", "request_id", "session_id"],
            ),
        ]
        old_value: Annotated[
            t.ConfigMapValue | None,
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

        data: t.Dict = Field(
            default_factory=t.Dict,
            description="Initial context data as key-value pairs",
        )
        metadata: Annotated[
            FlextModelFoundation.Metadata | t.Dict | None,
            BeforeValidator(_normalize_metadata_before),
        ] = Field(
            default=None,
            description="Context metadata (creation info, source, etc.)",
        )
        model_config = ConfigDict(
            extra=c.ModelConfig.EXTRA_IGNORE,
        )

        @staticmethod
        def normalize_to_general_value(
            val: t.GuardInputValue,
        ) -> t.GuardInputValue:
            """Normalize any value to t.GuardInputValue recursively."""
            return FlextRuntime.normalize_to_general_value(val)

        @classmethod
        def normalize_to_serializable_value(
            cls,
            val: t.GuardInputValue,
        ) -> t.GuardInputValue:
            normalized = cls.normalize_to_general_value(val)
            if normalized is None or isinstance(normalized, (str, int, float, bool)):
                return normalized
            if isinstance(normalized, Mapping):
                return {
                    str(k): cls.normalize_to_serializable_value(v)
                    for k, v in normalized.items()
                }
            if FlextRuntime.is_list_like(normalized):
                return [
                    cls.normalize_to_serializable_value(item) for item in normalized
                ]
            return str(normalized)

        @classmethod
        def check_json_serializable(
            cls,
            obj: t.GuardInputValue,
            path: str = "",
        ) -> None:
            """Recursively check if object is JSON-serializable."""
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return
            # is_dict_like already checks for Mapping protocol compliance
            if FlextRuntime.is_dict_like(obj):
                # Type narrowing: is_dict_like ensures Mapping protocol
                dict_obj: Mapping[str, t.GuardInputValue] = dict(obj)
                for key, val in dict_obj.items():
                    # Recursive call using cls for mypy compatibility
                    cls.check_json_serializable(val, f"{path}.{key}")
                return  # All dict items validated successfully
            # is_list_like already checks for Sequence protocol compliance
            # Exclude str/bytes which are also Sequence
            if FlextRuntime.is_list_like(obj) and not isinstance(obj, (str, bytes)):
                # Type narrowing: is_list_like ensures Sequence protocol
                seq_obj: Sequence[t.GuardInputValue] = obj
                for i, item in enumerate(seq_obj):
                    # Recursive call using cls for mypy compatibility
                    cls.check_json_serializable(item, f"{path}[{i}]")
                return  # All list items validated successfully
            msg = f"Non-JSON-serializable type {obj.__class__.__name__} at {path}"
            raise TypeError(msg)

        @field_validator("data", mode="before")
        @classmethod
        def validate_dict_serializable(
            cls,
            v: t.Dict | Mapping[str, t.ConfigMapValue] | BaseModel | None,
        ) -> Mapping[str, t.ConfigMapValue]:
            """Validate that ConfigurationMapping values are JSON-serializable.

            STRICT mode: Also accepts FlextModelFoundation.Metadata and converts to dict.
            Uses mode='before' to validate raw input before Pydantic processing.
            Only allows JSON-serializable types: str, int, float, bool, list, dict,
            None.
            """
            # Convert various input types to dict
            working_value: MutableMapping[str, t.ConfigMapValue]

            # STRICT mode: Accept FlextModelFoundation.Metadata and convert to dict
            if v is None:
                return {}

            if isinstance(v, FlextModelFoundation.Metadata):
                normalized = FlextModelsContext.ContextData.normalize_to_general_value(
                    v.attributes,
                )
            elif isinstance(v, BaseModel):
                dump_result = v.model_dump()
                if not isinstance(dump_result, dict):
                    type_name = v.__class__.__name__
                    msg = f"Value must be a dictionary or Metadata, got {type_name}"
                    raise TypeError(msg)
                dump_dict = dict(dump_result)
                normalized = FlextModelsContext.ContextData.normalize_to_general_value(
                    dump_dict,
                )
            elif FlextRuntime.is_dict_like(v):
                normalized = FlextModelsContext.ContextData.normalize_to_general_value(
                    v
                )
            else:
                type_name = v.__class__.__name__
                msg = f"Value must be a dictionary or Metadata, got {type_name}"
                raise TypeError(msg)

            if not isinstance(normalized, Mapping):
                msg = "Normalized value must be dict"
                raise TypeError(msg)

            working_value = {
                str(k): FlextModelsContext.ContextData.normalize_to_serializable_value(
                    val
                )
                for k, val in normalized.items()
            }

            # Validate JSON serializability
            FlextModelsContext.ContextData.check_json_serializable(working_value)

            return dict(working_value)

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

        data: Mapping[str, t.GuardInputValue] = Field(
            default_factory=dict,
            description="All context data from all scopes",
        )
        metadata: Annotated[
            FlextModelFoundation.Metadata | t.Dict | None,
            BeforeValidator(_normalize_metadata_before),
        ] = Field(
            default=None,
            description="Context metadata (creation info, source, etc.)",
        )
        statistics: Annotated[
            Mapping[str, t.GuardInputValue],
            BeforeValidator(_normalize_statistics_before),
        ] = Field(
            default_factory=dict,
            description="Usage statistics (operation counts, timing info)",
        )

        @classmethod
        def check_json_serializable(
            cls,
            obj: t.GuardInputValue,
            path: str = "",
        ) -> None:
            """Recursively check if object is JSON-serializable."""
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return
            # is_dict_like already checks for Mapping protocol compliance
            if FlextRuntime.is_dict_like(obj):
                # Type narrowing: is_dict_like ensures Mapping protocol
                dict_obj: Mapping[str, t.GuardInputValue] = dict(obj)
                for key, val in dict_obj.items():
                    # Recursive call using cls for mypy compatibility
                    cls.check_json_serializable(
                        val,
                        f"{path}.{key}",
                    )
            # is_list_like already checks for Sequence protocol compliance
            # Exclude str/bytes which are also Sequence
            elif FlextRuntime.is_list_like(obj) and not isinstance(obj, (str, bytes)):
                # Type narrowing: is_list_like ensures Sequence protocol
                seq_obj: Sequence[t.GuardInputValue] = obj
                for i, item in enumerate(seq_obj):
                    # Recursive call using cls for mypy compatibility
                    cls.check_json_serializable(
                        item,
                        f"{path}[{i}]",
                    )
            else:
                msg = f"Non-JSON-serializable type {obj.__class__.__name__} at {path}"
                raise TypeError(msg)

        @field_validator("data", mode="before")
        @classmethod
        def validate_dict_serializable(
            cls,
            v: t.Dict | Mapping[str, t.ConfigMapValue] | BaseModel | None,
        ) -> Mapping[str, t.ConfigMapValue]:
            """Validate that ConfigurationMapping values are JSON-serializable.

            Uses mode='before' to validate raw input before Pydantic processing.
            Accepts Pydantic models (converts via model_dump) or dict.
            Only allows JSON-serializable types: str, int, float, bool, list, dict,
            None.
            """
            # Convert various input types to dict
            working_value: MutableMapping[str, t.ConfigMapValue]

            # Handle m.Metadata specially - extract only attributes dict
            # (excludes datetime fields which aren't JSON-serializable)
            if v is None:
                return {}

            if isinstance(v, FlextModelFoundation.Metadata):
                normalized = FlextModelsContext.ContextData.normalize_to_general_value(
                    v.attributes,
                )
            elif isinstance(v, BaseModel):
                dump_result = v.model_dump()
                if not isinstance(dump_result, dict):
                    type_name = v.__class__.__name__
                    msg = f"Value must be a dict or Pydantic model, got {type_name}"
                    raise TypeError(msg)
                dump_dict = dict(dump_result)
                normalized = FlextModelsContext.ContextData.normalize_to_general_value(
                    dump_dict,
                )
            elif FlextRuntime.is_dict_like(v):
                normalized = FlextModelsContext.ContextData.normalize_to_general_value(
                    v
                )
            else:
                type_name = v.__class__.__name__
                msg = f"Value must be a dict or Pydantic model, got {type_name}"
                raise TypeError(msg)

            if not isinstance(normalized, Mapping):
                msg = "Normalized value must be dict"
                raise TypeError(msg)

            working_value = {
                str(k): FlextModelsContext.ContextData.normalize_to_serializable_value(
                    val
                )
                for k, val in normalized.items()
            }

            # Recursively check all values are JSON-serializable
            FlextModelsContext.ContextExport.check_json_serializable(working_value)

            # working_value is always dict from comprehensions above;
            # explicit dict() satisfies return mapping[str, PayloadValue]
            return dict(working_value)

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
            str,
            Field(default="", description="Type/category of scope"),
        ] = ""
        data: Annotated[
            Mapping[str, t.GuardInputValue],
            BeforeValidator(_normalize_to_mapping),
        ] = Field(default_factory=dict, description="Scope data")
        metadata: Annotated[
            Mapping[str, t.GuardInputValue],
            BeforeValidator(_normalize_to_mapping),
        ] = Field(default_factory=dict, description="Scope metadata")

    class ContextStatistics(BaseModel):
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
            Mapping[str, t.GuardInputValue],
            BeforeValidator(_normalize_to_mapping),
            Field(
                default_factory=dict,
                description="Extensible operation/metrics counts",
            ),
        ] = Field(default_factory=dict)

    class ContextMetadata(BaseModel):
        """Metadata storage for context objects with full tracing support.

        Enhanced to use t.GuardInputValue and Mapping patterns
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
            Mapping[str, t.GuardInputValue],
            BeforeValidator(_normalize_to_mapping),
            Field(
                default_factory=dict,
                description="Extensible custom metadata fields",
            ),
        ] = Field(default_factory=dict)

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
            Mapping[str, t.GuardInputValue],
            Field(default_factory=dict, description="Domain-specific data"),
        ] = Field(default_factory=dict)
        domain_metadata: Annotated[
            Mapping[str, t.GuardInputValue],
            Field(default_factory=dict, description="Domain metadata"),
        ] = Field(default_factory=dict)


# Resolve forward references created by `from __future__ import annotations`.
# Models using `t.GuardInputValue` (recursive PEP 695 type alias `_ContainerValue`)
# require explicit rebuild so Pydantic can resolve the deferred string annotations.
_ = FlextModelsContext.ContextScopeData.model_rebuild()
_ = FlextModelsContext.ContextStatistics.model_rebuild()
_ = FlextModelsContext.ContextMetadata.model_rebuild()
_ = FlextModelsContext.ContextDomainData.model_rebuild()

__all__ = ["FlextModelsContext"]
