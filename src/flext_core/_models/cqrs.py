"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
)

from flext_core import FlextModelFoundation, FlextRuntime, c, p, r, t


class FlextModelsCqrs:
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or
    directly via FlextModelsCqrs.*
    """

    class Command(FlextModelFoundation.ArbitraryTypesModel):
        """Base class for CQRS commands with validation."""

        tag: ClassVar[Literal["command"]] = "command"
        message_type: Annotated[
            Literal["command"],
            Field(
                default="command",
                frozen=True,
                description="Message type discriminator (always 'command')",
            ),
        ] = "command"
        command_type: Annotated[
            t.NonEmptyStr,
            Field(
                default=c.DEFAULT_COMMAND_TYPE,
                description="Command type identifier",
            ),
        ] = c.DEFAULT_COMMAND_TYPE
        command_id: Annotated[
            t.NonEmptyStr,
            Field(
                default_factory=lambda: FlextRuntime.generate_prefixed_id("cmd"),
                description="Unique command identifier used for tracing and idempotency checks.",
                title="Command Id",
                examples=["cmd_01HZX7Q0P5N6M2"],
            ),
        ]
        issuer_id: t.NonEmptyStr | None = None

        @property
        def event_type(self) -> str | None:
            """Event type identifier (always None for commands)."""
            return None

        @property
        def query_type(self) -> str | None:
            """Query type identifier (always None for commands)."""
            return None

    class Pagination(FlextModelFoundation.FlexibleInternalModel):
        """Pagination model for query results."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Pagination",
                "description": "Pagination model for query results with computed fields",
            },
        )
        page: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_PAGE_NUMBER,
                description="Page number (1-based indexing)",
                examples=[1, 2, 10, 100],
            ),
        ] = c.DEFAULT_PAGE_NUMBER
        size: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_PAGE_SIZE,
                le=c.MAX_PAGE_SIZE_EXAMPLE,
                description="Number of items per page (max 1000)",
                examples=[10, 20, 50, 100],
            ),
        ] = c.DEFAULT_PAGE_SIZE

        @computed_field
        def limit(self) -> int:
            """Get limit (same as size)."""
            return self.size

        @computed_field
        def offset(self) -> int:
            """Calculate offset from page and size."""
            return (self.page - 1) * self.size

    class Query(FlextModelFoundation.ArbitraryTypesModel):
        """Query model for CQRS query operations."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations",
            },
        )
        tag: ClassVar[Literal["query"]] = "query"
        _pagination_input_adapter: ClassVar[
            TypeAdapter[BaseModel | t.Dict | Mapping[str, t.Scalar] | None] | None
        ] = None
        message_type: Literal["query"] = Field(
            default="query",
            frozen=True,
            description="Message type discriminator",
        )
        filters: t.Dict = Field(
            default_factory=t.Dict,
            description="Filter values that restrict which records are returned by the query.",
            title="Query Filters",
            examples=[{"status": "active", "tenant": "acme"}],
        )
        pagination: BaseModel | t.Dict = Field(
            default_factory=t.Dict,
            description="Pagination settings controlling page number and page size for query results.",
            title="Pagination",
            examples=[{"page": 1, "size": 50}],
        )
        query_id: Annotated[
            t.NonEmptyStr,
            Field(
                default_factory=lambda: FlextRuntime.generate_prefixed_id("query"),
                description="Unique query identifier used for tracing and cache correlation.",
                title="Query Id",
                examples=["query_01HZX7Q0P5N6M2"],
            ),
        ]
        query_type: str | None = None

        @property
        def command_type(self) -> str | None:
            """Command type identifier (always None for queries)."""
            return None

        @property
        def event_type(self) -> str | None:
            """Event type identifier (always None for queries)."""
            return None

        @classmethod
        def _resolve_pagination_class(
            cls: type[FlextModelsCqrs.Query],
        ) -> type[BaseModel]:
            """Resolve correct Pagination class based on context."""
            if cls.__module__ != "flext_core.models" or "." not in cls.__qualname__:
                return FlextModelsCqrs.Pagination
            parts = cls.__qualname__.split(".")
            models_module = sys.modules.get("flext_core.models")
            min_qualname_parts = 2
            if not models_module or len(parts) < min_qualname_parts:
                return FlextModelsCqrs.Pagination
            obj: p.Base | BaseModel | type | ModuleType | None = (
                models_module.__dict__.get(parts[0])
            )
            for part in parts[1:-1]:
                if obj is None:
                    break
                if isinstance(obj, ModuleType):
                    obj = obj.__dict__.get(part)
                    continue
                if isinstance(obj, type):
                    obj = obj.__dict__.get(part)
                    continue
                obj = None
            if isinstance(obj, type):
                pagination_cls_attr = obj.__dict__.get("Pagination")
                if (
                    pagination_cls_attr is not None
                    and isinstance(pagination_cls_attr, type)
                    and issubclass(pagination_cls_attr, FlextModelsCqrs.Pagination)
                ):
                    result_cls: type[FlextModelsCqrs.Pagination] = pagination_cls_attr
                    return result_cls
            return FlextModelsCqrs.Pagination

        @classmethod
        def _pagination_adapter(
            cls,
        ) -> TypeAdapter[BaseModel | t.Dict | Mapping[str, t.Scalar] | None]:
            if cls._pagination_input_adapter is None:
                cls._pagination_input_adapter = TypeAdapter(
                    FlextModelsCqrs.Pagination | t.Dict | Mapping[str, t.Scalar] | None,
                )
            return cls._pagination_input_adapter

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: BaseModel | t.Dict | Mapping[str, t.Scalar] | None,
        ) -> BaseModel:
            """Convert pagination to Pagination instance."""
            pagination_cls = cls._resolve_pagination_class()
            parsed_input = cls._pagination_adapter().validate_python(v)
            if parsed_input is None:
                return pagination_cls()
            validate_result = r[BaseModel].create_from_callable(
                lambda: (
                    pagination_cls.model_validate(parsed_input.root)
                    if isinstance(parsed_input, t.Dict)
                    else pagination_cls.model_validate(parsed_input.model_dump())
                    if isinstance(parsed_input, FlextModelsCqrs.Pagination)
                    else pagination_cls.model_validate(dict(parsed_input))
                ),
            )
            if validate_result.is_failure:
                return pagination_cls()
            return validate_result.value

    class Bus(FlextModelFoundation.ArbitraryTypesModel):
        """Dispatcher configuration model for CQRS routing."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Dispatcher",
                "description": "CQRS dispatcher configuration",
            },
        )
        enable_middleware: Annotated[
            bool,
            Field(default=True, description="Enable middleware pipeline"),
        ] = True
        enable_metrics: Annotated[
            bool,
            Field(default=True, description="Enable metrics collection"),
        ] = True
        enable_caching: Annotated[
            bool,
            Field(default=True, description="Enable query result caching"),
        ] = True
        execution_timeout: Annotated[
            int,
            Field(
                default=c.TIMEOUT,
                description="Command execution timeout",
            ),
        ] = c.TIMEOUT
        max_cache_size: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_MAX_CACHE_SIZE,
                description="Maximum cache size",
            ),
        ] = c.DEFAULT_MAX_CACHE_SIZE
        implementation_path: Annotated[
            str,
            Field(
                default=c.DEFAULT_DISPATCHER_PATH,
                pattern=c.PATTERN_MODULE_PATH,
                description="Implementation path",
            ),
        ] = c.DEFAULT_DISPATCHER_PATH

    class Handler(FlextModelFoundation.ArbitraryTypesModel):
        """Handler configuration model with Builder pattern support."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Handler",
                "description": "CQRS handler configuration",
            },
        )
        handler_id: Annotated[
            t.NonEmptyStr, Field(description="Unique handler identifier")
        ]
        handler_name: Annotated[
            t.NonEmptyStr, Field(description="Human-readable handler name")
        ]
        handler_type: Annotated[
            c.HandlerType,
            Field(
                default=c.HandlerType.COMMAND,
                description="Handler type",
            ),
        ] = c.HandlerType.COMMAND
        handler_mode: Annotated[
            c.HandlerType,
            Field(
                default=c.HandlerType.COMMAND,
                description="Handler mode",
            ),
        ] = c.HandlerType.COMMAND
        command_timeout: Annotated[
            int,
            Field(
                default=c.DEFAULT_COMMAND_TIMEOUT,
                description="Command timeout from c (default). Models use Config values in initialization.",
            ),
        ] = c.DEFAULT_COMMAND_TIMEOUT
        max_command_retries: Annotated[
            int,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Maximum retry attempts from c (default). Models use Config values in initialization.",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        metadata: Annotated[
            FlextModelFoundation.Metadata | None,
            Field(
                default=None,
                description="Handler metadata (Pydantic model)",
            ),
        ] = None

        class Builder:
            """Builder pattern for Handler (reduces 8 params to fluent API).

            Example:
                config = (Handler.Builder(handler_type=c.HandlerType.COMMAND)
                         .with_name("MyHandler")
                         .with_timeout(30)
                         .build())

            """

            def __init__(self, handler_type: c.HandlerType) -> None:
                """Initialize builder with required handler_type."""
                super().__init__()
                handler_short_id = FlextRuntime.generate_prefixed_id("", length=8)
                self._data: t.Dict = t.Dict(
                    root={
                        "handler_type": handler_type,
                        c.FIELD_HANDLER_MODE: c.HANDLER_MODE_COMMAND
                        if handler_type == c.HandlerType.COMMAND
                        else c.HANDLER_MODE_QUERY,
                        "handler_id": f"{handler_type}_handler_{handler_short_id}",
                        "handler_name": f"{handler_type.title()} Handler",
                        "command_timeout": c.DEFAULT_COMMAND_TIMEOUT,
                        "max_command_retries": c.DEFAULT_MAX_COMMAND_RETRIES,
                        c.FIELD_METADATA: None,
                    },
                )

            def build(self) -> FlextModelsCqrs.Handler:
                """Build and validate Handler instance."""
                return FlextModelsCqrs.Handler.model_validate(self._data.root)

            def merge_config(self, config: t.ConfigMap) -> Self:
                """Merge additional config (fluent API)."""
                self._data.root.update(config.root)
                return self

            def with_id(self, handler_id: str) -> Self:
                """Set handler ID (fluent API)."""
                self._data.root["handler_id"] = handler_id
                return self

            def with_metadata(self, metadata: FlextModelFoundation.Metadata) -> Self:
                """Set metadata (fluent API - Pydantic model)."""
                self._data.root[c.FIELD_METADATA] = metadata
                return self

            def with_name(self, handler_name: str) -> Self:
                """Set handler name (fluent API)."""
                self._data.root["handler_name"] = handler_name
                return self

            def with_retries(self, max_retries: int) -> Self:
                """Set max retries (fluent API)."""
                self._data.root["max_command_retries"] = max_retries
                return self

            def with_timeout(self, timeout: int) -> Self:
                """Set command timeout (fluent API)."""
                self._data.root["command_timeout"] = timeout
                return self

    class Event(FlextModelFoundation.ArbitraryTypesModel):
        """Event model for CQRS event operations.

        Events represent domain events that occur as a result of command execution.
        They are immutable records of what happened in the system.
        """

        tag: ClassVar[Literal["event"]] = "event"
        message_type: Annotated[
            Literal["event"],
            Field(
                default="event",
                frozen=True,
                description="Message type discriminator (always 'event')",
            ),
        ] = "event"
        event_type: Annotated[t.NonEmptyStr, Field(description="Event type identifier")]

        @property
        def command_type(self) -> str | None:
            """Command type identifier (always None for events)."""
            return None

        @property
        def query_type(self) -> str | None:
            """Query type identifier (always None for events)."""
            return None

        aggregate_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="ID of the aggregate that generated this event",
            ),
        ]
        event_id: Annotated[
            t.NonEmptyStr,
            Field(
                default_factory=lambda: FlextRuntime.generate_prefixed_id("evt"),
                description="Unique event identifier used for deduplication and observability.",
                title="Event Id",
                examples=["evt_01HZX7Q0P5N6M2"],
            ),
        ]
        data: Annotated[
            t.Dict,
            Field(
                default_factory=t.Dict,
                description="Event payload data",
            ),
        ]
        metadata: Annotated[
            t.Dict,
            Field(
                default_factory=t.Dict,
                description="Event metadata (timestamps, correlation IDs, etc.)",
            ),
        ]

    type FlextMessage = Annotated[
        Command | Query | Event,
        Discriminator("message_type"),
    ]

    @staticmethod
    def parse_message(
        payload: p.Base | BaseModel | Mapping[str, t.NormalizedValue],
    ) -> FlextMessage:
        """Parse a message payload into a FlextMessage instance."""
        _ = payload
        msg = "parse_message must be implemented by subclasses"
        raise NotImplementedError(msg)

    class HandlerBatchRegistrationResult(FlextModelFoundation.ArbitraryTypesModel):
        """Result of batch handler registration."""

        status: str
        count: int
        handlers: Sequence[str]


__all__ = ["FlextModelsCqrs"]
