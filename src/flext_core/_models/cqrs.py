"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    ValidationError,
    computed_field,
    field_validator,
)

from flext_core import c, p, t
from flext_core._models.base import FlextModelFoundation
from flext_core._models.containers import FlextModelsContainers
from flext_core.runtime import FlextRuntime


class FlextModelsCqrs:
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or
    directly via FlextModelsCqrs.*
    """

    class Command(BaseModel):
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
            str,
            Field(
                default=c.Cqrs.DEFAULT_COMMAND_TYPE,
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Command type identifier",
            ),
        ] = c.Cqrs.DEFAULT_COMMAND_TYPE
        command_id: Annotated[
            str,
            Field(
                default_factory=lambda: FlextRuntime.generate_prefixed_id("cmd"),
                description="Unique command identifier used for tracing and idempotency checks.",
                title="Command Id",
                examples=["cmd_01HZX7Q0P5N6M2"],
            ),
        ]
        issuer_id: str | None = None

        @property
        def event_type(self) -> str | None:
            """Event type identifier (always None for commands)."""
            return None

        @property
        def query_type(self) -> str | None:
            """Query type identifier (always None for commands)."""
            return None

    class Pagination(BaseModel):
        """Pagination model for query results."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Pagination",
                "description": "Pagination model for query results with computed fields",
            }
        )
        page: Annotated[
            int,
            Field(
                default=c.Pagination.DEFAULT_PAGE_NUMBER,
                ge=c.Reliability.RETRY_COUNT_MIN,
                description="Page number (1-based indexing)",
                examples=[1, 2, 10, 100],
            ),
        ] = c.Pagination.DEFAULT_PAGE_NUMBER
        size: Annotated[
            int,
            Field(
                default=c.Pagination.DEFAULT_PAGE_SIZE,
                ge=c.Reliability.RETRY_COUNT_MIN,
                le=c.Pagination.MAX_PAGE_SIZE_EXAMPLE,
                description="Number of items per page (max 1000)",
                examples=[10, 20, 50, 100],
            ),
        ] = c.Pagination.DEFAULT_PAGE_SIZE

        @computed_field
        def limit(self) -> int:
            """Get limit (same as size)."""
            return self.size

        @computed_field
        def offset(self) -> int:
            """Calculate offset from page and size."""
            return (self.page - 1) * self.size

    class Query(BaseModel):
        """Query model for CQRS query operations."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations",
            },
        )
        tag: ClassVar[Literal["query"]] = "query"
        _pagination_input_adapter: ClassVar[
            TypeAdapter[
                BaseModel | FlextModelsContainers.Dict | Mapping[str, t.Scalar] | None
            ]
            | None
        ] = None
        message_type: Literal["query"] = Field(
            default="query",
            frozen=True,
            description="Message type discriminator",
        )
        filters: FlextModelsContainers.Dict = Field(
            default_factory=FlextModelsContainers.Dict,
            description="Filter values that restrict which records are returned by the query.",
            title="Query Filters",
            examples=[{"status": "active", "tenant": "acme"}],
        )
        pagination: BaseModel | FlextModelsContainers.Dict = Field(
            default_factory=FlextModelsContainers.Dict,
            description="Pagination settings controlling page number and page size for query results.",
            title="Pagination",
            examples=[{"page": 1, "size": 50}],
        )
        query_id: str = Field(
            default_factory=lambda: FlextRuntime.generate_prefixed_id("query"),
            description="Unique query identifier used for tracing and cache correlation.",
            title="Query Id",
            examples=["query_01HZX7Q0P5N6M2"],
        )
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
            obj: p.Base | BaseModel | None = getattr(models_module, parts[0], None)
            for part in parts[1:-1]:
                if obj and hasattr(obj, part):
                    obj = getattr(obj, part)
            if obj and hasattr(obj, "Pagination"):
                pagination_cls_attr = getattr(obj, "Pagination", None)
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
        ) -> TypeAdapter[
            BaseModel | FlextModelsContainers.Dict | Mapping[str, t.Scalar] | None
        ]:
            if cls._pagination_input_adapter is None:
                cls._pagination_input_adapter = TypeAdapter(
                    FlextModelsCqrs.Pagination
                    | FlextModelsContainers.Dict
                    | Mapping[str, t.Scalar]
                    | None
                )
            return cls._pagination_input_adapter

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: BaseModel | FlextModelsContainers.Dict | Mapping[str, t.Scalar] | None,
        ) -> BaseModel:
            """Convert pagination to Pagination instance."""
            pagination_cls = cls._resolve_pagination_class()
            parsed_input = cls._pagination_adapter().validate_python(v)
            if parsed_input is None:
                return pagination_cls()
            try:
                if isinstance(parsed_input, FlextModelsContainers.Dict):
                    return pagination_cls.model_validate(parsed_input.root)
                if isinstance(parsed_input, FlextModelsCqrs.Pagination):
                    return pagination_cls.model_validate(parsed_input.model_dump())
                return pagination_cls.model_validate(dict(parsed_input))
            except (ValidationError, TypeError, ValueError):
                return pagination_cls()

    class Bus(BaseModel):
        """Dispatcher configuration model for CQRS routing."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Dispatcher",
                "description": "CQRS dispatcher configuration",
            }
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
                default=c.Defaults.TIMEOUT,
                description="Command execution timeout",
            ),
        ] = c.Defaults.TIMEOUT
        max_cache_size: Annotated[
            int,
            Field(
                default=c.Defaults.DEFAULT_MAX_CACHE_SIZE,
                description="Maximum cache size",
            ),
        ] = c.Defaults.DEFAULT_MAX_CACHE_SIZE
        implementation_path: Annotated[
            str,
            Field(
                default=c.Dispatcher.DEFAULT_DISPATCHER_PATH,
                pattern=c.Platform.PATTERN_MODULE_PATH,
                description="Implementation path",
            ),
        ] = c.Dispatcher.DEFAULT_DISPATCHER_PATH

    class Handler(BaseModel):
        """Handler configuration model with Builder pattern support."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Handler",
                "description": "CQRS handler configuration",
            }
        )
        handler_id: Annotated[str, Field(description="Unique handler identifier")]
        handler_name: Annotated[str, Field(description="Human-readable handler name")]
        handler_type: Annotated[
            c.Cqrs.HandlerType,
            Field(
                default=c.Cqrs.HandlerType.COMMAND,
                description="Handler type",
            ),
        ] = c.Cqrs.HandlerType.COMMAND
        handler_mode: Annotated[
            c.Cqrs.HandlerType,
            Field(
                default=c.Cqrs.HandlerType.COMMAND,
                description="Handler mode",
            ),
        ] = c.Cqrs.HandlerType.COMMAND
        command_timeout: Annotated[
            int,
            Field(
                default=c.Cqrs.DEFAULT_COMMAND_TIMEOUT,
                description="Command timeout from c (default). Models use Config values in initialization.",
            ),
        ] = c.Cqrs.DEFAULT_COMMAND_TIMEOUT
        max_command_retries: Annotated[
            int,
            Field(
                default=c.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
                description="Maximum retry attempts from c (default). Models use Config values in initialization.",
            ),
        ] = c.Cqrs.DEFAULT_MAX_COMMAND_RETRIES
        metadata: Annotated[
            FlextModelFoundation.Metadata | None,
            Field(
                default=None,
                description="Handler metadata (Pydantic model)",
            ),
        ] = None

        class ConfigParams(BaseModel):
            """Parameter object for handler configuration (reduces parameter count)."""

            model_config = ConfigDict(
                arbitrary_types_allowed=True,
                json_schema_extra={
                    "title": "HandlerConfigParams",
                    "description": "Parameter object for handler configuration",
                },
            )
            default_name: str | None = None
            default_id: str | None = None
            handler_config: FlextModelsContainers.ConfigMap | None = None
            command_timeout: int = 0
            max_command_retries: int = 0

        class Builder:
            """Builder pattern for Handler (reduces 8 params to fluent API).

            Example:
                config = (Handler.Builder(handler_type=c.Cqrs.HandlerType.COMMAND)
                         .with_name("MyHandler")
                         .with_timeout(30)
                         .build())

            """

            def __init__(self, handler_type: c.Cqrs.HandlerType) -> None:
                """Initialize builder with required handler_type."""
                super().__init__()
                handler_short_id = FlextRuntime.generate_prefixed_id("", length=8)
                self._data: FlextModelsContainers.Dict = FlextModelsContainers.Dict(
                    root={
                        "handler_type": handler_type,
                        "handler_mode": c.Dispatcher.HANDLER_MODE_COMMAND
                        if handler_type == c.Cqrs.HandlerType.COMMAND
                        else c.Dispatcher.HANDLER_MODE_QUERY,
                        "handler_id": f"{handler_type}_handler_{handler_short_id}",
                        "handler_name": f"{handler_type.title()} Handler",
                        "command_timeout": c.Cqrs.DEFAULT_COMMAND_TIMEOUT,
                        "max_command_retries": c.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
                        "metadata": None,
                    }
                )

            def build(self) -> FlextModelsCqrs.Handler:
                """Build and validate Handler instance."""
                return FlextModelsCqrs.Handler.model_validate(self._data.root)

            def merge_config(self, config: FlextModelsContainers.ConfigMap) -> Self:
                """Merge additional config (fluent API)."""
                self._data.root.update(config.root)
                return self

            def with_id(self, handler_id: str) -> Self:
                """Set handler ID (fluent API)."""
                self._data.root["handler_id"] = handler_id
                return self

            def with_metadata(self, metadata: FlextModelFoundation.Metadata) -> Self:
                """Set metadata (fluent API - Pydantic model)."""
                self._data.root["metadata"] = metadata
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

    class Event(BaseModel):
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
        event_type: Annotated[str, Field(description="Event type identifier")]

        @property
        def command_type(self) -> str | None:
            """Command type identifier (always None for events)."""
            return None

        @property
        def query_type(self) -> str | None:
            """Query type identifier (always None for events)."""
            return None

        aggregate_id: Annotated[
            str,
            Field(
                description="ID of the aggregate that generated this event",
            ),
        ]
        event_id: Annotated[
            str,
            Field(
                default_factory=lambda: FlextRuntime.generate_prefixed_id("evt"),
                description="Unique event identifier used for deduplication and observability.",
                title="Event Id",
                examples=["evt_01HZX7Q0P5N6M2"],
            ),
        ]
        data: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Event payload data",
            ),
        ]
        metadata: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Event metadata (timestamps, correlation IDs, etc.)",
            ),
        ]

    type FlextMessage = Annotated[
        Command | Query | Event, Discriminator("message_type")
    ]

    @staticmethod
    def parse_message(
        payload: p.Base | BaseModel | Mapping[str, t.NormalizedValue],
    ) -> FlextMessage:
        """Parse a message payload into a FlextMessage instance."""
        _ = payload
        msg = "parse_message must be implemented by subclasses"
        raise NotImplementedError(msg)

    class HandlerBatchRegistrationResult(BaseModel):
        """Result of batch handler registration."""

        status: str
        count: int
        handlers: list[str]


__all__ = ["FlextModelsCqrs"]
