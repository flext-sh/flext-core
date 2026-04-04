"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from flext_core import (
    FlextModelsBase,
    FlextRuntime,
    FlextUtilitiesGenerators,
    c,
    p,
    r,
    t,
)


class FlextModelsCqrs:
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or
    directly via FlextModelsCqrs.*
    """

    class Command(FlextModelsBase.ArbitraryTypesModel):
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
                description="Unique command identifier used for tracing and idempotency checks.",
                title="Command Id",
                examples=["cmd_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: FlextUtilitiesGenerators.generate_prefixed_id("cmd")
        )
        issuer_id: Annotated[
            t.NonEmptyStr | None,
            Field(
                default=None,
                description="Identity of the principal that issued this command.",
            ),
        ] = None

        @property
        def event_type(self) -> str | None:
            """Event type identifier (always None for commands)."""
            return None

        @property
        def query_type(self) -> str | None:
            """Query type identifier (always None for commands)."""
            return None

    class Pagination(FlextModelsBase.FlexibleInternalModel):
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
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Page number (1-based indexing)",
                examples=[1, 2, 10, 100],
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS
        size: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_PAGE_SIZE,
                le=c.MAX_PAGE_SIZE,
                description="Number of items per page (max 1000)",
                examples=[10, 20, 50, 100],
            ),
        ] = c.DEFAULT_PAGE_SIZE

        @computed_field
        @property
        def limit(self) -> int:
            """Get limit (same as size)."""
            return self.size

        @computed_field
        @property
        def offset(self) -> int:
            """Calculate offset from page and size."""
            return (self.page - 1) * self.size

    class Query(FlextModelsBase.ArbitraryTypesModel):
        """Query model for CQRS query operations."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations",
            },
        )
        tag: ClassVar[Literal["query"]] = "query"
        message_type: Annotated[
            Literal["query"],
            Field(
                default="query",
                frozen=True,
                description="Message type discriminator",
            ),
        ] = "query"
        filters: Annotated[
            t.Dict,
            Field(
                description="Filter values that restrict which records are returned by the query.",
                title="Query Filters",
                examples=[{"status": "active", "tenant": "acme"}],
            ),
        ] = Field(default_factory=t.Dict)
        pagination: FlextModelsCqrs.Pagination | t.Dict = Field(
            default_factory=t.Dict,
            description="Pagination settings controlling page number and page size for query results.",
            title="Pagination",
            examples=[{"page": 1, "size": 50}],
        )
        query_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Unique query identifier used for tracing and cache correlation.",
                title="Query Id",
                examples=["query_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: FlextUtilitiesGenerators.generate_prefixed_id(
                "query"
            )
        )
        query_type: Annotated[
            str | None,
            Field(
                default=None,
                description="Query type identifier for dispatcher routing.",
            ),
        ] = None

        @property
        def command_type(self) -> str | None:
            """Command type identifier (always None for queries)."""
            return None

        @property
        def event_type(self) -> str | None:
            """Event type identifier (always None for queries)."""
            return None

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: BaseModel | t.Dict | t.ScalarMapping | None,
        ) -> BaseModel:
            """Convert pagination to Pagination instance."""
            pagination_cls = FlextRuntime.resolve_nested_model_class(
                module_name=cls.__module__,
                qualname=cls.__qualname__,
                models_module_name="flext_core.models",
                attribute_name="Pagination",
                fallback=FlextModelsCqrs.Pagination,
            )
            normalized_input = FlextRuntime.normalize_model_input_mapping(v)
            if normalized_input is None:
                return pagination_cls()
            validate_result = r[BaseModel].create_from_callable(
                lambda: pagination_cls.model_validate(normalized_input),
            )
            if validate_result.is_failure:
                return pagination_cls()
            return validate_result.value

    class Handler(FlextModelsBase.ArbitraryTypesModel):
        """Handler configuration model with Builder pattern support."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Handler",
                "description": "CQRS handler configuration",
            },
        )
        handler_id: Annotated[
            t.NonEmptyStr,
            Field(description="Unique handler identifier"),
        ]
        handler_name: Annotated[
            t.NonEmptyStr,
            Field(description="Human-readable handler name"),
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
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Command timeout from c (default). Models use Config values in initialization.",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        max_command_retries: Annotated[
            int,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Maximum retry attempts from c (default). Models use Config values in initialization.",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        metadata: Annotated[
            FlextModelsBase.Metadata | None,
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
                handler_short_id = FlextUtilitiesGenerators.generate_prefixed_id(
                    "",
                    length=8,
                )
                self._data: t.Dict = t.Dict(
                    root={
                        "handler_type": handler_type,
                        c.FIELD_HANDLER_MODE: c.DEFAULT_HANDLER_MODE
                        if handler_type == c.HandlerType.COMMAND
                        else c.HANDLER_MODE_QUERY,
                        "handler_id": f"{handler_type}_handler_{handler_short_id}",
                        "handler_name": f"{handler_type.title()} Handler",
                        "command_timeout": c.DEFAULT_MAX_COMMAND_RETRIES,
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

            def with_metadata(self, metadata: FlextModelsBase.Metadata) -> Self:
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

    class Event(FlextModelsBase.ArbitraryTypesModel):
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
                description="Unique event identifier used for deduplication and observability.",
                title="Event Id",
                examples=["evt_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: FlextUtilitiesGenerators.generate_prefixed_id("evt")
        )
        data: Annotated[
            t.Dict,
            Field(
                description="Event payload data",
            ),
        ] = Field(default_factory=t.Dict)
        metadata: Annotated[
            t.Dict,
            Field(
                description="Event metadata (timestamps, correlation IDs, etc.)",
            ),
        ] = Field(default_factory=t.Dict)

    type FlextMessage = t.MessageUnion[Command, Query, Event]

    @staticmethod
    def parse_message(
        payload: p.Base | BaseModel | t.ContainerMapping,
    ) -> FlextMessage:
        """Parse a message payload into a FlextMessage instance."""
        _ = payload
        msg = "parse_message must be implemented by subclasses"
        raise NotImplementedError(msg)


__all__ = ["FlextModelsCqrs"]
