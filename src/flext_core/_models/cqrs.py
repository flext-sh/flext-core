"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from flext_core import (
    FlextModelsBase as m,
    FlextRuntime as ur,
    FlextUtilitiesGenerators as ug,
    c,
    r,
    t,
)


class _CqrsPagination(m.FlexibleInternalModel):
    """Pagination model for query results.

    Defined at module level so it can be referenced in Query annotations
    without forward-reference issues (Pydantic can resolve it statically).
    Exposed as FlextModelsCqrs.Pagination.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={
            "title": "Pagination",
            "description": "Pagination model for query results with computed fields",
        },
    )
    page: Annotated[
        t.PositiveInt,
        Field(
            description="Page number (1-based indexing)",
            examples=[1, 2, 10, 100],
        ),
    ] = c.DEFAULT_RETRY_DELAY_SECONDS
    size: Annotated[
        t.PositiveInt,
        Field(
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


class FlextModelsCqrs:
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or
    directly via FlextModelsCqrs.*
    """

    class Command(m.ArbitraryTypesModel):
        """Base class for CQRS commands with validation."""

        tag: ClassVar[Literal["command"]] = "command"
        message_type: Annotated[
            Literal["command"],
            Field(
                frozen=True,
                description="Message type discriminator (always 'command')",
            ),
        ] = "command"
        command_type: Annotated[
            t.NonEmptyStr,
            Field(
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
            default_factory=lambda: ug.generate_prefixed_id("cmd"),
        )
        issuer_id: Annotated[
            t.NonEmptyStr | None,
            Field(
                description="Identity of the principal that issued this command.",
            ),
        ] = None

    Pagination = _CqrsPagination

    class Query(m.ArbitraryTypesModel):
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
                frozen=True,
                description="Message type discriminator",
            ),
        ] = "query"
        filters: Annotated[
            Mapping[str, t.Scalar],
            Field(
                description="Filter values that restrict which records are returned by the query.",
                title="Query Filters",
                examples=[{"status": "active", "tenant": "acme"}],
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))
        pagination: Annotated[
            _CqrsPagination,
            Field(
                description="Pagination settings controlling page number and page size for query results.",
                title="Pagination",
                examples=[{"page": 1, "size": 50}],
            ),
        ] = Field(default_factory=_CqrsPagination)
        query_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Unique query identifier used for tracing and cache correlation.",
                title="Query Id",
                examples=["query_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: ug.generate_prefixed_id(
                "query",
            ),
        )
        query_type: Annotated[
            str | None,
            Field(
                description="Query type identifier for dispatcher routing.",
            ),
        ] = None

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: BaseModel | Mapping[str, t.Scalar] | None,
        ) -> BaseModel:
            """Convert pagination to Pagination instance."""
            # Allow subclasses to override Pagination via class attribute,
            # fallback to the default _CqrsPagination
            pagination_cls = getattr(cls, "Pagination", _CqrsPagination)
            normalized_input = ur.normalize_model_input_mapping(v)
            if normalized_input is None:
                return pagination_cls()
            validate_result = r[BaseModel].create_from_callable(
                lambda: pagination_cls.model_validate(normalized_input),
            )
            if validate_result.failure:
                return pagination_cls()
            return validate_result.value

    class Handler(m.ArbitraryTypesModel):
        """Handler configuration model."""

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
                description="Handler type",
            ),
        ] = c.HandlerType.COMMAND
        handler_mode: Annotated[
            c.HandlerType,
            Field(
                description="Handler mode",
            ),
        ] = c.HandlerType.COMMAND
        command_timeout: Annotated[
            int,
            Field(
                description="Command timeout from c (default). Models use Config values in initialization.",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        max_command_retries: Annotated[
            int,
            Field(
                description="Maximum retry attempts from c (default). Models use Config values in initialization.",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        metadata: Annotated[
            m.Metadata | None,
            Field(
                description="Handler metadata (Pydantic model)",
            ),
        ] = None

    class Event(m.ArbitraryTypesModel):
        """Event model for CQRS event operations.

        Events represent domain events that occur as a result of command execution.
        They are immutable records of what happened in the system.
        """

        tag: ClassVar[Literal["event"]] = "event"
        message_type: Annotated[
            Literal["event"],
            Field(
                frozen=True,
                description="Message type discriminator (always 'event')",
            ),
        ] = "event"
        event_type: Annotated[t.NonEmptyStr, Field(description="Event type identifier")]

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
            default_factory=lambda: ug.generate_prefixed_id("evt"),
        )
        data: Annotated[
            Mapping[str, t.Scalar],
            Field(
                description="Event payload data",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))
        metadata: Annotated[
            Mapping[str, t.Scalar],
            Field(
                description="Event metadata (timestamps, correlation IDs, etc.)",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))

    type FlextMessage = t.MessageUnion[Command, Query, Event]


__all__: t.MutableSequenceOf[str] = ["FlextModelsCqrs"]
