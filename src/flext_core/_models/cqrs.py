"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

import flext_core
from flext_core._models import FlextModelsBase
from flext_core._models._cqrs import FlextModelsCqrsPart01


class FlextModelsCqrs(FlextModelsCqrsPart01):
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or
    directly via FlextModelsCqrs.*
    """

    class Command(FlextModelsBase.ArbitraryTypesModel):
        """Base class for CQRS commands with validation."""

        tag: ClassVar[Literal[flext_core.c.HandlerMode.COMMAND]] = (
            flext_core.c.HandlerMode.COMMAND
        )
        message_type: Annotated[
            Literal[flext_core.c.HandlerMode.COMMAND],
            Field(
                frozen=True,
                description="Message type discriminator (always 'command')",
            ),
        ] = flext_core.c.HandlerMode.COMMAND
        command_type: Annotated[
            flext_core.t.NonEmptyStr,
            Field(description="Command type identifier"),
        ] = flext_core.c.DEFAULT_COMMAND_TYPE
        command_id: Annotated[
            flext_core.t.NonEmptyStr,
            Field(
                description=(
                    "Unique command identifier used for tracing and idempotency checks."
                ),
                title="Command Id",
                examples=["cmd_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: (
                "cmd_" + str(uuid.uuid4())[: flext_core.c.SHORT_UUID_LENGTH]
            ),
        )
        issuer_id: Annotated[
            flext_core.t.NonEmptyStr | None,
            Field(description="Identity of the principal that issued this command."),
        ] = None

    class Query(FlextModelsBase.ArbitraryTypesModel):
        """Query model for CQRS query operations."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations",
            },
        )
        tag: ClassVar[Literal[flext_core.c.HandlerMode.QUERY]] = (
            flext_core.c.HandlerMode.QUERY
        )
        message_type: Annotated[
            Literal[flext_core.c.HandlerMode.QUERY],
            Field(frozen=True, description="Message type discriminator"),
        ] = flext_core.c.HandlerMode.QUERY
        filters: Annotated[
            Mapping[str, flext_core.t.Scalar],
            Field(
                description=(
                    "Filter values that restrict"
                    " which records are returned by the query."
                ),
                title="Query Filters",
                examples=[{"status": "active", "tenant": "acme"}],
            ),
        ] = Field(
            default_factory=lambda: MappingProxyType[str, flext_core.t.Scalar]({}),
            validate_default=True,
        )
        pagination: Annotated[
            FlextModelsCqrsPart01.Pagination,
            Field(
                description=(
                    "Pagination settings controlling page number"
                    " and page size for query results."
                ),
                title="Pagination",
                examples=[
                    {
                        "page": flext_core.c.FIRST_PAGE_INDEX,
                        "size": flext_core.c.DEFAULT_PAGE_SIZE,
                    }
                ],
            ),
        ] = Field(default_factory=FlextModelsCqrsPart01.Pagination)
        query_id: Annotated[
            flext_core.t.NonEmptyStr,
            Field(
                description=(
                    "Unique query identifier used for tracing and cache correlation."
                ),
                title="Query Id",
                examples=["query_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: (
                f"{flext_core.c.HandlerMode.QUERY}_"
                + str(uuid.uuid4())[: flext_core.c.SHORT_UUID_LENGTH]
            ),
        )
        query_type: Annotated[
            str | None,
            Field(description="Query type identifier for dispatcher routing."),
        ] = None

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: BaseModel | Mapping[str, flext_core.t.Scalar] | None,
        ) -> BaseModel:
            """Convert pagination to Pagination instance.

            Returns:
                The resulting ``BaseModel``.

            """
            # A query subclass may declare its own nested Pagination model;
            # otherwise the namespace Pagination model applies.
            pagination_cls: type[BaseModel] = getattr(
                cls,
                "Pagination",
                FlextModelsCqrsPart01.Pagination,
            )
            if v is None:
                return pagination_cls()
            return pagination_cls.model_validate(
                v.model_dump(mode="json") if isinstance(v, BaseModel) else v
            )

    class HandlerConfig(FlextModelsBase.ArbitraryTypesModel):
        """Handler configuration model."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            json_schema_extra={
                "title": "Handler",
                "description": "CQRS handler configuration",
            },
        )
        handler_id: Annotated[
            flext_core.t.NonEmptyStr,
            Field(frozen=True, description="Unique handler identifier"),
        ]
        handler_name: Annotated[
            flext_core.t.NonEmptyStr,
            Field(frozen=True, description="Human-readable handler name"),
        ]
        handler_type: Annotated[
            flext_core.c.HandlerType, Field(description="Handler type")
        ] = flext_core.c.HandlerType.COMMAND
        handler_mode: Annotated[
            flext_core.c.HandlerType, Field(description="Handler mode")
        ] = flext_core.c.HandlerType.COMMAND
        command_timeout: Annotated[
            int,
            Field(
                description=(
                    "Command timeout from c (default)."
                    " Models use Config values in initialization."
                ),
            ),
        ] = flext_core.c.DEFAULT_MAX_COMMAND_RETRIES
        max_command_retries: Annotated[
            int,
            Field(
                description=(
                    "Maximum retry attempts from c (default)."
                    " Models use Config values in initialization."
                ),
            ),
        ] = flext_core.c.DEFAULT_MAX_COMMAND_RETRIES
        metadata: Annotated[
            FlextModelsBase.Metadata | None,
            Field(description="Handler metadata (Pydantic model)"),
        ] = None

    class Event(FlextModelsBase.ArbitraryTypesModel):
        """Event model for CQRS event operations.

        Events represent domain events that occur as a result of command execution.
        They are immutable records of what happened in the system.
        """

        tag: ClassVar[Literal[flext_core.c.HandlerType.EVENT]] = (
            flext_core.c.HandlerType.EVENT
        )
        message_type: Annotated[
            Literal[flext_core.c.HandlerType.EVENT],
            Field(
                frozen=True,
                description="Message type discriminator (always 'event')",
            ),
        ] = flext_core.c.HandlerType.EVENT
        event_type: Annotated[
            flext_core.t.NonEmptyStr, Field(description="Event type identifier")
        ]

        aggregate_id: Annotated[
            flext_core.t.NonEmptyStr,
            Field(description="ID of the aggregate that generated this event"),
        ]
        event_id: Annotated[
            flext_core.t.NonEmptyStr,
            Field(
                description=(
                    "Unique event identifier used for deduplication and observability."
                ),
                title="Event Id",
                examples=["evt_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: (
                "evt_" + str(uuid.uuid4())[: flext_core.c.SHORT_UUID_LENGTH]
            ),
        )
        data: Annotated[
            Mapping[str, flext_core.t.Scalar],
            Field(description="Event payload data"),
        ] = Field(
            default_factory=lambda: MappingProxyType[str, flext_core.t.Scalar]({}),
            validate_default=True,
        )
        metadata: Annotated[
            Mapping[str, flext_core.t.Scalar],
            Field(description="Event metadata (timestamps, correlation IDs, etc.)"),
        ] = Field(
            default_factory=lambda: MappingProxyType[str, flext_core.t.Scalar]({}),
            validate_default=True,
        )

    type FlextMessage = flext_core.t.MessageUnion[Command, Query, Event]


__all__: list[str] = ["FlextModelsCqrs"]
