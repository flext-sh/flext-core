"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core._models import FlextModelsBase
from flext_core._models._cqrs import flextmodelscqrs_part_01
from flext_core._utilities import FlextUtilitiesGenerators
from flext_core.constants import c
from flext_core.typings import t


class FlextModelsCqrs(flextmodelscqrs_part_01.FlextModelsCqrsPart01):
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or
    directly via FlextModelsCqrs.*
    """

    class Command(FlextModelsBase.ArbitraryTypesModel):
        """Base class for CQRS commands with validation."""

        tag: ClassVar[Literal[c.HandlerMode.COMMAND]] = c.HandlerMode.COMMAND
        message_type: Annotated[
            Literal[c.HandlerMode.COMMAND],
            Field(
                frozen=True,
                description="Message type discriminator (always 'command')",
            ),
        ] = c.HandlerMode.COMMAND
        command_type: Annotated[
            t.NonEmptyStr,
            Field(description="Command type identifier"),
        ] = c.DEFAULT_COMMAND_TYPE
        command_id: Annotated[
            t.NonEmptyStr,
            Field(
                description=(
                    "Unique command identifier used for tracing and idempotency checks."
                ),
                title="Command Id",
                examples=["cmd_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: FlextUtilitiesGenerators.generate_prefixed_id(
                "cmd",
            ),
        )
        issuer_id: Annotated[
            t.NonEmptyStr | None,
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
        tag: ClassVar[Literal[c.HandlerMode.QUERY]] = c.HandlerMode.QUERY
        message_type: Annotated[
            Literal[c.HandlerMode.QUERY],
            Field(frozen=True, description="Message type discriminator"),
        ] = c.HandlerMode.QUERY
        filters: Annotated[
            t.MappingKV[str, t.Scalar],
            Field(
                description=(
                    "Filter values that restrict"
                    " which records are returned by the query."
                ),
                title="Query Filters",
                examples=[{"status": "active", "tenant": "acme"}],
            ),
        ] = Field(
            default_factory=lambda: MappingProxyType[str, t.Scalar]({}),
            validate_default=True,
        )
        pagination: Annotated[
            flextmodelscqrs_part_01.FlextModelsCqrsPart01.Pagination,
            Field(
                description=(
                    "Pagination settings controlling page number"
                    " and page size for query results."
                ),
                title="Pagination",
                examples=[{"page": c.FIRST_PAGE_INDEX, "size": c.DEFAULT_PAGE_SIZE}],
            ),
        ] = Field(
            default_factory=flextmodelscqrs_part_01.FlextModelsCqrsPart01.Pagination
        )
        query_id: Annotated[
            t.NonEmptyStr,
            Field(
                description=(
                    "Unique query identifier used for tracing and cache correlation."
                ),
                title="Query Id",
                examples=["query_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: FlextUtilitiesGenerators.generate_prefixed_id(
                c.HandlerMode.QUERY,
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
            v: BaseModel | t.MappingKV[str, t.Scalar] | None,
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
                flextmodelscqrs_part_01.FlextModelsCqrsPart01.Pagination,
            )
            if v is None:
                return pagination_cls()
            return pagination_cls.model_validate(
                v.model_dump(mode="json") if isinstance(v, BaseModel) else v
            )

    class Handler(FlextModelsBase.ArbitraryTypesModel):
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
        handler_type: Annotated[c.HandlerType, Field(description="Handler type")] = (
            c.HandlerType.COMMAND
        )
        handler_mode: Annotated[c.HandlerType, Field(description="Handler mode")] = (
            c.HandlerType.COMMAND
        )
        command_timeout: Annotated[
            int,
            Field(
                description=(
                    "Command timeout from c (default)."
                    " Models use Config values in initialization."
                ),
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        max_command_retries: Annotated[
            int,
            Field(
                description=(
                    "Maximum retry attempts from c (default)."
                    " Models use Config values in initialization."
                ),
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        metadata: Annotated[
            FlextModelsBase.Metadata | None,
            Field(description="Handler metadata (Pydantic model)"),
        ] = None

    class Event(FlextModelsBase.ArbitraryTypesModel):
        """Event model for CQRS event operations.

        Events represent domain events that occur as a result of command execution.
        They are immutable records of what happened in the system.
        """

        tag: ClassVar[Literal[c.HandlerType.EVENT]] = c.HandlerType.EVENT
        message_type: Annotated[
            Literal[c.HandlerType.EVENT],
            Field(
                frozen=True,
                description="Message type discriminator (always 'event')",
            ),
        ] = c.HandlerType.EVENT
        event_type: Annotated[t.NonEmptyStr, Field(description="Event type identifier")]

        aggregate_id: Annotated[
            t.NonEmptyStr,
            Field(description="ID of the aggregate that generated this event"),
        ]
        event_id: Annotated[
            t.NonEmptyStr,
            Field(
                description=(
                    "Unique event identifier used for deduplication and observability."
                ),
                title="Event Id",
                examples=["evt_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: FlextUtilitiesGenerators.generate_prefixed_id(
                "evt",
            ),
        )
        data: Annotated[
            t.MappingKV[str, t.Scalar],
            Field(description="Event payload data"),
        ] = Field(
            default_factory=lambda: MappingProxyType[str, t.Scalar]({}),
            validate_default=True,
        )
        metadata: Annotated[
            t.MappingKV[str, t.Scalar],
            Field(description="Event metadata (timestamps, correlation IDs, etc.)"),
        ] = Field(
            default_factory=lambda: MappingProxyType[str, t.Scalar]({}),
            validate_default=True,
        )

    type FlextMessage = t.MessageUnion[Command, Query, Event]


__all__: list[str] = ["FlextModelsCqrs"]
