"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, ClassVar, Literal

from pydantic import (
    ConfigDict,
    Field,
)

from flext_core import (
    FlextConstants as c,
    FlextModelsBase as m,
    FlextTypes as t,
    FlextUtilitiesGenerators as ug,
)

from .flextmodelscqrs_part_01 import (
    FlextModelsCqrs as FlextModelsCqrsPart01,
)


class FlextModelsCqrs(FlextModelsCqrsPart01):
    Command = FlextModelsCqrsPart01.Command
    Query = FlextModelsCqrsPart01.Query

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
            t.MappingKV[str, t.Scalar],
            Field(
                description="Event payload data",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))
        metadata: Annotated[
            t.MappingKV[str, t.Scalar],
            Field(
                description="Event metadata (timestamps, correlation IDs, etc.)",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))

    type FlextMessage = t.MessageUnion[Command, Query, Event]


__all__: list[str] = ["FlextModelsCqrs"]
