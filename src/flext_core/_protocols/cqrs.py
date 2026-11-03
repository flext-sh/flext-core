# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/cqrs
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase
from flext_core._protocols._cqrs import FlextProtocolsCqrsFlextmodelscqrsPart01

if TYPE_CHECKING:
    from flext_core._constants import FlextConstantsCqrs, FlextConstantsInfrastructure
    from flext_core._typings import FlextTypesAnnotateds, FlextTypingBase


class FlextProtocolsCqrs(FlextProtocolsCqrsFlextmodelscqrsPart01):
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Command(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(
            self,
        ) -> Literal[FlextConstantsInfrastructure.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        @property
        def command_type(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Command type identifier."""
            ...

        @property
        def command_id(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Unique command identifier used for tracing and idempotency checks."""
            ...

        @property
        def issuer_id(self) -> FlextTypesAnnotateds.NonEmptyStr | None:
            """Identity of the principal that issued this command."""
            ...

    @runtime_checkable
    class Query(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(
            self,
        ) -> Literal[FlextConstantsInfrastructure.HandlerMode.QUERY]:
            """Message type discriminator."""
            ...

        @property
        def filters(self) -> Mapping[str, FlextTypingBase.Scalar]:
            """Filter values that restrict which records are returned by the query."""
            ...

        @property
        def pagination(self) -> FlextProtocolsCqrsFlextmodelscqrsPart01.Pagination:
            """Structural member projected from the owning model.

            Pagination settings controlling page number and page size for query results.
            """
            ...

        @property
        def query_id(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Unique query identifier used for tracing and cache correlation."""
            ...

        query_type: str | None

    @runtime_checkable
    class HandlerConfig(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def handler_id(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Unique handler identifier."""
            ...

        @property
        def handler_name(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Human-readable handler name."""
            ...

        @property
        def handler_type(self) -> FlextConstantsCqrs.HandlerType:
            """Handler type."""
            ...

        @property
        def handler_mode(self) -> FlextConstantsCqrs.HandlerType:
            """Handler mode."""
            ...

        @property
        def command_timeout(self) -> int:
            """Structural member projected from the owning model.

            Command timeout from c (default). Models use Config values in
            initialization.
            """
            ...

        @property
        def max_command_retries(self) -> int:
            """Structural member projected from the owning model.

            Maximum retry attempts from c (default). Models use Config values in
            initialization.
            """
            ...

        @property
        def metadata(self) -> FlextProtocolsBase.Metadata | None:
            """Handler metadata (Pydantic model)."""
            ...

    @runtime_checkable
    class Event(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[FlextConstantsCqrs.HandlerType.EVENT]:
            """Message type discriminator (always 'event')."""
            ...

        @property
        def event_type(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Event type identifier."""
            ...

        @property
        def aggregate_id(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """ID of the aggregate that generated this event."""
            ...

        @property
        def event_id(self) -> FlextTypesAnnotateds.NonEmptyStr:
            """Unique event identifier used for deduplication and observability."""
            ...

        @property
        def data(self) -> Mapping[str, FlextTypingBase.Scalar]:
            """Event payload data."""
            ...

        @property
        def metadata(self) -> Mapping[str, FlextTypingBase.Scalar]:
            """Event metadata (timestamps, correlation IDs, etc.)."""
            ...

    type FlextMessage = FlextTypesAnnotateds.MessageUnion[Command, Query, Event]


__all__: list[str] = [
    "FlextProtocolsCqrs",
]
