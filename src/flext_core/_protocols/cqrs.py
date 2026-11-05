# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/cqrs
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

from flext_core import c, t
from flext_core._protocols import FlextProtocolsBase
from flext_core._protocols._cqrs import FlextProtocolsCqrsFlextmodelscqrsPart01


class FlextProtocolsCqrs(FlextProtocolsCqrsFlextmodelscqrsPart01):
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Command(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.COMMAND]:
            """Message type discriminator (always 'command')."""
            ...

        command_type: t.NonEmptyStr
        command_id: t.NonEmptyStr
        issuer_id: t.NonEmptyStr | None

    @runtime_checkable
    class Query(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.QUERY]:
            """Message type discriminator."""
            ...

        filters: Mapping[str, t.Scalar]

        @property
        def pagination(self) -> FlextProtocolsCqrsFlextmodelscqrsPart01.Pagination:
            """Structural member projected from the owning model.

            Pagination settings controlling page number and page size for query results.
            """
            ...

        query_id: t.NonEmptyStr
        query_type: str | None

    @runtime_checkable
    class HandlerConfig(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def handler_id(self) -> t.NonEmptyStr:
            """Unique handler identifier."""
            ...

        @property
        def handler_name(self) -> t.NonEmptyStr:
            """Human-readable handler name."""
            ...

        @property
        def handler_type(self) -> c.HandlerType:
            """Handler type."""
            ...

        @property
        def handler_mode(self) -> c.HandlerType:
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
        def message_type(self) -> Literal[c.HandlerType.EVENT]:
            """Message type discriminator (always 'event')."""
            ...

        event_type: t.NonEmptyStr
        aggregate_id: t.NonEmptyStr
        event_id: t.NonEmptyStr
        data: Mapping[str, t.Scalar]
        metadata: Mapping[str, t.Scalar]

    type FlextMessage = t.MessageUnion[Command, Query, Event]


__all__: list[str] = [
    "FlextProtocolsCqrs",
]
