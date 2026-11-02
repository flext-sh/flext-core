# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/cqrs
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import c, t

if TYPE_CHECKING:
    from flext_core import p


class FlextProtocolsCqrs:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Command(Protocol):
        """Structural data contract projected from the owning model."""

        message_type: Literal[c.HandlerMode.COMMAND]
        command_type: t.NonEmptyStr
        command_id: t.NonEmptyStr
        issuer_id: t.NonEmptyStr | None

    @runtime_checkable
    class Query(Protocol):
        """Structural data contract projected from the owning model."""

        message_type: Literal[c.HandlerMode.QUERY]
        filters: Mapping[str, t.Scalar]
        pagination: p.Pagination
        query_id: t.NonEmptyStr
        query_type: str | None

    @runtime_checkable
    class HandlerConfig(Protocol):
        """Structural data contract projected from the owning model."""

        handler_id: t.NonEmptyStr
        handler_name: t.NonEmptyStr
        handler_type: c.HandlerType
        handler_mode: c.HandlerType
        command_timeout: int
        max_command_retries: int
        metadata: p.Metadata | None

    @runtime_checkable
    class Event(Protocol):
        """Structural data contract projected from the owning model."""

        message_type: Literal[c.HandlerType.EVENT]
        event_type: t.NonEmptyStr
        aggregate_id: t.NonEmptyStr
        event_id: t.NonEmptyStr
        data: Mapping[str, t.Scalar]
        metadata: Mapping[str, t.Scalar]


__all__: list[str] = [
    "FlextProtocolsCqrs",
]
