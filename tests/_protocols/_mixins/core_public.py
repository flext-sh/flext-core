# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/core_public
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols, t

from flext_core import c

if TYPE_CHECKING:
    from flext_tests import p


class TestsFlextProtocolsMixinsCorePublic:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class RegistryQuery(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> Literal[c.HandlerMode.QUERY]:
            """Message type discriminator."""
            ...

        @property
        def filters(self) -> Mapping[str, t.Scalar]:
            """Structural member projected from the owning model.

            Filter values that restrict which records are returned by the query.
            """
            ...

        @property
        def pagination(self) -> p.Pagination:
            """Structural member projected from the owning model.

            Pagination settings controlling page number and page size for query
            results.
            """
            ...

        @property
        def query_id(self) -> t.NonEmptyStr:
            """Unique query identifier used for tracing and cache correlation."""
            ...

        query_type: str | None

    @runtime_checkable
    class HandlerObservations(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        handled: MutableSequence[str]

    @runtime_checkable
    class BinaryContract(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def payload(self) -> bytes:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class DispatchRequest(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def command_name(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def tenant(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def environment(self) -> str:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class DispatchAudit(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def correlation_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def command_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def event_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def replay_key(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def opaque_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def aggregate_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def emitted_at(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def generated_at(self) -> datetime:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class QueryAudit(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def request_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def explicit_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def lookup_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def query_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def event_channel_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def ulid_token(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def manual_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def external_token(self) -> str:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class OrchestrationAudit(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def entity_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def batch_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def transaction_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def saga_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def timestamped_batch_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def explicit_uuid(self) -> str:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class DispatchEnvelope(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def command_name(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def correlation_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def attempt_count(self) -> int:
            """Structural member projected from the owning model."""
            ...

        @property
        def tags(self) -> Sequence[str]:
            """Structural member projected from the owning model."""
            ...

        @property
        def started_at(self) -> datetime:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class ManifestSnapshot(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def app_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def normalized_key(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def manifest_path(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def manifest_content(self) -> str:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class BootstrapSnapshot(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def env_file(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def process_environment(self) -> Mapping[str, str]:
            """Structural member projected from the owning model."""
            ...

        @property
        def log_level(self) -> str:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class PublicPayload(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def raw_name(self) -> str:
            """Structural member projected from the owning model."""
            ...

        visits: int

        @property
        def label(self) -> str:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class KwargsComputedPayload(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        raw: str

        @property
        def label(self) -> str:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class PlainSerializedPayload(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        name: str


__all__: list[str] = [
    "TestsFlextProtocolsMixinsCorePublic",
]
