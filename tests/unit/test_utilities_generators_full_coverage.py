"""Behavior contract for public generator utilities in dispatch workflows.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from tests import c, m, tm, u


class TestsFlextCoreUtilitiesGenerators:
    """Tests for ``FlextCoreUtilitiesGenerators``."""

    @staticmethod
    def test_public_generators_build_dispatch_audit_metadata() -> None:
        """Test public generators build dispatch audit metadata."""
        request = m.Tests.DispatchRequest(
            command_name="sync-users",
            tenant="tenant-a",
            environment="prod",
        )

        audit = m.Tests.DispatchAudit(
            correlation_id=u.generate("correlation"),
            command_id=u.generate(
                kind=c.HandlerType.COMMAND,
                options=m.GenerateOptions(
                    include_timestamp=True,
                    separator="-",
                    parts=(request.command_name, request.tenant),
                    length=8,
                ),
            ),
            event_id=u.generate(
                kind=c.HandlerType.EVENT,
                options=m.GenerateOptions(
                    separator="-",
                    parts=(request.environment,),
                    length=6,
                ),
            ),
            replay_key=u.generate_prefixed_id("replay", length=6),
            opaque_id=u.generate(kind="id"),
            aggregate_id=u.generate(kind="aggregate"),
            emitted_at=u.generate_iso_timestamp(),
            generated_at=u.generate_datetime_utc(),
        )

        correlation_prefix, correlation_suffix = audit.correlation_id.split("_", 1)
        command_body, command_suffix = audit.command_id.removeprefix("cmd-").rsplit(
            "-",
            1,
        )
        command_timestamp, command_metadata = command_body.split("-", 1)
        event_body, event_suffix = audit.event_id.removeprefix("evt-").rsplit("-", 1)
        replay_prefix, replay_suffix = audit.replay_key.split("_", 1)

        tm.that(correlation_prefix, eq="corr")
        tm.that(len(correlation_suffix), eq=c.SHORT_UUID_LENGTH)
        tm.that(audit.command_id.startswith("cmd-"), truth=True)
        tm.that(command_timestamp.isdigit(), truth=True)
        tm.that(command_metadata, eq=f"{request.command_name}-{request.tenant}")
        tm.that(len(command_suffix), eq=8)
        tm.that(event_body, eq=request.environment)
        tm.that(len(event_suffix), eq=6)
        tm.that(replay_prefix, eq="replay")
        tm.that(len(replay_suffix), eq=6)
        tm.that(str(UUID(audit.opaque_id)), eq=audit.opaque_id)
        tm.that(str(UUID(audit.aggregate_id)), eq=audit.aggregate_id)
        tm.that(datetime.fromisoformat(audit.emitted_at).tzinfo, eq=UTC)
        tm.that(audit.generated_at.tzinfo, eq=UTC)

    @staticmethod
    def test_public_generators_accept_prefix_override_for_custom_batches() -> None:
        """Test public generators accept prefix override for custom batches."""
        batch_id = u.generate(
            kind="aggregate",
            options=m.GenerateOptions(
                prefix="agg",
                parts=("ldap", "delta"),
                separator="-",
                length=10,
            ),
        )

        tm.that(batch_id.startswith("agg-ldap-delta-"), truth=True)
        tm.that(len(batch_id.removeprefix("agg-ldap-delta-")), eq=10)

    @staticmethod
    def test_public_generators_cover_query_and_external_provider_ids() -> None:
        """Test public generators cover query and external provider ids."""
        audit = m.Tests.QueryAudit(
            request_id=u.generate(),
            explicit_id=u.generate(kind="id"),
            lookup_id=u.generate(
                kind=c.HandlerType.QUERY,
                options=m.GenerateOptions(parts=("status",), length=7),
            ),
            query_id=u.generate(
                kind=c.HandlerType.QUERY,
                options=m.GenerateOptions(length=5),
            ),
            event_channel_id=u.generate(
                kind=c.HandlerType.EVENT,
                options=m.GenerateOptions(separator="-", length=6),
            ),
            ulid_token=u.generate("ulid", options=m.GenerateOptions(length=12)),
            manual_id=u.generate_id(),
            external_token=u.generate_prefixed_id("", length=8),
        )

        lookup_prefix, lookup_name, lookup_suffix = audit.lookup_id.split("_")
        query_prefix, query_suffix = audit.query_id.split("_")

        tm.that(str(UUID(audit.request_id)), eq=audit.request_id)
        tm.that(str(UUID(audit.explicit_id)), eq=audit.explicit_id)
        tm.that(lookup_prefix, eq="qry")
        tm.that(lookup_name, eq="status")
        tm.that(len(lookup_suffix), eq=7)
        tm.that(query_prefix, eq="qry")
        tm.that(len(query_suffix), eq=5)
        tm.that(audit.event_channel_id.startswith("evt-"), truth=True)
        tm.that(len(audit.event_channel_id.removeprefix("evt-")), eq=6)
        tm.that(audit.ulid_token.isalnum(), truth=True)
        tm.that(len(audit.ulid_token), eq=12)
        tm.that(str(UUID(audit.manual_id)), eq=audit.manual_id)
        tm.that(len(audit.external_token), eq=8)

    @staticmethod
    def test_public_generators_cover_orchestration_identifier_families() -> None:
        """Test public generators cover orchestration identifier families."""
        audit = m.Tests.OrchestrationAudit(
            entity_id=u.generate(
                kind="entity",
                options=m.GenerateOptions(parts=("customer",), length=6),
            ),
            batch_id=u.generate(
                kind=c.ProcessingMode.BATCH,
                options=m.GenerateOptions(parts=("users",), length=7),
            ),
            transaction_id=u.generate(
                kind=c.ContextScope.TRANSACTION,
                options=m.GenerateOptions(parts=("sync", 42), length=9),
            ),
            saga_id=u.generate(
                kind=c.HandlerType.SAGA,
                options=m.GenerateOptions(parts=("ldap", "full"), length=5),
            ),
            timestamped_batch_id=u.generate(
                kind=c.ProcessingMode.BATCH,
                options=m.GenerateOptions(include_timestamp=True, length=4),
            ),
            explicit_uuid=u.generate(kind="uuid"),
        )

        entity_prefix, entity_name, entity_suffix = audit.entity_id.split("_")
        batch_prefix, batch_name, batch_suffix = audit.batch_id.split("_")
        transaction_body, transaction_suffix = audit.transaction_id.removeprefix(
            "txn_",
        ).rsplit("_", 1)
        saga_body, saga_suffix = audit.saga_id.removeprefix("saga_").rsplit("_", 1)
        timestamped_prefix, timestamped_body = audit.timestamped_batch_id.split("_", 1)
        timestamped_value, timestamped_suffix = timestamped_body.rsplit("_", 1)

        tm.that(entity_prefix, eq="ent")
        tm.that(entity_name, eq="customer")
        tm.that(len(entity_suffix), eq=6)
        tm.that(batch_prefix, eq=c.ProcessingMode.BATCH)
        tm.that(batch_name, eq="users")
        tm.that(len(batch_suffix), eq=7)
        tm.that(transaction_body, eq="sync_42")
        tm.that(len(transaction_suffix), eq=9)
        tm.that(saga_body, eq="ldap_full")
        tm.that(len(saga_suffix), eq=5)
        tm.that(timestamped_prefix, eq=c.ProcessingMode.BATCH)
        tm.that(timestamped_value.isdigit(), truth=True)
        tm.that(len(timestamped_suffix), eq=4)
        tm.that(str(UUID(audit.explicit_uuid)), eq=audit.explicit_uuid)
