"""Behavior contract for public generator utilities in dispatch workflows."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from flext_core import u
from tests.constants import c
from tests.models import m


class TestsFlextUtilitiesGenerators:
    def test_public_generators_build_dispatch_audit_metadata(self) -> None:
        request = m.Tests.DispatchRequest(
            command_name="sync-users",
            tenant="tenant-a",
            environment="prod",
        )

        audit = m.Tests.DispatchAudit(
            correlation_id=u.generate("correlation"),
            command_id=u.generate(
                kind=c.HandlerType.COMMAND,
                options=u.GenerateOptions(
                    include_timestamp=True,
                    separator="-",
                    parts=(request.command_name, request.tenant),
                    length=8,
                ),
            ),
            event_id=u.generate(
                kind=c.HandlerType.EVENT,
                options=u.GenerateOptions(
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

        assert correlation_prefix == "corr"
        assert len(correlation_suffix) == c.SHORT_UUID_LENGTH
        assert audit.command_id.startswith("cmd-")
        assert command_timestamp.isdigit()
        assert command_metadata == f"{request.command_name}-{request.tenant}"
        assert len(command_suffix) == 8
        assert event_body == request.environment
        assert len(event_suffix) == 6
        assert replay_prefix == "replay"
        assert len(replay_suffix) == 6
        assert str(UUID(audit.opaque_id)) == audit.opaque_id
        assert str(UUID(audit.aggregate_id)) == audit.aggregate_id
        assert datetime.fromisoformat(audit.emitted_at).tzinfo == UTC
        assert audit.generated_at.tzinfo == UTC

    def test_public_generators_accept_prefix_override_for_custom_batches(self) -> None:
        batch_id = u.generate(
            kind="aggregate",
            options=u.GenerateOptions(
                prefix="agg",
                parts=("ldap", "delta"),
                separator="-",
                length=10,
            ),
        )

        assert batch_id.startswith("agg-ldap-delta-")
        assert len(batch_id.removeprefix("agg-ldap-delta-")) == 10

    def test_public_generators_cover_query_and_external_provider_ids(self) -> None:
        audit = m.Tests.QueryAudit(
            request_id=u.generate(),
            explicit_id=u.generate(kind="id"),
            lookup_id=u.generate(
                kind=c.HandlerType.QUERY,
                options=u.GenerateOptions(parts=("status",), length=7),
            ),
            query_id=u.generate(
                kind=c.HandlerType.QUERY,
                options=u.GenerateOptions(length=5),
            ),
            event_channel_id=u.generate(
                kind=c.HandlerType.EVENT,
                options=u.GenerateOptions(separator="-", length=6),
            ),
            ulid_token=u.generate("ulid", options=u.GenerateOptions(length=12)),
            manual_id=u.generate_id(),
            external_token=u.generate_prefixed_id("", length=8),
        )

        lookup_prefix, lookup_name, lookup_suffix = audit.lookup_id.split("_")
        query_prefix, query_suffix = audit.query_id.split("_")

        assert str(UUID(audit.request_id)) == audit.request_id
        assert str(UUID(audit.explicit_id)) == audit.explicit_id
        assert lookup_prefix == "qry"
        assert lookup_name == "status"
        assert len(lookup_suffix) == 7
        assert query_prefix == "qry"
        assert len(query_suffix) == 5
        assert audit.event_channel_id.startswith("evt-")
        assert len(audit.event_channel_id.removeprefix("evt-")) == 6
        assert audit.ulid_token.isalnum()
        assert len(audit.ulid_token) == 12
        assert str(UUID(audit.manual_id)) == audit.manual_id
        assert len(audit.external_token) == 8

    def test_public_generators_cover_orchestration_identifier_families(self) -> None:
        audit = m.Tests.OrchestrationAudit(
            entity_id=u.generate(
                kind="entity",
                options=u.GenerateOptions(parts=("customer",), length=6),
            ),
            batch_id=u.generate(
                kind="batch",
                options=u.GenerateOptions(parts=("users",), length=7),
            ),
            transaction_id=u.generate(
                kind="transaction",
                options=u.GenerateOptions(parts=("sync", 42), length=9),
            ),
            saga_id=u.generate(
                kind="saga",
                options=u.GenerateOptions(parts=("ldap", "full"), length=5),
            ),
            timestamped_batch_id=u.generate(
                kind="batch",
                options=u.GenerateOptions(include_timestamp=True, length=4),
            ),
            explicit_uuid=u.generate(kind="uuid"),
        )

        entity_prefix, entity_name, entity_suffix = audit.entity_id.split("_")
        batch_prefix, batch_name, batch_suffix = audit.batch_id.split("_")
        transaction_body, transaction_suffix = audit.transaction_id.removeprefix(
            "txn_"
        ).rsplit("_", 1)
        saga_body, saga_suffix = audit.saga_id.removeprefix("saga_").rsplit("_", 1)
        timestamped_prefix, timestamped_body = audit.timestamped_batch_id.split("_", 1)
        timestamped_value, timestamped_suffix = timestamped_body.rsplit("_", 1)

        assert entity_prefix == "ent"
        assert entity_name == "customer"
        assert len(entity_suffix) == 6
        assert batch_prefix == c.ProcessingMode.BATCH
        assert batch_name == "users"
        assert len(batch_suffix) == 7
        assert transaction_body == "sync_42"
        assert len(transaction_suffix) == 9
        assert saga_body == "ldap_full"
        assert len(saga_suffix) == 5
        assert timestamped_prefix == c.ProcessingMode.BATCH
        assert timestamped_value.isdigit()
        assert len(timestamped_suffix) == 4
        assert str(UUID(audit.explicit_uuid)) == audit.explicit_uuid
