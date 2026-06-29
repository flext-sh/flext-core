"""Core public model helpers."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Self

from flext_core import m, u
from tests.typings import t


class TestsFlextModelsCorePublicMixin:
    """Core public model helpers."""

    class DispatchRequest(m.BaseModel):
        """Request shape used by generator behavior tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        command_name: str
        tenant: str
        environment: str

    class DispatchAudit(m.BaseModel):
        """Dispatch audit payload used by generator behavior tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        correlation_id: str
        command_id: str
        event_id: str
        replay_key: str
        opaque_id: str
        aggregate_id: str
        emitted_at: str
        generated_at: datetime

    class QueryAudit(m.BaseModel):
        """Query audit payload used by generator behavior tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        request_id: str
        explicit_id: str
        lookup_id: str
        query_id: str
        event_channel_id: str
        ulid_token: str
        manual_id: str
        external_token: str

    class OrchestrationAudit(m.BaseModel):
        """Orchestration audit payload used by generator behavior tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        entity_id: str
        batch_id: str
        transaction_id: str
        saga_id: str
        timestamped_batch_id: str
        explicit_uuid: str

    class DispatchEnvelope(m.BaseModel):
        """Normalized dispatch metadata payload used by type-guard tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        command_name: str
        correlation_id: str
        attempt_count: int
        tags: t.StrSequence
        started_at: datetime

    class ManifestSnapshot(m.BaseModel):
        """Manifest snapshot used by public text-helper tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        app_id: str
        normalized_key: str
        manifest_path: str
        manifest_content: str

    class BootstrapSnapshot(m.BaseModel):
        """Bootstrap snapshot used by public settings-helper tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        env_file: str
        process_environment: dict[str, str]
        log_level: str

    class PublicPayload(m.BaseModel):
        """Payload model used to exercise the public Pydantic facade."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(populate_by_name=True)

        raw_name: str = u.Field(alias="rawName")
        visits: int = 0
        _events: list[str] = u.PrivateAttr(default_factory=list)

        @u.field_validator("raw_name")
        @classmethod
        def normalize_name(cls, value: str) -> str:
            return value.strip().title()

        @u.model_validator(mode="after")
        def record_validation(self) -> Self:
            self._events.append("validated")
            return self

        @u.computed_field()
        @property
        def label(self) -> str:
            return f"{self.raw_name}:{self.visits}"

        @u.field_serializer("visits")
        def serialize_visits(self, value: int) -> str:
            return f"{value} visits"


__all__: list[str] = ["TestsFlextModelsCorePublicMixin"]
