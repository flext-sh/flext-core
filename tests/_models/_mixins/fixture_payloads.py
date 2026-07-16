"""Fixture payload model helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_core import m

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextModelsFixturePayloadsMixin:
    """Fixture payload model helpers."""

    class UserProfileDict(m.BaseModel):
        """User profile for property-based testing."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        id: str
        name: str
        email: str

    class SettingsTestCaseDict(m.BaseModel):
        """Configuration test case."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        domain: str
        port: int
        timeout: float
        debug: bool

    class PerformanceMetricsDict(m.BaseModel):
        """Performance metrics from testing."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        total_operations: int
        time_elapsed: float
        ops_per_second: float
        memory_peak_mb: float

    class StressTestResultDict(m.BaseModel):
        """Result from stress testing."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        iterations: int
        success_count: int
        failure_count: int
        average_time_ms: float

    class AsyncPayloadDict(m.BaseModel):
        """Async event payload."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        data: str
        status: str

    class AsyncTestDataDict(m.BaseModel):
        """Async test data."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        event_type: str
        timestamp: str
        payload: t.MappingKV[
            str,
            TestsFlextModelsFixturePayloadsMixin.AsyncPayloadDict,
        ]

    class UserPayloadDict(m.BaseModel):
        """User command payload."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        username: str
        email: str

    class UpdateFieldDict(m.BaseModel):
        """Individual update field."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        field_name: str
        new_value: t.Primitives

    class UpdatePayloadDict(m.BaseModel):
        """Update command payload."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        target_user_id: str
        updates: t.MappingKV[
            str,
            TestsFlextModelsFixturePayloadsMixin.UpdateFieldDict,
        ]

    class UserDataDict(m.BaseModel):
        """User data response."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        id: str
        username: str
        email: str
        status: str

    class UpdateResultDict(m.BaseModel):
        """Update operation result."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        user_id: str
        updated_fields: t.StrSequence
        update_count: int

    class CommandPayloadDict(m.BaseModel):
        """Generic command payload."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        id: str = ""
        username: str = ""
        email: str = ""


__all__: list[str] = ["TestsFlextModelsFixturePayloadsMixin"]
