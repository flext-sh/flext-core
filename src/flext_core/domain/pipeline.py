"""Pipeline domain - COMPLETE implementation.

All pipeline-related domain logic in ONE place.
Zero duplication, maximum performance.
"""

from __future__ import annotations

import uuid as _uuid_module
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from flext_core.domain.core import (
    AggregateRoot,
    DomainEvent,
    Entity,
    ValueObject,
)

if TYPE_CHECKING:
    from uuid import UUID
else:
    from uuid import UUID


def _generate_uuid() -> Any:  # noqa: ANN401
    """Generate UUID without importing at module level."""
    return _uuid_module.uuid4()


class ExecutionStatus(StrEnum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineId(ValueObject):
    """Strongly typed pipeline ID."""

    value: UUID = Field(default_factory=_generate_uuid)

    def __hash__(self) -> int:
        """Generate hash based on UUID value."""
        return hash(self.value)

    def __str__(self) -> str:
        """Return string representation of pipeline ID."""
        return str(self.value)


class PipelineName(ValueObject):
    """Validated pipeline name."""

    value: str = Field(min_length=1, max_length=100)

    @field_validator("value")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate pipeline name is not empty."""
        if not v.strip():
            msg = "Pipeline name cannot be empty"
            raise ValueError(msg)
        return v.strip()

    def __str__(self) -> str:
        """Return string representation of pipeline name."""
        return self.value


class ExecutionId(ValueObject):
    """Strongly typed execution ID."""

    value: UUID = Field(default_factory=_generate_uuid)

    def __hash__(self) -> int:
        """Generate hash based on UUID value."""
        return hash(self.value)

    def __str__(self) -> str:
        """Return string representation of execution ID."""
        return str(self.value)


# Domain Events
class PipelineCreated(DomainEvent):
    """Pipeline was created."""

    pipeline_id: PipelineId
    name: PipelineName


class PipelineExecuted(DomainEvent):
    """Pipeline execution started."""

    pipeline_id: PipelineId
    execution_id: ExecutionId


# Entities
class PipelineExecution(Entity[Any]):
    """Pipeline execution entity."""

    id: ExecutionId = Field(default_factory=ExecutionId)
    pipeline_id: PipelineId
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None

    def complete(self, *, success: bool, result: dict[str, Any] | None = None) -> None:
        """Complete execution."""
        self.completed_at = datetime.now(UTC)
        self.status = ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED
        if result:
            self.result = result
        self.updated_at = self.completed_at

    def fail(self, error: str) -> None:
        """Mark as failed."""
        self.completed_at = datetime.now(UTC)
        self.status = ExecutionStatus.FAILED
        self.error_message = error
        self.updated_at = self.completed_at


class Pipeline(AggregateRoot[Any]):
    """Pipeline aggregate root."""

    id: PipelineId = Field(default_factory=PipelineId)
    name: PipelineName
    description: str = ""
    is_active: bool = True

    def create(self) -> None:
        """Create pipeline and emit event."""
        self.add_event(PipelineCreated(pipeline_id=self.id, name=self.name))
        self.updated_at = datetime.now(UTC)

    def execute(self) -> PipelineExecution:
        """Start execution."""
        execution = PipelineExecution(
            pipeline_id=self.id,
            status=ExecutionStatus.RUNNING,
        )
        self.add_event(
            PipelineExecuted(
                pipeline_id=self.id,
                execution_id=execution.id,
            ),
        )
        self.updated_at = datetime.now(UTC)
        return execution

    def deactivate(self) -> None:
        """Deactivate pipeline."""
        self.is_active = False
        self.updated_at = datetime.now(UTC)


# Rebuild models to ensure UUID is resolved
PipelineId.model_rebuild()
ExecutionId.model_rebuild()
PipelineExecution.model_rebuild()
Pipeline.model_rebuild()
