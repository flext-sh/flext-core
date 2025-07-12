"""Pipeline domain.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import Field
from pydantic import field_validator

from flext_core.domain.pydantic_base import DomainAggregateRoot
from flext_core.domain.pydantic_base import DomainEntity
from flext_core.domain.pydantic_base import DomainEvent
from flext_core.domain.pydantic_base import DomainValueObject
from flext_core.domain.types import EntityId


class ExecutionStatus(StrEnum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineId(DomainValueObject):
    """Strongly typed pipeline ID."""

    value: EntityId = Field(default_factory=uuid4)

    def __str__(self) -> str:
        """Convert pipeline ID to string.

        Returns:
            The pipeline ID as a string.

        """
        return str(self.value)


class PipelineName(DomainValueObject):
    """Validated pipeline name."""

    value: str = Field(max_length=100)

    @field_validator("value")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate pipeline name.

        Arguments:
            v: The pipeline name to validate.

        Raises:
            ValueError: If the pipeline name is empty.

        Returns:
            The validated pipeline name.

        """
        if not v or not v.strip():
            msg = "Pipeline name cannot be empty"
            raise ValueError(msg)
        return v.strip()

    def __str__(self) -> str:
        """Convert pipeline name to string.

        Returns:
            The pipeline name as a string.

        """
        return self.value


class ExecutionId(DomainValueObject):
    """Strongly typed execution ID."""

    value: EntityId = Field(default_factory=uuid4)

    def __str__(self) -> str:
        """Convert execution ID to string.

        Returns:
            The execution ID as a string.

        """
        return str(self.value)


# Domain Events
class PipelineCreated(DomainEvent):
    """Pipeline was created."""

    pipeline_id: PipelineId
    name: PipelineName


class PipelineExecuted(DomainEvent):
    """Pipeline was executed."""

    pipeline_id: PipelineId
    execution_id: ExecutionId


# Entities
class PipelineExecution(DomainEntity):
    """Pipeline execution entity."""

    execution_id: ExecutionId = Field(default_factory=ExecutionId)
    pipeline_id: PipelineId
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class Pipeline(DomainAggregateRoot):
    """Pipeline aggregate root."""

    pipeline_id: PipelineId = Field(default_factory=PipelineId)
    pipeline_name: PipelineName
    pipeline_description: str = ""
    pipeline_is_active: bool = True

    def create(self) -> None:
        """Create pipeline and emit event."""
        self.add_event(
            PipelineCreated(pipeline_id=self.pipeline_id, name=self.pipeline_name),
        )
        self.updated_at = datetime.now(UTC)

    def execute(self) -> PipelineExecution:
        """Execute pipeline and return execution.

        Returns:
            The pipeline execution.

        """
        execution = PipelineExecution(
            pipeline_id=self.pipeline_id,
        )
        execution.execution_status = ExecutionStatus.RUNNING
        execution.started_at = datetime.now(UTC)

        self.add_event(
            PipelineExecuted(
                pipeline_id=self.pipeline_id,
                execution_id=execution.execution_id,
            ),
        )
        self.updated_at = datetime.now(UTC)
        return execution

    def deactivate(self) -> None:
        """Deactivate pipeline."""
        self.pipeline_is_active = False
        self.updated_at = datetime.now(UTC)


# Models are automatically rebuilt by Pydantic v2
