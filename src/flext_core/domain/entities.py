"""Core domain entities for FLEXT framework."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .base import (
    PipelineId,
    create_pipeline_id,
)


class PipelineName(BaseModel):
    """Pipeline name value object with validation."""

    value: str = Field(..., min_length=1, max_length=100, description="Pipeline name")

    @field_validator('value')
    @classmethod
    def validate_name(cls, v):
        """Validate pipeline name format."""
        if not v.replace('-', '').replace('_', '').replace('.', '').isalnum():
            raise ValueError('Pipeline name must contain only alphanumeric characters, hyphens, underscores, and dots')
        return v

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"PipelineName('{self.value}')"


class ExecutionStatus(str, Enum):
    """Pipeline execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

    def is_terminal(self) -> bool:
        """Check if status is terminal (execution finished)."""
        return self in [self.SUCCESS, self.FAILED, self.CANCELLED, self.TIMEOUT]

    def is_active(self) -> bool:
        """Check if execution is currently active."""
        return self in [self.PENDING, self.RUNNING]


class Pipeline(BaseModel):
    """Core pipeline entity."""

    id: PipelineId = Field(default_factory=create_pipeline_id, description="Unique pipeline identifier")
    name: PipelineName = Field(..., description="Pipeline name")
    description: str = Field(default="", max_length=500, description="Pipeline description")
    configuration: dict[str, Any] = Field(default_factory=dict, description="Pipeline configuration")
    tags: list[str] = Field(default_factory=list, description="Pipeline tags")
    is_active: bool = Field(default=True, description="Whether pipeline is active")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")

    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            PipelineId: str,
        }
    }

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update pipeline configuration."""
        self.configuration.update(config)
        self.updated_at = datetime.now(timezone.utc)

    def add_tag(self, tag: str) -> None:
        """Add tag to pipeline."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now(timezone.utc)

    def remove_tag(self, tag: str) -> None:
        """Remove tag from pipeline."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now(timezone.utc)

    def deactivate(self) -> None:
        """Deactivate pipeline."""
        self.is_active = False
        self.updated_at = datetime.now(timezone.utc)

    def activate(self) -> None:
        """Activate pipeline."""
        self.is_active = True
        self.updated_at = datetime.now(timezone.utc)
