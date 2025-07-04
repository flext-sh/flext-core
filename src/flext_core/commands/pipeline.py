"""Pipeline commands for FLEXT Core."""

from typing import Any

from pydantic import Field

from flext_core.domain.pydantic_base import DomainBaseModel


class CreatePipelineCommand(DomainBaseModel):
    """Command to create a new pipeline."""

    name: str = Field(..., description="Pipeline name")
    description: str | None = Field(None, description="Pipeline description")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Pipeline configuration"
    )
    tags: dict[str, str] | None = Field(None, description="Pipeline tags")


class UpdatePipelineCommand(DomainBaseModel):
    """Command to update an existing pipeline."""

    pipeline_id: str = Field(..., description="Pipeline ID to update")
    name: str | None = Field(None, description="New pipeline name")
    description: str | None = Field(None, description="New pipeline description")
    config: dict[str, Any] | None = Field(
        None, description="New pipeline configuration"
    )
    tags: dict[str, str] | None = Field(None, description="New pipeline tags")


class ExecutePipelineCommand(DomainBaseModel):
    """Command to execute a pipeline."""

    pipeline_id: str = Field(..., description="Pipeline ID to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Execution parameters"
    )
    async_execution: bool = Field(default=True, description="Whether to execute asynchronously")
