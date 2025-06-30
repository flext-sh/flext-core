"""Pipeline service for FLEXT Core."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from flext_core.commands.pipeline import CreatePipelineCommand, UpdatePipelineCommand
from flext_core.domain.entities import Pipeline
from flext_core.domain.value_objects import PipelineId, PipelineName


class PipelineService(ABC):
    """Abstract pipeline service."""

    @abstractmethod
    async def create_pipeline(self, command: CreatePipelineCommand) -> Pipeline:
        """Create a new pipeline."""

    @abstractmethod
    async def update_pipeline(self, command: UpdatePipelineCommand) -> Pipeline:
        """Update an existing pipeline."""

    @abstractmethod
    async def get_pipeline(self, pipeline_id: str) -> Pipeline | None:
        """Get pipeline by ID."""

    @abstractmethod
    async def list_pipelines(self) -> list[Pipeline]:
        """List all pipelines."""

    @abstractmethod
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""

    @abstractmethod
    async def validate_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        """Validate pipeline configuration."""


class DefaultPipelineService(PipelineService):
    """Default implementation of pipeline service."""

    def __init__(self) -> None:
        self._pipelines: dict[str, Pipeline] = {}

    async def create_pipeline(self, command: CreatePipelineCommand) -> Pipeline:
        """Create a new pipeline."""
        pipeline = Pipeline(
            id=PipelineId(str(uuid4())),
            name=PipelineName(command.name),
            description=command.description,
            configuration=command.config,
            steps=[],
            is_active=True,
        )
        self._pipelines[str(pipeline.id)] = pipeline
        return pipeline

    async def update_pipeline(self, command: UpdatePipelineCommand) -> Pipeline:
        """Update an existing pipeline."""
        pipeline = self._pipelines.get(str(command.pipeline_id))
        if not pipeline:
            msg = f"Pipeline not found: {command.pipeline_id}"
            raise ValueError(msg)

        # Update pipeline fields
        if command.name:
            pipeline.name = PipelineName(command.name)
        if command.description is not None:
            pipeline.description = command.description
        if command.config:
            pipeline.configuration.update(command.config)

        return pipeline

    async def get_pipeline(self, pipeline_id: str) -> Pipeline | None:
        """Get pipeline by ID."""
        return self._pipelines.get(pipeline_id)

    async def list_pipelines(self) -> list[Pipeline]:
        """List all pipelines."""
        return list(self._pipelines.values())

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""
        if pipeline_id in self._pipelines:
            del self._pipelines[pipeline_id]
            return True
        return False

    async def validate_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        """Validate pipeline configuration."""
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return {"valid": False, "errors": ["Pipeline not found"]}

        errors = []
        if not pipeline.name:
            errors.append("Pipeline name is required")
        if not pipeline.steps:
            errors.append("Pipeline must have at least one step")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }
