"""Pipeline services for FLEXT Core."""

from abc import ABC, abstractmethod

from flext_core.commands.pipeline import (
    CreatePipelineCommand,
    ExecutePipelineCommand,
    UpdatePipelineCommand,
)
from flext_core.domain.entities import Pipeline, PipelineExecution


class PipelineManagementService(ABC):
    """Abstract service for pipeline management operations."""

    @abstractmethod
    async def create_pipeline(self, command: CreatePipelineCommand) -> Pipeline:
        """Create a new pipeline."""

    @abstractmethod
    async def update_pipeline(self, command: UpdatePipelineCommand) -> Pipeline:
        """Update an existing pipeline."""

    @abstractmethod
    async def get_pipeline(self, pipeline_id: str) -> Pipeline | None:
        """Get a pipeline by ID."""

    @abstractmethod
    async def list_pipelines(self) -> list[Pipeline]:
        """List all pipelines."""

    @abstractmethod
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""


class PipelineExecutionService(ABC):
    """Abstract service for pipeline execution operations."""

    @abstractmethod
    async def execute_pipeline(
        self, command: ExecutePipelineCommand
    ) -> PipelineExecution:
        """Execute a pipeline."""

    @abstractmethod
    async def get_execution(self, execution_id: str) -> PipelineExecution | None:
        """Get a pipeline execution by ID."""

    @abstractmethod
    async def list_executions(
        self, pipeline_id: str | None = None
    ) -> list[PipelineExecution]:
        """List pipeline executions."""

    @abstractmethod
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
