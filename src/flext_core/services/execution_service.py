"""Execution service for FLEXT Core."""

from abc import ABC, abstractmethod
from uuid import uuid4

from flext_core.commands.pipeline import ExecutePipelineCommand
from flext_core.domain.entities import PipelineExecution
from flext_core.domain.value_objects import ExecutionId, ExecutionStatus


class ExecutionService(ABC):
    """Abstract execution service."""

    @abstractmethod
    async def execute_pipeline(self, command: ExecutePipelineCommand) -> PipelineExecution:
        """Execute a pipeline."""

    @abstractmethod
    async def get_execution(self, execution_id: str) -> PipelineExecution | None:
        """Get execution by ID."""

    @abstractmethod
    async def list_executions(self, pipeline_id: str | None = None) -> list[PipelineExecution]:
        """List executions."""

    @abstractmethod
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel execution."""


class DefaultExecutionService(ExecutionService):
    """Default implementation of execution service."""

    def __init__(self) -> None:
        self._executions: dict[str, PipelineExecution] = {}

    async def execute_pipeline(self, command: ExecutePipelineCommand) -> PipelineExecution:
        """Execute a pipeline."""
        execution = PipelineExecution(
            id=ExecutionId(str(uuid4())),
            pipeline_id=command.pipeline_id,
            status=ExecutionStatus.RUNNING,
            parameters=command.parameters,
        )
        self._executions[str(execution.id)] = execution
        return execution

    async def get_execution(self, execution_id: str) -> PipelineExecution | None:
        """Get execution by ID."""
        return self._executions.get(execution_id)

    async def list_executions(self, pipeline_id: str | None = None) -> list[PipelineExecution]:
        """List executions."""
        executions = list(self._executions.values())
        if pipeline_id:
            executions = [e for e in executions if str(e.pipeline_id) == pipeline_id]
        return executions

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel execution."""
        if execution_id in self._executions:
            execution = self._executions[execution_id]
            execution.status = ExecutionStatus.CANCELLED
            return True
        return False
