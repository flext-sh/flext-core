"""Unified execution engine for FLEXT Core."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar
from uuid import uuid4

from flext_core.commands.pipeline import (
    CreatePipelineCommand,
    ExecutePipelineCommand,
    UpdatePipelineCommand,
)
from flext_core.domain.entities import PipelineExecution
from flext_core.domain.value_objects import ExecutionStatus

# Type definitions
CommandType = TypeVar("CommandType", bound=ExecutePipelineCommand | CreatePipelineCommand | UpdatePipelineCommand)


class OutputMode(Enum):
    """Output modes for execution."""

    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""

    output_mode: OutputMode = OutputMode.ASYNC
    timeout: int | None = None
    max_retries: int = 3


class ExecutionEngineCore(ABC):
    """Core execution engine interface."""

    @abstractmethod
    async def execute(self, command: ExecutePipelineCommand) -> PipelineExecution:
        """Execute a pipeline command."""


class AsyncExecutionEngine(ExecutionEngineCore):
    """Asynchronous execution engine."""

    async def execute(self, command: ExecutePipelineCommand) -> PipelineExecution:
        """Execute pipeline asynchronously."""
        # Implementation placeholder
        return PipelineExecution(
            id=str(uuid4()),
            pipeline_id=command.pipeline_id,
            status=ExecutionStatus.RUNNING,
            parameters=command.parameters,
        )


class SyncExecutionEngine(ExecutionEngineCore):
    """Synchronous execution engine."""

    async def execute(self, command: ExecutePipelineCommand) -> PipelineExecution:
        """Execute pipeline synchronously."""
        # Implementation placeholder
        return PipelineExecution(
            id=str(uuid4()),
            pipeline_id=command.pipeline_id,
            status=ExecutionStatus.COMPLETED,
            parameters=command.parameters,
        )


# Default engine alias
ExecutionEngine = AsyncExecutionEngine


def get_execution_engine(config: ExecutionConfig = None) -> ExecutionEngineCore:
    """Get execution engine based on configuration."""
    if config is None:
        config = ExecutionConfig()

    if config.output_mode == OutputMode.SYNC:
        return SyncExecutionEngine()
    return AsyncExecutionEngine()
