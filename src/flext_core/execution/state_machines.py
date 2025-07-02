"""State machines for execution engine."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from flext_core.domain.value_objects import ExecutionStatus


class JobState(Enum):
    """States for job execution."""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineState(Enum):
    """States for pipeline execution."""

    INITIALIZED = "initialized"
    PREPARING = "preparing"
    EXECUTING = "executing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobStateMachine:
    """State machine for job execution."""

    current_state: JobState = JobState.CREATED
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def transition_to(self, new_state: JobState) -> bool:
        """Transition to a new state if valid."""
        valid_transitions = {
            JobState.CREATED: [JobState.QUEUED, JobState.CANCELLED],
            JobState.QUEUED: [JobState.RUNNING, JobState.CANCELLED],
            JobState.RUNNING: [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED],
            JobState.COMPLETED: [],
            JobState.FAILED: [],
            JobState.CANCELLED: [],
        }

        if new_state in valid_transitions.get(self.current_state, []):
            self.current_state = new_state
            return True
        return False

    def can_transition_to(self, state: JobState) -> bool:
        """Check if transition to state is valid."""
        valid_transitions = {
            JobState.CREATED: [JobState.QUEUED, JobState.CANCELLED],
            JobState.QUEUED: [JobState.RUNNING, JobState.CANCELLED],
            JobState.RUNNING: [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED],
            JobState.COMPLETED: [],
            JobState.FAILED: [],
            JobState.CANCELLED: [],
        }
        return state in valid_transitions.get(self.current_state, [])


@dataclass
class PipelineExecutionStateMachine:
    """State machine for pipeline execution."""

    current_state: PipelineState = PipelineState.INITIALIZED
    execution_id: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def transition_to(self, new_state: PipelineState) -> bool:
        """Transition to a new state if valid."""
        valid_transitions = {
            PipelineState.INITIALIZED: [
                PipelineState.PREPARING,
                PipelineState.CANCELLED,
            ],
            PipelineState.PREPARING: [
                PipelineState.EXECUTING,
                PipelineState.FAILED,
                PipelineState.CANCELLED,
            ],
            PipelineState.EXECUTING: [
                PipelineState.FINALIZING,
                PipelineState.FAILED,
                PipelineState.CANCELLED,
            ],
            PipelineState.FINALIZING: [PipelineState.COMPLETED, PipelineState.FAILED],
            PipelineState.COMPLETED: [],
            PipelineState.FAILED: [],
            PipelineState.CANCELLED: [],
        }

        if new_state in valid_transitions.get(self.current_state, []):
            self.current_state = new_state
            return True
        return False

    def can_transition_to(self, state: PipelineState) -> bool:
        """Check if transition to state is valid."""
        valid_transitions = {
            PipelineState.INITIALIZED: [
                PipelineState.PREPARING,
                PipelineState.CANCELLED,
            ],
            PipelineState.PREPARING: [
                PipelineState.EXECUTING,
                PipelineState.FAILED,
                PipelineState.CANCELLED,
            ],
            PipelineState.EXECUTING: [
                PipelineState.FINALIZING,
                PipelineState.FAILED,
                PipelineState.CANCELLED,
            ],
            PipelineState.FINALIZING: [PipelineState.COMPLETED, PipelineState.FAILED],
            PipelineState.COMPLETED: [],
            PipelineState.FAILED: [],
            PipelineState.CANCELLED: [],
        }
        return state in valid_transitions.get(self.current_state, [])

    def to_execution_status(self) -> ExecutionStatus:
        """Convert pipeline state to execution status."""
        mapping = {
            PipelineState.INITIALIZED: ExecutionStatus.PENDING,
            PipelineState.PREPARING: ExecutionStatus.PENDING,
            PipelineState.EXECUTING: ExecutionStatus.RUNNING,
            PipelineState.FINALIZING: ExecutionStatus.RUNNING,
            PipelineState.COMPLETED: ExecutionStatus.COMPLETED,
            PipelineState.FAILED: ExecutionStatus.FAILED,
            PipelineState.CANCELLED: ExecutionStatus.CANCELLED,
        }
        return mapping.get(self.current_state, ExecutionStatus.UNKNOWN)
