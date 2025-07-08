"""Domain layer - Pure business logic with zero dependencies.

Modern Python 3.13 + Pydantic v2 implementation.
SOLID principles with maximum performance.
"""

from __future__ import annotations

# Single imports - no duplicates allowed
from flext_core.domain.core import (
    AggregateRoot,
    DomainError,
    DomainEvent,
    Entity,
    NotFoundError,
    Repository,
    RepositoryError,
    ServiceResult,
    ValidationError,
    ValueObject,
)
from flext_core.domain.pipeline import (
    ExecutionStatus,
    Pipeline,
    PipelineCreated,
    PipelineExecuted,
    PipelineExecution,
    PipelineId,
    PipelineName,
)

__all__ = [
    "AggregateRoot",
    "DomainError",
    "DomainEvent",
    "Entity",
    "ExecutionStatus",
    "NotFoundError",
    "Pipeline",
    "PipelineCreated",
    "PipelineExecuted",
    "PipelineExecution",
    "PipelineId",
    "PipelineName",
    "Repository",
    "RepositoryError",
    "ServiceResult",
    "ValidationError",
    "ValueObject",
]
