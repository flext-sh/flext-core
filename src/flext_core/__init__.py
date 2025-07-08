"""FLext Core - Enterprise Foundation Framework.

Modern Python 3.13 + Pydantic v2 + Clean Architecture.
Zero tolerance for code duplication and technical debt.

Key Principles:
- SOLID: Single responsibility, Open/closed, Liskov substitution,
  Interface segregation, Dependency inversion
- KISS: Keep it simple, stupid
- DRY: Don't repeat yourself
- Performance: Zero overhead abstractions
- Type Safety: 100% typed with modern Python 3.13 syntax
"""

from __future__ import annotations

# Version
__version__ = "0.6.0"

# Core Domain exports - SINGLE SOURCE OF TRUTH
# Application Layer - Command/Query patterns
from flext_core.application import PipelineService
from flext_core.domain import (
    ExecutionStatus,
    Pipeline,
    PipelineExecution,
    PipelineId,
    PipelineName,
    ServiceResult,
)

# Domain Base Classes - Foundation for all projects
from flext_core.domain.core import (
    AggregateRoot,
    DomainError,
    DomainEvent,
    Entity,
    NotFoundError,
    Repository,
    RepositoryError,
    ValidationError,
    ValueObject,
)

# Infrastructure - Base implementations
from flext_core.infrastructure import InMemoryRepository

# Configuration will be available when needed
_config_available = False
get_config = None
get_domain_constants = None

# Public API - Everything other projects need
__all__ = [
    # Foundation Classes
    "AggregateRoot",
    "DomainError",
    "DomainEvent",
    "Entity",
    "ExecutionStatus",
    "InMemoryRepository",
    "NotFoundError",
    "Pipeline",
    "PipelineExecution",
    "PipelineId",
    "PipelineName",
    "PipelineService",
    "Repository",
    "RepositoryError",
    "ServiceResult",
    "ValidationError",
    "ValueObject",
    # Version
    "__version__",
]

# Configuration exports will be added when implemented
