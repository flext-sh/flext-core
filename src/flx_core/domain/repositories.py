"""Ultimate Domain Repository Interfaces - Python 3.13 + DDD + ZERO TOLERANCE.

This module provides the SINGLE domain repository interface using Python 3.13 generics.
with strict validation
with strict validation

ARCHITECTURAL PRINCIPLE:
- Single generic repository interface for ALL domain entities
- Python 3.13 type parameters for complete type safety
- Domain-driven repository contracts with business-focused methods
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

# Import UnitOfWork from contracts for compatibility
from flx_core.domain.entities import (
    Pipeline,
    PipelineExecution,
    PipelineId,
    Plugin,
    PluginId,
)

# Re-export the ultimate repository from infrastructure for domain use
from flx_core.infrastructure.persistence.repositories_core import (
    CoreDomainRepository as DomainRepository,
)

# Python 3.13 Type Aliases for Domain Context - Using new type alias syntax
type PipelineRepository = DomainRepository[Pipeline, PipelineId]
type PipelineExecutionRepository = DomainRepository[PipelineExecution, UUID]
type PluginRepository = DomainRepository[Plugin, PluginId]


class DomainPipelineRepository(DomainRepository[Pipeline, PipelineId]):
    """Domain-specific pipeline repository with business methods."""

    @abstractmethod
    async def find_by_name(self, name: str) -> Pipeline | None:
        """Find pipeline by name - domain business rule."""

    @abstractmethod
    async def find_active_pipelines(self) -> list[Pipeline]:
        """Find all active pipelines - domain business rule."""

    @abstractmethod
    async def find_scheduled_pipelines(self) -> list[Pipeline]:
        """Find pipelines with scheduling enabled - domain business rule."""


class DomainExecutionRepository(DomainRepository[PipelineExecution, UUID]):
    """Domain-specific execution repository with business methods."""

    @abstractmethod
    async def find_by_pipeline(
        self, pipeline_id: PipelineId, limit: int = 50
    ) -> list[PipelineExecution]:
        """Find executions by pipeline - domain business rule."""

    @abstractmethod
    async def find_running_executions(self) -> list[PipelineExecution]:
        """Find currently running executions - domain business rule."""

    @abstractmethod
    async def get_next_execution_number(self, pipeline_id: PipelineId) -> int:
        """Get next execution number - domain business rule."""


class DomainPluginRepository(DomainRepository[Plugin, PluginId]):
    """Domain-specific plugin repository with business methods."""

    @abstractmethod
    async def find_by_type(self, plugin_type: str) -> list[Plugin]:
        """Find plugins by type - domain business rule."""

    @abstractmethod
    async def search_plugins(
        self, query: str, plugin_type: str | None = None
    ) -> list[Plugin]:
        """Search plugins - domain business rule."""


# Update existing type aliases to point to domain classes for legacy compatibility
PipelineRepository = DomainPipelineRepository  # type: ignore[assignment,misc]
PipelineExecutionRepository = DomainExecutionRepository  # type: ignore[assignment,misc]
PluginRepository = DomainPluginRepository  # type: ignore[assignment,misc]
