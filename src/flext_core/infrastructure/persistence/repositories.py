"""Repository backward compatibility module.

This module provides backward compatibility for imports that reference
the old repositories.py module. All functionality has been moved to
repositories.py for better organization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import entities and models that were previously imported lazily
from flext_core.domain.entities import Pipeline, PipelineExecution, Plugin
from flext_core.models import PipelineExecutionModel, PipelineModel, PluginModel

# Import all repository classes from the ultimate implementation
from flext_core.repositories_core import (  # Repository base classes; Repository errors
    CoreDomainRepository,
    DomainSpecificRepository,
    EntityNotFoundError,
    RepositoryError,
    SqlAlchemyRepository,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

# Backward compatibility aliases
SqlAlchemyPluginRepository = DomainSpecificRepository


# Factory functions for backward compatibility
def create_pipeline_repository(
    session: AsyncSession,
    entity_class: type[Pipeline] | None = None,
    model_class: type[PipelineModel] | None = None,
) -> DomainSpecificRepository[Pipeline, PipelineModel, UUID]:
    """Create pipeline repository for backward compatibility."""
    return DomainSpecificRepository(
        session=session,
        entity_class=entity_class or Pipeline,
        model_class=model_class or PipelineModel,
        id_field="id",  # SQLAlchemy model uses "id" field
    )


def create_execution_repository(
    session: AsyncSession,
    entity_class: type[PipelineExecution] | None = None,
    model_class: type[PipelineExecutionModel] | None = None,
) -> DomainSpecificRepository[PipelineExecution, PipelineExecutionModel, UUID]:
    """Create execution repository for backward compatibility."""
    return DomainSpecificRepository(
        session=session,
        entity_class=entity_class or PipelineExecution,
        model_class=model_class or PipelineExecutionModel,
        id_field="id",
    )


def create_plugin_repository(
    session: AsyncSession,
    entity_class: type[Plugin] | None = None,
    model_class: type[PluginModel] | None = None,
) -> DomainSpecificRepository[Plugin, PluginModel, UUID]:
    """Create plugin repository for backward compatibility."""
    return DomainSpecificRepository(
        session=session,
        entity_class=entity_class or Plugin,
        model_class=model_class or PluginModel,
        id_field="id",
    )


# Export all symbols for backward compatibility
__all__ = [
    "CoreDomainRepository",
    "DomainSpecificRepository",
    "EntityNotFoundError",
    "RepositoryError",
    "SqlAlchemyPluginRepository",
    "SqlAlchemyRepository",
    "create_execution_repository",
    "create_pipeline_repository",
    "create_plugin_repository",
]
