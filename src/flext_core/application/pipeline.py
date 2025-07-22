"""Pipeline application services.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Commands, Queries, and Service all together.
Zero duplication, maximum cohesion.

.. deprecated:: 0.7.0
    This module has been moved to flext.services.application.pipeline.
    Please use 'from flext.services.application import PipelineService' instead.
    This compatibility layer will be removed in v0.8.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from uuid import UUID

from pydantic import Field
from pydantic import ValidationError

from flext_core.domain.core import DatabaseError

# Move imports out of TYPE_CHECKING block since they're used in runtime
from flext_core.domain.core import DataError
from flext_core.domain.core import NotFoundError
from flext_core.domain.core import RepositoryError
from flext_core.domain.core import ServiceError
from flext_core.domain.core import TransformationError
from flext_core.domain.pipeline import Pipeline
from flext_core.domain.pipeline import PipelineId
from flext_core.domain.pipeline import PipelineName
from flext_core.domain.pydantic_base import DomainBaseModel
from flext_core.domain.shared_types import ServiceResult

# NOTE: This module is part of flext_core.application layer
# No deprecation warning needed - this is the correct import path

if TYPE_CHECKING:  # pragma: no cover
    from flext_core.domain.core import Repository


# Commands
class CreatePipelineCommand(DomainBaseModel):
    """Create pipeline command."""

    name: str = Field(..., description="Pipeline name", max_length=100)
    description: str = Field(
        default="",
        description="Pipeline description",
        max_length=500,
    )


class ExecutePipelineCommand(DomainBaseModel):
    """Execute pipeline command."""

    pipeline_id: str = Field(..., description="Pipeline ID to execute")


# Queries
class GetPipelineQuery(DomainBaseModel):
    """Get pipeline query."""

    pipeline_id: str = Field(..., description="Pipeline ID to retrieve")


class ListPipelinesQuery(DomainBaseModel):
    """List pipelines query."""

    limit: int = Field(
        100,
        description="Number of pipelines to return",
        ge=1,
        le=1000,
    )
    offset: int = Field(default=0, description="Offset for pagination", ge=0)
    active_only: bool = Field(default=True, description="Return only active pipelines")


# Service
class PipelineService:
    """Pipeline application service - SOLID principles."""

    def __init__(self, pipeline_repo: Repository[Pipeline, object]) -> None:
        """Initialize pipeline service.

        Args:
            pipeline_repo: Pipeline repository for data access

        """
        self._repo = pipeline_repo

    async def create_pipeline(
        self,
        command: CreatePipelineCommand,
    ) -> ServiceResult[dict[str, Any]]:
        """Create a new pipeline.

        Args:
            command: Create pipeline command

        Returns:
            Service result with created pipeline

        """
        try:
            pipeline_name = PipelineName(value=command.name)
            pipeline = Pipeline(
                pipeline_name=pipeline_name,
                pipeline_description=command.description,
            )
            pipeline.create()  # Emit domain event

            saved = await self._repo.save(pipeline)
            return ServiceResult.ok(
                data={
                    "result": saved.model_dump()
                    if hasattr(saved, "model_dump")
                    else str(saved),
                },
            )

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e}")
        except (ValueError, TypeError) as e:
            return ServiceResult.fail(f"Input error: {e}")
        except (
            RepositoryError,
            DatabaseError,
            NotFoundError,
            ServiceError,
            DataError,
            TransformationError,
            RuntimeError,
            AttributeError,
            ConnectionError,
            OSError,
        ) as e:
            return ServiceResult.fail(f"Repository error: {e}")

    async def execute_pipeline(
        self,
        command: ExecutePipelineCommand,
    ) -> ServiceResult[dict[str, Any]]:
        """Execute a pipeline.

        Args:
            command: Execute pipeline command

        Returns:
            Service result with pipeline execution

        """
        try:
            pipeline_id = PipelineId(value=UUID(command.pipeline_id))
            pipeline = await self._repo.find_by_id(pipeline_id)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            if not pipeline.pipeline_is_active:
                return ServiceResult.fail("Pipeline is inactive")

            execution = pipeline.execute()  # Emit domain event
            return ServiceResult.ok(
                data={
                    "result": execution.model_dump()
                    if hasattr(execution, "model_dump")
                    else str(execution),
                },
            )

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e}")
        except (ValueError, TypeError) as e:
            return ServiceResult.fail(f"Input error: {e}")
        except (
            RepositoryError,
            DatabaseError,
            NotFoundError,
            ServiceError,
            DataError,
            TransformationError,
            RuntimeError,
            AttributeError,
            ConnectionError,
            OSError,
        ) as e:
            return ServiceResult.fail(f"Execution error: {e}")

    async def get_pipeline(
        self,
        query: GetPipelineQuery,
    ) -> ServiceResult[dict[str, Any]]:
        """Get a pipeline by ID.

        Args:
            query: Get pipeline query

        Returns:
            Service result with pipeline

        """
        try:
            pipeline_id = PipelineId(value=UUID(query.pipeline_id))
            pipeline = await self._repo.find_by_id(pipeline_id)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            return ServiceResult.ok(
                data={
                    "result": pipeline.model_dump()
                    if hasattr(pipeline, "model_dump")
                    else str(pipeline),
                },
            )

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e}")
        except (ValueError, TypeError) as e:
            return ServiceResult.fail(f"Input error: {e}")
        except (
            RepositoryError,
            DatabaseError,
            NotFoundError,
            ServiceError,
            DataError,
            TransformationError,
            RuntimeError,
            AttributeError,
            ConnectionError,
            OSError,
        ) as e:
            return ServiceResult.fail(f"Repository error: {e}")

    async def deactivate_pipeline(
        self,
        pipeline_id: str,
    ) -> ServiceResult[dict[str, Any]]:
        """Deactivate a pipeline.

        Args:
            pipeline_id: Pipeline ID to deactivate

        Returns:
            Service result with deactivated pipeline

        """
        try:
            pid = PipelineId(value=UUID(pipeline_id))
            pipeline = await self._repo.find_by_id(pid)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            pipeline.deactivate()
            saved = await self._repo.save(pipeline)
            return ServiceResult.ok(
                data={
                    "result": saved.model_dump()
                    if hasattr(saved, "model_dump")
                    else str(saved),
                },
            )

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e}")
        except (ValueError, TypeError) as e:
            return ServiceResult.fail(f"Input error: {e}")
        except (
            RepositoryError,
            DatabaseError,
            NotFoundError,
            ServiceError,
            DataError,
            TransformationError,
            RuntimeError,
            AttributeError,
            ConnectionError,
            OSError,
        ) as e:
            return ServiceResult.fail(f"Repository error: {e}")


__all__ = [
    "CreatePipelineCommand",
    "ExecutePipelineCommand",
    "GetPipelineQuery",
    "ListPipelinesQuery",
    "PipelineService",
]
