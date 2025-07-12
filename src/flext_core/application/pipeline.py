"""Pipeline application services.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Commands, Queries, and Service all together.
Zero duplication, maximum cohesion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import Field
from pydantic import ValidationError

# Move imports out of TYPE_CHECKING block since they're used in runtime
from flext_core.domain.pipeline import Pipeline
from flext_core.domain.pipeline import PipelineId
from flext_core.domain.pipeline import PipelineName
from flext_core.domain.pydantic_base import APIRequest
from flext_core.domain.types import ServiceResult

if TYPE_CHECKING:
    from flext_core.domain.pipeline import PipelineExecution
    from flext_core.infrastructure.persistence.base import Repository


# Commands
class CreatePipelineCommand(APIRequest):
    """Create pipeline command."""

    name: str = Field(..., description="Pipeline name", max_length=100)
    description: str = Field(
        default="",
        description="Pipeline description",
        max_length=500,
    )


class ExecutePipelineCommand(APIRequest):
    """Execute pipeline command."""

    pipeline_id: str = Field(..., description="Pipeline ID to execute")


# Queries
class GetPipelineQuery(APIRequest):
    """Get pipeline query."""

    pipeline_id: str = Field(..., description="Pipeline ID to retrieve")


class ListPipelinesQuery(APIRequest):
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
    ) -> ServiceResult[Pipeline]:
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
            return ServiceResult.ok(saved)

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e}")
        except (ValueError, TypeError) as e:
            return ServiceResult.fail(f"Input error: {e}")
        except OSError as e:
            return ServiceResult.fail(f"Repository error: {e}")

    async def execute_pipeline(
        self,
        command: ExecutePipelineCommand,
    ) -> ServiceResult[PipelineExecution]:
        """Execute a pipeline.

        Args:
            command: Execute pipeline command

        Returns:
            Service result with pipeline execution

        """
        try:
            pipeline_id = PipelineId(value=UUID(command.pipeline_id))
            pipeline = await self._repo.get_by_id(pipeline_id)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            if not pipeline.pipeline_is_active:
                return ServiceResult.fail("Pipeline is inactive")

            execution = pipeline.execute()  # Emit domain event
            return ServiceResult.ok(execution)

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e}")
        except (ValueError, TypeError) as e:
            return ServiceResult.fail(f"Input error: {e}")
        except OSError as e:
            return ServiceResult.fail(f"Execution error: {e}")

    async def get_pipeline(self, query: GetPipelineQuery) -> ServiceResult[Pipeline]:
        """Get a pipeline by ID.

        Args:
            query: Get pipeline query

        Returns:
            Service result with pipeline

        """
        try:
            pipeline_id = PipelineId(value=UUID(query.pipeline_id))
            pipeline = await self._repo.get_by_id(pipeline_id)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            return ServiceResult.ok(pipeline)

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e}")
        except (ValueError, TypeError) as e:
            return ServiceResult.fail(f"Input error: {e}")
        except OSError as e:
            return ServiceResult.fail(f"Repository error: {e}")

    async def deactivate_pipeline(
        self,
        pipeline_id: str,
    ) -> ServiceResult[Pipeline]:
        """Deactivate a pipeline.

        Args:
            pipeline_id: Pipeline ID to deactivate

        Returns:
            Service result with deactivated pipeline

        """
        try:
            pid = PipelineId(value=UUID(pipeline_id))
            pipeline = await self._repo.get_by_id(pid)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            pipeline.deactivate()
            saved = await self._repo.save(pipeline)
            return ServiceResult.ok(saved)

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e}")
        except (ValueError, TypeError) as e:
            return ServiceResult.fail(f"Input error: {e}")
        except OSError as e:
            return ServiceResult.fail(f"Repository error: {e}")


__all__ = [
    "CreatePipelineCommand",
    "ExecutePipelineCommand",
    "GetPipelineQuery",
    "ListPipelinesQuery",
    "PipelineService",
]
