"""Pipeline application services - COMPLETE in ONE file.

Commands, Queries, and Service all together.
Zero duplication, maximum cohesion.
"""

from __future__ import annotations

from typing import Any

from flext_core.domain import (
    NotFoundError,
    Pipeline,
    PipelineExecution,
    PipelineId,
    PipelineName,
    Repository,
    RepositoryError,
    ServiceResult,
    ValidationError,
    ValueObject,
)


# Commands
class CreatePipelineCommand(ValueObject):
    """Create pipeline command."""

    name: PipelineName
    description: str = ""


class ExecutePipelineCommand(ValueObject):
    """Execute pipeline command."""

    pipeline_id: PipelineId


# Queries
class GetPipelineQuery(ValueObject):
    """Get pipeline query."""

    pipeline_id: PipelineId


class ListPipelinesQuery(ValueObject):
    """List pipelines query."""

    limit: int = 100
    offset: int = 0
    active_only: bool = True


# Service
class PipelineService:
    """Pipeline application service - SOLID principles."""

    def __init__(self, pipeline_repo: Repository[Pipeline, Any]) -> None:
        """Initialize pipeline service with repository."""
        self._repo = pipeline_repo

    async def create_pipeline(
        self,
        command: CreatePipelineCommand,
    ) -> ServiceResult[Pipeline]:
        """Create new pipeline."""
        try:
            pipeline = Pipeline(
                name=command.name,
                description=command.description,
            )
            pipeline.create()  # Emit domain event

            saved = await self._repo.save(pipeline)
            return ServiceResult.ok(saved)

        except ValidationError as e:
            return ServiceResult.fail(f"Validation failed: {e!s}")
        except RepositoryError as e:
            return ServiceResult.fail(f"Repository error: {e!s}")

    async def execute_pipeline(
        self,
        command: ExecutePipelineCommand,
    ) -> ServiceResult[PipelineExecution]:
        """Execute pipeline."""
        try:
            pipeline = await self._repo.get(command.pipeline_id)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            if not pipeline.is_active:
                return ServiceResult.fail("Pipeline is inactive")

            execution = pipeline.execute()  # Emit domain event
            return ServiceResult.ok(execution)

        except NotFoundError:
            return ServiceResult.fail("Pipeline not found")
        except RepositoryError as e:
            return ServiceResult.fail(f"Repository error: {e!s}")

    async def get_pipeline(self, query: GetPipelineQuery) -> ServiceResult[Pipeline]:
        """Get specific pipeline."""
        try:
            pipeline = await self._repo.get(query.pipeline_id)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            return ServiceResult.ok(pipeline)

        except NotFoundError:
            return ServiceResult.fail("Pipeline not found")
        except RepositoryError as e:
            return ServiceResult.fail(f"Repository error: {e!s}")

    async def deactivate_pipeline(self, pipeline_id: PipelineId) -> ServiceResult[Pipeline]:
        """Deactivate pipeline."""
        try:
            pipeline = await self._repo.get(pipeline_id)
            if not pipeline:
                return ServiceResult.fail("Pipeline not found")

            pipeline.deactivate()
            saved = await self._repo.save(pipeline)
            return ServiceResult.ok(saved)

        except NotFoundError:
            return ServiceResult.fail("Pipeline not found")
        except RepositoryError as e:
            return ServiceResult.fail(f"Repository error: {e!s}")
