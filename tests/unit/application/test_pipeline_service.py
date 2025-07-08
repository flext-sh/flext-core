"""Pipeline application service tests - 100% coverage.

Tests for commands, queries, and service operations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

from flext_core import (
    NotFoundError,
    Pipeline,
    PipelineExecution,
    PipelineId,
    PipelineName,
    PipelineService,
    Repository,
    RepositoryError,
    ValidationError,
)
from flext_core.application.pipeline import (
    CreatePipelineCommand,
    ExecutePipelineCommand,
    GetPipelineQuery,
    ListPipelinesQuery,
)
from flext_core.domain.pipeline import ExecutionStatus


class TestCommands:
    """Test command value objects."""

    def test_create_pipeline_command(self) -> None:
        """Test CreatePipelineCommand."""
        name = PipelineName(value="test-pipeline")
        cmd = CreatePipelineCommand(
            name=name,
            description="Test description",
        )

        assert cmd.name == name
        assert cmd.description == "Test description"

        # Test with default description
        cmd2 = CreatePipelineCommand(name=name)
        assert cmd2.description == ""

    def test_execute_pipeline_command(self) -> None:
        """Test ExecutePipelineCommand."""
        pipeline_id = PipelineId()
        cmd = ExecutePipelineCommand(pipeline_id=pipeline_id)

        assert cmd.pipeline_id == pipeline_id


class TestQueries:
    """Test query value objects."""

    def test_get_pipeline_query(self) -> None:
        """Test GetPipelineQuery."""
        pipeline_id = PipelineId()
        query = GetPipelineQuery(pipeline_id=pipeline_id)

        assert query.pipeline_id == pipeline_id

    def test_list_pipelines_query(self) -> None:
        """Test ListPipelinesQuery with defaults."""
        query = ListPipelinesQuery()

        assert query.limit == 100
        assert query.offset == 0
        assert query.active_only is True

        # Test with custom values
        query2 = ListPipelinesQuery(
            limit=50,
            offset=10,
            active_only=False,
        )

        assert query2.limit == 50
        assert query2.offset == 10
        assert query2.active_only is False


class TestPipelineService:
    """Test PipelineService operations."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.mock_repo: Mock = Mock(spec=Repository[Pipeline, Any])
        self.service = PipelineService(self.mock_repo)

    async def test_create_pipeline_success(self) -> None:
        """Test successful pipeline creation."""
        # Setup
        name = PipelineName(value="test-pipeline")
        command = CreatePipelineCommand(
            name=name,
            description="Test pipeline",
        )

        # Mock repository to return saved pipeline
        async def mock_save(pipeline: Pipeline) -> Pipeline:
            return pipeline

        self.mock_repo.save = AsyncMock(side_effect=mock_save)

        # Execute
        result = await self.service.create_pipeline(command)

        # Verify
        assert result.success is True
        pipeline = result.unwrap()
        assert pipeline.name == name
        assert pipeline.description == "Test pipeline"

        # Check domain events at class level
        events = Pipeline.get_events()
        assert len(events) == 1  # Should have PipelineCreated event

        # Verify repository was called
        self.mock_repo.save.assert_called_once()

    async def test_create_pipeline_validation_error(self) -> None:
        """Test pipeline creation with validation error."""
        # Setup to raise ValidationError
        self.mock_repo.save = AsyncMock(
            side_effect=ValidationError("Invalid pipeline data"),
        )

        command = CreatePipelineCommand(
            name=PipelineName(value="test"),
        )

        # Execute
        result = await self.service.create_pipeline(command)

        # Verify
        assert result.success is False
        assert result.error == "Validation failed: Invalid pipeline data"

    async def test_create_pipeline_repository_error(self) -> None:
        """Test pipeline creation with repository error."""
        # Setup to raise RepositoryError
        self.mock_repo.save = AsyncMock(
            side_effect=RepositoryError("Database connection failed"),
        )

        command = CreatePipelineCommand(
            name=PipelineName(value="test"),
        )

        # Execute
        result = await self.service.create_pipeline(command)

        # Verify
        assert result.success is False
        assert result.error == "Repository error: Database connection failed"

    async def test_execute_pipeline_success(self) -> None:
        """Test successful pipeline execution."""
        # Setup
        pipeline_id = PipelineId()
        pipeline = Pipeline(
            id=pipeline_id,
            name=PipelineName(value="test"),
            is_active=True,
        )

        self.mock_repo.get = AsyncMock(return_value=pipeline)

        command = ExecutePipelineCommand(pipeline_id=pipeline_id)

        # Execute
        result = await self.service.execute_pipeline(command)

        # Verify
        assert result.success is True
        execution = result.unwrap()
        assert isinstance(execution, PipelineExecution)
        assert execution.pipeline_id == pipeline_id
        assert execution.status == ExecutionStatus.RUNNING

        # Verify pipeline has execution event at class level
        events = Pipeline.get_events()
        assert len(events) == 1

    async def test_execute_pipeline_not_found(self) -> None:
        """Test executing non-existent pipeline."""
        # Setup - repository returns None
        self.mock_repo.get = AsyncMock(return_value=None)

        command = ExecutePipelineCommand(pipeline_id=PipelineId())

        # Execute
        result = await self.service.execute_pipeline(command)

        # Verify
        assert result.success is False
        assert result.error == "Pipeline not found"

    async def test_execute_pipeline_not_found_exception(self) -> None:
        """Test executing pipeline with NotFoundError."""
        # Setup - repository raises NotFoundError
        self.mock_repo.get = AsyncMock(
            side_effect=NotFoundError("Pipeline not found"),
        )

        command = ExecutePipelineCommand(pipeline_id=PipelineId())

        # Execute
        result = await self.service.execute_pipeline(command)

        # Verify
        assert result.success is False
        assert result.error == "Pipeline not found"

    async def test_execute_inactive_pipeline(self) -> None:
        """Test executing inactive pipeline."""
        # Setup
        pipeline = Pipeline(
            name=PipelineName(value="test"),
            is_active=False,
        )

        self.mock_repo.get = AsyncMock(return_value=pipeline)

        command = ExecutePipelineCommand(pipeline_id=pipeline.id)

        # Execute
        result = await self.service.execute_pipeline(command)

        # Verify
        assert result.success is False
        assert result.error == "Pipeline is inactive"

    async def test_execute_pipeline_repository_error(self) -> None:
        """Test pipeline execution with repository error."""
        # Setup
        self.mock_repo.get = AsyncMock(
            side_effect=RepositoryError("Connection lost"),
        )

        command = ExecutePipelineCommand(pipeline_id=PipelineId())

        # Execute
        result = await self.service.execute_pipeline(command)

        # Verify
        assert result.success is False
        assert result.error == "Repository error: Connection lost"

    async def test_get_pipeline_success(self) -> None:
        """Test successful pipeline retrieval."""
        # Setup
        pipeline_id = PipelineId()
        pipeline = Pipeline(
            id=pipeline_id,
            name=PipelineName(value="test"),
        )

        self.mock_repo.get = AsyncMock(return_value=pipeline)

        query = GetPipelineQuery(pipeline_id=pipeline_id)

        # Execute
        result = await self.service.get_pipeline(query)

        # Verify
        assert result.success is True
        assert result.unwrap() == pipeline

    async def test_get_pipeline_not_found(self) -> None:
        """Test getting non-existent pipeline."""
        # Setup
        self.mock_repo.get = AsyncMock(return_value=None)

        query = GetPipelineQuery(pipeline_id=PipelineId())

        # Execute
        result = await self.service.get_pipeline(query)

        # Verify
        assert result.success is False
        assert result.error == "Pipeline not found"

    async def test_get_pipeline_exception(self) -> None:
        """Test getting pipeline with exceptions."""
        # Test NotFoundError
        self.mock_repo.get = AsyncMock(
            side_effect=NotFoundError("Not found"),
        )

        query = GetPipelineQuery(pipeline_id=PipelineId())
        result = await self.service.get_pipeline(query)

        assert result.success is False
        assert result.error == "Pipeline not found"

        # Test RepositoryError
        self.mock_repo.get = AsyncMock(
            side_effect=RepositoryError("DB error"),
        )

        result = await self.service.get_pipeline(query)

        assert result.success is False
        assert result.error == "Repository error: DB error"

    async def test_deactivate_pipeline_success(self) -> None:
        """Test successful pipeline deactivation."""
        # Setup
        pipeline_id = PipelineId()
        pipeline = Pipeline(
            id=pipeline_id,
            name=PipelineName(value="test"),
            is_active=True,
        )

        self.mock_repo.get = AsyncMock(return_value=pipeline)
        self.mock_repo.save = AsyncMock(return_value=pipeline)

        # Execute
        result = await self.service.deactivate_pipeline(pipeline_id)

        # Verify
        assert result.success is True
        deactivated = result.unwrap()
        assert deactivated.is_active is False

        # Verify repository calls
        self.mock_repo.get.assert_called_once_with(pipeline_id)
        self.mock_repo.save.assert_called_once()

    async def test_deactivate_pipeline_not_found(self) -> None:
        """Test deactivating non-existent pipeline."""
        # Setup
        self.mock_repo.get = AsyncMock(return_value=None)

        # Execute
        result = await self.service.deactivate_pipeline(PipelineId())

        # Verify
        assert result.success is False
        assert result.error == "Pipeline not found"

    async def test_deactivate_pipeline_exceptions(self) -> None:
        """Test deactivating pipeline with exceptions."""
        pipeline_id = PipelineId()

        # Test NotFoundError
        self.mock_repo.get = AsyncMock(
            side_effect=NotFoundError("Not found"),
        )

        result = await self.service.deactivate_pipeline(pipeline_id)
        assert result.success is False
        assert result.error == "Pipeline not found"

        # Test RepositoryError
        self.mock_repo.get = AsyncMock(
            side_effect=RepositoryError("DB error"),
        )

        result = await self.service.deactivate_pipeline(pipeline_id)
        assert result.success is False
        assert result.error == "Repository error: DB error"
