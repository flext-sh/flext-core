"""Tests to cover missing lines in application/pipeline.py.

This file targets specific missing exception handlers and edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from flext_core.application.pipeline import (
    CreatePipelineCommand,
    ExecutePipelineCommand,
    GetPipelineQuery,
    ListPipelinesQuery,
    PipelineService,
)

# DeactivatePipelineCommand doesn't exist in current implementation
from flext_core.domain.pipeline import Pipeline, PipelineId, PipelineName


class TestMissingExceptionHandlers:
    """Test specific exception handlers that are missing coverage."""

    @pytest.fixture
    def mock_repository(self) -> AsyncMock:
        """Create a mock repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_repository: AsyncMock) -> PipelineService:
        """Create pipeline service with mock repository."""
        return PipelineService(pipeline_repo=mock_repository)

    @pytest.mark.asyncio
    async def test_create_pipeline_runtime_error(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test create_pipeline RuntimeError handling (line 112)."""
        # Make repository.save raise RuntimeError
        mock_repository.save.side_effect = RuntimeError("Runtime error occurred")

        command = CreatePipelineCommand(name="test", description="test")
        result = await service.create_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Runtime error occurred" in result.error

    @pytest.mark.asyncio
    async def test_create_pipeline_attribute_error(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test create_pipeline AttributeError handling (line 112)."""
        # Make repository.save raise AttributeError
        mock_repository.save.side_effect = AttributeError("Attribute error occurred")

        command = CreatePipelineCommand(name="test", description="test")
        result = await service.create_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Attribute error occurred" in result.error

    @pytest.mark.asyncio
    async def test_create_pipeline_connection_error(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test create_pipeline ConnectionError handling (line 112)."""
        # Make repository.save raise ConnectionError
        mock_repository.save.side_effect = ConnectionError("Connection error occurred")

        command = CreatePipelineCommand(name="test", description="test")
        result = await service.create_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Connection error occurred" in result.error

    @pytest.mark.asyncio
    async def test_execute_pipeline_validation_error(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test execute_pipeline ValidationError handling (line 142)."""
        # Test ValidationError by providing invalid UUID format to trigger line 142
        command = ExecutePipelineCommand(pipeline_id="invalid-uuid-format")
        result = await service.execute_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Input error:" in result.error

    @pytest.mark.asyncio
    async def test_get_pipeline_runtime_error(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test get_pipeline RuntimeError handling (line 167)."""
        # Make repository.get_by_id raise RuntimeError
        mock_repository.get_by_id.side_effect = RuntimeError("Runtime error in get")

        from uuid import uuid4

        pipeline_id = uuid4()  # PipelineId is a type alias for UUID
        command = GetPipelineQuery(pipeline_id=str(pipeline_id))
        result = await service.get_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Runtime error in get" in result.error

    @pytest.mark.asyncio
    async def test_get_pipeline_connection_error(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test get_pipeline ConnectionError handling (line 167)."""
        # Make repository.get_by_id raise ConnectionError
        mock_repository.get_by_id.side_effect = ConnectionError(
            "Connection failed in get"
        )

        from uuid import uuid4

        pipeline_id = uuid4()  # PipelineId is a type alias for UUID
        command = GetPipelineQuery(pipeline_id=str(pipeline_id))
        result = await service.get_pipeline(command)

        assert not result.is_success
        assert result.error
        assert "Repository error: Connection failed in get" in result.error

    @pytest.mark.asyncio
    async def test_get_pipeline_general_exception(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test get_pipeline general Exception handling (line 173)."""
        # Make repository.get_by_id raise a generic Exception
        mock_repository.get_by_id.side_effect = Exception("Generic error in get")

        from uuid import uuid4

        pipeline_id = uuid4()  # PipelineId is a type alias for UUID
        command = GetPipelineQuery(pipeline_id=str(pipeline_id))
        result = await service.get_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Generic error in get" in result.error

    @pytest.mark.asyncio
    async def test_deactivate_pipeline_runtime_error(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test deactivate_pipeline RuntimeError handling (line 201, 207)."""
        # Setup pipeline first - need to mock get_by_id method
        pipeline = Pipeline(
            pipeline_name=PipelineName(value="test"), pipeline_description="test"
        )
        mock_repository.get_by_id.return_value = pipeline

        # Make repository.save raise RuntimeError
        mock_repository.save.side_effect = RuntimeError("Runtime error in deactivate")

        pipeline_id = str(pipeline.pipeline_id.value)
        result = await service.deactivate_pipeline(pipeline_id)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Runtime error in deactivate" in result.error

    @pytest.mark.asyncio
    async def test_deactivate_pipeline_general_exception(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test deactivate_pipeline general Exception handling (line 207)."""
        # Setup pipeline first
        pipeline = Pipeline(
            pipeline_name=PipelineName(value="test"), pipeline_description="test"
        )
        mock_repository.get_by_id.return_value = pipeline

        # Make repository.save raise a generic Exception
        mock_repository.save.side_effect = Exception("Generic error in deactivate")

        pipeline_id = str(pipeline.pipeline_id.value)
        result = await service.deactivate_pipeline(pipeline_id)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Generic error in deactivate" in result.error


class TestEdgeCaseExceptionPaths:
    """Test edge case exception paths."""

    @pytest.fixture
    def mock_repository(self) -> AsyncMock:
        """Create a mock repository."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_repository: AsyncMock) -> PipelineService:
        """Create pipeline service with mock repository."""
        return PipelineService(pipeline_repo=mock_repository)

    @pytest.mark.asyncio
    async def test_create_pipeline_complex_exception_chain(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test complex exception chains in create_pipeline."""
        # Chain exceptions to hit multiple handlers
        mock_repository.save.side_effect = AttributeError("Test attribute error")

        command = CreatePipelineCommand(name="test", description="test")
        result = await service.create_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Test attribute error" in result.error

    @pytest.mark.asyncio
    async def test_execute_pipeline_nested_validation_error(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test nested validation errors in execute_pipeline."""
        # Test input validation error path with invalid UUID format
        command = ExecutePipelineCommand(pipeline_id="not-a-valid-uuid")
        result = await service.execute_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Input error:" in result.error

    @pytest.mark.asyncio
    async def test_repository_state_consistency_during_exceptions(
        self, service: PipelineService, mock_repository: AsyncMock
    ) -> None:
        """Test repository state consistency when exceptions occur."""
        # Test that repository state remains consistent even with exceptions

        # Setup a scenario where get succeeds but save fails
        pipeline = Pipeline(
            pipeline_name=PipelineName(value="test"), pipeline_description="test"
        )
        mock_repository.get_by_id.return_value = pipeline
        mock_repository.save.side_effect = ConnectionError("Connection lost")

        # Test deactivate_pipeline method that should have these exception handlers
        pipeline_id = str(pipeline.pipeline_id.value)
        result = await service.deactivate_pipeline(pipeline_id)

        # Should handle the error gracefully
        assert not result.is_success
        assert result.error is not None
        assert "Repository error: Connection lost" in result.error

        # Repository get_by_id should have been called
        mock_repository.get_by_id.assert_called_once()
        # Repository save should have been attempted
        mock_repository.save.assert_called_once()


class TestModuleCompleteness:
    """Test module completeness and exports."""

    def test_module_exports_completeness(self) -> None:
        """Test that all expected classes are properly exported."""
        from flext_core.application import pipeline

        # Test that all major classes are available
        assert hasattr(pipeline, "PipelineService")
        assert hasattr(pipeline, "CreatePipelineCommand")
        assert hasattr(pipeline, "ExecutePipelineCommand")
        assert hasattr(pipeline, "GetPipelineQuery")
        assert hasattr(pipeline, "ListPipelinesQuery")
        # DeactivatePipelineCommand doesn't exist in current implementation

    def test_service_initialization_edge_cases(self) -> None:
        """Test service initialization with edge cases."""
        from flext_core.infrastructure.memory import InMemoryRepository

        # Test with different repository types
        repo: InMemoryRepository[Pipeline, str] = InMemoryRepository()
        service = PipelineService(pipeline_repo=repo)  # type: ignore[arg-type]

        assert service._repo is repo  # type: ignore[comparison-overlap]

    def test_command_query_model_validation(self) -> None:
        """Test command and query model validation completeness."""
        # Test edge cases in model validation

        # Test CreatePipelineCommand with edge values
        command = CreatePipelineCommand(name="a", description="")  # Minimal values
        assert command.name == "a"
        assert command.description == ""

        # Test ExecutePipelineCommand
        from uuid import uuid4

        pipeline_id = uuid4()  # PipelineId is a type alias for UUID
        execute_cmd = ExecutePipelineCommand(pipeline_id=str(pipeline_id))
        assert execute_cmd.pipeline_id == str(pipeline_id)

        # Test queries
        get_query = GetPipelineQuery(pipeline_id=str(pipeline_id))
        assert get_query.pipeline_id == str(pipeline_id)

        list_query = ListPipelinesQuery(limit=1, offset=0)
        assert list_query.limit == 1
        assert list_query.offset == 0
