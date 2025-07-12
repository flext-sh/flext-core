"""Tests for flext_core.application.pipeline module."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from uuid import uuid4

import pytest

from flext_core.application.pipeline import CreatePipelineCommand
from flext_core.application.pipeline import ExecutePipelineCommand
from flext_core.application.pipeline import GetPipelineQuery
from flext_core.application.pipeline import ListPipelinesQuery
from flext_core.application.pipeline import PipelineService


class TestCreatePipelineCommand:
    """Test CreatePipelineCommand validation."""

    def test_create_command_valid(self) -> None:
        """Test valid create command."""
        command = CreatePipelineCommand(
            name="Test Pipeline",
            description="Test Description",
        )
        assert command.name == "Test Pipeline"
        assert command.description == "Test Description"

    def test_create_command_defaults(self) -> None:
        """Test create command with defaults."""
        command = CreatePipelineCommand(name="Test Pipeline")
        assert command.name == "Test Pipeline"
        assert command.description == ""

    def test_create_command_validation_error(self) -> None:
        """Test create command validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            CreatePipelineCommand(name="x" * 101)  # Too long


class TestExecutePipelineCommand:
    """Test ExecutePipelineCommand validation."""

    def test_execute_command_valid(self) -> None:
        """Test valid execute command."""
        pipeline_id = str(uuid4())
        command = ExecutePipelineCommand(pipeline_id=pipeline_id)
        assert command.pipeline_id == pipeline_id


class TestGetPipelineQuery:
    """Test GetPipelineQuery validation."""

    def test_get_query_valid(self) -> None:
        """Test valid get query."""
        pipeline_id = str(uuid4())
        query = GetPipelineQuery(pipeline_id=pipeline_id)
        assert query.pipeline_id == pipeline_id


class TestListPipelinesQuery:
    """Test ListPipelinesQuery validation."""

    def test_list_query_defaults(self) -> None:
        """Test list query with defaults."""
        query = ListPipelinesQuery()
        assert query.limit == 100
        assert query.offset == 0
        assert query.active_only is True

    def test_list_query_custom_values(self) -> None:
        """Test list query with custom values."""
        query = ListPipelinesQuery(limit=50, offset=10, active_only=False)
        assert query.limit == 50
        assert query.offset == 10
        assert query.active_only is False


class TestPipelineService:
    """Test PipelineService functionality."""

    @pytest.fixture
    def mock_repository(self) -> Mock:
        """Create mock pipeline repository."""
        repo = Mock()
        repo.save = AsyncMock()
        repo.get_by_id = AsyncMock()
        return repo

    @pytest.fixture
    def pipeline_service(self, mock_repository: Mock) -> PipelineService:
        """Create PipelineService with mock repository."""
        return PipelineService(mock_repository)

    @pytest.mark.asyncio
    async def test_create_pipeline_success(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test successful pipeline creation."""
        command = CreatePipelineCommand(
            name="Test Pipeline",
            description="Test Description",
        )

        # Mock successful save
        mock_repository.save.return_value = Mock()  # Return some pipeline object

        result = await pipeline_service.create_pipeline(command)

        assert result.is_success
        mock_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_pipeline_repository_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test pipeline creation with repository error."""
        command = CreatePipelineCommand(name="Test Pipeline")

        # Mock repository error
        mock_repository.save.side_effect = Exception("Database error")

        result = await pipeline_service.create_pipeline(command)

        assert not result.is_success
        assert "Repository error" in result.error

    @pytest.mark.asyncio
    async def test_get_pipeline_success(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test successful pipeline retrieval."""
        pipeline_id = str(uuid4())
        query = GetPipelineQuery(pipeline_id=pipeline_id)

        # Mock successful get
        mock_pipeline = Mock()
        mock_repository.get_by_id.return_value = mock_pipeline

        result = await pipeline_service.get_pipeline(query)

        assert result.is_success
        assert result.value == mock_pipeline
        mock_repository.get_by_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pipeline_not_found(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test pipeline retrieval when not found."""
        pipeline_id = str(uuid4())
        query = GetPipelineQuery(pipeline_id=pipeline_id)

        # Mock not found
        mock_repository.get_by_id.return_value = None

        result = await pipeline_service.get_pipeline(query)

        assert not result.is_success
        assert "Pipeline not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_pipeline_success(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test successful pipeline execution."""
        pipeline_id = str(uuid4())
        command = ExecutePipelineCommand(pipeline_id=pipeline_id)

        # Mock active pipeline
        mock_pipeline = Mock()
        mock_pipeline.pipeline_is_active = True
        mock_pipeline.execute.return_value = Mock()  # Mock execution
        mock_repository.get_by_id.return_value = mock_pipeline

        result = await pipeline_service.execute_pipeline(command)

        assert result.is_success
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_pipeline_not_found(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test pipeline execution when pipeline not found."""
        pipeline_id = str(uuid4())
        command = ExecutePipelineCommand(pipeline_id=pipeline_id)

        # Mock not found
        mock_repository.get_by_id.return_value = None

        result = await pipeline_service.execute_pipeline(command)

        assert not result.is_success
        assert "Pipeline not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_pipeline_inactive(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test pipeline execution when pipeline is inactive."""
        pipeline_id = str(uuid4())
        command = ExecutePipelineCommand(pipeline_id=pipeline_id)

        # Mock inactive pipeline
        mock_pipeline = Mock()
        mock_pipeline.pipeline_is_active = False
        mock_repository.get_by_id.return_value = mock_pipeline

        result = await pipeline_service.execute_pipeline(command)

        assert not result.is_success
        assert "Pipeline is inactive" in result.error

    @pytest.mark.asyncio
    async def test_deactivate_pipeline_success(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test successful pipeline deactivation."""
        pipeline_id = str(uuid4())

        # Mock active pipeline
        mock_pipeline = Mock()
        mock_repository.get_by_id.return_value = mock_pipeline
        mock_repository.save.return_value = mock_pipeline

        result = await pipeline_service.deactivate_pipeline(pipeline_id)

        assert result.is_success
        mock_pipeline.deactivate.assert_called_once()
        mock_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_pipeline_not_found(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test pipeline deactivation when not found."""
        pipeline_id = str(uuid4())

        # Mock not found
        mock_repository.get_by_id.return_value = None

        result = await pipeline_service.deactivate_pipeline(pipeline_id)

        assert not result.is_success
        assert "Pipeline not found" in result.error


class TestPipelineServiceIntegration:
    """Test pipeline service integration scenarios."""

    @pytest.fixture
    def integration_service(self) -> tuple[PipelineService, Mock]:
        """Create pipeline service for integration testing."""
        mock_repo = Mock()
        mock_repo.save = AsyncMock()
        mock_repo.get_by_id = AsyncMock()
        return PipelineService(mock_repo), mock_repo

    @pytest.mark.asyncio
    async def test_create_and_execute_pipeline_flow(
        self, integration_service: tuple[PipelineService, Mock]
    ) -> None:
        """Test complete pipeline creation and execution flow."""
        service, mock_repo = integration_service

        # Step 1: Create pipeline
        create_command = CreatePipelineCommand(
            name="Integration Test Pipeline",
            description="Full flow test",
        )

        mock_pipeline = Mock()
        mock_pipeline.pipeline_is_active = True
        mock_pipeline.execute.return_value = Mock()
        mock_repo.save.return_value = mock_pipeline
        mock_repo.get_by_id.return_value = mock_pipeline

        create_result = await service.create_pipeline(create_command)
        assert create_result.is_success

        # Step 2: Execute pipeline
        execute_command = ExecutePipelineCommand(
            pipeline_id=str(uuid4()),
        )

        execute_result = await service.execute_pipeline(execute_command)
        assert execute_result.is_success

    @pytest.mark.asyncio
    async def test_error_handling_chain(
        self, integration_service: tuple[PipelineService, Mock]
    ) -> None:
        """Test error handling across service operations."""
        service, mock_repo = integration_service

        # Test repository error propagation
        mock_repo.save.side_effect = Exception("Database connection failed")

        command = CreatePipelineCommand(name="Test Pipeline")
        result = await service.create_pipeline(command)

        assert not result.is_success
        assert "Repository error" in result.error
        assert "Database connection failed" in result.error
