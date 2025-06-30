"""Unit tests for application services.

Tests for pipeline and execution services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from flx_core.application.services import ExecutionService, PipelineService
from flx_core.domain.advanced_types import ServiceResult
from flx_core.domain.entities import (
    Pipeline,
    PipelineExecution,
    PipelineId,
    PipelineName,
)
from flx_core.domain.value_objects import ExecutionStatus

if TYPE_CHECKING:
    pass

# Python 3.13 type aliases
type MockUnitOfWork = AsyncMock
type MockRepository = AsyncMock
type MockEventBus = AsyncMock


class TestPipelineService:
    """Test PipelineService."""

    @pytest.fixture
    def mock_unit_of_work(self) -> MockUnitOfWork:
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.__aenter__.return_value = uow
        uow.__aexit__.return_value = None

        # Create mock repositories
        uow.pipelines = AsyncMock()
        uow.executions = AsyncMock()
        uow.plugins = AsyncMock()

        return uow

    @pytest.fixture
    def mock_event_bus(self) -> MockEventBus:
        """Create mock event bus."""
        return AsyncMock()

    @pytest.fixture
    def pipeline_service(
        self, mock_unit_of_work: MockUnitOfWork, mock_event_bus: MockEventBus
    ) -> PipelineService:
        """Create pipeline service with mocks."""
        return PipelineService(unit_of_work=mock_unit_of_work, event_bus=mock_event_bus)

    async def test_create_pipeline_success(
        self,
        pipeline_service: PipelineService,
        mock_unit_of_work: MockUnitOfWork,
        mock_event_bus: MockEventBus,
    ) -> None:
        """Test successful pipeline creation."""
        # Arrange
        pipeline_data = {
            "name": "test_pipeline",
            "description": "Test pipeline",
            "steps": [],
        }

        mock_unit_of_work.pipelines.find_by_name.return_value = None
        mock_unit_of_work.pipelines.save.return_value = None
        mock_unit_of_work.commit.return_value = None

        # Act
        result = await pipeline_service.create_pipeline(pipeline_data)

        # Assert
        assert result.is_success is True
        assert result.value is not None
        assert isinstance(result.value, Pipeline)
        assert result.value.name.value == "test_pipeline"

        mock_unit_of_work.pipelines.save.assert_called_once()
        mock_unit_of_work.commit.assert_called_once()
        mock_event_bus.publish.assert_called_once()

    async def test_create_pipeline_duplicate_name(
        self, pipeline_service: PipelineService, mock_unit_of_work: MockUnitOfWork
    ) -> None:
        """Test pipeline creation with duplicate name."""
        # Arrange
        pipeline_data = {
            "name": "existing_pipeline",
            "description": "Test pipeline",
            "steps": [],
        }

        existing_pipeline = Pipeline(
            pipeline_id=PipelineId(),
            name=PipelineName(value="existing_pipeline"),
            description="Existing pipeline",
        )

        mock_unit_of_work.pipelines.find_by_name.return_value = existing_pipeline

        # Act
        result = await pipeline_service.create_pipeline(pipeline_data)

        # Assert
        assert result.is_failure is True
        assert result.error.code == "DUPLICATE_NAME"
        assert "already exists" in result.error.message

        mock_unit_of_work.pipelines.save.assert_not_called()
        mock_unit_of_work.commit.assert_not_called()

    async def test_get_pipeline_success(
        self,
        pipeline_service: PipelineService,
        mock_unit_of_work: MockUnitOfWork,
        sample_pipeline: Pipeline,
    ) -> None:
        """Test successful pipeline retrieval."""
        # Arrange
        pipeline_id = sample_pipeline.pipeline_id
        mock_unit_of_work.pipelines.find_by_id.return_value = sample_pipeline

        # Act
        result = await pipeline_service.get_pipeline(str(pipeline_id.value))

        # Assert
        assert result.is_success is True
        assert result.value == sample_pipeline

        mock_unit_of_work.pipelines.find_by_id.assert_called_once_with(pipeline_id)

    async def test_get_pipeline_not_found(
        self, pipeline_service: PipelineService, mock_unit_of_work: MockUnitOfWork
    ) -> None:
        """Test pipeline retrieval when not found."""
        # Arrange
        pipeline_id = str(uuid4())
        mock_unit_of_work.pipelines.find_by_id.return_value = None

        # Act
        result = await pipeline_service.get_pipeline(pipeline_id)

        # Assert
        assert result.is_failure is True
        assert result.error.code == "NOT_FOUND"
        assert "Pipeline" in result.error.message

    async def test_update_pipeline_success(
        self,
        pipeline_service: PipelineService,
        mock_unit_of_work: MockUnitOfWork,
        mock_event_bus: MockEventBus,
        sample_pipeline: Pipeline,
    ) -> None:
        """Test successful pipeline update."""
        # Arrange
        pipeline_id = str(sample_pipeline.pipeline_id.value)
        update_data = {
            "description": "Updated description",
            "max_concurrent_executions": 5,
        }

        mock_unit_of_work.pipelines.find_by_id.return_value = sample_pipeline
        mock_unit_of_work.pipelines.update.return_value = None
        mock_unit_of_work.commit.return_value = None

        # Act
        result = await pipeline_service.update_pipeline(pipeline_id, update_data)

        # Assert
        assert result.is_success is True
        assert result.value.description == "Updated description"
        assert result.value.max_concurrent_executions == 5

        mock_unit_of_work.pipelines.update.assert_called_once()
        mock_unit_of_work.commit.assert_called_once()
        mock_event_bus.publish.assert_called_once()

    async def test_delete_pipeline_success(
        self,
        pipeline_service: PipelineService,
        mock_unit_of_work: MockUnitOfWork,
        mock_event_bus: MockEventBus,
        sample_pipeline: Pipeline,
    ) -> None:
        """Test successful pipeline deletion."""
        # Arrange
        pipeline_id = str(sample_pipeline.pipeline_id.value)

        mock_unit_of_work.pipelines.find_by_id.return_value = sample_pipeline
        mock_unit_of_work.executions.find_active_by_pipeline.return_value = []
        mock_unit_of_work.pipelines.delete.return_value = None
        mock_unit_of_work.commit.return_value = None

        # Act
        result = await pipeline_service.delete_pipeline(pipeline_id)

        # Assert
        assert result.is_success is True

        mock_unit_of_work.pipelines.delete.assert_called_once_with(sample_pipeline)
        mock_unit_of_work.commit.assert_called_once()
        mock_event_bus.publish.assert_called_once()

    async def test_delete_pipeline_with_active_executions(
        self,
        pipeline_service: PipelineService,
        mock_unit_of_work: MockUnitOfWork,
        sample_pipeline: Pipeline,
        sample_execution: PipelineExecution,
    ) -> None:
        """Test pipeline deletion with active executions."""
        # Arrange
        pipeline_id = str(sample_pipeline.pipeline_id.value)
        sample_execution.status = ExecutionStatus.RUNNING

        mock_unit_of_work.pipelines.find_by_id.return_value = sample_pipeline
        mock_unit_of_work.executions.find_active_by_pipeline.return_value = [
            sample_execution
        ]

        # Act
        result = await pipeline_service.delete_pipeline(pipeline_id)

        # Assert
        assert result.is_failure is True
        assert result.error.code == "ACTIVE_EXECUTIONS"
        assert "active executions" in result.error.message

        mock_unit_of_work.pipelines.delete.assert_not_called()
        mock_unit_of_work.commit.assert_not_called()


class TestExecutionService:
    """Test ExecutionService."""

    @pytest.fixture
    def mock_unit_of_work(self) -> MockUnitOfWork:
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.__aenter__.return_value = uow
        uow.__aexit__.return_value = None

        # Create mock repositories
        uow.pipelines = AsyncMock()
        uow.executions = AsyncMock()

        return uow

    @pytest.fixture
    def mock_event_bus(self) -> MockEventBus:
        """Create mock event bus."""
        return AsyncMock()

    @pytest.fixture
    def mock_executor(self) -> AsyncMock:
        """Create mock executor."""
        return AsyncMock()

    @pytest.fixture
    def execution_service(
        self,
        mock_unit_of_work: MockUnitOfWork,
        mock_event_bus: MockEventBus,
        mock_executor: AsyncMock,
    ) -> ExecutionService:
        """Create execution service with mocks."""
        return ExecutionService(
            unit_of_work=mock_unit_of_work,
            event_bus=mock_event_bus,
            executor=mock_executor,
        )

    async def test_start_execution_success(
        self,
        execution_service: ExecutionService,
        mock_unit_of_work: MockUnitOfWork,
        mock_event_bus: MockEventBus,
        mock_executor: AsyncMock,
        sample_pipeline: Pipeline,
    ) -> None:
        """Test successful execution start."""
        # Arrange
        pipeline_id = str(sample_pipeline.pipeline_id.value)
        sample_pipeline.is_active = True

        mock_unit_of_work.pipelines.find_by_id.return_value = sample_pipeline
        mock_unit_of_work.executions.count_active_by_pipeline.return_value = 0
        mock_unit_of_work.executions.save.return_value = None
        mock_unit_of_work.commit.return_value = None
        mock_executor.execute_pipeline.return_value = ServiceResult.ok(None)

        # Act
        result = await execution_service.start_execution(pipeline_id)

        # Assert
        assert result.is_success is True
        assert isinstance(result.value, PipelineExecution)
        assert result.value.status == ExecutionStatus.PENDING

        mock_unit_of_work.executions.save.assert_called_once()
        mock_unit_of_work.commit.assert_called()
        mock_event_bus.publish.assert_called()
        mock_executor.execute_pipeline.assert_called_once()

    async def test_start_execution_inactive_pipeline(
        self,
        execution_service: ExecutionService,
        mock_unit_of_work: MockUnitOfWork,
        sample_pipeline: Pipeline,
    ) -> None:
        """Test starting execution for inactive pipeline."""
        # Arrange
        pipeline_id = str(sample_pipeline.pipeline_id.value)
        sample_pipeline.is_active = False

        mock_unit_of_work.pipelines.find_by_id.return_value = sample_pipeline

        # Act
        result = await execution_service.start_execution(pipeline_id)

        # Assert
        assert result.is_failure is True
        assert result.error.code == "PIPELINE_INACTIVE"

        mock_unit_of_work.executions.save.assert_not_called()

    async def test_start_execution_max_concurrent_reached(
        self,
        execution_service: ExecutionService,
        mock_unit_of_work: MockUnitOfWork,
        sample_pipeline: Pipeline,
    ) -> None:
        """Test starting execution when max concurrent limit reached."""
        # Arrange
        pipeline_id = str(sample_pipeline.pipeline_id.value)
        sample_pipeline.is_active = True
        sample_pipeline.max_concurrent_executions = 2

        mock_unit_of_work.pipelines.find_by_id.return_value = sample_pipeline
        mock_unit_of_work.executions.count_active_by_pipeline.return_value = 2

        # Act
        result = await execution_service.start_execution(pipeline_id)

        # Assert
        assert result.is_failure is True
        assert result.error.code == "MAX_CONCURRENT_EXECUTIONS"
        assert "concurrent executions limit" in result.error.message

        mock_unit_of_work.executions.save.assert_not_called()

    async def test_cancel_execution_success(
        self,
        execution_service: ExecutionService,
        mock_unit_of_work: MockUnitOfWork,
        mock_event_bus: MockEventBus,
        sample_execution: PipelineExecution,
    ) -> None:
        """Test successful execution cancellation."""
        # Arrange
        execution_id = str(sample_execution.execution_id.value)
        sample_execution.status = ExecutionStatus.RUNNING

        mock_unit_of_work.executions.find_by_id.return_value = sample_execution
        mock_unit_of_work.executions.update.return_value = None
        mock_unit_of_work.commit.return_value = None

        # Act
        result = await execution_service.cancel_execution(execution_id)

        # Assert
        assert result.is_success is True
        assert result.value.status == ExecutionStatus.CANCELLED

        mock_unit_of_work.executions.update.assert_called_once()
        mock_unit_of_work.commit.assert_called_once()
        mock_event_bus.publish.assert_called()

    async def test_cancel_execution_already_completed(
        self,
        execution_service: ExecutionService,
        mock_unit_of_work: MockUnitOfWork,
        sample_execution: PipelineExecution,
    ) -> None:
        """Test cancelling already completed execution."""
        # Arrange
        execution_id = str(sample_execution.execution_id.value)
        sample_execution.status = ExecutionStatus.COMPLETED

        mock_unit_of_work.executions.find_by_id.return_value = sample_execution

        # Act
        result = await execution_service.cancel_execution(execution_id)

        # Assert
        assert result.is_failure is True
        assert result.error.code == "INVALID_STATE"
        assert "Cannot cancel" in result.error.message

        mock_unit_of_work.executions.update.assert_not_called()
        mock_unit_of_work.commit.assert_not_called()
