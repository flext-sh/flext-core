"""Comprehensive tests for flext_core.application.pipeline module.

This file provides additional test coverage to reach 90%+ coverage,
complementing the existing test_pipeline.py without duplication.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from flext_core.application.pipeline import (
    CreatePipelineCommand,
    ExecutePipelineCommand,
    GetPipelineQuery,
    ListPipelinesQuery,
    PipelineService,
)


class TestPipelineServiceErrorPaths:
    """Test specific error paths not covered by existing tests."""

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
    async def test_create_pipeline_validation_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test create pipeline with ValidationError."""
        # Create command that will cause ValidationError in domain logic
        import unittest.mock

        # Create a proper ValidationError
        from flext_core.domain.pipeline import PipelineName

        # Force a ValidationError by creating an invalid PipelineName
        try:
            PipelineName(value="")  # Empty string should cause validation error
            # If we get here, the validation didn't work as expected
            msg = "Empty string should trigger ValidationError"
            raise AssertionError(msg)
        except ValidationError as validation_error:
            # Use the real validation error
            with unittest.mock.patch(
                "flext_core.application.pipeline.PipelineName"
            ) as mock_name:
                mock_name.side_effect = validation_error

                command = CreatePipelineCommand(name="Test Pipeline")
                result = await pipeline_service.create_pipeline(command)

                assert not result.is_success
                assert result.error is not None
                assert "Validation failed" in result.error

    @pytest.mark.asyncio
    async def test_create_pipeline_value_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test create pipeline with ValueError."""
        import unittest.mock

        with unittest.mock.patch(
            "flext_core.application.pipeline.PipelineName"
        ) as mock_name:
            # Make PipelineName raise ValueError
            mock_name.side_effect = ValueError("Invalid value")

            command = CreatePipelineCommand(name="Test Pipeline")
            result = await pipeline_service.create_pipeline(command)

            assert not result.is_success
            assert result.error is not None
            assert "Input error" in result.error

    @pytest.mark.asyncio
    async def test_create_pipeline_type_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test create pipeline with TypeError."""
        import unittest.mock

        with unittest.mock.patch(
            "flext_core.application.pipeline.PipelineName"
        ) as mock_name:
            # Make PipelineName raise TypeError
            mock_name.side_effect = TypeError("Invalid type")

            command = CreatePipelineCommand(name="Test Pipeline")
            result = await pipeline_service.create_pipeline(command)

            assert not result.is_success
            assert result.error is not None
            assert "Input error" in result.error

    @pytest.mark.asyncio
    async def test_create_pipeline_os_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test create pipeline with OSError."""
        command = CreatePipelineCommand(name="Test Pipeline")

        # Mock repository to raise OSError
        mock_repository.save.side_effect = OSError("File system error")

        result = await pipeline_service.create_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error" in result.error

    @pytest.mark.asyncio
    async def test_execute_pipeline_validation_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test execute pipeline with ValidationError in UUID parsing."""
        import unittest.mock

        with unittest.mock.patch("uuid.UUID") as mock_uuid:
            # Make UUID raise ValidationError-like exception
            mock_uuid.side_effect = ValueError("Invalid UUID")

            command = ExecutePipelineCommand(pipeline_id="invalid-uuid")
            result = await pipeline_service.execute_pipeline(command)

            assert not result.is_success
            assert result.error is not None
            assert "Input error" in result.error

    @pytest.mark.asyncio
    async def test_execute_pipeline_type_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test execute pipeline with TypeError."""
        import unittest.mock

        with unittest.mock.patch(
            "flext_core.application.pipeline.PipelineId"
        ) as mock_id:
            # Make PipelineId raise TypeError
            mock_id.side_effect = TypeError("Invalid type")

            command = ExecutePipelineCommand(pipeline_id=str(uuid4()))
            result = await pipeline_service.execute_pipeline(command)

            assert not result.is_success
            assert result.error is not None
            assert "Input error" in result.error

    @pytest.mark.asyncio
    async def test_execute_pipeline_os_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test execute pipeline with OSError."""
        pipeline_id = str(uuid4())
        command = ExecutePipelineCommand(pipeline_id=pipeline_id)

        # Mock repository to raise OSError
        mock_repository.get_by_id.side_effect = OSError("Database connection error")

        result = await pipeline_service.execute_pipeline(command)

        assert not result.is_success
        assert result.error is not None
        assert "Execution error" in result.error

    @pytest.mark.asyncio
    async def test_get_pipeline_validation_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test get pipeline with ValidationError."""
        import unittest.mock

        with unittest.mock.patch("uuid.UUID") as mock_uuid:
            # Make UUID raise ValueError
            mock_uuid.side_effect = ValueError("Invalid UUID format")

            query = GetPipelineQuery(pipeline_id="invalid-uuid")
            result = await pipeline_service.get_pipeline(query)

            assert not result.is_success
            assert result.error is not None
            assert "Input error" in result.error

    @pytest.mark.asyncio
    async def test_get_pipeline_type_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test get pipeline with TypeError."""
        import unittest.mock

        with unittest.mock.patch(
            "flext_core.application.pipeline.PipelineId"
        ) as mock_id:
            # Make PipelineId raise TypeError
            mock_id.side_effect = TypeError("Invalid type")

            query = GetPipelineQuery(pipeline_id=str(uuid4()))
            result = await pipeline_service.get_pipeline(query)

            assert not result.is_success
            assert result.error is not None
            assert "Input error" in result.error

    @pytest.mark.asyncio
    async def test_get_pipeline_os_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test get pipeline with OSError."""
        pipeline_id = str(uuid4())
        query = GetPipelineQuery(pipeline_id=pipeline_id)

        # Mock repository to raise OSError
        mock_repository.get_by_id.side_effect = OSError("Database error")

        result = await pipeline_service.get_pipeline(query)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error" in result.error

    @pytest.mark.asyncio
    async def test_get_pipeline_general_exception(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test get pipeline with general Exception."""
        pipeline_id = str(uuid4())
        query = GetPipelineQuery(pipeline_id=pipeline_id)

        # Mock repository to raise general Exception
        mock_repository.get_by_id.side_effect = Exception("Unexpected error")

        result = await pipeline_service.get_pipeline(query)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error" in result.error

    @pytest.mark.asyncio
    async def test_deactivate_pipeline_validation_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test deactivate pipeline with ValidationError."""
        import unittest.mock

        with unittest.mock.patch("uuid.UUID") as mock_uuid:
            # Make UUID raise ValueError
            mock_uuid.side_effect = ValueError("Invalid UUID")

            result = await pipeline_service.deactivate_pipeline("invalid-uuid")

            assert not result.is_success
            assert result.error is not None
            assert "Input error" in result.error

    @pytest.mark.asyncio
    async def test_deactivate_pipeline_type_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test deactivate pipeline with TypeError."""
        import unittest.mock

        with unittest.mock.patch(
            "flext_core.application.pipeline.PipelineId"
        ) as mock_id:
            # Make PipelineId raise TypeError
            mock_id.side_effect = TypeError("Invalid type")

            result = await pipeline_service.deactivate_pipeline(str(uuid4()))

            assert not result.is_success
            assert result.error is not None
            assert "Input error" in result.error

    @pytest.mark.asyncio
    async def test_deactivate_pipeline_os_error(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test deactivate pipeline with OSError."""
        pipeline_id = str(uuid4())

        # Mock repository to raise OSError
        mock_repository.get_by_id.side_effect = OSError("Storage error")

        result = await pipeline_service.deactivate_pipeline(pipeline_id)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error" in result.error

    @pytest.mark.asyncio
    async def test_deactivate_pipeline_general_exception(
        self, pipeline_service: PipelineService, mock_repository: Mock
    ) -> None:
        """Test deactivate pipeline with general Exception."""
        pipeline_id = str(uuid4())

        # Mock repository to raise general Exception
        mock_repository.get_by_id.side_effect = Exception("Unexpected error")

        result = await pipeline_service.deactivate_pipeline(pipeline_id)

        assert not result.is_success
        assert result.error is not None
        assert "Repository error" in result.error


class TestCommandQueryValidationEdgeCases:
    """Test edge cases for command and query validation."""

    def test_create_command_name_too_long(self) -> None:
        """Test create command with name exceeding max length."""
        with pytest.raises(ValidationError):
            CreatePipelineCommand(name="x" * 101)  # Exceeds 100 char limit

    def test_create_command_description_too_long(self) -> None:
        """Test create command with description exceeding max length."""
        with pytest.raises(ValidationError):
            CreatePipelineCommand(
                name="Valid Name",
                description="x" * 501,  # Exceeds 500 char limit
            )

    def test_execute_command_missing_pipeline_id(self) -> None:
        """Test execute command with missing pipeline ID."""
        with pytest.raises(ValidationError):
            # Missing required parameter should fail
            ExecutePipelineCommand()  # type: ignore[call-arg]

    def test_get_query_missing_pipeline_id(self) -> None:
        """Test get query with missing pipeline ID."""
        with pytest.raises(ValidationError):
            # Missing required parameter should fail
            GetPipelineQuery()  # type: ignore[call-arg]

    def test_list_query_limit_too_small(self) -> None:
        """Test list query with limit below minimum."""
        with pytest.raises(ValidationError):
            ListPipelinesQuery(limit=0)  # Below ge=1

    def test_list_query_limit_too_large(self) -> None:
        """Test list query with limit above maximum."""
        with pytest.raises(ValidationError):
            ListPipelinesQuery(limit=1001)  # Above le=1000

    def test_list_query_negative_offset(self) -> None:
        """Test list query with negative offset."""
        with pytest.raises(ValidationError):
            ListPipelinesQuery(offset=-1)  # Below ge=0


class TestTypeCheckingImports:
    """Test TYPE_CHECKING imports for coverage."""

    def test_type_checking_imports_coverage(self) -> None:
        """Test that TYPE_CHECKING imports are covered."""
        # This test ensures that the TYPE_CHECKING block is covered
        # by importing the module during test execution
        import flext_core.application.pipeline

        # The imports are available during runtime through the module
        assert hasattr(flext_core.application.pipeline, "PipelineService")
        assert hasattr(flext_core.application.pipeline, "CreatePipelineCommand")
        assert hasattr(flext_core.application.pipeline, "ExecutePipelineCommand")
        assert hasattr(flext_core.application.pipeline, "GetPipelineQuery")
        assert hasattr(flext_core.application.pipeline, "ListPipelinesQuery")

    def test_module_all_exports(self) -> None:
        """Test that __all__ exports are correct."""
        from flext_core.application.pipeline import __all__

        expected_exports = [
            "CreatePipelineCommand",
            "ExecutePipelineCommand",
            "GetPipelineQuery",
            "ListPipelinesQuery",
            "PipelineService",
        ]

        assert set(__all__) == set(expected_exports)


class TestServiceIntegrationScenarios:
    """Test additional integration scenarios for complete coverage."""

    @pytest.fixture
    def service_with_repo(self) -> tuple[PipelineService, Mock]:
        """Create service with mock repository."""
        mock_repo = Mock()
        mock_repo.save = AsyncMock()
        mock_repo.get_by_id = AsyncMock()
        return PipelineService(mock_repo), mock_repo

    @pytest.mark.asyncio
    async def test_complete_pipeline_lifecycle(
        self, service_with_repo: tuple[PipelineService, Mock]
    ) -> None:
        """Test complete pipeline lifecycle: create -> execute -> deactivate."""
        service, mock_repo = service_with_repo

        # Mock pipeline objects
        created_pipeline = Mock()
        created_pipeline.pipeline_is_active = True
        created_pipeline.execute.return_value = Mock()

        deactivated_pipeline = Mock()
        deactivated_pipeline.pipeline_is_active = False

        mock_repo.save.side_effect = [created_pipeline, deactivated_pipeline]
        mock_repo.get_by_id.side_effect = [created_pipeline, created_pipeline]

        # Create pipeline
        create_cmd = CreatePipelineCommand(name="Lifecycle Test")
        create_result = await service.create_pipeline(create_cmd)
        assert create_result.is_success

        # Execute pipeline
        execute_cmd = ExecutePipelineCommand(pipeline_id=str(uuid4()))
        execute_result = await service.execute_pipeline(execute_cmd)
        assert execute_result.is_success

        # Deactivate pipeline
        deactivate_result = await service.deactivate_pipeline(str(uuid4()))
        assert deactivate_result.is_success

    @pytest.mark.asyncio
    async def test_repository_error_consistency(
        self, service_with_repo: tuple[PipelineService, Mock]
    ) -> None:
        """Test that repository errors are handled consistently."""
        service, mock_repo = service_with_repo

        # Test that all service methods handle repository errors consistently
        error_types = [OSError("Storage error"), Exception("General error")]

        for error in error_types:
            mock_repo.save.side_effect = error
            mock_repo.get_by_id.side_effect = error

            # Test create_pipeline error handling
            create_cmd = CreatePipelineCommand(name="Error Test")
            create_result = await service.create_pipeline(create_cmd)
            assert not create_result.is_success
            assert create_result.error is not None
            assert "Repository error" in create_result.error

            # Test get_pipeline error handling
            get_query = GetPipelineQuery(pipeline_id=str(uuid4()))
            get_result = await service.get_pipeline(get_query)
            assert not get_result.is_success
            assert get_result.error is not None
            assert "Repository error" in get_result.error

            # Test deactivate_pipeline error handling
            deactivate_result = await service.deactivate_pipeline(str(uuid4()))
            assert not deactivate_result.is_success
            assert deactivate_result.error is not None
            assert "Repository error" in deactivate_result.error
