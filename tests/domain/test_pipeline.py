"""Tests for Pipeline domain entities."""

import pytest

from flext_core import Pipeline
from flext_core import PipelineId
from flext_core import PipelineName
from flext_core import ServiceResult
from flext_core.domain.pipeline import ExecutionId


def test_pipeline_name_string_conversion() -> None:
    """Test PipelineName __str__ method (line 80)."""
    pipeline_name = PipelineName(value="Test Pipeline String")
    # Test the __str__ method directly
    str_result = str(pipeline_name)
    assert str_result == "Test Pipeline String"


def test_execution_id_string_conversion() -> None:
    """Test ExecutionId __str__ method (line 95)."""
    execution_id = ExecutionId()
    # Test the __str__ method directly
    str_result = str(execution_id)
    # Should be a valid UUID string
    assert len(str_result) == 36  # UUID format
    assert "-" in str_result


def test_pipeline_deactivate_method() -> None:
    """Test Pipeline deactivate method (lines 165-166)."""
    pipeline_name = PipelineName(value="Deactivation Test")
    pipeline = Pipeline(
        pipeline_name=pipeline_name,
        pipeline_description="Test deactivate method",
    )

    # Initially active
    assert pipeline.pipeline_is_active is True

    # Call deactivate method directly
    pipeline.deactivate()

    # Should be deactivated
    assert pipeline.pipeline_is_active is False
    # updated_at should be set
    assert pipeline.updated_at is not None


def test_pipeline_creation() -> None:
    """Test basic pipeline creation."""
    pipeline_name = PipelineName(value="Test Pipeline")
    pipeline = Pipeline(
        pipeline_name=pipeline_name,
        pipeline_description="A test pipeline",
    )

    assert pipeline.pipeline_name == pipeline_name
    assert pipeline.pipeline_description == "A test pipeline"
    assert pipeline.pipeline_is_active is True


def test_pipeline_name_validation() -> None:
    """Test pipeline name validation."""
    # Valid name
    valid_name = PipelineName(value="Test Pipeline")
    assert valid_name.value == "Test Pipeline"

    # Test that empty string raises validation error
    with pytest.raises(ValueError):
        PipelineName(value="")

    # Test that whitespace-only string raises validation error
    with pytest.raises(ValueError):
        PipelineName(value="   ")


def test_pipeline_id_creation() -> None:
    """Test pipeline ID creation."""
    # Default factory creates UUID
    pipeline_id = PipelineId()
    assert pipeline_id.value is not None

    # Can be converted to string
    pipeline_id_str = str(pipeline_id)
    assert isinstance(pipeline_id_str, str)


def test_pipeline_execution() -> None:
    """Test pipeline execution."""
    pipeline_name = PipelineName(value="Test Pipeline")
    pipeline = Pipeline(
        pipeline_name=pipeline_name,
        pipeline_description="A test pipeline",
    )

    execution = pipeline.execute()

    assert execution.pipeline_id == pipeline.pipeline_id
    assert execution.execution_status == "running"
    assert execution.started_at is not None


def test_pipeline_create_event() -> None:
    """Test pipeline create event emission."""
    pipeline_name = PipelineName(value="Test Pipeline")
    pipeline = Pipeline(
        pipeline_name=pipeline_name,
        pipeline_description="A test pipeline",
    )

    pipeline.create()

    # Check that event was added
    assert len(pipeline.events) > 0
    assert pipeline.updated_at is not None


def test_service_result_success() -> None:
    """Test ServiceResult success case."""
    result = ServiceResult.success("Operation completed")

    assert result.is_success
    assert result.value == "Operation completed"
    assert result.error is None


def test_service_result_failure() -> None:
    """Test ServiceResult failure case."""
    result: ServiceResult[str] = ServiceResult.failure("Something went wrong")

    assert not result.is_success
    assert result.value is None
    assert result.error == "Something went wrong"
