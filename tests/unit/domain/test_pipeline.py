"""Pipeline domain tests - 100% coverage.

Tests for pipeline entities, value objects, and events matching actual implementation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError as PydanticValidationError

from flext_core.domain.pipeline import (
    ExecutionId,
    ExecutionStatus,
    Pipeline,
    PipelineCreated,
    PipelineExecuted,
    PipelineExecution,
    PipelineId,
    PipelineName,
)


class TestExecutionStatus:
    """Test ExecutionStatus enum."""

    def test_execution_status_values(self) -> None:
        """Test all execution status values."""
        # ExecutionStatus is a StrEnum, so values ARE strings
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"

    def test_execution_status_is_str_enum(self) -> None:
        """Test ExecutionStatus behaves as string."""
        status = ExecutionStatus.RUNNING
        assert isinstance(status, str)
        assert status == ExecutionStatus.RUNNING
        assert status.upper() == "RUNNING"

    def test_execution_status_comparison(self) -> None:
        """Test ExecutionStatus comparisons."""
        assert ExecutionStatus.PENDING == ExecutionStatus.PENDING
        # Test that different statuses are different
        pending = ExecutionStatus.PENDING
        running = ExecutionStatus.RUNNING
        success = ExecutionStatus.SUCCESS
        failed = ExecutionStatus.FAILED
        assert pending != running
        assert success != failed


class TestPipelineId:
    """Test PipelineId value object."""

    def test_pipeline_id_generation(self) -> None:
        """Test PipelineId auto-generates UUID."""
        id1 = PipelineId()
        id2 = PipelineId()

        assert isinstance(id1.value, UUID)
        assert isinstance(id2.value, UUID)
        assert id1.value != id2.value

    def test_pipeline_id_with_value(self) -> None:
        """Test PipelineId with specific UUID."""
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        pid = PipelineId(value=test_uuid)

        assert pid.value == test_uuid
        assert str(pid) == "12345678-1234-5678-1234-567812345678"

    def test_pipeline_id_hash_and_equality(self) -> None:
        """Test PipelineId hashing and equality."""
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        id1 = PipelineId(value=test_uuid)
        id2 = PipelineId(value=test_uuid)
        id3 = PipelineId()  # Different UUID

        assert id1 == id2
        assert id1 != id3
        assert hash(id1) == hash(id2)
        assert hash(id1) != hash(id3)

    def test_pipeline_id_immutable(self) -> None:
        """Test PipelineId is immutable."""
        pid = PipelineId()
        with pytest.raises(PydanticValidationError):
            pid.__dict__["value"] = UUID()  # Bypass type checking for test


class TestPipelineName:
    """Test PipelineName value object."""

    def test_valid_pipeline_name(self) -> None:
        """Test valid pipeline names."""
        name = PipelineName(value="my-pipeline")
        assert name.value == "my-pipeline"
        assert str(name) == "my-pipeline"

    def test_pipeline_name_strips_whitespace(self) -> None:
        """Test pipeline name strips whitespace."""
        name = PipelineName(value="  spaced-name  ")
        assert name.value == "spaced-name"

    def test_pipeline_name_validation(self) -> None:
        """Test pipeline name validation."""
        # Empty string
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            PipelineName(value="")

        # Only whitespace
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            PipelineName(value="   ")

        # Too long (>100 chars)
        with pytest.raises(PydanticValidationError):
            PipelineName(value="a" * 101)

    def test_pipeline_name_min_length(self) -> None:
        """Test pipeline name minimum length."""
        # Single character should work (min_length=1)
        name = PipelineName(value="a")
        assert name.value == "a"

    def test_pipeline_name_immutable(self) -> None:
        """Test PipelineName is immutable."""
        name = PipelineName(value="test")
        with pytest.raises(PydanticValidationError):
            name.__dict__["value"] = "changed"  # Bypass type checking for test


class TestExecutionId:
    """Test ExecutionId value object."""

    def test_execution_id_generation(self) -> None:
        """Test ExecutionId auto-generates UUID."""
        id1 = ExecutionId()
        id2 = ExecutionId()

        assert isinstance(id1.value, UUID)
        assert id1.value != id2.value
        assert str(id1) == str(id1.value)

    def test_execution_id_hash(self) -> None:
        """Test ExecutionId hashing."""
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        id1 = ExecutionId(value=test_uuid)
        id2 = ExecutionId(value=test_uuid)

        assert hash(id1) == hash(id2)
        assert id1 == id2


class TestPipelineExecution:
    """Test PipelineExecution entity."""

    def test_execution_creation(self) -> None:
        """Test execution entity creation."""
        pipeline_id = PipelineId()
        execution = PipelineExecution(pipeline_id=pipeline_id)

        assert execution.pipeline_id == pipeline_id
        assert execution.status == ExecutionStatus.PENDING
        assert execution.started_at <= datetime.now(UTC)
        assert execution.completed_at is None
        assert execution.result == {}
        assert execution.error_message is None
        assert isinstance(execution.id, ExecutionId)

    def test_execution_complete_success(self) -> None:
        """Test completing execution successfully."""
        execution = PipelineExecution(pipeline_id=PipelineId())
        result_data = {"output": "data", "count": 42}

        execution.complete(success=True, result=result_data)

        assert execution.status == ExecutionStatus.SUCCESS
        assert execution.completed_at is not None
        assert execution.result == result_data
        assert execution.error_message is None
        assert execution.updated_at == execution.completed_at

    def test_execution_complete_failure(self) -> None:
        """Test completing execution with failure."""
        execution = PipelineExecution(pipeline_id=PipelineId())

        execution.complete(success=False)

        assert execution.status == ExecutionStatus.FAILED
        assert execution.completed_at is not None
        assert execution.result == {}
        assert execution.error_message is None

    def test_execution_fail_with_error(self) -> None:
        """Test marking execution as failed with error."""
        execution = PipelineExecution(pipeline_id=PipelineId())
        error_msg = "Connection timeout"

        execution.fail(error_msg)

        assert execution.status == ExecutionStatus.FAILED
        assert execution.completed_at is not None
        assert execution.error_message == error_msg
        assert execution.updated_at == execution.completed_at

    def test_execution_with_custom_status(self) -> None:
        """Test execution with custom initial status."""
        execution = PipelineExecution(pipeline_id=PipelineId(), status=ExecutionStatus.RUNNING)

        assert execution.status == ExecutionStatus.RUNNING

    def test_execution_with_custom_id(self) -> None:
        """Test execution with specific ID."""
        exec_id = ExecutionId()
        execution = PipelineExecution(id=exec_id, pipeline_id=PipelineId())

        assert execution.id == exec_id


class TestPipeline:
    """Test Pipeline aggregate root."""

    def test_pipeline_creation(self) -> None:
        """Test pipeline aggregate creation."""
        name = PipelineName(value="test-pipeline")
        pipeline = Pipeline(name=name, description="Test description")

        assert isinstance(pipeline.id, PipelineId)
        assert pipeline.name == name
        assert pipeline.description == "Test description"
        assert pipeline.is_active is True
        # Check class-level events
        Pipeline.get_events()  # Clear any existing

    def test_pipeline_create_method(self) -> None:
        """Test pipeline create method emits event."""
        name = PipelineName(value="test-pipeline")
        pipeline = Pipeline(name=name)

        # Clear events
        Pipeline.get_events()

        pipeline.create()

        events = Pipeline.get_events()
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, PipelineCreated)
        assert event.pipeline_id == pipeline.id
        assert event.name == name
        assert pipeline.updated_at is not None
        assert pipeline.updated_at > pipeline.created_at

    def test_pipeline_execute_method(self) -> None:
        """Test pipeline execute method."""
        pipeline = Pipeline(name=PipelineName(value="test"))
        Pipeline.get_events()  # Clear

        execution = pipeline.execute()

        assert isinstance(execution, PipelineExecution)
        assert execution.pipeline_id == pipeline.id
        assert execution.status == ExecutionStatus.RUNNING

        events = Pipeline.get_events()
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, PipelineExecuted)
        assert event.pipeline_id == pipeline.id
        assert event.execution_id == execution.id

    def test_pipeline_deactivate(self) -> None:
        """Test pipeline deactivation."""
        pipeline = Pipeline(name=PipelineName(value="test"))
        original_updated = pipeline.updated_at

        assert pipeline.is_active is True

        pipeline.deactivate()

        assert pipeline.is_active is False
        assert pipeline.updated_at > original_updated

    def test_pipeline_with_custom_id(self) -> None:
        """Test pipeline with specific ID."""
        custom_id = PipelineId()
        pipeline = Pipeline(id=custom_id, name=PipelineName(value="test"))

        assert pipeline.id == custom_id

    def test_pipeline_with_empty_description(self) -> None:
        """Test pipeline with default empty description."""
        pipeline = Pipeline(name=PipelineName(value="test"))
        assert pipeline.description == ""


class TestDomainEvents:
    """Test domain event classes."""

    def test_pipeline_created_event(self) -> None:
        """Test PipelineCreated event."""
        pipeline_id = PipelineId()
        name = PipelineName(value="test")

        event = PipelineCreated(pipeline_id=pipeline_id, name=name)

        assert event.pipeline_id == pipeline_id
        assert event.name == name
        assert event.occurred_at <= datetime.now(UTC)
        assert isinstance(event.event_id, UUID)
        assert event.event_type == "PipelineCreated"

    def test_pipeline_executed_event(self) -> None:
        """Test PipelineExecuted event."""
        pipeline_id = PipelineId()
        execution_id = ExecutionId()

        event = PipelineExecuted(pipeline_id=pipeline_id, execution_id=execution_id)

        assert event.pipeline_id == pipeline_id
        assert event.execution_id == execution_id
        assert event.occurred_at <= datetime.now(UTC)
        assert isinstance(event.event_id, UUID)
        assert event.event_type == "PipelineExecuted"

    def test_events_are_immutable(self) -> None:
        """Test that domain events are immutable."""
        event = PipelineCreated(pipeline_id=PipelineId(), name=PipelineName(value="test"))

        with pytest.raises(PydanticValidationError):
            event.__dict__["pipeline_id"] = PipelineId()  # Bypass type checking for test
