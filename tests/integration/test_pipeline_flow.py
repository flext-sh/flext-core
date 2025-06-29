"""Integration test for complete pipeline flow.

Tests the full pipeline lifecycle from creation to execution.
"""

from __future__ import annotations

import pytest
from flx_core.application.services import ExecutionService, PipelineService
from flx_core.domain.value_objects import ExecutionStatus
from flx_core.events.event_bus import InMemoryEventBus
from flx_core.infrastructure.persistence.unit_of_work import UnitOfWork


@pytest.mark.integration
class TestPipelineIntegrationFlow:
    """Test complete pipeline flow integration."""

    @pytest.fixture
    async def pipeline_service(self, unit_of_work: UnitOfWork) -> PipelineService:
        """Create pipeline service."""
        event_bus = InMemoryEventBus()
        return PipelineService(unit_of_work=unit_of_work, event_bus=event_bus)

    @pytest.fixture
    async def execution_service(self, unit_of_work: UnitOfWork) -> ExecutionService:
        """Create execution service."""
        event_bus = InMemoryEventBus()
        # Mock executor for testing
        from unittest.mock import AsyncMock

        executor = AsyncMock()
        executor.execute_pipeline.return_value = None

        return ExecutionService(
            unit_of_work=unit_of_work, event_bus=event_bus, executor=executor
        )

    async def test_complete_pipeline_lifecycle(
        self,
        pipeline_service: PipelineService,
        execution_service: ExecutionService,
        sample_plugin,
    ) -> None:
        """Test complete pipeline lifecycle from creation to execution."""
        # Step 1: Create pipeline
        pipeline_data = {
            "name": "integration_test_pipeline",
            "description": "Pipeline for integration testing",
            "steps": [
                {
                    "name": "extract",
                    "plugin_id": str(sample_plugin.plugin_id.value),
                    "configuration": {"source": "test_source"},
                }
            ],
            "environment_variables": {"TEST_VAR": "test_value"},
            "max_concurrent_executions": 2,
        }

        create_result = await pipeline_service.create_pipeline(pipeline_data)
        assert create_result.is_success is True

        pipeline = create_result.value
        pipeline_id = str(pipeline.pipeline_id.value)

        # Step 2: Verify pipeline was created
        get_result = await pipeline_service.get_pipeline(pipeline_id)
        assert get_result.is_success is True
        assert get_result.value.name.value == "integration_test_pipeline"
        assert len(get_result.value.steps) == 1

        # Step 3: Update pipeline
        update_data = {
            "description": "Updated integration test pipeline",
            "retry_attempts": 3,
        }

        update_result = await pipeline_service.update_pipeline(pipeline_id, update_data)
        assert update_result.is_success is True
        assert update_result.value.description == "Updated integration test pipeline"
        assert update_result.value.retry_attempts == 3

        # Step 4: Start execution
        exec_result = await execution_service.start_execution(pipeline_id)
        assert exec_result.is_success is True

        execution = exec_result.value
        execution_id = str(execution.execution_id.value)

        # Step 5: Verify execution was created
        get_exec_result = await execution_service.get_execution(execution_id)
        assert get_exec_result.is_success is True
        assert get_exec_result.value.pipeline_id == pipeline.pipeline_id
        assert get_exec_result.value.status == ExecutionStatus.PENDING

        # Step 6: List pipeline executions
        list_result = await execution_service.list_pipeline_executions(pipeline_id)
        assert list_result.is_success is True
        assert len(list_result.value) >= 1
        assert any(e.execution_id == execution.execution_id for e in list_result.value)

        # Step 7: Test concurrent execution limit
        # Start another execution
        exec_result2 = await execution_service.start_execution(pipeline_id)
        assert exec_result2.is_success is True

        # Try to start third execution (should fail due to limit)
        exec_result3 = await execution_service.start_execution(pipeline_id)
        assert exec_result3.is_failure is True
        assert exec_result3.error.code == "MAX_CONCURRENT_EXECUTIONS"

        # Step 8: Cancel execution
        cancel_result = await execution_service.cancel_execution(execution_id)
        assert cancel_result.is_success is True
        assert cancel_result.value.status == ExecutionStatus.CANCELLED

        # Step 9: Deactivate pipeline
        deactivate_result = await pipeline_service.deactivate_pipeline(pipeline_id)
        assert deactivate_result.is_success is True
        assert deactivate_result.value.is_active is False

        # Step 10: Try to execute inactive pipeline (should fail)
        exec_inactive = await execution_service.start_execution(pipeline_id)
        assert exec_inactive.is_failure is True
        assert exec_inactive.error.code == "PIPELINE_INACTIVE"

        # Step 11: Reactivate pipeline
        activate_result = await pipeline_service.activate_pipeline(pipeline_id)
        assert activate_result.is_success is True
        assert activate_result.value.is_active is True

        # Step 12: Delete pipeline (should fail with executions)
        delete_result = await pipeline_service.delete_pipeline(pipeline_id)
        assert delete_result.is_failure is True
        assert delete_result.error.code == "HAS_EXECUTIONS"

        # Step 13: Archive executions and delete pipeline
        archive_result = await execution_service.archive_executions(pipeline_id)
        assert archive_result.is_success is True

        delete_result = await pipeline_service.delete_pipeline(pipeline_id)
        assert delete_result.is_success is True

        # Step 14: Verify pipeline was deleted
        get_deleted = await pipeline_service.get_pipeline(pipeline_id)
        assert get_deleted.is_failure is True
        assert get_deleted.error.code == "NOT_FOUND"

    async def test_pipeline_with_multiple_steps(
        self, pipeline_service: PipelineService, sample_plugin
    ) -> None:
        """Test pipeline with multiple steps and dependencies."""
        # Create pipeline with multiple steps
        pipeline_data = {
            "name": "multi_step_pipeline",
            "description": "Pipeline with multiple steps",
            "steps": [
                {
                    "name": "extract",
                    "plugin_id": str(sample_plugin.plugin_id.value),
                    "configuration": {"source": "database"},
                    "depends_on": [],
                },
                {
                    "name": "transform",
                    "plugin_id": str(sample_plugin.plugin_id.value),
                    "configuration": {"operation": "clean"},
                    "depends_on": ["extract"],
                },
                {
                    "name": "load",
                    "plugin_id": str(sample_plugin.plugin_id.value),
                    "configuration": {"target": "warehouse"},
                    "depends_on": ["transform"],
                },
            ],
        }

        result = await pipeline_service.create_pipeline(pipeline_data)
        assert result.is_success is True

        pipeline = result.value
        assert len(pipeline.steps) == 3

        # Verify step dependencies
        extract_step = next(s for s in pipeline.steps if s.name == "extract")
        transform_step = next(s for s in pipeline.steps if s.name == "transform")
        load_step = next(s for s in pipeline.steps if s.name == "load")

        assert extract_step.depends_on == []
        assert transform_step.depends_on == ["extract"]
        assert load_step.depends_on == ["transform"]

        # Add a new step
        add_step_result = await pipeline_service.add_step(
            str(pipeline.pipeline_id.value),
            {
                "name": "validate",
                "plugin_id": str(sample_plugin.plugin_id.value),
                "configuration": {"rules": "quality"},
                "depends_on": ["load"],
            },
        )
        assert add_step_result.is_success is True
        assert len(add_step_result.value.steps) == 4

        # Remove a step
        remove_step_result = await pipeline_service.remove_step(
            str(pipeline.pipeline_id.value), "validate"
        )
        assert remove_step_result.is_success is True
        assert len(remove_step_result.value.steps) == 3
