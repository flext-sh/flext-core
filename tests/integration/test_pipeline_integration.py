"""Pipeline integration tests - End-to-end scenarios.

Tests complete workflows with real components.
"""

from __future__ import annotations

import pytest

from flext_core import (
    ExecutionStatus,
    InMemoryRepository,
    Pipeline,
    PipelineExecution,
    PipelineId,
    PipelineName,
    PipelineService,
)
from flext_core.application.pipeline import (
    CreatePipelineCommand,
    ExecutePipelineCommand,
    GetPipelineQuery,
)


class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""

    def setup_method(self) -> None:
        """Setup integration test environment."""
        self.repository = InMemoryRepository[Pipeline]()
        self.service = PipelineService(self.repository)

    async def test_complete_pipeline_lifecycle(self) -> None:
        """Test complete pipeline lifecycle from creation to execution."""
        # 1. Create pipeline
        create_cmd = CreatePipelineCommand(
            name=PipelineName(value="integration-test-pipeline"),
            description="Integration test pipeline",
        )

        create_result = await self.service.create_pipeline(create_cmd)
        assert create_result.success

        pipeline = create_result.unwrap()
        pipeline_id = pipeline.id

        # Verify pipeline was created
        assert pipeline.name.value == "integration-test-pipeline"
        assert pipeline.description == "Integration test pipeline"
        assert pipeline.is_active

        # Check domain events at class level
        events = Pipeline.get_events()
        assert len(events) == 1  # PipelineCreated event

        # 2. Get pipeline
        get_query = GetPipelineQuery(pipeline_id=pipeline_id)
        get_result = await self.service.get_pipeline(get_query)

        assert get_result.success
        retrieved = get_result.unwrap()
        assert retrieved.id == pipeline_id

        # 3. Execute pipeline
        exec_cmd = ExecutePipelineCommand(pipeline_id=pipeline_id)
        exec_result = await self.service.execute_pipeline(exec_cmd)

        assert exec_result.success
        execution = exec_result.unwrap()
        assert isinstance(execution, PipelineExecution)
        assert execution.pipeline_id == pipeline_id
        assert execution.status == ExecutionStatus.RUNNING

        # 4. Deactivate pipeline
        deactivate_result = await self.service.deactivate_pipeline(pipeline_id)
        assert deactivate_result.success

        deactivated = deactivate_result.unwrap()
        assert not deactivated.is_active

        # 5. Try to execute deactivated pipeline
        exec_result2 = await self.service.execute_pipeline(exec_cmd)
        assert not exec_result2.success
        assert exec_result2.error == "Pipeline is inactive"

    async def test_multiple_pipelines_management(self) -> None:
        """Test managing multiple pipelines."""
        # Create multiple pipelines
        pipelines_data = [
            ("data-ingestion", "Ingests data from sources"),
            ("data-transformation", "Transforms raw data"),
            ("data-export", "Exports processed data"),
            ("data-validation", "Validates data quality"),
            ("data-archive", "Archives old data"),
        ]

        created_pipelines = []
        for name, desc in pipelines_data:
            cmd = CreatePipelineCommand(
                name=PipelineName(value=name),
                description=desc,
            )
            result = await self.service.create_pipeline(cmd)
            assert result.success
            created_pipelines.append(result.unwrap())

        # List all pipelines
        all_pipelines = await self.repository.list_all()
        assert len(all_pipelines) == 5

        # Check data prefix pipelines
        data_pipelines = [p for p in all_pipelines if p.name.value.startswith("data-")]
        assert len(data_pipelines) == 5

        # Deactivate some pipelines
        for i in [1, 3]:  # Deactivate 2nd and 4th pipelines
            await self.service.deactivate_pipeline(created_pipelines[i].id)

        # Get updated pipeline list and filter active ones
        updated_pipelines = await self.repository.list_all()
        active_pipelines = [p for p in updated_pipelines if p.is_active]
        assert len(active_pipelines) == 3

        # Execute all active pipelines
        executions = []
        for pipeline in active_pipelines:
            exec_cmd = ExecutePipelineCommand(pipeline_id=pipeline.id)
            exec_result = await self.service.execute_pipeline(exec_cmd)
            assert exec_result.success
            executions.append(exec_result.unwrap())

        assert len(executions) == 3
        assert all(e.status == ExecutionStatus.RUNNING for e in executions)

    async def test_concurrent_pipeline_operations(self) -> None:
        """Test concurrent operations on same pipeline."""
        # Create pipeline
        create_cmd = CreatePipelineCommand(
            name=PipelineName(value="concurrent-test"),
            description="Testing concurrent access",
        )

        create_result = await self.service.create_pipeline(create_cmd)
        pipeline = create_result.unwrap()
        pipeline_id = pipeline.id

        # Simulate concurrent executions
        exec_cmd = ExecutePipelineCommand(pipeline_id=pipeline_id)

        # Execute pipeline multiple times
        execution_results = []
        for _ in range(5):
            result = await self.service.execute_pipeline(exec_cmd)
            assert result.success
            execution_results.append(result.unwrap())

        # All executions should have unique IDs
        execution_ids = {e.id for e in execution_results}
        assert len(execution_ids) == 5

        # All should be for the same pipeline
        assert all(e.pipeline_id == pipeline_id for e in execution_results)

    async def test_error_recovery_workflow(self) -> None:
        """Test error handling and recovery."""
        # Try to execute non-existent pipeline
        fake_id = PipelineId()
        exec_cmd = ExecutePipelineCommand(pipeline_id=fake_id)

        result = await self.service.execute_pipeline(exec_cmd)
        assert not result.success
        assert result.error == "Pipeline not found"

        # Try to get non-existent pipeline
        get_query = GetPipelineQuery(pipeline_id=fake_id)
        get_result = await self.service.get_pipeline(get_query)
        assert not get_result.success
        assert get_result.error == "Pipeline not found"

        # Try to deactivate non-existent pipeline
        deactivate_result = await self.service.deactivate_pipeline(fake_id)
        assert not deactivate_result.success
        assert deactivate_result.error == "Pipeline not found"

        # Create pipeline with empty name should fail validation
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            CreatePipelineCommand(
                name=PipelineName(value="   "),
                description="Invalid pipeline",
            )

    async def test_repository_persistence(self) -> None:
        """Test that repository maintains state correctly."""
        # Create and save pipeline
        pipeline = Pipeline(
            name=PipelineName(value="persistence-test"),
            description="Testing persistence",
        )

        original_created = pipeline.created_at
        original_updated = pipeline.updated_at

        saved = await self.repository.save(pipeline)
        saved_id = saved.id

        # Retrieve and modify
        retrieved = await self.repository.get(saved_id)
        assert retrieved is not None
        assert retrieved.created_at == original_created

        # Deactivate and save again
        retrieved.deactivate()
        await self.repository.save(retrieved)

        # Retrieve again to verify persistence
        final = await self.repository.get(saved_id)
        assert final is not None
        assert not final.is_active
        assert final.updated_at is not None
        assert original_updated is not None
        assert final.updated_at > original_updated
        assert final.created_at == original_created  # Created date unchanged

    async def test_pipeline_event_accumulation(self) -> None:
        """Test that pipeline events accumulate correctly at class level."""
        # Clear any existing events
        Pipeline.get_events()

        # Create pipeline
        pipeline = Pipeline(
            name=PipelineName(value="event-test"),
            description="Testing events",
        )

        # Initially no events at class level
        events = Pipeline.get_events()
        assert len(events) == 0

        # Create event
        pipeline.create()
        events = Pipeline.get_events()
        assert len(events) == 1

        # Execute multiple times
        for _ in range(3):
            pipeline.execute()

        events = Pipeline.get_events()
        assert len(events) == 3  # Only execute events (create was cleared)

        # Clear events by getting them
        Pipeline.get_events()
        events = Pipeline.get_events()
        assert len(events) == 0

        # New events still work
        pipeline.execute()
        events = Pipeline.get_events()
        assert len(events) == 1
