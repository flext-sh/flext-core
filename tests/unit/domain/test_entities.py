"""Comprehensive unit tests for domain entities.

Adapted from flx-meltano-enterprise with focus on Pipeline and Plugin entities.
"""

from __future__ import annotations

import pytest

from flx_core.domain.entities import (
    Pipeline,
    PipelineExecution,
    PipelineId,
    PipelineName,
    PipelineStep,
    Plugin,
    PluginConfiguration,
    PluginId,
    PluginType,
)
from flx_core.domain.value_objects import Duration, ExecutionStatus

# Python 3.13 type aliases
TestResult = bool
TestPipelineData = dict[str, str | int]
TestMetrics = dict[str, int | float]

# Constants
MAX_CONCURRENT_EXECUTIONS = 5
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 60
TIMEOUT_SECONDS = 3600
STEP_COUNT = 2
PLUGIN_VERSION = "1.2.3"
ROWS_PROCESSED = 1000
CPU_USAGE = 0.5
MEMORY_USAGE_MB = 256
EXECUTION_NUMBER = 1


class TestPipelineEntity:
    """Test Pipeline domain entity."""

    def test_pipeline_creation_minimal(self, sample_pipeline_id: PipelineId) -> None:
        """Test pipeline creation with minimal required fields."""
        pipeline = Pipeline(
            pipeline_id=sample_pipeline_id,
            name=PipelineName(value="test_pipeline"),
            description="Test pipeline description",
        )

        assert pipeline.pipeline_id == sample_pipeline_id
        assert str(pipeline.name) == "test_pipeline"
        assert pipeline.description == "Test pipeline description"
        assert pipeline.steps == []
        assert pipeline.is_active is True
        assert pipeline.max_concurrent_executions == 1
        assert pipeline.retry_attempts == 0

    def test_pipeline_creation_complete(self) -> None:
        """Test pipeline creation with all fields."""
        pipeline_id = PipelineId()
        name = PipelineName(value="complete_pipeline")

        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=name,
            description="Complete pipeline",
            environment_variables={"ENV": "test"},
            schedule_expression="0 * * * *",
            timezone="UTC",
            max_concurrent_executions=MAX_CONCURRENT_EXECUTIONS,
            timeout=Duration(seconds=TIMEOUT_SECONDS),
            retry_attempts=RETRY_ATTEMPTS,
            retry_delay=Duration(seconds=RETRY_DELAY_SECONDS),
            created_by="test_user",
            updated_by="test_user",
        )

        assert pipeline.pipeline_id == pipeline_id
        assert pipeline.name == name
        assert pipeline.max_concurrent_executions == MAX_CONCURRENT_EXECUTIONS
        assert pipeline.retry_attempts == RETRY_ATTEMPTS

    def test_pipeline_add_step(
        self, sample_pipeline: Pipeline, sample_plugin: Plugin
    ) -> None:
        """Test adding steps to pipeline."""
        # Pipeline fixture already has one step
        initial_count = len(sample_pipeline.steps)

        # Add another step
        new_step = PipelineStep(
            name="transform",
            plugin=sample_plugin,
            configuration={"transform": "upper"},
            depends_on=["extract"],
        )

        sample_pipeline.add_step(new_step)

        assert len(sample_pipeline.steps) == initial_count + 1
        assert sample_pipeline.steps[-1].name == "transform"
        assert sample_pipeline.steps[-1].depends_on == ["extract"]

    def test_pipeline_duplicate_step_name(
        self, sample_pipeline: Pipeline, sample_plugin: Plugin
    ) -> None:
        """Test that duplicate step names raise error."""
        # Pipeline already has "extract" step
        duplicate_step = PipelineStep(
            name="extract",  # Duplicate name
            plugin=sample_plugin,
            configuration={},
            depends_on=[],
        )

        with pytest.raises(ValueError, match="Step with name 'extract' already exists"):
            sample_pipeline.add_step(duplicate_step)

    def test_pipeline_remove_step(self, sample_pipeline: Pipeline) -> None:
        """Test removing step from pipeline."""
        initial_count = len(sample_pipeline.steps)

        sample_pipeline.remove_step("extract")

        assert len(sample_pipeline.steps) == initial_count - 1
        assert not any(step.name == "extract" for step in sample_pipeline.steps)

    def test_pipeline_remove_nonexistent_step(self, sample_pipeline: Pipeline) -> None:
        """Test removing non-existent step raises error."""
        with pytest.raises(ValueError, match="Step 'nonexistent' not found"):
            sample_pipeline.remove_step("nonexistent")

    def test_pipeline_deactivate(self, sample_pipeline: Pipeline) -> None:
        """Test pipeline deactivation."""
        assert sample_pipeline.is_active is True

        sample_pipeline.deactivate()

        assert sample_pipeline.is_active is False

    def test_pipeline_activate(self, sample_pipeline: Pipeline) -> None:
        """Test pipeline activation."""
        sample_pipeline.deactivate()
        assert sample_pipeline.is_active is False

        sample_pipeline.activate()

        assert sample_pipeline.is_active is True

    def test_pipeline_update_configuration(self, sample_pipeline: Pipeline) -> None:
        """Test updating pipeline configuration."""
        new_config = {
            "max_concurrent_executions": 10,
            "retry_attempts": 5,
            "environment_variables": {"NEW_VAR": "value"},
        }

        sample_pipeline.update_configuration(new_config)

        assert sample_pipeline.max_concurrent_executions == 10
        assert sample_pipeline.retry_attempts == 5
        assert sample_pipeline.environment_variables == {"NEW_VAR": "value"}


class TestPluginEntity:
    """Test Plugin domain entity."""

    def test_plugin_creation_minimal(self) -> None:
        """Test plugin creation with minimal fields."""
        plugin = Plugin(
            plugin_id=PluginId(),
            plugin_type=PluginType.EXTRACTOR,
            name="test_plugin",
            description="Test plugin",
            configuration=PluginConfiguration(
                plugin_name="test_plugin",
                namespace="test",
                pip_url="test-plugin==1.0.0",
                executable="test-plugin",
            ),
        )

        assert plugin.name == "test_plugin"
        assert plugin.plugin_type == PluginType.EXTRACTOR
        assert plugin.is_builtin is False
        assert plugin.is_enabled is True

    def test_plugin_creation_complete(self) -> None:
        """Test plugin creation with all fields."""
        plugin_id = PluginId()
        config = PluginConfiguration(
            plugin_name="advanced_plugin",
            namespace="tap",
            pip_url="tap-advanced==2.0.0",
            executable="tap-advanced",
            settings={"api_key": "secret"},
            select=["users", "orders"],
            catalog="catalog.json",
            state="state.json",
        )

        plugin = Plugin(
            plugin_id=plugin_id,
            plugin_type=PluginType.EXTRACTOR,
            name="advanced_plugin",
            description="Advanced plugin with full config",
            configuration=config,
            version=PLUGIN_VERSION,
            author="Test Author",
            homepage="https://example.com",
            keywords=["etl", "data"],
            is_builtin=True,
            is_enabled=False,
        )

        assert plugin.plugin_id == plugin_id
        assert plugin.version == PLUGIN_VERSION
        assert plugin.is_builtin is True
        assert plugin.is_enabled is False
        assert plugin.configuration.settings == {"api_key": "secret"}

    def test_plugin_enable_disable(self) -> None:
        """Test enabling and disabling plugin."""
        plugin = Plugin(
            plugin_id=PluginId(),
            plugin_type=PluginType.LOADER,
            name="test_loader",
            description="Test loader",
            configuration=PluginConfiguration(
                plugin_name="test_loader",
                namespace="target",
                pip_url="target-test==1.0.0",
                executable="target-test",
            ),
            is_enabled=False,
        )

        assert plugin.is_enabled is False

        plugin.enable()
        assert plugin.is_enabled is True

        plugin.disable()
        assert plugin.is_enabled is False

    def test_plugin_update_configuration(self) -> None:
        """Test updating plugin configuration."""
        plugin = Plugin(
            plugin_id=PluginId(),
            plugin_type=PluginType.TRANSFORMER,
            name="test_transformer",
            description="Test transformer",
            configuration=PluginConfiguration(
                plugin_name="test_transformer",
                namespace="transform",
                pip_url="transform-test==1.0.0",
                executable="transform-test",
                settings={"mode": "basic"},
            ),
        )

        new_settings = {"mode": "advanced", "threads": 4}
        plugin.update_configuration(settings=new_settings)

        assert plugin.configuration.settings == new_settings


class TestPipelineExecution:
    """Test PipelineExecution entity."""

    def test_execution_creation(self, sample_pipeline: Pipeline) -> None:
        """Test pipeline execution creation."""
        execution = PipelineExecution(
            pipeline_id=sample_pipeline.pipeline_id,
            status=ExecutionStatus.PENDING,
        )

        assert execution.pipeline_id == sample_pipeline.pipeline_id
        assert execution.status == ExecutionStatus.PENDING
        assert execution.started_at is None
        assert execution.completed_at is None
        assert execution.error_message is None

    def test_execution_start(self, sample_execution: PipelineExecution) -> None:
        """Test starting pipeline execution."""
        assert sample_execution.status == ExecutionStatus.PENDING
        assert sample_execution.started_at is None

        sample_execution.start()

        assert sample_execution.status == ExecutionStatus.RUNNING
        assert sample_execution.started_at is not None

    def test_execution_complete_success(
        self, sample_execution: PipelineExecution
    ) -> None:
        """Test completing execution successfully."""
        sample_execution.start()

        sample_execution.complete()

        assert sample_execution.status == ExecutionStatus.COMPLETED
        assert sample_execution.completed_at is not None
        assert sample_execution.error_message is None

    def test_execution_complete_failure(
        self, sample_execution: PipelineExecution
    ) -> None:
        """Test completing execution with failure."""
        sample_execution.start()

        error_msg = "Pipeline failed due to connection error"
        sample_execution.fail(error_msg)

        assert sample_execution.status == ExecutionStatus.FAILED
        assert sample_execution.completed_at is not None
        assert sample_execution.error_message == error_msg

    def test_execution_cancel(self, sample_execution: PipelineExecution) -> None:
        """Test cancelling execution."""
        sample_execution.start()

        sample_execution.cancel()

        assert sample_execution.status == ExecutionStatus.CANCELLED
        assert sample_execution.completed_at is not None

    def test_execution_duration(self, sample_execution: PipelineExecution) -> None:
        """Test execution duration calculation."""
        # Before start, duration is None
        assert sample_execution.duration is None

        # Start execution
        sample_execution.start()

        # Complete execution
        sample_execution.complete()

        # Now duration should be available
        assert sample_execution.duration is not None
        assert sample_execution.duration.total_seconds >= 0

    def test_execution_invalid_transitions(
        self, sample_execution: PipelineExecution
    ) -> None:
        """Test invalid status transitions."""
        # Complete execution
        sample_execution.start()
        sample_execution.complete()

        # Cannot start completed execution
        with pytest.raises(ValueError, match="Cannot start execution in status"):
            sample_execution.start()

        # Cannot complete already completed execution
        with pytest.raises(ValueError, match="Cannot complete execution in status"):
            sample_execution.complete()


class TestPipelineStep:
    """Test PipelineStep value object."""

    def test_step_creation(self, sample_plugin: Plugin) -> None:
        """Test pipeline step creation."""
        step = PipelineStep(
            name="extract_users",
            plugin=sample_plugin,
            configuration={"table": "users", "batch_size": 1000},
            depends_on=[],
        )

        assert step.name == "extract_users"
        assert step.plugin == sample_plugin
        assert step.configuration["table"] == "users"
        assert step.depends_on == []

    def test_step_with_dependencies(self, sample_plugin: Plugin) -> None:
        """Test step with dependencies."""
        step = PipelineStep(
            name="load_users",
            plugin=sample_plugin,
            configuration={"target": "warehouse"},
            depends_on=["extract_users", "transform_users"],
        )

        assert len(step.depends_on) == 2
        assert "extract_users" in step.depends_on
        assert "transform_users" in step.depends_on

    def test_step_equality(self, sample_plugin: Plugin) -> None:
        """Test step equality based on name."""
        step1 = PipelineStep(
            name="test_step",
            plugin=sample_plugin,
            configuration={},
            depends_on=[],
        )

        step2 = PipelineStep(
            name="test_step",
            plugin=sample_plugin,
            configuration={"different": "config"},
            depends_on=["other"],
        )

        step3 = PipelineStep(
            name="different_step",
            plugin=sample_plugin,
            configuration={},
            depends_on=[],
        )

        # Same name = equal (even with different config)
        assert step1 == step2

        # Different name = not equal
        assert step1 != step3
