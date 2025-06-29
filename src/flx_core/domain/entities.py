"""Domain entities implementing domain-driven design patterns - Pydantic + Python 3.13.

This module contains the core domain model of the FLX platform, including
aggregates, entities, and value objects that represent the business concepts
of pipelines, plugins, and executions.

ZERO TOLERANCE ARCHITECTURAL CHANGE:
- Converted from dataclasses to Pydantic models for Ultimate Repository compatibility
- Maintains ALL domain logic and validation rules
- Enhanced with Python 3.13 type system and Pydantic v2 performance
- Complete backward compatibility preserved
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from flx_core.domain.advanced_types import ConfigurationDict
from flx_core.events.event_bus import (
    DomainEvent,
    PipelineCreated,
    PipelineExecutionCompleted,
    PipelineExecutionStarted,
    StepAdded,
)

# ZERO TOLERANCE: Import ConfigurationDict at runtime for Pydantic model_rebuild()
from flx_core.value_objects import (
    CanExecuteSpecification,
    Duration,
    ExecutionId,
    ExecutionStatus,
    HasValidDependenciesSpecification,
    PipelineId,
    PipelineName,
    PipelineStep,
    PluginConfiguration,
    PluginId,
    PluginType,
)

if TYPE_CHECKING:
    from flx_core.advanced_types import ConfigurationDict


class PipelineExecution(BaseModel):
    """Represents a single run of a `Pipeline`.

    This entity tracks the status, timing, and outcome of a pipeline execution.
    It acts as a record of a specific instance of a pipeline run.
    Python 3.13 + Pydantic v2 mutable entity.
    """

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    pipeline_id: PipelineId
    execution_id: ExecutionId = Field(default_factory=ExecutionId)
    execution_number: int = Field(default=0, ge=0)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    triggered_by: str = Field(default="system", min_length=1)
    trigger_type: str = Field(default="manual", min_length=1)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_data: ConfigurationDict = Field(default_factory=dict)
    output_data: ConfigurationDict = Field(default_factory=dict)
    log_messages: list[str] = Field(default_factory=list)
    error_message: str | None = None
    cpu_usage: float | None = Field(default=None, ge=0.0, le=100.0)
    memory_usage: float | None = Field(default=None, ge=0.0)
    uncommitted_events: list[DomainEvent] = Field(
        default_factory=list,
        exclude=True,
        alias="_uncommitted_events",
    )

    @property
    def id(self) -> ExecutionId:
        """Return the execution ID.

        This property provides a convenient way to access the execution ID,
        which is the primary identifier for this entity.

        Returns:
        -------
            ExecutionId: The unique identifier of the execution.

        Note:
        ----
            Implements primary key accessor pattern.

        """
        return self.execution_id

    def add_log_message(self, message: str) -> None:
        """Add a log message to the execution record."""
        self.log_messages.append(message)

    def raise_event(self, event: DomainEvent) -> None:
        """Raise a domain event for this execution."""
        self.uncommitted_events.append(event)

    def mark_events_as_committed(self) -> None:
        """Mark all uncommitted events as committed."""
        self.uncommitted_events.clear()

    def is_completed(self) -> bool:
        """Check if the execution is completed (success, failure, or cancelled)."""
        return self.status in {
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }

    def start(self) -> None:
        """Mark execution as started.

        This method sets the execution status to `RUNNING` and records the
        start time. It ensures that the execution can only be started if it
        is in a `PENDING` state.

        Raises
        ------
            ValueError: If the execution is not in a pending state.

        """
        if self.status != ExecutionStatus.PENDING:
            msg = "Can only start pending executions"
            raise ValueError(msg)
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def start_execution(self) -> None:
        """Alias for `start()` for backward compatibility."""
        self.start()

    def is_running(self) -> bool:
        """Check if the execution is currently running."""
        return self.status == ExecutionStatus.RUNNING

    def complete_successfully(self, output: ConfigurationDict | None = None) -> None:
        """Mark execution as successful.

        This method sets the execution status to `SUCCESS`, records the
        completion time, and stores any output data. It ensures that the
        execution is in a running state before completion.

        Args:
        ----
            output: A dictionary of output data from the execution.

        Raises:
        ------
            ValueError: If the execution is not in a running state.

        """
        if not self.is_running():
            msg = "Cannot complete an execution that is not running"
            raise ValueError(msg)
        self.status = ExecutionStatus.SUCCESS
        self.completed_at = datetime.now(UTC)
        if output:
            self.output_data = output
        self.add_log_message("Execution completed successfully.")

        # Raise domain event for pipeline execution completion
        completion_event = PipelineExecutionCompleted(
            execution_id=str(self.execution_id.value),
            pipeline_id=str(self.pipeline_id.value),
            pipeline_name="",  # Will be populated by the service if available
            status="success",
            duration_seconds=(
                (self.completed_at - self.started_at).total_seconds()
                if self.started_at and self.completed_at
                else 0.0
            ),
            result_data=dict(
                self.output_data,
            ),  # Convert to dict[str, object] for event
        )
        self.raise_event(completion_event)

    def cancel(self, cancelled_by: str) -> None:
        """Cancel a running execution.

        Args:
        ----
            cancelled_by: The identifier of the user or system that cancelled the execution.

        """
        if not self.is_running():
            msg = "Cannot cancel an execution that is not running"
            raise ValueError(msg)
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.now(UTC)
        self.add_log_message(f"Execution cancelled by {cancelled_by}.")

    def fail(self, error: str) -> None:
        """Mark execution as failed.

        This method sets the execution status to `FAILURE`, records the
        completion time, and stores the error message. It can transition from
        either `PENDING` or `RUNNING` status to `FAILED`.

        Args:
        ----
            error: The error message describing the cause of failure.

        Raises:
        ------
            ValueError: If the execution is already completed.

        """
        if self.is_completed():
            msg = "Cannot fail an already completed execution"
            raise ValueError(msg)
        self.status = ExecutionStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.error_message = error
        self.add_log_message(f"Execution failed: {error}")

    @property
    def duration(self) -> Duration | None:
        """Calculate the duration of the execution.

        This property computes the duration of the execution if it has both
        a start and completion time. The duration is returned as a `Duration`
        value object.

        Returns
        -------
            Duration | None: The duration of the execution, or `None` if not completed.

        """
        if self.started_at and self.completed_at:
            return Duration.from_timedelta(self.completed_at - self.started_at)
        return None

    def track_resource_usage(self, resource_usage: dict[str, Any]) -> None:
        """Track resource usage for this execution."""
        self._resource_usage = resource_usage
        # Update individual fields if present
        if "cpu_usage" in resource_usage:
            self.cpu_usage = resource_usage["cpu_usage"]
        if "memory_mb" in resource_usage:
            self.memory_usage = resource_usage["memory_mb"]

    @property
    def resource_usage(self) -> dict[str, Any]:
        """Get the tracked resource usage."""
        return getattr(self, "_resource_usage", {})


class Plugin(BaseModel):
    """Plugin domain entity for the FLX enterprise plugin system.

    Python 3.13 + Pydantic v2 mutable entity representing Meltano ecosystem plugins.

    📋 Documentation: docs/architecture/003-plugin-system-architecture/02-plugin-interfaces.md#241
    🔗 Plugin Interfaces: src/flx_core/plugins/base.py - Enterprise plugin foundation
    🎯 Usage: Domain entity for Meltano plugin metadata and configuration

    Note: This represents external Meltano plugins (taps/targets), not the
    internal Python plugin system documented in the architecture.

    See IMPLEMENTATION-REALITY-MAP.md for distinction between:
    - This Plugin entity (Meltano ecosystem tools)
    - PluginInterface system (internal Python extensions)
    """

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "str_strip_whitespace": True,
    }

    plugin_id: PluginId = Field(default_factory=PluginId)
    name: str = Field(min_length=1)
    plugin_type: PluginType = Field(default=PluginType.EXTRACTOR)
    namespace: str = Field(default="meltano", min_length=1)
    pip_url: str | None = None
    configuration: PluginConfiguration = Field(default_factory=PluginConfiguration)
    version: str | None = None
    description: str | None = None
    documentation_url: str | None = None
    keywords: list[str] = Field(default_factory=list)

    @property
    def id(self) -> PluginId:
        """Return the plugin ID.

        This property provides a convenient way to access the plugin ID,
        which is the primary identifier for this entity.

        Returns:
        -------
            PluginId: The unique identifier of the plugin.

        Note:
        ----
            Implements primary key accessor pattern.

        """
        return self.plugin_id

    def update_configuration(self, config: PluginConfiguration) -> None:
        """Update plugin configuration.

        This method updates the plugin's configuration with a new set of
        settings. It marks the plugin as updated.

        Args:
        ----
            config: The new plugin configuration.

        Note:
        ----
            Implements configuration update pattern.

        """
        self.configuration = config

    def __eq__(self, other: object) -> bool:
        """Compare plugins by plugin_id equality."""
        if not isinstance(other, Plugin):
            return False
        return self.plugin_id == other.plugin_id

    def __hash__(self) -> int:
        """Hash based on plugin_id."""
        return hash(self.plugin_id)


class Pipeline(BaseModel):
    """The `Pipeline` aggregate root.

    This is the central entity in the domain model, representing a data pipeline
    with its definition, schedule, and associated business rules. It enforces
    invariants, such as preventing circular dependencies in steps.
    Python 3.13 + Pydantic v2 aggregate root with domain events.
    """

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    name: PipelineName
    pipeline_id: PipelineId = Field(default_factory=PipelineId)
    description: str = Field(default="")
    steps: list[PipelineStep] = Field(default_factory=list)
    environment_variables: ConfigurationDict = Field(default_factory=dict)
    schedule_expression: str | None = None
    timezone: str = Field(default="UTC")
    max_concurrent_executions: int = Field(default=1, ge=1)
    timeout: Duration | None = None
    retry_attempts: int = Field(default=0, ge=0)
    retry_delay: Duration = Field(default_factory=lambda: Duration(seconds=30))
    is_active: bool = Field(default=True)
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = Field(
        default=1,
        ge=1,
    )  # Aggregate version for optimistic concurrency
    uncommitted_events: list[DomainEvent] = Field(
        default_factory=list,
        exclude=True,
        alias="_uncommitted_events",
    )

    def model_post_init(self, __context) -> None:
        """Post-initialization hook to raise domain events."""
        # Raise pipeline created event
        pipeline_created_event = PipelineCreated(
            aggregate_id=self.pipeline_id.value,
            aggregate_type="Pipeline",
            pipeline_id=str(self.pipeline_id.value),
            pipeline_name=str(self.name.value),
            created_by=self.created_by,
            pipeline_data={
                "description": self.description,
                "max_concurrent_executions": self.max_concurrent_executions,
                "is_active": self.is_active,
            },
        )
        self.raise_event(pipeline_created_event)

    @property
    def id(self) -> PipelineId:
        """Return the pipeline ID.

        This property provides a convenient way to access the pipeline ID,
        which is the primary identifier for this aggregate root.

        Returns:
        -------
            PipelineId: The unique identifier of the pipeline.

        Note:
        ----
            Implements primary key accessor pattern.

        """
        return self.pipeline_id

    def raise_event(self, event: DomainEvent) -> None:
        """Raise a domain event.

        This method adds a domain event to the list of uncommitted events
        which can then be dispatched by the application layer.

        Args:
        ----
            event: The domain event to raise.

        """
        self.uncommitted_events.append(event)

    @property
    def uncommitted_events_copy(self) -> list[DomainEvent]:
        """Return a copy of uncommitted events."""
        return self.uncommitted_events.copy()

    def mark_events_as_committed(self) -> None:
        """Clear uncommitted events."""
        self.uncommitted_events.clear()

    def can_execute(self) -> bool:
        """Check if the pipeline can be executed.

        This method uses the specification pattern to determine if the
        pipeline is in a state where it can be executed.

        Returns
        -------
            bool: True if the pipeline can be executed, False otherwise.

        """
        return (
            CanExecuteSpecification() & HasValidDependenciesSpecification()
        ).is_satisfied_by(self)

    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline.

        This method adds a new step to the pipeline and validates that it
        does not introduce a circular dependency. It also raises a `StepAdded`
        domain event.

        Args:
        ----
            step: The pipeline step to add.

        Raises:
        ------
            ValueError: If a step with the same ID already exists or if there are
                       missing dependencies or circular dependencies.

        """
        if step.step_id in {s.step_id for s in self.steps}:
            msg = f"Step with ID '{step.step_id}' already exists"
            raise ValueError(msg)

        # Validate dependencies exist
        step_ids = {existing.step_id for existing in self.steps}
        step_deps = (
            frozenset(step.depends_on)
            if isinstance(step.depends_on, list)
            else step.depends_on
        )
        missing_deps = step_deps - step_ids
        if missing_deps:
            msg = f"Missing dependencies: {missing_deps}"
            raise ValueError(msg)

        self._validate_no_circular_dependencies(step)
        self.steps.append(step)
        self.raise_event(
            StepAdded(
                pipeline_id=str(self.pipeline_id.value),
                step_id=step.step_id,
                plugin_id=str(step.plugin_id.value),
            ),
        )

    def add_or_replace_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline or replace if it already exists.

        This method is used for testing circular dependencies by allowing
        replacement of existing steps. It validates both missing dependencies
        and circular dependencies.

        Args:
        ----
            step: The pipeline step to add or replace.

        Raises:
        ------
            ValueError: If there are missing dependencies or circular dependencies.

        """
        # Remove existing step if it exists (for replacement)
        existing_step_index = None
        for i, existing in enumerate(self.steps):
            if existing.step_id == step.step_id:
                existing_step_index = i
                break

        # Validate dependencies exist (excluding the step being replaced)
        step_ids = {s.step_id for s in self.steps if s.step_id != step.step_id}
        step_deps = (
            frozenset(step.depends_on)
            if isinstance(step.depends_on, list)
            else step.depends_on
        )
        missing_deps = step_deps - step_ids
        if missing_deps:
            msg = f"Missing dependencies: {missing_deps}"
            raise ValueError(msg)

        # Validate circular dependencies (with replacement logic)
        self._validate_no_circular_dependencies_with_replacement(step)

        # Replace or add the step
        if existing_step_index is not None:
            self.steps[existing_step_index] = step
        else:
            self.steps.append(step)

        self.raise_event(
            StepAdded(
                pipeline_id=str(self.pipeline_id.value),
                step_id=step.step_id,
                plugin_id=str(step.plugin_id.value),
            ),
        )

    def remove_step(self, step_id: str) -> None:
        """Remove a step from the pipeline.

        This method removes a step by its ID and also removes it from the
        dependencies of other steps.

        Args:
        ----
            step_id: The ID of the step to remove.

        Raises:
        ------
            ValueError: If the step does not exist.

        """
        if step_id not in {s.step_id for s in self.steps}:
            msg = f"Step with ID '{step_id}' not found"
            raise ValueError(msg)

        # Check if any other step depends on this step
        dependent_steps = [s.step_id for s in self.steps if step_id in s.depends_on]
        if dependent_steps:
            msg = f"Step '{step_id}' is a dependency for other steps: {dependent_steps}"
            raise ValueError(msg)

        self.steps = [s for s in self.steps if s.step_id != step_id]
        for step in self.steps:
            if step_id in step.depends_on:
                new_deps = step.depends_on - {step_id}
                self.steps[self.steps.index(step)] = step.copy(
                    update={"depends_on": new_deps},
                )

    def _validate_no_circular_dependencies(self, new_step: PipelineStep) -> None:
        """Validate that adding a step does not create a circular dependency."""
        dependencies = {s.step_id: s.depends_on for s in self.steps}
        dependencies[new_step.step_id] = new_step.depends_on

        def has_cycle(node: str, visited: set[str], rec_stack: set[str]) -> bool:
            """Check for cycles using DFS."""
            visited.add(node)
            rec_stack.add(node)

            for neighbour in dependencies.get(node, []):
                if neighbour not in visited:
                    if has_cycle(neighbour, visited, rec_stack):
                        return True
                elif neighbour in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_id in dependencies:
            if has_cycle(step_id, set(), set()):
                msg = f"Circular dependency detected involving step '{step_id}'"
                raise ValueError(msg)

    def _validate_no_circular_dependencies_with_replacement(
        self, new_step: PipelineStep
    ) -> None:
        """Validate that adding/replacing a step does not create a circular dependency."""
        # Build dependencies including the replacement (exclude existing step with same ID)
        dependencies = {
            s.step_id: s.depends_on for s in self.steps if s.step_id != new_step.step_id
        }
        dependencies[new_step.step_id] = new_step.depends_on

        def has_cycle(node: str, visited: set[str], rec_stack: set[str]) -> bool:
            """Check for cycles using DFS."""
            visited.add(node)
            rec_stack.add(node)

            for neighbour in dependencies.get(node, []):
                if neighbour not in visited:
                    if has_cycle(neighbour, visited, rec_stack):
                        return True
                elif neighbour in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_id in dependencies:
            if has_cycle(step_id, set(), set()):
                msg = f"Circular dependency detected involving step '{step_id}'"
                raise ValueError(msg)

    def activate(self) -> None:
        """Activate the pipeline."""
        self.is_active = True

    def deactivate(self) -> None:
        """Deactivate the pipeline."""
        self.is_active = False

    def create_execution(
        self,
        triggered_by: str,
        execution_number: int,
        input_data: ConfigurationDict | None = None,
        trigger_type: str = "manual",
    ) -> PipelineExecution:
        """Create a new execution for this pipeline."""
        execution = PipelineExecution(
            pipeline_id=self.pipeline_id,
            execution_number=execution_number,
            triggered_by=triggered_by,
            trigger_type=trigger_type,
            input_data=input_data or {},
        )
        self.raise_event(
            PipelineExecutionStarted(
                execution_id=str(execution.execution_id.value),
                pipeline_id=str(self.pipeline_id.value),
                pipeline_name=str(self.name.value),
                started_by=triggered_by,
            ),
        )
        return execution

    def __eq__(self, other: object) -> bool:
        """Compare pipelines by pipeline_id equality."""
        if not isinstance(other, Pipeline):
            return False
        return self.pipeline_id == other.pipeline_id

    def __hash__(self) -> int:
        """Hash based on pipeline_id."""
        return hash(self.pipeline_id)

    def __str__(self) -> str:
        """Return string representation."""
        return f"Pipeline(name='{self.name.value}', id='{self.pipeline_id.value}')"


# ZERO TOLERANCE - ConfigurationDict import resolution completed
# Forward references resolved at import time

# Rebuild models to resolve forward references
Pipeline.model_rebuild()
PipelineStep.model_rebuild()
PipelineExecution.model_rebuild()
Plugin.model_rebuild()
