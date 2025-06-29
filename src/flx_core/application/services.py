"""ADR-001 Compliant Application Services - Clean Architecture Use Cases.

Application Services implementing use cases following ADR-001 Clean Architecture and
Domain-Driven Design principles. These services orchestrate domain operations while
maintaining strict architectural boundaries and implementing CLAUDE.md ZERO TOLERANCE standards.

ARCHITECTURAL COMPLIANCE:
- ADR-001: Application Services as use case coordinators
- Clean Architecture: Use cases implementing business rules at application boundary
- Hexagonal Architecture: Primary port implementations driving domain operations
- DDD: Application services coordinating domain aggregates and services
- CLAUDE.md: ZERO TOLERANCE - Real implementations with comprehensive error handling

This module contains application services that orchestrate high-level operations
and business workflows. These services do not contain complex business logic
themselves but delegate to domain models and repositories, acting as a facade
over the domain layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict
from uuid import UUID

import structlog

# === MODERN PYTHON 3.13 TYPE DEFINITIONS ===
# Type aliases for pipeline configuration - replacing Any types
# Import from domain types for consistency
from flx_core.domain.advanced_types import ConfigurationDict
from flx_core.domain.entities import Pipeline, PipelineExecution, Plugin
from flx_core.domain.value_objects import (
    PipelineId,
    PipelineName,
    PipelineStep,
    PluginId,
)
from flx_core.infrastructure.persistence.models import (
    PipelineExecutionModel,
    PipelineModel,
    PluginModel,
)

type EnvironmentVariables = ConfigurationDict
type PipelineInputData = ConfigurationDict


class PipelineStepData(TypedDict):
    """Type-safe structure for pipeline step configuration.

    Replaces dict[str, Any] with structured typing.
    """

    step_id: str
    plugin_id: str
    order: int
    configuration: ConfigurationDict
    depends_on: list[str]  # List of step IDs this step depends on


# Commands imported above for use in services

if TYPE_CHECKING:
    from flx_meltano.unified_anti_corruption_layer import (
        UnifiedMeltanoAntiCorruptionLayer as MeltanoAntiCorruptionLayer,
    )

    from flx_core.commands.pipeline import (
        CreatePipelineCommand,
        ExecutePipelineCommand,
        UpdatePipelineCommand,
    )
    from flx_core.contracts.repository_contracts import (
        RepositoryInterface,
        UnitOfWorkInterface,
    )
    from flx_core.domain.value_objects import ExecutionStatus
    from flx_core.events.event_bus import DomainEventBus


# Import the canonical CreatePipelineCommand from the unified commands module
# Import the canonical UpdatePipelineCommand from the unified commands module
# Import the canonical ExecutePipelineCommand from the unified commands module


class PipelineManagementService:
    """Provides a high-level API for managing data pipelines.

    This service orchestrates the creation, modification, and deletion of
    pipelines, ensuring that all operations are transactional and that relevant
    domain events are dispatched.
    """

    def __init__(
        self, unit_of_work: UnitOfWorkInterface, event_bus: DomainEventBus
    ) -> None:
        """Initialize the service with its dependencies."""
        self._uow = unit_of_work
        self._event_bus = event_bus

    def _convert_to_configuration_dict(
        self, data: dict[str, Any] | None
    ) -> ConfigurationDict:
        """Convert dictionary data to ConfigurationDict with proper type handling.  # noqa: PLR0911 - type conversion requires multiple returns.

        Args:
        ----
            data: Input dictionary to convert

        Returns:
        -------
            ConfigurationDict with properly typed values

        """
        result: ConfigurationDict = {}
        if not data:
            return result

        for k, v in data.items():
            if v is not None:
                if isinstance(v, str | int | float | bool | list | dict):
                    result[k] = v
                else:
                    result[k] = str(v)
        return result

    async def _create_pipeline_step(
        self, step_data: dict[str, Any], plugin_repo: Any
    ) -> PipelineStep:
        """Create a pipeline step from step data.  # noqa: PLR0911 - step creation requires multiple returns.

        Args:
        ----
            step_data: Raw step data dictionary
            plugin_repo: Plugin repository for validation

        Returns:
        -------
            Configured PipelineStep instance

        """
        plugin_id = PluginId(value=UUID(str(step_data["plugin_id"])))
        plugin = await plugin_repo.find_by_id(plugin_id.value)
        if not plugin:
            msg = f"Plugin {plugin_id.value} not found"
            raise ValueError(msg)

        # Convert step data with proper type handling
        order_val = step_data["order"]
        order = int(order_val) if isinstance(order_val, int | str) else 1

        configuration = self._convert_to_configuration_dict(
            step_data.get("configuration", {}),
        )

        # Handle depends_on with proper type conversion
        depends_on_data = step_data.get("depends_on", [])
        depends_on: frozenset[str] = frozenset()
        if isinstance(depends_on_data, list | tuple):
            depends_on = frozenset(str(item) for item in depends_on_data)

        return PipelineStep(
            step_id=str(step_data["step_id"]),
            plugin_id=plugin_id,
            order=order,
            configuration=configuration,
            depends_on=depends_on,
        )

    async def create_pipeline(self, command: CreatePipelineCommand) -> Pipeline:
        """Create a new data pipeline."""
        async with self._uow as uow:
            pipeline_repo = uow.get_repository(Pipeline, PipelineModel)
            plugin_repo = uow.get_repository(Plugin, PluginModel)

            existing_pipeline = await pipeline_repo.find_by_name(command.name)
            if existing_pipeline:
                msg = f"Pipeline with name '{command.name}' already exists"
                raise ValueError(msg)

            env_vars = self._convert_to_configuration_dict(
                command.environment_variables,
            )

            pipeline = Pipeline(
                name=PipelineName(value=command.name),
                description=command.description,
                environment_variables=env_vars,
                schedule_expression=command.schedule_expression,
                timezone=command.timezone,
                max_concurrent_executions=command.max_concurrent_executions,
                created_by=command.created_by,
            )

            for step_data in command.steps:
                step = await self._create_pipeline_step(step_data, plugin_repo)
                pipeline.add_step(step)

            await pipeline_repo.save(pipeline)

            for event in pipeline.uncommitted_events:
                await self._event_bus.publish(event)
            pipeline.mark_events_as_committed()

            return pipeline

    async def update_pipeline(self, command: UpdatePipelineCommand) -> Pipeline:
        """Update an existing data pipeline."""
        async with self._uow as uow:
            pipeline_repo = uow.get_repository(Pipeline, PipelineModel)
            uow.get_repository(Plugin, PluginModel)

            pipeline = await pipeline_repo.find_by_id(command.pipeline_id)
            if not pipeline:
                msg = f"Pipeline {command.pipeline_id} not found"
                raise ValueError(msg)

            # Update basic pipeline properties
            self._update_pipeline_properties(pipeline, command)

            # Note: Step updates are handled through separate commands to maintain clear responsibilities

            pipeline.updated_by = command.updated_by
            await pipeline_repo.save(pipeline)

            return pipeline

    def _update_pipeline_properties(
        self, pipeline: Pipeline, command: UpdatePipelineCommand
    ) -> None:
        """Update basic pipeline properties from command."""
        if command.name is not None:
            pipeline.name = PipelineName(value=command.name)
        if command.description is not None:
            pipeline.description = command.description
        if command.environment_variables is not None:
            # Convert environment variables to ConfigurationDict type with proper value conversion
            env_vars: ConfigurationDict = {}
            for k, v in command.environment_variables.items():
                if v is not None:
                    if isinstance(v, str | int | float | bool | list | dict):
                        env_vars[k] = v
                    else:
                        env_vars[k] = str(v)
            pipeline.environment_variables = env_vars
        if command.schedule_expression is not None:
            pipeline.schedule_expression = command.schedule_expression
        if command.is_active is not None:
            if command.is_active:
                pipeline.activate()
            else:
                pipeline.deactivate()

    async def _update_pipeline_steps(
        self,
        pipeline: Pipeline,
        steps_data: list[PipelineStepData],
        plugin_repo: RepositoryInterface[Any, UUID],
    ) -> None:
        """Update pipeline steps from command data."""
        pipeline.steps.clear()
        for step_data in steps_data:
            plugin_id = PluginId(value=UUID(str(step_data["plugin_id"])))
            plugin = await plugin_repo.find_by_id(plugin_id.value)
            if not plugin:
                msg = f"Plugin {plugin_id.value} not found"
                raise ValueError(msg)

            step = PipelineStep(
                step_id=str(step_data["step_id"]),
                plugin_id=plugin_id,
                order=int(step_data["order"]),
                configuration=dict(step_data.get("configuration", {})),
                depends_on=frozenset(step_data.get("depends_on", [])),
            )
            pipeline.add_step(step)

    async def delete_pipeline(self, pipeline_id: PipelineId) -> None:
        """Delete a data pipeline."""
        async with self._uow as uow:
            pipeline_repo = uow.get_repository(Pipeline, PipelineModel)
            pipeline = await pipeline_repo.find_by_id(pipeline_id)
            if not pipeline:
                msg = f"Pipeline {pipeline_id} not found"
                raise ValueError(msg)
            await pipeline_repo.delete(pipeline_id)

    async def get_pipeline(self, pipeline_id: PipelineId) -> Pipeline | None:
        """Retrieve a single pipeline by its ID."""
        async with self._uow as uow:
            pipeline_repo = uow.get_repository(Pipeline, PipelineModel)
            return await pipeline_repo.find_by_id(pipeline_id)

    async def list_pipelines(
        self, *, active_only: bool = False, limit: int = 100, offset: int = 0
    ) -> list[Pipeline]:
        """List all available data pipelines."""
        async with self._uow as uow:
            pipeline_repo = uow.get_repository(Pipeline, PipelineModel)
            if active_only:
                criteria_result = await pipeline_repo.find_by_criteria(
                    {"is_active": True},
                    limit=limit,
                    offset=offset,
                )
                return list(criteria_result)
            all_result = await pipeline_repo.find_all(limit=limit, offset=offset)
            return list(all_result)


class PipelineExecutionService:
    """Service for orchestrating pipeline executions."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkInterface,
        event_bus: DomainEventBus,
        meltano_acl: MeltanoAntiCorruptionLayer | None = None,
    ) -> None:
        """Initialize the service with its dependencies."""
        self._uow = unit_of_work
        self._event_bus = event_bus
        self._meltano_acl = meltano_acl
        self._logger = structlog.get_logger(self.__class__.__name__)

    def _convert_to_configuration_dict(
        self, data: dict[str, Any] | None
    ) -> ConfigurationDict:
        """Convert dictionary data to ConfigurationDict with proper type handling.

        Args:
        ----
            data: Input dictionary to convert

        Returns:
        -------
            ConfigurationDict with properly typed values

        """
        result: ConfigurationDict = {}
        if not data:
            return result

        for k, v in data.items():
            if v is not None:
                if isinstance(v, str | int | float | bool | list | dict):
                    result[k] = v
                else:
                    result[k] = str(v)
        return result

    async def _validate_pipeline_for_execution(
        self, pipeline_id: PipelineId, uow: Any
    ) -> Pipeline:
        """Validate pipeline exists and can be executed.  # noqa: PLR0911 - validation requires multiple returns.

        Args:
        ----
            pipeline_id: ID of pipeline to validate
            uow: Unit of work for repository access

        Returns:
        -------
            Pipeline instance if valid

        Raises:
        ------
            ValueError: If pipeline not found or cannot execute

        """
        pipeline_repo = uow.get_repository(Pipeline, PipelineModel)
        pipeline = await pipeline_repo.find_by_id(pipeline_id)
        if not pipeline:
            msg = f"Pipeline {pipeline_id} not found for execution"
            raise ValueError(msg)

        if not pipeline.can_execute():
            msg = f"Pipeline {pipeline.name.value} cannot be executed"
            raise ValueError(msg)

        return pipeline

    async def _check_concurrent_execution_limit(
        self, pipeline: Pipeline, execution_repo: Any
    ) -> None:
        """Check if pipeline can start new execution based on concurrent limits.  # noqa: PLR0911 - limit check requires multiple returns.

        Args:
        ----
            pipeline: Pipeline to check limits for
            execution_repo: Repository for execution queries

        Raises:
        ------
            ValueError: If concurrent execution limit reached

        """
        running_executions = await execution_repo.find_by_criteria(
            {
                "pipeline_id": pipeline.id.value,
                "status": "RUNNING",
            },
        )
        if len(running_executions) >= pipeline.max_concurrent_executions:
            msg = f"Maximum concurrent executions ({pipeline.max_concurrent_executions}) reached"
            raise ValueError(msg)

    async def _get_next_execution_number(self, execution_repo: Any) -> int:
        """Get next execution number for pipeline.  # noqa: PLR0911 - number generation requires multiple returns.

        Args:
        ----
            execution_repo: Repository for counting executions

        Returns:
        -------
            Next execution number

        """
        try:
            execution_count = await execution_repo.count()
            return execution_count + 1
        except AttributeError:
            # Fallback for repositories without count method
            return 1

    async def execute_pipeline(
        self, command: ExecutePipelineCommand
    ) -> PipelineExecution:
        """Trigger the execution of a data pipeline."""
        logger = structlog.get_logger(self.__class__.__name__)
        logger.info("Executing pipeline", pipeline_id=command.pipeline_id)

        async with self._uow as uow:
            pipeline = await self._validate_pipeline_for_execution(
                command.pipeline_id,
                uow,
            )
            execution_repo = uow.get_repository(
                PipelineExecution,
                PipelineExecutionModel,
            )

            await self._check_concurrent_execution_limit(pipeline, execution_repo)
            next_execution_number = await self._get_next_execution_number(
                execution_repo,
            )

            execution = pipeline.create_execution(
                triggered_by=command.triggered_by,
                execution_number=next_execution_number,
            )

            execution.input_data = self._convert_to_configuration_dict(
                command.input_data,
            )

            await execution_repo.save(execution)

            for event in pipeline.uncommitted_events:
                await self._event_bus.publish(event)
            pipeline.mark_events_as_committed()

            if self._meltano_acl:
                # ZERO TOLERANCE IMPLEMENTATION: Execute immediately for synchronous operation
                # Background task execution is handled by dependency injection container
                # EventBus will handle async task distribution to worker pools
                await self._run_pipeline_with_meltano(pipeline, execution)

            return execution

    async def get_execution(self, execution_id: UUID) -> PipelineExecution | None:
        """Retrieve a single pipeline execution by its ID."""
        async with self._uow as uow:
            execution_repo = uow.get_repository(
                PipelineExecution,
                PipelineExecutionModel,
            )
            return await execution_repo.find_by_id(execution_id)

    async def list_executions(
        self,
        pipeline_id: PipelineId | None = None,
        status: ExecutionStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PipelineExecution]:
        """List executions for a given pipeline or status."""
        async with self._uow as uow:
            execution_repo = uow.get_repository(
                PipelineExecution,
                PipelineExecutionModel,
            )
            criteria: dict[str, object] = {}
            if pipeline_id:
                criteria["pipeline_id"] = pipeline_id
            if status:
                criteria["status"] = status.value

            # Repository interface may not support limit/offset parameters
            # Apply basic criteria filtering
            executions = await execution_repo.find_by_criteria(criteria)

            # Manual pagination if needed
            if isinstance(executions, list):
                start_idx = offset
                end_idx = offset + limit
                return executions[start_idx:end_idx]
            return list(executions)[:limit]

    async def cancel_execution(
        self, execution_id: UUID, cancelled_by: str
    ) -> PipelineExecution:
        """Cancel a running pipeline execution."""
        async with self._uow as uow:
            repo = uow.get_repository(PipelineExecution, PipelineExecutionModel)
            execution = await repo.find_by_id(execution_id)
            if not execution:
                msg = f"Execution {execution_id} not found"
                raise ValueError(msg)

            execution.cancel(cancelled_by=cancelled_by)
            await repo.save(execution)
            return execution

    async def _run_pipeline_with_meltano(
        self, pipeline: Pipeline, execution: PipelineExecution
    ) -> None:
        """Run the pipeline using the Meltano anti-corruption layer."""
        if not self._meltano_acl:
            return

        logger = structlog.get_logger(self.__class__.__name__)

        try:
            execution.start()
            async with self._uow as uow:
                repo = uow.get_repository(PipelineExecution, PipelineExecutionModel)
                await repo.save(execution)

            # ZERO TOLERANCE P1 FIX: Enhanced async/await patterns with ServiceResult handling
            try:
                service_result = await self._meltano_acl.translate_and_run_pipeline(
                    pipeline,
                    execution,
                )

                # Handle ServiceResult pattern with proper error checking
                if service_result.is_ok() and service_result.data:
                    meltano_result = service_result.data
                    try:
                        success_flag = meltano_result.success
                        if success_flag:
                            output_data = getattr(meltano_result, "output", {})
                    except AttributeError:
                        output_data = {}
                        if isinstance(output_data, dict):
                            execution.complete_successfully(output_data)
                        else:
                            execution.complete_successfully({})
                    else:
                        error_msg = getattr(
                            meltano_result,
                            "stderr",
                            "Unknown execution error",
                        )
                        execution.fail(str(error_msg))
                else:
                    # Handle ServiceResult failure
                    error_details = (
                        service_result.error.message
                        if service_result.error
                        else "Service execution failed"
                    )
                    execution.fail(str(error_details))

            except AttributeError as e:
                # Handle case where meltano ACL doesn't have expected interface
                execution.fail(f"Meltano ACL interface error: {e}")
            except (
                RuntimeError,
                ValueError,
                TypeError,
                ImportError,
                ModuleNotFoundError,
            ) as e:
                # Handle known execution errors with specific exception types
                execution.fail(f"Execution runtime error: {e}")

        except (
            RuntimeError,
            ValueError,
            OSError,
            ImportError,
            TypeError,
            AttributeError,
        ) as e:
            logger.exception(
                "Pipeline execution failed",
                execution_id=execution.id.value,
            )
            execution.fail(str(e))

        finally:
            async with self._uow as uow:
                repo = uow.get_repository(PipelineExecution, PipelineExecutionModel)
                await repo.save(execution)
