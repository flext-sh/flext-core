"""ADR-001 Hexagonal Architecture Ports - Clean Architecture Interfaces.

Defines the ports (interfaces) for hexagonal architecture following ADR-001 Clean Architecture
and Domain-Driven Design principles. Implements primary and secondary ports that enable
the dependency inversion principle and technology independence.

ARCHITECTURAL COMPLIANCE:
- ADR-001: Hexagonal Architecture with strict port/adapter separation
- Clean Architecture: Dependency rule enforcement - domain defines interfaces
- DDD: Repository and Service patterns for aggregate persistence and coordination
- CLAUDE.md: ZERO TOLERANCE - Real interfaces with comprehensive contracts
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flext_core.domain.advanced_types import (
        ConfigurationDict,
        MetadataDict,
        QueryParameters,
        ServiceResult,
    )
    from flext_core.domain.entities import Pipeline, PipelineExecution, Plugin
    from flext_core.domain.specifications import CompositeSpecification
    from flext_core.domain.value_objects import ExecutionId, PipelineId, PluginId
    from flext_core.events.event_bus import DomainEvent


# PRIMARY PORTS - Driven by external actors (UI, API, CLI)
class PipelineManagementPort(ABC):
    """Primary port for pipeline management operations.

    Defines the contract for pipeline management use cases, driven by external
    actors such as web UI, REST API, or CLI commands. Follows ADR-001 Clean
    Architecture with use case interfaces at the application boundary.
    """

    @abstractmethod
    async def create_pipeline(
        self,
        name: str,
        description: str | None,
        steps: list[dict[str, Any]],
        created_by: str,
        metadata: MetadataDict | None = None,
    ) -> ServiceResult[PipelineId]:
        """Create a new pipeline with validation and business rules.

        Args:
        ----
            name: Human-readable pipeline name
            description: Optional pipeline description
            steps: List of pipeline step configurations
            created_by: User identifier who created the pipeline
            metadata: Optional metadata for the pipeline

        Returns:
        -------
            ServiceResult containing the created pipeline ID or error details

        """

    @abstractmethod
    async def update_pipeline(
        self, pipeline_id: PipelineId, updates: dict[str, Any], updated_by: str,
    ) -> ServiceResult[Pipeline]:
        """Update existing pipeline with business validation.

        Args:
        ----
            pipeline_id: Unique identifier of pipeline to update
            updates: Dictionary of fields to update
            updated_by: User identifier who updated the pipeline

        Returns:
        -------
            ServiceResult containing updated pipeline or error details

        """

    @abstractmethod
    async def delete_pipeline(
        self, pipeline_id: PipelineId, deleted_by: str, force: bool = False,
    ) -> ServiceResult[bool]:
        """Delete pipeline with safety checks.

        Args:
        ----
            pipeline_id: Unique identifier of pipeline to delete
            deleted_by: User identifier who deleted the pipeline
            force: Whether to force deletion even if executions exist

        Returns:
        -------
            ServiceResult indicating success or failure

        """

    @abstractmethod
    async def execute_pipeline(
        self,
        pipeline_id: PipelineId,
        triggered_by: str,
        parameters: ConfigurationDict | None = None,
        environment: str | None = None,
    ) -> ServiceResult[ExecutionId]:
        """Execute pipeline with runtime parameters.

        Args:
        ----
            pipeline_id: Unique identifier of pipeline to execute
            triggered_by: User identifier who triggered execution
            parameters: Optional runtime parameters for execution
            environment: Optional environment specification

        Returns:
        -------
            ServiceResult containing execution ID or error details

        """

    @abstractmethod
    async def get_pipeline(self, pipeline_id: PipelineId) -> ServiceResult[Pipeline]:
        """Retrieve pipeline by ID.

        Args:
        ----
            pipeline_id: Unique identifier of pipeline to retrieve

        Returns:
        -------
            ServiceResult containing pipeline or error if not found

        """

    @abstractmethod
    async def list_pipelines(
        self,
        filters: dict[str, Any] | None = None,
        pagination: dict[str, int] | None = None,
    ) -> ServiceResult[list[Pipeline]]:
        """List pipelines with optional filtering and pagination.

        Args:
        ----
            filters: Optional filters to apply (status, created_by, etc.)
            pagination: Optional pagination parameters (page, size)

        Returns:
        -------
            ServiceResult containing list of pipelines matching criteria

        """


class ExecutionMonitoringPort(ABC):
    """Primary port for execution monitoring and control.

    Provides interfaces for monitoring and controlling pipeline executions,
    including real-time status updates and execution management operations.
    """

    @abstractmethod
    async def get_execution_status(
        self, execution_id: ExecutionId,
    ) -> ServiceResult[PipelineExecution]:
        """Get current execution status and details.

        Args:
        ----
            execution_id: Unique identifier of execution to monitor

        Returns:
        -------
            ServiceResult containing execution details or error

        """

    @abstractmethod
    async def cancel_execution(
        self,
        execution_id: ExecutionId,
        cancelled_by: str,
        reason: str | None = None,
    ) -> ServiceResult[bool]:
        """Cancel running execution.

        Args:
        ----
            execution_id: Unique identifier of execution to cancel
            cancelled_by: User identifier who cancelled execution
            reason: Optional reason for cancellation

        Returns:
        -------
            ServiceResult indicating success or failure

        """

    @abstractmethod
    async def retry_execution(
        self,
        execution_id: ExecutionId,
        retried_by: str,
        retry_parameters: ConfigurationDict | None = None,
    ) -> ServiceResult[ExecutionId]:
        """Retry failed execution with optional parameter overrides.

        Args:
        ----
            execution_id: Unique identifier of execution to retry
            retried_by: User identifier who initiated retry
            retry_parameters: Optional parameter overrides for retry

        Returns:
        -------
            ServiceResult containing new execution ID or error

        """

    @abstractmethod
    async def get_execution_logs(
        self, execution_id: ExecutionId, log_level: str | None = None,
    ) -> ServiceResult[list[str]]:
        """Retrieve execution logs with optional filtering.

        Args:
        ----
            execution_id: Unique identifier of execution
            log_level: Optional log level filter (DEBUG, INFO, WARNING, ERROR)

        Returns:
        -------
            ServiceResult containing execution logs or error

        """


class PluginManagementPort(ABC):
    """Primary port for plugin management operations.

    Provides interfaces for managing plugins including installation,
    configuration, and lifecycle management operations.
    """

    @abstractmethod
    async def install_plugin(
        self,
        plugin_name: str,
        plugin_type: str,
        configuration: ConfigurationDict,
        installed_by: str,
    ) -> ServiceResult[PluginId]:
        """Install and configure a new plugin.

        Args:
        ----
            plugin_name: Name of plugin to install
            plugin_type: Type of plugin (tap, target, transform, etc.)
            configuration: Plugin configuration parameters
            installed_by: User identifier who installed plugin

        Returns:
        -------
            ServiceResult containing plugin ID or error details

        """

    @abstractmethod
    async def configure_plugin(
        self,
        plugin_id: PluginId,
        configuration: ConfigurationDict,
        configured_by: str,
    ) -> ServiceResult[Plugin]:
        """Update plugin configuration.

        Args:
        ----
            plugin_id: Unique identifier of plugin to configure
            configuration: New configuration parameters
            configured_by: User identifier who updated configuration

        Returns:
        -------
            ServiceResult containing updated plugin or error

        """

    @abstractmethod
    async def uninstall_plugin(
        self, plugin_id: PluginId, uninstalled_by: str, force: bool = False,
    ) -> ServiceResult[bool]:
        """Uninstall plugin with dependency checks.

        Args:
        ----
            plugin_id: Unique identifier of plugin to uninstall
            uninstalled_by: User identifier who uninstalled plugin
            force: Whether to force uninstall despite dependencies

        Returns:
        -------
            ServiceResult indicating success or failure

        """


# SECONDARY PORTS - Drive external systems (Database, Message Queue, External APIs)
class PipelineRepositoryPort(ABC):
    """Secondary port for pipeline persistence operations.

    Defines the contract for pipeline aggregate persistence, following DDD
    Repository pattern with aggregate-oriented operations.
    """

    @abstractmethod
    async def save(self, pipeline: Pipeline) -> Pipeline:
        """Persist pipeline aggregate with event handling.

        Args:
        ----
            pipeline: Pipeline aggregate to persist

        Returns:
        -------
            Persisted pipeline with updated metadata

        Raises:
        ------
            ConcurrencyError: If optimistic locking detects conflicts
            ValidationError: If persistence validation fails

        """

    @abstractmethod
    async def get_by_id(self, pipeline_id: PipelineId) -> Pipeline | None:
        """Retrieve pipeline aggregate by unique identifier.

        Args:
        ----
            pipeline_id: Unique identifier of pipeline

        Returns:
        -------
            Pipeline aggregate or None if not found

        """

    @abstractmethod
    async def find_by_specification(
        self, specification: CompositeSpecification[Pipeline],
    ) -> list[Pipeline]:
        """Query pipelines using domain specifications.

        Args:
        ----
            specification: Business rule specification to evaluate

        Returns:
        -------
            List of pipelines satisfying the specification

        """

    @abstractmethod
    async def delete(self, pipeline_id: PipelineId) -> bool:
        """Delete pipeline aggregate.

        Args:
        ----
            pipeline_id: Unique identifier of pipeline to delete

        Returns:
        -------
            True if deletion successful, False if not found

        """

    @abstractmethod
    async def exists(self, pipeline_id: PipelineId) -> bool:
        """Check if pipeline exists.

        Args:
        ----
            pipeline_id: Unique identifier to check

        Returns:
        -------
            True if pipeline exists, False otherwise

        """


class ExecutionRepositoryPort(ABC):
    """Secondary port for execution persistence operations.

    Manages pipeline execution entity persistence with support for execution
    lifecycle tracking and historical queries.
    """

    @abstractmethod
    async def save_execution(self, execution: PipelineExecution) -> PipelineExecution:
        """Persist execution entity with status tracking.

        Args:
        ----
            execution: Execution entity to persist

        Returns:
        -------
            Persisted execution with updated metadata

        """

    @abstractmethod
    async def get_execution_by_id(
        self, execution_id: ExecutionId,
    ) -> PipelineExecution | None:
        """Retrieve execution by unique identifier.

        Args:
        ----
            execution_id: Unique identifier of execution

        Returns:
        -------
            Execution entity or None if not found

        """

    @abstractmethod
    async def find_executions_by_pipeline(
        self, pipeline_id: PipelineId, limit: int | None = None,
    ) -> list[PipelineExecution]:
        """Find executions for specific pipeline.

        Args:
        ----
            pipeline_id: Pipeline to find executions for
            limit: Optional limit on number of results

        Returns:
        -------
            List of executions for the pipeline

        """

    @abstractmethod
    async def find_active_executions(self) -> list[PipelineExecution]:
        """Find all currently active executions.

        Returns
        -------
            List of executions with non-terminal status

        """


class PluginRepositoryPort(ABC):
    """Secondary port for plugin persistence operations.

    Manages plugin entity persistence with configuration management and
    dependency tracking capabilities.
    """

    @abstractmethod
    async def save_plugin(self, plugin: Plugin) -> Plugin:
        """Persist plugin entity with configuration.

        Args:
        ----
            plugin: Plugin entity to persist

        Returns:
        -------
            Persisted plugin with updated metadata

        """

    @abstractmethod
    async def get_plugin_by_id(self, plugin_id: PluginId) -> Plugin | None:
        """Retrieve plugin by unique identifier.

        Args:
        ----
            plugin_id: Unique identifier of plugin

        Returns:
        -------
            Plugin entity or None if not found

        """

    @abstractmethod
    async def find_plugins_by_type(self, plugin_type: str) -> list[Plugin]:
        """Find plugins by type classification.

        Args:
        ----
            plugin_type: Type of plugins to find (tap, target, etc.)

        Returns:
        -------
            List of plugins matching the specified type

        """

    @abstractmethod
    async def find_available_plugins(self) -> list[Plugin]:
        """Find all available and compatible plugins.

        Returns
        -------
            List of plugins available for use

        """


class EventBusPort(ABC):
    """Secondary port for domain event publishing and subscription.

    Provides event-driven architecture capabilities for domain event handling
    and cross-aggregate communication.
    """

    @abstractmethod
    async def publish_event(self, event: DomainEvent) -> None:
        """Publish domain event for eventual consistency.

        Args:
        ----
            event: Domain event to publish

        Raises:
        ------
            EventPublishingError: If event publishing fails

        """

    @abstractmethod
    async def publish_events(self, events: list[DomainEvent]) -> None:
        """Publish multiple domain events atomically.

        Args:
        ----
            events: List of domain events to publish

        Raises:
        ------
            EventPublishingError: If any event publishing fails

        """

    @abstractmethod
    async def subscribe_to_events(self, event_type: str, handler: Any) -> None:
        """Subscribe to specific domain event types.

        Args:
        ----
            event_type: Type of events to subscribe to
            handler: Event handler function

        Raises:
        ------
            SubscriptionError: If subscription setup fails

        """


class DistributedExecutionPort(ABC):
    """Secondary port for distributed execution orchestration.

    Interfaces with external distributed computing systems (Ray, Kubernetes)
    for scalable pipeline execution and resource management.
    """

    @abstractmethod
    async def execute_distributed_task(
        self,
        task_definition: dict[str, Any],
        resource_requirements: dict[str, Any] | None = None,
    ) -> ServiceResult[dict[str, Any]]:
        """Execute task in distributed computing environment.

        Args:
        ----
            task_definition: Task configuration and parameters
            resource_requirements: CPU, memory, GPU requirements

        Returns:
        -------
            ServiceResult containing task result or error

        """

    @abstractmethod
    async def scale_resources(
        self, target_nodes: int, resource_type: str = "cpu",
    ) -> ServiceResult[dict[str, Any]]:
        """Scale distributed computing resources.

        Args:
        ----
            target_nodes: Desired number of compute nodes
            resource_type: Type of resources to scale

        Returns:
        -------
            ServiceResult containing scaling status

        """

    @abstractmethod
    async def get_cluster_status(self) -> ServiceResult[dict[str, Any]]:
        """Get current distributed cluster status.

        Returns
        -------
            ServiceResult containing cluster information

        """


class ExternalIntegrationPort(ABC):
    """Secondary port for external system integrations.

    Provides anti-corruption layer interfaces for integrating with external
    systems while maintaining domain model integrity.
    """

    @abstractmethod
    async def send_notification(
        self,
        notification_type: str,
        recipient: str,
        message: str,
        metadata: MetadataDict | None = None,
    ) -> ServiceResult[bool]:
        """Send notification through external system.

        Args:
        ----
            notification_type: Type of notification (email, slack, etc.)
            recipient: Notification recipient identifier
            message: Notification message content
            metadata: Optional notification metadata

        Returns:
        -------
            ServiceResult indicating notification success

        """

    @abstractmethod
    async def fetch_external_data(
        self, source: str, query: QueryParameters,
    ) -> ServiceResult[dict[str, Any]]:
        """Fetch data from external systems.

        Args:
        ----
            source: External data source identifier
            query: Query parameters for data retrieval

        Returns:
        -------
            ServiceResult containing external data or error

        """

    @abstractmethod
    async def validate_external_resource(
        self, resource_type: str, resource_identifier: str,
    ) -> ServiceResult[bool]:
        """Validate external resource availability.

        Args:
        ----
            resource_type: Type of external resource
            resource_identifier: Unique identifier of resource

        Returns:
        -------
            ServiceResult indicating resource validity

        """


class AuditLogPort(ABC):
    """Secondary port for audit logging and compliance.

    Provides interfaces for comprehensive audit trail management and
    regulatory compliance requirements.
    """

    @abstractmethod
    async def log_user_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        metadata: MetadataDict | None = None,
    ) -> None:
        """Log user action for audit trail.

        Args:
        ----
            user_id: Identifier of user performing action
            action: Action performed (create, update, delete, etc.)
            resource_type: Type of resource affected
            resource_id: Unique identifier of affected resource
            metadata: Optional action metadata

        """

    @abstractmethod
    async def log_system_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        metadata: MetadataDict | None = None,
    ) -> None:
        """Log system event for monitoring and audit.

        Args:
        ----
            event_type: Type of system event
            severity: Event severity (info, warning, error, critical)
            message: Event description
            metadata: Optional event metadata

        """

    @abstractmethod
    async def query_audit_logs(
        self,
        filters: dict[str, Any],
        pagination: dict[str, int] | None = None,
    ) -> ServiceResult[list[dict[str, Any]]]:
        """Query audit logs with filtering and pagination.

        Args:
        ----
            filters: Filters for log retrieval
            pagination: Optional pagination parameters

        Returns:
        -------
            ServiceResult containing matching audit log entries

        """
