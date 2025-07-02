"""Enterprise-grade dependency injection container using dependency-injector.

This module implements a comprehensive dependency injection system using the
dependency-injector library to replace manual service instantiation with
declarative, type-safe dependency management.

📋 Architecture Documentation: docs/architecture/003-plugin-system-architecture/
🔗 Plugin Integration Ready: docs/architecture/003-plugin-system-architecture/IMPLEMENTATION-REALITY-MAP.md#available-integration-infrastructure
📊 DI Analysis: docs/architecture/003-plugin-system-architecture/01-context-and-vision.md#122-available-architecture

ARCHITECTURAL PRINCIPLES:
- Declarative container configuration
- Automatic resource lifecycle management
- Type-safe dependency resolution
- Environment-specific configuration polymorphism
- Lazy loading and singleton patterns where appropriate
- Integration with existing domain architecture

PERFORMANCE BENEFITS:
- Automatic dependency graph optimization
- Resource pooling and connection management
- Lazy initialization of expensive resources
- Proper cleanup and resource disposal
- Reduced boilerplate and manual wiring

PLUGIN SYSTEM INTEGRATION:
✅ Ready for plugin service registration via ServiceContainer
✅ Plugin lifecycle management through ApplicationContainer
✅ Plugin dependency injection through providers system
❌ Missing: Plugin Manager integration (see IMPLEMENTATION-REALITY-MAP.md)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog
from dependency_injector import containers, providers
from flx_meltano.job_manager import FlxMeltanoJobManager
from flx_meltano.orchestrator import FlxMeltanoOrchestrator
from flx_meltano.project_manager import FlxMeltanoProjectManager
from flx_meltano.state_manager import FlxMeltanoStateManager
from flx_observability.health import HealthChecker
from flx_observability.metrics import MetricsCollector
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from flx_core.config.domain_config import get_config
from flx_core.domain.entities import Pipeline, PipelineExecution, Plugin
from flx_core.events.event_bus import HybridEventBus
from flx_core.execution.state_machines import (
    JobStateMachine,
    PipelineExecutionStateMachine,
)
from flx_core.infrastructure.persistence.repositories_core import (
    DomainSpecificRepository,
    SqlAlchemyRepository,
)
from flx_core.infrastructure.persistence.unit_of_work import UnitOfWork
from flx_core.serialization.msgspec_adapters import (
    HighPerformanceSerializer,
    get_serializer,
)
from flx_core.services.analytics_service import AnalyticsService
from flx_core.services.execution_service import ExecutionService
from flx_core.services.export_service import ExportService
from flx_core.services.notification_service import NotificationService
from flx_core.services.pipeline_service import PipelineService
from flx_core.services.plugin_service import PluginService
from flx_core.services.scheduler_service import SchedulerService
from flx_core.services.validation_service import ValidationService

if TYPE_CHECKING:
    from flx_core.engine.meltano_wrapper import MeltanoEngine
    from flx_core.infrastructure.persistence.models import (
        PipelineExecutionModel,
        PipelineModel,
        PluginModel,
    )

logger = structlog.get_logger()

# Python 3.13 type aliases for dependency injection
DatabaseEngine = object
SessionFactory = async_sessionmaker[AsyncSession]
type RepositoryFactory[T] = Callable[[], SqlAlchemyRepository[T, object, object]]
type ServiceFactory[T] = Callable[[], T]


class DatabaseContainer(containers.DeclarativeContainer):
    """Database-specific dependency injection container.

    Manages database connections, session factories, and repository instances
    with proper resource lifecycle management and connection pooling.
    """

    # Configuration injection
    config = providers.Configuration()

    # Database engine with connection pooling - ZERO TOLERANCE P0 FIX: conditional pooling for SQLite
    database_engine: providers.Resource[DatabaseEngine] = providers.Resource(
        lambda url, **kwargs: create_async_engine(
            url,
            echo=kwargs.get("echo", False),
            # Only include pooling options for non-SQLite databases
            **(
                {
                    "pool_pre_ping": True,
                    "pool_size": kwargs.get("pool_size", 10),
                    "max_overflow": kwargs.get("max_overflow", 20),
                    "pool_timeout": kwargs.get("pool_timeout", 30),
                    "pool_recycle": kwargs.get("pool_recycle", 3600),
                }
                if not url.startswith("sqlite")
                else {}
            ),
        ),
        config.database.url,
        echo=config.debug,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
        pool_timeout=config.database.pool_timeout,
        pool_recycle=config.database.pool_recycle,
    )

    # Session factory for async database operations
    session_factory: providers.Singleton[SessionFactory] = providers.Singleton(
        async_sessionmaker,
        database_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Unit of Work pattern implementation - ZERO TOLERANCE P0 FIX: pass session instance
    unit_of_work: providers.Factory[UnitOfWork] = providers.Factory(
        UnitOfWork,
        session=session_factory.provided.call(),
    )


class EventingContainer(containers.DeclarativeContainer):
    """Event system dependency injection container.

    Manages event bus instances, event handlers, and messaging infrastructure
    with proper integration to the domain event system.
    """

    # Configuration injection
    config = providers.Configuration()

    # Centralized event bus for domain events
    event_bus: providers.Singleton[HybridEventBus] = providers.Singleton(
        HybridEventBus,
    )

    # High-performance serializer for event serialization
    serializer: providers.Singleton[HighPerformanceSerializer] = providers.Singleton(
        get_serializer,
    )


class RepositoryContainer(containers.DeclarativeContainer):
    """Repository pattern dependency injection container.

    Provides type-safe repository instances with automatic session management
    and proper domain entity mapping.
    """

    # Database dependencies
    database = providers.DependenciesContainer()

    # Repository factories for domain entities
    pipeline_repository: providers.Factory[
        DomainSpecificRepository[Pipeline, PipelineModel, str]
    ] = providers.Factory(
        DomainSpecificRepository,
        session=database.session_factory.provided.call(),
        entity_class=Pipeline,
        model_class=providers.Object(
            "flx_core.infrastructure.persistence.models.PipelineModel",
        ),
        id_field="pipeline_id",
    )

    pipeline_execution_repository: providers.Factory[
        DomainSpecificRepository[PipelineExecution, PipelineExecutionModel, str]
    ] = providers.Factory(
        DomainSpecificRepository,
        session=database.session_factory.provided.call(),
        entity_class=PipelineExecution,
        model_class=providers.Object(
            "flx_core.infrastructure.persistence.models.PipelineExecutionModel",
        ),
        id_field="execution_id",
    )

    plugin_repository: providers.Factory[
        DomainSpecificRepository[Plugin, PluginModel, str]
    ] = providers.Factory(
        DomainSpecificRepository,
        session=database.session_factory.provided.call(),
        entity_class=Plugin,
        model_class=providers.Object(
            "flx_core.infrastructure.persistence.models.PluginModel",
        ),
        id_field="plugin_id",
    )


class MeltanoContainer(containers.DeclarativeContainer):
    """Meltano ecosystem dependency injection container.

    Manages Meltano-specific services including project management,
    job orchestration, state management, and event integration.
    """

    # Configuration injection
    config = providers.Configuration()

    # Event dependencies
    eventing = providers.DependenciesContainer()

    # Meltano project manager
    project_manager: providers.Singleton[FlxMeltanoProjectManager] = (
        providers.Singleton(
            FlxMeltanoProjectManager,
            project_root=config.meltano.project_root,
            event_bus=eventing.event_bus,
        )
    )

    # Meltano state manager with event integration
    state_manager: providers.Singleton[FlxMeltanoStateManager] = providers.Singleton(
        FlxMeltanoStateManager,
        event_bus=eventing.event_bus,
    )

    # Meltano job manager for job lifecycle
    job_manager: providers.Singleton[FlxMeltanoJobManager] = providers.Singleton(
        FlxMeltanoJobManager,
        event_bus=eventing.event_bus,
    )

    # Meltano orchestrator for pipeline execution
    orchestrator: providers.Singleton[FlxMeltanoOrchestrator] = providers.Singleton(
        FlxMeltanoOrchestrator,
        project_manager=project_manager,
        state_manager=state_manager,
        event_bus=eventing.event_bus,
    )

    # Meltano engine (dynamically imported to avoid circular dependencies)
    meltano_engine: providers.Factory[MeltanoEngine] = providers.Factory(
        providers.Object("flx_core.engine.meltano_wrapper.MeltanoEngine"),
        project_root=config.meltano.project_root,
        event_bus=eventing.event_bus,
    )


class StateMachineContainer(containers.DeclarativeContainer):
    """State machine dependency injection container.

    Provides enterprise-grade state machines for pipeline execution
    and job lifecycle management with proper event integration.
    """

    # Configuration injection
    config = providers.Configuration()

    # Event dependencies
    eventing = providers.DependenciesContainer()

    # Pipeline execution state machine factory - ZERO TOLERANCE P0 FIX: requires execution_id parameter
    pipeline_execution_state_machine: providers.Factory[
        PipelineExecutionStateMachine
    ] = providers.Factory(
        PipelineExecutionStateMachine,
        # execution_id must be provided when creating instance
        event_bus=eventing.event_bus,
        timeout_seconds=config.meltano.pipeline_timeout_seconds,
    )

    # Job state machine factory
    job_state_machine: providers.Factory[JobStateMachine] = providers.Factory(
        JobStateMachine,
        event_bus=eventing.event_bus,
        heartbeat_interval=config.monitoring.heartbeat_interval_seconds,
    )


class ServiceContainer(containers.DeclarativeContainer):
    """Application service dependency injection container.

    Provides business logic services with proper dependency injection
    and integration with repositories, event bus, and external systems.
    """

    # Configuration injection
    config = providers.Configuration()

    # Dependency containers
    database = providers.DependenciesContainer()
    repositories = providers.DependenciesContainer()
    eventing = providers.DependenciesContainer()
    meltano = providers.DependenciesContainer()
    state_machines = providers.DependenciesContainer()

    # Core application services
    pipeline_service: providers.Factory[PipelineService] = providers.Factory(
        PipelineService,
        uow=database.unit_of_work,
        event_bus=eventing.event_bus,
        pipeline_repository=repositories.pipeline_repository,
        execution_repository=repositories.pipeline_execution_repository,
        state_machine_factory=state_machines.pipeline_execution_state_machine,
    )

    execution_service: providers.Factory[ExecutionService] = providers.Factory(
        ExecutionService,
        uow=database.unit_of_work,
        event_bus=eventing.event_bus,
        meltano_engine=meltano.meltano_engine,
        orchestrator=meltano.orchestrator,
        state_machine_factory=state_machines.pipeline_execution_state_machine,
    )

    plugin_service: providers.Factory[PluginService] = providers.Factory(
        PluginService,
        uow=database.unit_of_work,
        event_bus=eventing.event_bus,
        plugin_repository=repositories.plugin_repository,
        meltano_engine=meltano.meltano_engine,
    )

    scheduler_service: providers.Factory[SchedulerService] = providers.Factory(
        SchedulerService,
        uow=database.unit_of_work,
        event_bus=eventing.event_bus,
        pipeline_service=pipeline_service,
        execution_service=execution_service,
    )

    notification_service: providers.Factory[NotificationService] = providers.Factory(
        NotificationService,
        # No constructor dependencies - service uses get_config() internally
    )

    export_service: providers.Factory[ExportService] = providers.Factory(
        ExportService,
        session=database.session_factory.provided.call(),
    )

    # Analytics service for business intelligence and reporting
    analytics_service: providers.Factory[AnalyticsService] = providers.Factory(
        AnalyticsService,
        session=database.session_factory.provided.call(),
    )

    # Validation service for comprehensive parameter validation
    validation_service: providers.Factory[ValidationService] = providers.Factory(
        ValidationService,
        # Optional dependency - pass None if not available
        meltano_acl=None,
    )


class MonitoringContainer(containers.DeclarativeContainer):
    """Monitoring and observability dependency injection container.

    Manages health checkers, metrics collectors, and observability
    infrastructure with proper integration to the domain architecture.
    """

    # Configuration injection
    config = providers.Configuration()

    # Event dependencies
    eventing = providers.DependenciesContainer()

    # Health checker for system health monitoring
    health_checker: providers.Singleton[HealthChecker] = providers.Singleton(
        HealthChecker,
        config=config,
    )

    # Metrics collector for performance monitoring
    metrics_collector: providers.Singleton[MetricsCollector] = providers.Singleton(
        MetricsCollector,
        config=config,
        event_bus=eventing.event_bus,
    )


class ApplicationContainer(containers.DeclarativeContainer):
    """Main application dependency injection container.

    Orchestrates all domain containers and provides unified access
    to the complete dependency graph with proper lifecycle management.

    Features:
        - Hierarchical container composition
        - Environment-specific configuration
        - Resource lifecycle management
        - Type-safe dependency resolution
        - Integration with existing architecture
    """

    # Configuration management
    config: providers.Configuration = providers.Configuration()

    # Logging configuration
    logger: providers.Singleton[structlog.BoundLogger] = providers.Singleton(
        structlog.get_logger,
    )

    # Domain containers composition
    database: providers.Container[DatabaseContainer] = providers.Container(
        DatabaseContainer,
        config=config,
    )

    eventing: providers.Container[EventingContainer] = providers.Container(
        EventingContainer,
        config=config,
    )

    repositories: providers.Container[RepositoryContainer] = providers.Container(
        RepositoryContainer,
        database=database,
    )

    meltano: providers.Container[MeltanoContainer] = providers.Container(
        MeltanoContainer,
        config=config,
        eventing=eventing,
    )

    state_machines: providers.Container[StateMachineContainer] = providers.Container(
        StateMachineContainer,
        config=config,
        eventing=eventing,
    )

    monitoring: providers.Container[MonitoringContainer] = providers.Container(
        MonitoringContainer,
        config=config,
        eventing=eventing,
    )

    services: providers.Container[ServiceContainer] = providers.Container(
        ServiceContainer,
        config=config,
        database=database,
        repositories=repositories,
        eventing=eventing,
        meltano=meltano,
        state_machines=state_machines,
    )

    # Application lifecycle management
    def wire_dependencies(self: ApplicationContainer, *modules: str) -> None:
        """Wire dependencies for automatic injection in specified modules.

        Args:
        ----
            *modules: Module names to wire for dependency injection

        Example:
        -------
            container.wire_dependencies(
                "flx_api.dependencies",
                "flx_core.daemon.daemon",
                "flx_web.apps.dashboard.views",
            )

        """
        from flx_core.utils.import_fallback_patterns import OptionalDependency

        # ZERO TOLERANCE CONSOLIDATION: Use centralized dependency wiring management
        OptionalDependency("Dependency Injection Wiring", fallback_value=None)

        try:
            self.wire(modules=modules)
            logger.info(
                "Dependencies wired successfully",
                modules=modules,
                component="container.wiring",
            )
        except (RuntimeError, ValueError, AttributeError, TypeError) as e:
            logger.exception(
                "Failed to wire dependencies",
                error=str(e),
                modules=modules,
                component="container.wiring",
            )
            raise

    def shutdown_resources(self: ApplicationContainer) -> None:
        """Gracefully shutdown all managed resources.

        Ensures proper cleanup of database connections, event handlers,
        and other managed resources during application shutdown.
        """
        try:
            # Shutdown database connections
            try:
                shutdown_method = self.database.shutdown
                if callable(shutdown_method):
                    shutdown_method()
            except AttributeError:
                pass

            # Shutdown event system
            try:
                shutdown_method = self.eventing.shutdown
                if callable(shutdown_method):
                    shutdown_method()
            except AttributeError:
                pass

            # Shutdown monitoring services
            try:
                shutdown_method = self.monitoring.shutdown
                if callable(shutdown_method):
                    shutdown_method()
            except AttributeError:
                pass

            # Shutdown Meltano resources
            try:
                shutdown_method = self.meltano.shutdown
                if callable(shutdown_method):
                    shutdown_method()
            except AttributeError:
                pass

            logger.info(
                "Application resources shutdown successfully",
                component="container.shutdown",
            )
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.exception(
                "Error during resource shutdown",
                error=str(e),
                component="container.shutdown",
            )
            raise


# Configuration factories for different environments
def create_development_container() -> ApplicationContainer:
    """Create dependency injection container for development environment.

    Returns
    -------
        ApplicationContainer: Configured container for development

    """
    container = ApplicationContainer()
    config = get_config()
    container.config.from_pydantic(config)

    # Development-specific overrides
    container.config.debug.override(True)
    container.config.monitoring.metrics_enabled.override(True)

    logger.info("Development dependency injection container created")
    return container


def create_production_container() -> ApplicationContainer:
    """Create dependency injection container for production environment.

    Returns
    -------
        ApplicationContainer: Configured container for production

    """
    container = ApplicationContainer()
    config = get_config()
    container.config.from_pydantic(config)

    # Production-specific overrides
    container.config.debug.override(False)
    container.config.monitoring.metrics_enabled.override(True)
    container.config.monitoring.profiling_enabled.override(False)

    logger.info("Production dependency injection container created")
    return container


def create_testing_container() -> ApplicationContainer:
    """Create dependency injection container for testing environment.

    Returns
    -------
        ApplicationContainer: Configured container for testing

    """
    container = ApplicationContainer()
    config = get_config()
    container.config.from_pydantic(config)

    # Testing-specific overrides
    container.config.database.url.override("sqlite+aiosqlite:///:memory:")
    container.config.monitoring.metrics_enabled.override(False)
    # ZERO TOLERANCE P0 FIX: Override the actual event bus provider instead of config
    container.eventing.event_bus.override(providers.Singleton(lambda: None))

    logger.info("Testing dependency injection container created")
    return container


# ZERO TOLERANCE - Modern Python 3.13 singleton pattern for DI container
@functools.lru_cache(maxsize=1)
def get_application_container() -> ApplicationContainer:
    """Get the global application container instance.

    Returns:
    -------
        ApplicationContainer: The global dependency injection container

    Note:
    ----
        Creates container based on current environment configuration.

    """
    config = get_config()

    if config.environment == "development":
        return create_development_container()
    if config.environment == "production":
        return create_production_container()
    if config.environment == "testing":
        return create_testing_container()
    # Default to development
    return create_development_container()


def reset_application_container() -> None:
    """Reset the global application container instance.

    Used primarily for testing to ensure clean container state.
    """
    # ZERO TOLERANCE: Use functools.lru_cache.cache_clear() instead of global statement
    get_application_container.cache_clear()
    logger.info("Application container reset")


# Export container classes and factory functions
__all__ = [
    "ApplicationContainer",
    "DatabaseContainer",
    "EventingContainer",
    "MeltanoContainer",
    "MonitoringContainer",
    "RepositoryContainer",
    "ServiceContainer",
    "StateMachineContainer",
    "create_development_container",
    "create_production_container",
    "create_testing_container",
    "get_application_container",
    "reset_application_container",
]
