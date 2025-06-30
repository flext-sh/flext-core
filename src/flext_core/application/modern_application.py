"""Modern enterprise application with dependency-injector integration.

This module implements a next-generation application architecture using
dependency-injector for comprehensive dependency management while maintaining
compatibility with the existing lato framework integration.

ARCHITECTURAL PRINCIPLES:
- Dependency injection over manual instantiation
- Declarative service configuration
- Environment-aware polymorphic configuration
- Resource lifecycle management
- Type-safe dependency resolution
- Integration with existing domain architecture

PERFORMANCE BENEFITS:
- Lazy loading of expensive resources
- Automatic connection pooling and resource management
- Reduced startup time through deferred initialization
- Memory-efficient service instantiation
- Proper resource cleanup and disposal
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from dependency_injector.wiring import Provide
from dependency_injector.wiring import inject as di_inject
from flext_observability.monitoring.rich_error_handler import error_context
from lato import Application, DependencyProvider

from flext_core.config.domain_config import FlextConfiguration, get_config
from flext_core.infrastructure.containers import (
    ApplicationContainer,
    get_application_container,
    reset_application_container,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from flext_observability.health import HealthChecker
    from flext_observability.metrics import MetricsCollector

    from flext_core.engine.meltano_wrapper import MeltanoEngine
    from flext_core.events.event_bus import HybridEventBus
    from flext_core.infrastructure.persistence.unit_of_work import UnitOfWork
    from flext_core.services.execution_service import ExecutionService
    from flext_core.services.pipeline_service import PipelineService
    from flext_core.services.plugin_service import PluginService
    from flext_core.services.scheduler_service import SchedulerService

logger = structlog.get_logger()

# Python 3.13 type aliases for modern application architecture
type ApplicationConfig = FlextConfiguration
type ServiceFactory = object  # Service factory callable type
type ResourceManager = object  # Resource management interface


class ModernFlxApplication(Application):
    """Modern enterprise application integrating lato with dependency-injector.

    This application class provides professional integration between lato Application
    patterns and dependency-injector container management, following enterprise
    architectural standards.

    Features:
        - lato Application for DDD command/query/event handling
        - dependency-injector for professional service resolution
        - Type-safe dependency injection throughout
        - Domain handlers with DI integration
        - Enterprise monitoring and observability
        - Professional resource lifecycle management
    """

    def __init__(
        self,
        *,
        name: str = "flext-enterprise-platform",
        config: ApplicationConfig | None = None,
        container: ApplicationContainer | None = None,
        dependency_provider: DependencyProvider | None = None,
    ) -> None:
        """Initialize modern FLEXT application with dependency injection.

        Args:
        ----
            name: Application name for identification
            config: Application configuration (uses global config if None)
            container: Dependency injection container (creates if None)
            dependency_provider: Lato dependency provider (optional)

        """
        super().__init__(name=name, dependency_provider=dependency_provider)

        # Configuration management
        self._config = config or get_config()

        # Dependency injection container
        self._container = container or get_application_container()
        self._container.config.from_pydantic(self._config)

        # Application state
        self._is_initialized = False
        self._startup_tasks: list[asyncio.Task[None]] = []

        self.logger = logger.bind(
            component="modern_application",
            name=name,
            environment=self._config.environment,
        )

        self.logger.info(
            "Modern FLEXT application initialized",
            config_environment=self._config.environment,
            debug_mode=self._config.debug,
        )

    @property
    def config(self) -> ApplicationConfig:
        """Get application configuration."""
        return self._config

    @property
    def container(self) -> ApplicationContainer:
        """Get dependency injection container."""
        return self._container

    @property
    def is_initialized(self) -> bool:
        """Check if application is fully initialized."""
        return self._is_initialized

    # Dependency-injected service accessors

    @di_inject
    def get_event_bus(
        self,
        event_bus: HybridEventBus = Provide[ApplicationContainer.eventing.event_bus],
    ) -> HybridEventBus:
        """Get the application event bus with dependency injection."""
        return event_bus

    @di_inject
    def get_unit_of_work(
        self, uow: UnitOfWork = Provide[ApplicationContainer.database.unit_of_work],
    ) -> UnitOfWork:
        """Get unit of work instance with dependency injection."""
        return uow

    @di_inject
    def get_meltano_engine(
        self,
        engine: MeltanoEngine = Provide[ApplicationContainer.meltano.meltano_engine],
    ) -> MeltanoEngine:
        """Get Meltano engine with dependency injection."""
        return engine

    @di_inject
    def get_health_checker(
        self,
        health_checker: HealthChecker = Provide[
            ApplicationContainer.monitoring.health_checker
        ],
    ) -> HealthChecker:
        """Get health checker with dependency injection."""
        return health_checker

    @di_inject
    def get_metrics_collector(
        self,
        metrics: MetricsCollector = Provide[
            ApplicationContainer.monitoring.metrics_collector
        ],
    ) -> MetricsCollector:
        """Get metrics collector with dependency injection."""
        return metrics

    # Application service accessors

    @di_inject
    def get_pipeline_service(
        self,
        service: PipelineService = Provide[
            ApplicationContainer.services.pipeline_service
        ],
    ) -> PipelineService:
        """Get pipeline service with dependency injection."""
        return service

    @di_inject
    def get_execution_service(
        self,
        service: ExecutionService = Provide[
            ApplicationContainer.services.execution_service
        ],
    ) -> ExecutionService:
        """Get execution service with dependency injection."""
        return service

    @di_inject
    def get_plugin_service(
        self,
        service: PluginService = Provide[ApplicationContainer.services.plugin_service],
    ) -> PluginService:
        """Get plugin service with dependency injection."""
        return service

    @di_inject
    def get_scheduler_service(
        self,
        service: SchedulerService = Provide[
            ApplicationContainer.services.scheduler_service
        ],
    ) -> SchedulerService:
        """Get scheduler service with dependency injection."""
        return service

    # Application lifecycle management

    async def initialize(self) -> None:
        """Initialize application with all dependencies.

        Performs comprehensive application initialization including:
        - Dependency injection container setup
        - Database connection initialization
        - Event system startup
        - Monitoring and health check setup
        - Service registration and configuration
        """
        if self._is_initialized:
            self.logger.warning("Application already initialized")
            return

        try:
            self.logger.info("Starting application initialization")

            # Wire dependency injection for relevant modules
            self._container.wire_dependencies(
                "flext_api.dependencies",
                "flext_core.daemon.daemon",
                "flext_web.apps.dashboard.views",
                "flext_core.application.modern_application",
            )

            # Initialize core services
            await self._initialize_core_services()

            # Start monitoring and health checks
            await self._initialize_monitoring()

            # Initialize Meltano integration
            await self._initialize_meltano()

            # Setup event subscriptions
            await self._setup_event_subscriptions()

            self._is_initialized = True

            self.logger.info(
                "Application initialization completed successfully",
                services_initialized=len(self._startup_tasks),
            )

        except (
            RuntimeError,
            ValueError,
            TypeError,
            ImportError,
            OSError,
            ConnectionError,
        ):
            with error_context(
                "Application initialization failed",
                application_name=self.name,
                environment=self._config.environment,
                re_raise=False,
            ):
                pass  # Error already logged and formatted by context

            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown application and cleanup resources.

        Performs orderly shutdown including:
        - Service cleanup and resource disposal
        - Database connection closure
        - Event system shutdown
        - Monitoring cleanup
        - Container resource management
        """
        try:
            self.logger.info("Starting application shutdown")

            # Cancel all startup tasks
            for task in self._startup_tasks:
                if not task.done():
                    task.cancel()

            # Wait for task cancellation
            if self._startup_tasks:
                await asyncio.gather(*self._startup_tasks, return_exceptions=True)

            # Shutdown container resources
            self._container.shutdown_resources()

            # Reset application state
            self._is_initialized = False
            self._startup_tasks.clear()

            self.logger.info("Application shutdown completed successfully")

        except (
            RuntimeError,
            ValueError,
            TypeError,
            ImportError,
            OSError,
            ConnectionError,
        ):
            with error_context(
                "Application shutdown failed",
                application_name=self.name,
                re_raise=False,
            ):
                pass  # Error already logged and formatted by context

            raise

    @asynccontextmanager
    async def lifespan(self) -> AsyncGenerator[None]:
        """Application lifespan context manager for FastAPI integration.

        Provides proper application lifecycle management for web frameworks
        with automatic initialization and cleanup.

        Yields:
        ------
            None: Application is ready for requests

        Example:
        -------
            app = ModernFlxApplication()

            @asynccontextmanager
            async def lifespan(fastapi_app: object):
                async with app.lifespan():
                    yield

        """
        await self.initialize()
        try:
            yield
        finally:
            await self.shutdown()

    # Private initialization methods

    async def _initialize_core_services(self) -> None:
        """Initialize core application services."""
        self.logger.debug("Initializing core services")

        # Initialize database connections
        database_task = asyncio.create_task(self._initialize_database())
        self._startup_tasks.append(database_task)

        # Initialize event system
        event_task = asyncio.create_task(self._initialize_event_system())
        self._startup_tasks.append(event_task)

        # Wait for core services
        await asyncio.gather(database_task, event_task)

        self.logger.debug("Core services initialized successfully")

    async def _initialize_database(self) -> None:
        """Initialize database connections and session factory."""
        try:
            # Database engine initialization is handled by dependency injection
            # Test connection by creating a session
            async with self.get_unit_of_work() as uow:
                await uow.test_connection()

            self.logger.debug("Database initialization completed")

        except (RuntimeError, ValueError, ImportError, ConnectionError, OSError) as e:
            self.logger.exception("Database initialization failed", error=str(e))
            raise

    async def _initialize_event_system(self) -> None:
        """Initialize event bus and message handling."""
        try:
            event_bus = self.get_event_bus()
            await event_bus.initialize()

            self.logger.debug("Event system initialization completed")

        except (RuntimeError, ValueError, ImportError, ConnectionError, OSError) as e:
            self.logger.exception("Event system initialization failed", error=str(e))
            raise

    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring and observability systems."""
        try:
            # Initialize health checker
            health_checker = self.get_health_checker()
            await health_checker.initialize()

            # Initialize metrics collector
            metrics_collector = self.get_metrics_collector()
            await metrics_collector.start_collection()

            self.logger.debug("Monitoring systems initialized successfully")

        except (RuntimeError, ValueError, ImportError, ConnectionError, OSError) as e:
            self.logger.exception("Monitoring initialization failed", error=str(e))
            raise

    async def _initialize_meltano(self) -> None:
        """Initialize Meltano engine and integration."""
        try:
            meltano_engine = self.get_meltano_engine()
            await meltano_engine.initialize()

            self.logger.debug("Meltano integration initialized successfully")

        except (RuntimeError, ValueError, ImportError, ConnectionError, OSError) as e:
            self.logger.exception("Meltano initialization failed", error=str(e))
            raise

    async def _setup_event_subscriptions(self) -> None:
        """Set up application-level event subscriptions."""
        try:
            event_bus = self.get_event_bus()

            # Subscribe to application lifecycle events
            event_bus.subscribe("application.startup", self._handle_startup_event)
            event_bus.subscribe("application.shutdown", self._handle_shutdown_event)

            # Subscribe to health monitoring events
            event_bus.subscribe("health.check.failed", self._handle_health_failure)

            self.logger.debug("Event subscriptions setup completed")

        except (RuntimeError, ValueError, ImportError, ConnectionError, OSError) as e:
            self.logger.exception("Event subscription setup failed", error=str(e))
            raise

    # Event handlers

    async def _handle_startup_event(self, event_data: object) -> None:
        """Handle application startup events."""
        self.logger.info("Application startup event received", event_data=event_data)

    async def _handle_shutdown_event(self, event_data: object) -> None:
        """Handle application shutdown events."""
        self.logger.info("Application shutdown event received", event_data=event_data)

    async def _handle_health_failure(self, event_data: object) -> None:
        """Handle health check failure events."""
        self.logger.warning("Health check failure detected", event_data=event_data)

    # Compatibility methods for existing code

    def get_hybrid_event_bus(self) -> HybridEventBus:
        """Get hybrid event bus for backward compatibility."""
        return self.get_event_bus()

    def get_settings(self) -> ApplicationConfig:
        """Get application settings for backward compatibility."""
        return self.config


# Factory functions for different environments


def create_application(
    name: str = "flext-enterprise-platform", config: ApplicationConfig | None = None,
) -> ModernFlxApplication:
    """Create a modern FLEXT application instance.

    Args:
    ----
        name: Application name for identification
        config: Application configuration (uses global config if None)

    Returns:
    -------
        ModernFlxApplication: Configured application instance

    """
    app_config = config or get_config()

    # Wire container dependencies for application modules
    container = get_application_container()

    app = ModernFlxApplication(
        name=name,
        config=app_config,
        container=container,
    )

    logger.info(
        "Modern FLEXT application created",
        name=name,
        environment=app_config.environment,
    )

    return app


def create_testing_application(
    name: str = "flext-test-application",
) -> ModernFlxApplication:
    """Create application instance configured for testing.

    Args:
    ----
        name: Application name for identification

    Returns:
    -------
        ModernFlxApplication: Testing-configured application instance

    """
    # Reset container to ensure clean state
    reset_application_container()

    # Create testing configuration
    config = get_config()
    config.environment = "testing"
    config.database.url = "sqlite+aiosqlite:///:memory:"

    app = create_application(name=name, config=config)

    logger.info("Testing application created", name=name)
    return app


# Export application classes and factory functions
__all__ = [
    "ApplicationConfig",
    "ModernFlxApplication",
    "create_application",
    "create_testing_application",
]
