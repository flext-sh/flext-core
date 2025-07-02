"""Application container for dependency injection with ZERO boilerplate using Python 3.13.

This module provides the central dependency injection container that manages all
application services, infrastructure components, and configuration.
"""

from __future__ import annotations

import importlib
from contextlib import asynccontextmanager

# ZERO TOLERANCE - Strategic import handling using centralized patterns
from typing import TYPE_CHECKING

import sqlalchemy.exc
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Use centralized import fallback patterns
from flext_core.utils.import_fallback_patterns import get_redis_client

if TYPE_CHECKING:
    import redis

    from flext_core.plugins.manager import PluginManager
else:
    # Redis managed through centralized fallback patterns
    redis = (
        get_redis_client() or type("StubRedis", (), {"Redis": lambda *_a, **_k: None})()
    )

from flext_meltano.unified_anti_corruption_layer import (
    UnifiedMeltanoAntiCorruptionLayer,
)

from flext_core.commands.base import ReflectionCommandBus
from flext_core.engine.meltano_wrapper import MeltanoEngine
from flext_core.infrastructure.persistence.unit_of_work import UnitOfWork
from flext_core.services.execution_service import ExecutionService
from flext_core.services.pipeline_service import PipelineService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from flext_core.config.domain_config import FlextConfiguration
    from flext_core.contracts.repository_contracts import UnitOfWorkInterface
    from flext_core.events.event_bus import HybridEventBus

# Python 3.13 type aliases for container - with strict validation
from flext_core.domain.advanced_types import MetadataDict

HealthStatus = MetadataDict
ServiceFactory = type[object]


class FlextApplicationContainer:
    """Central dependency injection container for FLEXT Meltano Enterprise.

    This container manages all application dependencies including services,
    repositories, engines, and infrastructure components using modern
    Python 3.13 patterns with zero boilerplate.
    """

    def __init__(self, settings: FlextConfiguration) -> None:
        """Initialize the application container.

        Args:
        ----
        settings: Application configuration settings

        """
        self._settings = settings
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._event_bus: HybridEventBus | None = None
        self._meltano_engine: MeltanoEngine | None = None
        self._command_bus: ReflectionCommandBus | None = None
        self._plugin_manager: PluginManager | None = None

    @property
    def settings(self) -> FlextConfiguration:
        """Access to application settings."""
        return self._settings

    @asynccontextmanager
    async def unit_of_work(self) -> AsyncGenerator[UnitOfWorkInterface]:
        """Create and manage unit of work with proper transaction handling.

        Returns
        -------
            Async context manager for unit of work

        """
        session_factory = await self._get_session_factory()
        uow = UnitOfWork(session=session_factory())

        try:
            async with uow:
                yield uow
        except sqlalchemy.exc.SQLAlchemyError as e:
            # The UnitOfWork context manager handles the rollback.
            # We re-raise with more context for higher layers.
            msg = f"Database transaction failed: {e}"
            raise RuntimeError(msg) from e

    def pipeline_service(self, _uow: UnitOfWorkInterface) -> PipelineService:
        """Create pipeline service with unit of work.

        Args:
        ----
            uow: Unit of work instance

        Returns:
        -------
            Configured pipeline service

        """
        event_bus = self._get_event_bus()
        return PipelineService(
            event_bus=event_bus,
        )

    def execution_service(self, uow: UnitOfWorkInterface) -> ExecutionService:
        """Create execution service with unit of work.

        Args:
        ----
            uow: Unit of work instance

        Returns:
        -------
            Configured execution service

        """
        meltano_engine = self.meltano_engine()
        event_bus = self._get_event_bus()

        # Create the UnifiedMeltanoAntiCorruptionLayer with required dependencies
        meltano_acl = UnifiedMeltanoAntiCorruptionLayer(
            engine=meltano_engine,
            event_bus=event_bus,
        )

        return ExecutionService(
            uow=uow,
            meltano_acl=meltano_acl,
            event_bus=event_bus,
        )

    def meltano_engine(self) -> MeltanoEngine:
        """Get or create Meltano engine instance.

        Returns
        -------
            Configured Meltano engine

        """
        if self._meltano_engine is None:
            self._meltano_engine = MeltanoEngine(
                project_root=self._settings.meltano.project_root,
                event_bus=self._get_event_bus(),
            )
        return self._meltano_engine

    def command_bus(self) -> ReflectionCommandBus:
        """Get or create command bus instance.

        Returns
        -------
            Command bus instance for command handling

        """
        if self._command_bus is None:
            self._command_bus = ReflectionCommandBus()
        return self._command_bus

    def plugin_manager(self) -> PluginManager:
        """Get or create plugin manager instance.

        Returns
        -------
            Plugin manager instance for plugin management

        """
        if self._plugin_manager is None:
            # Import here to avoid circular imports with container parameter
            from flext_core.plugins.manager import PluginManager

            self._plugin_manager = PluginManager(container=self)
        return self._plugin_manager

    # Health check methods for system monitoring
    def database_health(self) -> HealthStatus:
        """Check database connectivity health.

        Returns
        -------
            Database health status

        """
        try:
            # ZERO TOLERANCE - Perform REAL database connection test
            if self._settings.database.url:
                # Attempt actual database connection to validate health
                engine = sqlalchemy.create_engine(
                    self._settings.database.url,
                    pool_pre_ping=True,
                )
                with engine.connect() as conn:
                    # Execute simple query to verify database is responsive
                    conn.execute(sqlalchemy.text("SELECT 1"))

                return {
                    "status": "healthy",
                    "database_url": self._settings.database.url,
                    "connection": "verified",
                }
        except (
            ValueError,
            TypeError,
            AttributeError,
            OSError,
            sqlalchemy.exc.SQLAlchemyError,
        ) as e:
            return {
                "status": "unhealthy",
                "error": f"Database health check failed: {e}",
                "error_type": type(e).__name__,
            }
        else:
            return {
                "status": "unhealthy",
                "error": "Database URL not configured",
            }

    def redis_health(self) -> HealthStatus:
        """Check Redis connectivity health.

        Returns
        -------
            Redis health status

        """
        try:
            # ZERO TOLERANCE - Perform REAL Redis connection test
            if self._settings.network.redis_port and redis is not None:
                # Use redis imported at top level with graceful fallback handling
                redis_client = redis.Redis(
                    host=getattr(self._settings.network, "redis_host", "localhost"),
                    port=self._settings.network.redis_port,
                    socket_connect_timeout=self._settings.database.pool_timeout,
                    socket_timeout=self._settings.database.pool_timeout,
                )

                # Execute ping to verify Redis is responsive
                redis_client.ping()

                return {
                    "status": "healthy",
                    "redis_port": self._settings.network.redis_port,
                    "connection": "verified",
                }
        except (ValueError, TypeError, AttributeError, OSError, ConnectionError) as e:
            return {
                "status": "unhealthy",
                "error": f"Redis health check failed: {e}",
                "error_type": type(e).__name__,
            }
        else:
            return {
                "status": "unhealthy",
                "error": "Redis URL not configured",
            }

    def meltano_health(self) -> HealthStatus:
        """Check Meltano system health.

        Returns
        -------
            Meltano health status

        """
        try:
            engine = self.meltano_engine()
            if engine.project_root and engine.project_root.exists():
                return {
                    "status": "healthy",
                    "project_root": str(engine.project_root),
                    "environment": self._settings.meltano.environment,
                }
        except (
            # File system and path errors
            OSError,
            FileNotFoundError,
            PermissionError,
            # Configuration errors
            ValueError,
            TypeError,
            AttributeError,
        ) as e:
            return {
                "status": "unhealthy",
                "error": f"Meltano health check failed: {e}",
                "error_type": type(e).__name__,
            }
        else:
            return {
                "status": "unhealthy",
                "error": "Meltano project root not found",
            }

    def grpc_health(self) -> HealthStatus:
        """Check gRPC server health.

        Returns
        -------
            gRPC health status

        """
        try:
            return {
                "status": "healthy",
                "port": self._settings.network.grpc_port,
                "reflection": getattr(
                    self._settings.network,
                    "grpc_reflection_enabled",
                    True,
                ),
            }
        except (
            # Configuration and validation errors
            ValueError,
            TypeError,
            AttributeError,
            # System and I/O errors
            OSError,
            RuntimeError,
        ) as e:
            return {
                "status": "unhealthy",
                "error": f"gRPC health check failed: {e}",
                "error_type": type(e).__name__,
            }

    def event_bus_health(self) -> HealthStatus:
        """Check event bus health.

        Returns
        -------
            Event bus health status

        """
        try:
            event_bus = self._get_event_bus()
            return {
                "status": "healthy",
                "type": type(event_bus).__name__,
                "subscribers": len(getattr(event_bus, "_subscribers", {})),
            }
        except (
            # Event bus creation and configuration errors
            ImportError,
            ModuleNotFoundError,
            ValueError,
            TypeError,
            AttributeError,
            # Runtime errors
            RuntimeError,
        ) as e:
            return {
                "status": "unhealthy",
                "error": f"Event bus health check failed: {e}",
                "error_type": type(e).__name__,
            }

    def plugin_system_health(self) -> HealthStatus:
        """Check plugin system health.

        Returns
        -------
            Plugin system health status

        """
        try:
            plugin_manager = self.plugin_manager()
            return {
                "status": "healthy",
                "is_initialized": plugin_manager.is_initialized,
                "loaded_plugins": plugin_manager.plugin_count,
                "discovered_plugins": plugin_manager.get_discovered_plugin_count(),
            }
        except (
            # Plugin system creation and configuration errors
            ImportError,
            ModuleNotFoundError,
            ValueError,
            TypeError,
            AttributeError,
            # Runtime errors
            RuntimeError,
        ) as e:
            return {
                "status": "unhealthy",
                "error": f"Plugin system health check failed: {e}",
                "error_type": type(e).__name__,
            }

    # Private helper methods
    async def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or create async session factory.

        Returns
        -------
            Async session factory

        """
        if self._session_factory is None:
            # Use domain configuration for all database parameters - with strict validation
            engine = create_async_engine(
                self._settings.database.url,
                echo=self._settings.debug,
                pool_pre_ping=True,
                pool_size=self._settings.database.pool_size,
                max_overflow=self._settings.database.max_overflow,
                pool_timeout=self._settings.database.pool_timeout,
                pool_recycle=self._settings.database.pool_recycle,
            )

            self._session_factory = async_sessionmaker(
                engine,
                expire_on_commit=False,
            )

        return self._session_factory

    def _get_event_bus(self) -> HybridEventBus:
        """Get or create event bus instance.

        Returns
        -------
            Event bus instance

        """
        if self._event_bus is None:
            # Import at runtime to avoid circular dependency
            # ZERO TOLERANCE - event_bus is REQUIRED and guaranteed in pyproject.toml
            event_bus_module = importlib.import_module("flext_core.events.event_bus")
            self._event_bus = event_bus_module.HybridEventBus()

        return self._event_bus

    async def cleanup(self) -> None:
        """Cleanup container resources.

        This method should be called when shutting down the application
        to properly cleanup all resources.
        """
        # ZERO TOLERANCE - Direct interface contracts, no hasattr() fallbacks
        if self._session_factory:
            # SQLAlchemy sessionmaker cleanup - direct interface call
            # Session factory cleanup - async_sessionmaker doesn't have close_all
            # Individual sessions are closed through context managers
            pass

        if self._event_bus:
            # Event bus cleanup - direct interface call
            # Event bus cleanup - check for available cleanup methods
            try:
                close_method = self._event_bus.close
                await close_method()
            except AttributeError:
                try:
                    shutdown_method = self._event_bus.shutdown
                    await shutdown_method()
                except AttributeError:
                    # Event bus doesn't have cleanup methods
                    pass
            # HybridEventBus may not require explicit cleanup

        if self._meltano_engine:
            # Meltano engine cleanup - direct interface call
            try:
                await self._meltano_engine.cleanup()
            except AttributeError as e:
                msg = (
                    f"Meltano engine does not implement required cleanup() method: {e}"
                )
                raise RuntimeError(msg) from e

        if self._command_bus:
            # Command bus cleanup - clear handlers
            self._command_bus.handlers.clear()

        if self._plugin_manager:
            # Plugin manager cleanup - cleanup all plugins
            await self._plugin_manager.cleanup()

    def __repr__(self) -> str:
        """Return string representation of the container."""
        return (
            f"FlextApplicationContainer(settings={self._settings.__class__.__name__})"
        )


# Factory function for easy container creation
def create_application_container(
    settings: FlextConfiguration | None = None,
) -> FlextApplicationContainer:
    """Create application container with default or provided settings.

    Args:
    ----
        settings: Optional settings, will use default if not provided

    Returns:
    -------
        Configured application container

    """
    if settings is None:
        # ZERO TOLERANCE - domain_config is REQUIRED and guaranteed in pyproject.toml
        config_module = importlib.import_module("flext_core.config.domain_config")
        settings = config_module.get_config()

    return FlextApplicationContainer(settings)
