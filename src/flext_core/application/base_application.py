"""Base Application class with common functionality."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import structlog

from flext_core.domain.advanced_types import ServiceError, ServiceResult
from flext_core.utils.import_fallback_patterns import OptionalDependency

if TYPE_CHECKING:
    from flext_observability.monitoring.system_monitor import SystemMonitor


class BaseApplicationMixin(ABC):
    """Mixin class providing common application functionality."""

    async def _initialize_monitoring(self) -> None:
        """Initialize application monitoring capabilities."""
        logger = structlog.get_logger(__name__)
        OptionalDependency("System Monitor", fallback_value=None)

        try:
            monitor = self.get_system_monitor()
            if monitor:
                await monitor.start_monitoring()
                monitor.register_health_check(
                    "application",
                    self._application_health_check,
                )
                logger.debug("System monitoring initialized successfully")
            else:
                logger.debug(
                    "System monitor not available - continuing without monitoring",
                )
        except (AttributeError, RuntimeError, OSError) as e:
            logger.warning(
                "Failed to initialize monitoring",
                error=str(e),
                component="application.monitoring",
            )

    async def _setup_event_subscriptions(self) -> None:
        """Set up application-level event subscriptions."""
        logger = structlog.get_logger(__name__)
        OptionalDependency("Event Bus", fallback_value=None)

        try:
            event_bus = self.get_event_bus()
            if not event_bus:
                logger.debug("Event bus not available - skipping event subscriptions")
                return

            # Subscribe to application lifecycle events
            event_bus.subscribe("application.startup", self._handle_startup_event)
            event_bus.subscribe(
                "application.shutdown",
                self._handle_shutdown_event,
            )
            event_bus.subscribe("application.error", self._handle_error_event)

            # Subscribe to system events
            event_bus.subscribe(
                "system.health.critical",
                self._handle_critical_health_event,
            )
            event_bus.subscribe(
                "system.resource.exhausted",
                self._handle_resource_exhausted_event,
            )

            logger.debug(
                "Event subscriptions configured successfully",
                subscriptions=5,
                component="application.events",
            )

        except (AttributeError, RuntimeError, OSError) as e:
            logger.warning(
                "Failed to setup event subscriptions",
                error=str(e),
                component="application.events",
            )

    async def _application_health_check(self) -> ServiceResult[bool]:
        """Perform application health check."""
        try:
            # Basic health checks
            try:
                container = self._container
                container_healthy = container is not None
            except AttributeError:
                container_healthy = False
            services_healthy = await self._check_services_health()

            if container_healthy and services_healthy:
                return ServiceResult.ok(True)

            return ServiceResult.fail(
                ServiceError(
                    code="HEALTH_CHECK_FAILED",
                    message="Application health check failed",
                ),
            )
        except Exception as e:
            return ServiceResult.fail(
                ServiceError(
                    code="HEALTH_CHECK_ERROR",
                    message=f"Health check error: {e}",
                ),
            )

    async def _check_services_health(self) -> bool:
        """Check if core services are healthy."""
        # Placeholder implementation - override in subclasses
        return True

    async def _handle_startup_event(self, event: object) -> None:
        """Handle application startup event."""

    async def _handle_shutdown_event(self, event: object) -> None:
        """Handle application shutdown event."""

    async def _handle_error_event(self, event: object) -> None:
        """Handle application error event."""

    async def _handle_critical_health_event(self, event: object) -> None:
        """Handle critical health event."""

    async def _handle_resource_exhausted_event(self, event: object) -> None:
        """Handle resource exhausted event."""

    # Abstract methods that must be implemented by subclasses
    def get_system_monitor(self) -> SystemMonitor | None:
        """Get system monitor instance."""
        return getattr(self, "_monitor", None)

    def get_event_bus(self) -> Any:
        """Get event bus instance."""
        return getattr(self, "_event_bus", None)
