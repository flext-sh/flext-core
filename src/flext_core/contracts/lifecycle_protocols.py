"""Lifecycle protocols for FLEXT Core."""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ResultProtocol(Protocol):
    """Protocol for result objects."""

    def is_success(self) -> bool:
        """Check if result is successful."""
        ...

    def is_failure(self) -> bool:
        """Check if result is a failure."""
        ...

    def get_value(self) -> object:
        """Get the result value."""
        ...

    def get_error(self) -> object:
        """Get the error if any."""
        ...


@runtime_checkable
class LifecycleProtocol(Protocol):
    """Protocol for lifecycle management."""

    async def start(self) -> None:
        """Start the component."""
        ...

    async def stop(self) -> None:
        """Stop the component."""
        ...

    async def restart(self) -> None:
        """Restart the component."""
        ...

    def is_running(self) -> bool:
        """Check if component is running."""
        ...


@runtime_checkable
class HealthCheckProtocol(Protocol):
    """Protocol for health checks."""

    async def check_health(self) -> dict[str, Any]:
        """Perform health check."""
        ...

    def get_status(self) -> str:
        """Get current status."""
        ...


class AsyncContextManagerProtocol(Protocol):
    """Protocol for async context managers."""

    async def __aenter__(self) -> object:
        """Enter async context."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool | None:
        """Exit async context."""
        ...


class ServiceLifecycle(ABC):
    """Abstract base class for service lifecycle management."""

    def __init__(self) -> None:
        """Initialize service lifecycle management."""
        self._running = False

    @abstractmethod
    async def _do_start(self) -> None:
        """Implementation-specific start logic."""

    @abstractmethod
    async def _do_stop(self) -> None:
        """Implementation-specific stop logic."""

    async def start(self) -> None:
        """Start the service."""
        if not self._running:
            await self._do_start()
            self._running = True

    async def stop(self) -> None:
        """Stop the service."""
        if self._running:
            await self._do_stop()
            self._running = False

    async def restart(self) -> None:
        """Restart the service."""
        await self.stop()
        await self.start()

    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running


def is_initializable(obj: object) -> bool:
    """Check if object has initialization methods."""
    return (
        hasattr(obj, "start")
        or hasattr(obj, "initialize")
        or hasattr(obj, "__aenter__")
    )


def is_shutdownable(obj: object) -> bool:
    """Check if object has shutdown methods."""
    return hasattr(obj, "stop") or hasattr(obj, "shutdown") or hasattr(obj, "__aexit__")
