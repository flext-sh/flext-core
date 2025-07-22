"""Infrastructure layer protocols for Dependency Inversion Principle compliance.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

These protocols define abstractions that the domain layer can depend on,
ensuring infrastructure depends on domain, not the reverse.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

# Type variables for generic protocols
T = TypeVar("T")
ID = TypeVar("ID")
TEvent = TypeVar("TEvent")


# ==============================================================================
# DEPENDENCY INVERSION PRINCIPLE - INFRASTRUCTURE ABSTRACTIONS
# ==============================================================================


@runtime_checkable
class PersistenceProtocol[T, ID](Protocol):
    """Protocol for persistence operations - infrastructure abstraction."""

    async def save(self, entity: T) -> T:
        """Save entity to persistence layer."""
        ...

    async def find_by_id(self, entity_id: ID) -> T | None:
        """Find entity by ID from persistence layer."""
        ...

    async def delete(self, entity_id: ID) -> bool:
        """Delete entity from persistence layer."""
        ...

    async def find_all(self) -> list[T]:
        """Find all entities in persistence layer."""
        ...

    async def count(self) -> int:
        """Count entities in persistence layer."""
        ...


@runtime_checkable
class EventPublishingProtocol[TEvent](Protocol):
    """Protocol for event publishing - infrastructure abstraction."""

    async def publish(self, event: TEvent) -> None:
        """Publish domain event to infrastructure."""
        ...

    async def publish_batch(self, events: list[TEvent]) -> None:
        """Publish multiple events as a batch."""
        ...


@runtime_checkable
class CacheProtocol[T](Protocol):
    """Protocol for caching operations - infrastructure abstraction."""

    async def get(self, key: str) -> T | None:
        """Get cached value by key."""
        ...

    async def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """Set cached value with optional TTL."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete cached value by key."""
        ...

    async def clear(self) -> None:
        """Clear all cached values."""
        ...


@runtime_checkable
class LoggingProtocol(Protocol):
    """Protocol for logging operations - infrastructure abstraction."""

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log debug message with formatting support."""
        ...

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log info message with formatting support."""
        ...

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log warning message with formatting support."""
        ...

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log error message with formatting support."""
        ...

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log exception with traceback and formatting support."""
        ...


@runtime_checkable
class ConnectionProtocol(Protocol):
    """Protocol for external connections - infrastructure abstraction."""

    async def connect(self) -> None:
        """Establish connection to external system."""
        ...

    async def disconnect(self) -> None:
        """Close connection to external system."""
        ...

    def is_connected(self) -> bool:
        """Check if connection is active."""
        ...

    async def ping(self) -> bool:
        """Test connection health."""
        ...


@runtime_checkable
class SerializationProtocol[T](Protocol):
    """Protocol for serialization operations - infrastructure abstraction."""

    def serialize(self, obj: T) -> str | bytes:
        """Serialize object to string or bytes."""
        ...

    def deserialize(self, data: str | bytes) -> T:
        """Deserialize string or bytes to object."""
        ...


@runtime_checkable
class ConfigurationProviderProtocol(Protocol):
    """Protocol for configuration providers - infrastructure abstraction."""

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        ...

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...

    def has_config(self, key: str) -> bool:
        """Check if configuration key exists."""
        ...


# ==============================================================================
# ABSTRACT BASE CLASSES FOR COMMON INFRASTRUCTURE PATTERNS
# ==============================================================================


class BaseInfrastructureService(ABC):
    """Base abstract class for infrastructure services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the infrastructure service."""
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the service."""
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        ...


class BaseAsyncContextManager[T](ABC):
    """Base abstract class for async context managers."""

    @abstractmethod
    async def __aenter__(self) -> T:
        """Enter async context manager."""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool | None:
        """Exit async context manager."""
        ...


__all__ = [
    "BaseAsyncContextManager",
    # Abstract base classes
    "BaseInfrastructureService",
    "CacheProtocol",
    "ConfigurationProviderProtocol",
    "ConnectionProtocol",
    "EventPublishingProtocol",
    "LoggingProtocol",
    # Protocols
    "PersistenceProtocol",
    "SerializationProtocol",
]
