"""Protocol Definitions - Structural Type Interfaces.

Defines protocols (structural typing) for common interfaces
that components can implement without inheritance.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dictionaries."""

    def to_dict(self) -> dict[str, Any]:
        """Convert object to dictionary representation."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Serializable:
        """Create object from dictionary representation."""


@runtime_checkable
class Validatable(Protocol):
    """Protocol for objects that can validate themselves."""

    def validate(self) -> list[str]:
        """Validate object and return list of error messages.

        Returns:
            Empty list if valid, list of error messages if invalid.

        """

    def is_valid(self) -> bool:
        """Check if object is valid."""


@runtime_checkable
class EventBus(Protocol):
    """Protocol for event bus implementations."""

    async def publish(self, event: Any) -> None:
        """Publish an event to the bus."""

    async def subscribe(self, event_type: type[Any], handler: Any) -> None:
        """Subscribe a handler to a specific event type."""

    async def unsubscribe(self, event_type: type[Any], handler: Any) -> None:
        """Unsubscribe a handler from a specific event type."""
