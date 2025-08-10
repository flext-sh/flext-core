"""Base command abstractions following SOLID and CQRS principles.

This module provides abstract base classes for CQRS command patterns used across
the FLEXT ecosystem. Concrete implementations are in commands.py.

Classes:
    FlextAbstractCommand: Base class for all commands.
    FlextAbstractCommandHandler: Abstract command handler.
    FlextAbstractCommandBus: Abstract command bus.
    FlextAbstractQueryHandler: Abstract query handler.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from flext_core.result import FlextResult

# Type variables for CQRS patterns
TCommand = TypeVar("TCommand")
TQuery = TypeVar("TQuery")
TResult = TypeVar("TResult")
TResponse = TypeVar("TResponse")

# =============================================================================
# ABSTRACT COMMAND BASE
# =============================================================================


class FlextAbstractCommand(ABC):
    """Abstract base class for all FLEXT commands following CQRS principles.

    Concrete command implementations should expose at least the attributes
    `command_id`, `command_type`, and `correlation_id`, but this base class
    does not enforce them as abstract properties to avoid conflicts with
    Pydantic model fields.
    """

    @abstractmethod
    def validate_command(self) -> FlextResult[None]:
        """Validate command - must be implemented by subclasses."""
        ...


# =============================================================================
# ABSTRACT COMMAND HANDLER
# =============================================================================


class FlextAbstractCommandHandler[TCommand, TResult](ABC):
    """Abstract command handler for CQRS command processing.

    Provides foundation for implementing command handlers with proper
    command handling patterns and error management.
    """

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Get handler name - must be implemented by subclasses."""
        ...

    @abstractmethod
    def can_handle(self, command: TCommand) -> bool:
        """Check if handler can handle command - must be implemented by subclasses."""
        ...

    @abstractmethod
    def validate_command(self, command: TCommand) -> FlextResult[None]:
        """Validate command - must be implemented by subclasses."""
        ...

    @abstractmethod
    def handle(self, command: TCommand) -> FlextResult[TResult]:
        """Handle command - must be implemented by subclasses."""
        ...


# =============================================================================
# ABSTRACT QUERY HANDLER
# =============================================================================


class FlextAbstractQueryHandler[TQuery, TResponse](ABC):
    """Abstract query handler for CQRS query processing.

    Provides foundation for implementing query handlers with proper
    query handling patterns and response management.
    """

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Get handler name - must be implemented by subclasses."""
        ...

    @abstractmethod
    def can_handle(self, query: TQuery) -> bool:
        """Check if handler can handle query - must be implemented by subclasses."""
        ...

    @abstractmethod
    def validate_query(self, query: TQuery) -> FlextResult[None]:
        """Validate query - must be implemented by subclasses."""
        ...

    @abstractmethod
    def handle(self, query: TQuery) -> FlextResult[TResponse]:
        """Handle query - must be implemented by subclasses."""
        ...


# =============================================================================
# ABSTRACT COMMAND BUS
# =============================================================================


class FlextAbstractCommandBus(ABC):
    """Abstract command bus for CQRS command routing.

    Provides foundation for implementing command buses with proper
    command routing and handler management.
    """

    @abstractmethod
    def register_handler(
        self,
        command_type: str,
        handler: FlextAbstractCommandHandler[object, object],
    ) -> None:
        """Register command handler - must be implemented by subclasses."""
        ...

    @abstractmethod
    def unregister_handler(self, command_type: str) -> bool:
        """Unregister command handler - must be implemented by subclasses."""
        ...

    @abstractmethod
    def send_command(self, command: FlextAbstractCommand) -> FlextResult[object]:
        """Send command - must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_registered_handlers(self) -> dict[str, object]:
        """Get registered handlers - must be implemented by subclasses."""
        ...


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextAbstractCommand",
    "FlextAbstractCommandBus",
    "FlextAbstractCommandHandler",
    "FlextAbstractQueryHandler",
]
