"""Base handler abstractions following SOLID and CQRS principles.

This module provides abstract base classes for handler patterns used across
the FLEXT ecosystem. Concrete implementations are in handlers.py.

Classes:
    FlextAbstractHandler: Base class for all handlers.
    FlextAbstractValidatingHandler: Abstract validating handler.
    FlextAbstractHandlerChain: Abstract handler chain.
    FlextAbstractHandlerRegistry: Abstract handler registry.
    FlextAbstractMetricsHandler: Abstract metrics handler.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from flext_core.result import FlextResult

# Type variables for handler patterns
TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")
THandler = TypeVar("THandler")

# =============================================================================
# ABSTRACT HANDLER BASE
# =============================================================================


class FlextAbstractHandler[TRequest, TResponse](ABC):
    """Abstract base class for all FLEXT handlers following CQRS principles.

    Provides foundation for implementing handlers with proper separation
    of concerns and command/query responsibility segregation.
    """

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Get handler name - must be implemented by subclasses."""
        ...

    @abstractmethod
    def can_handle(self, request: TRequest) -> bool:
        """Check if handler can handle request - must be implemented by subclasses."""
        ...

    @abstractmethod
    def handle(self, request: TRequest) -> FlextResult[TResponse]:
        """Handle request - must be implemented by subclasses."""
        ...

    @abstractmethod
    def validate_request(self, request: TRequest) -> FlextResult[None]:
        """Validate request - must be implemented by subclasses."""
        ...


# =============================================================================
# ABSTRACT VALIDATING HANDLER
# =============================================================================


class FlextAbstractValidatingHandler[TRequest, TResponse](
    FlextAbstractHandler[TRequest, TResponse],
):
    """Abstract validating handler with validation logic.

    Extends base handler to provide validation capabilities
    for request processing with proper error handling.
    """

    @abstractmethod
    def validate_input(self, request: TRequest) -> FlextResult[None]:
        """Validate input - must be implemented by subclasses."""
        ...

    @abstractmethod
    def validate_output(self, response: TResponse) -> FlextResult[TResponse]:
        """Validate output - must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_validation_rules(self) -> list[object]:
        """Get validation rules - must be implemented by subclasses."""
        ...


# =============================================================================
# ABSTRACT HANDLER CHAIN
# =============================================================================


class FlextAbstractHandlerChain[TRequest, TResponse](ABC):
    """Abstract handler chain for chain of responsibility pattern.

    Provides foundation for implementing handler chains with proper
    request processing and handler coordination.
    """

    @abstractmethod
    def add_handler(self, handler: FlextAbstractHandler[TRequest, TResponse]) -> None:
        """Add handler to chain - must be implemented by subclasses."""
        ...

    @abstractmethod
    def remove_handler(
        self,
        handler: FlextAbstractHandler[TRequest, TResponse],
    ) -> bool:
        """Remove handler from chain - must be implemented by subclasses."""
        ...

    @abstractmethod
    def handle_request(self, request: TRequest) -> FlextResult[TResponse]:
        """Handle request through chain - must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_handlers(self) -> list[FlextAbstractHandler[TRequest, TResponse]]:
        """Get all handlers - must be implemented by subclasses."""
        ...


# =============================================================================
# ABSTRACT HANDLER REGISTRY
# =============================================================================


class FlextAbstractHandlerRegistry[THandler](ABC):
    """Abstract handler registry for handler management.

    Provides foundation for implementing handler registries with proper
    handler registration and lookup capabilities.
    """

    @abstractmethod
    def register_handler(self, name: str, handler: THandler) -> None:
        """Register handler - must be implemented by subclasses."""
        ...

    @abstractmethod
    def unregister_handler(self, name: str) -> bool:
        """Unregister handler - must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_handler(self, name: str) -> FlextResult[THandler]:
        """Get handler by name - must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_all_handlers(self) -> dict[str, THandler]:
        """Get all handlers - must be implemented by subclasses."""
        ...

    @abstractmethod
    def clear_handlers(self) -> None:
        """Clear all handlers - must be implemented by subclasses."""
        ...


# =============================================================================
# ABSTRACT METRICS HANDLER
# =============================================================================


class FlextAbstractMetricsHandler[TRequest, TResponse](
    FlextAbstractHandler[TRequest, TResponse],
):
    """Abstract metrics handler with metrics collection.

    Extends base handler to provide metrics collection capabilities
    for performance monitoring and observability.
    """

    @abstractmethod
    def start_metrics(self, request: TRequest) -> None:
        """Start metrics collection - must be implemented by subclasses."""
        ...

    @abstractmethod
    def stop_metrics(self, request: TRequest, response: TResponse) -> None:
        """Stop metrics collection - must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, object]:
        """Get collected metrics - must be implemented by subclasses."""
        ...

    @abstractmethod
    def clear_metrics(self) -> None:
        """Clear metrics - must be implemented by subclasses."""
        ...


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextAbstractHandler",
    "FlextAbstractHandlerChain",
    "FlextAbstractHandlerRegistry",
    "FlextAbstractMetricsHandler",
    "FlextAbstractValidatingHandler",
]
