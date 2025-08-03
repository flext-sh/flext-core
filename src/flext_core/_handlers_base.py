"""FLEXT Core Handlers - Internal Implementation Module.

Internal implementation providing the foundational logic for handler patterns.
This module is part of the Internal Implementation Layer and should not be imported
directly by ecosystem projects. Use the public API through handlers module instead.

Module Role in Architecture:
    Internal Implementation Layer → Handler Patterns → Public API Layer

    This internal module provides:
    - Base handler interface with type-safe message processing
    - Specialized handlers for commands, events, and queries
    - Composition-based delegation patterns eliminating multiple inheritance
    - Lifecycle hooks for pre/post processing with logging integration

Implementation Patterns:
    Handler Pattern: Template method with customizable lifecycle hooks
    Composition-Based: Delegation instead of multiple inheritance complexity

Design Principles:
    - Single responsibility for internal handler implementation concerns
    - No external dependencies beyond standard library and sibling modules
    - Performance-optimized implementations for public API consumption
    - Type safety maintained through internal validation

Access Restrictions:
    - This module is internal and not exported in __init__.py
    - Use handlers module for all external access to handler functionality
    - Breaking changes may occur without notice in internal modules
    - No compatibility guarantees for internal implementation details

Quality Standards:
    - Internal implementation must maintain public API contracts
    - Performance optimizations must not break type safety
    - Code must be thoroughly tested through public API surface
    - Internal changes must not affect public behavior

See Also:
    handlers: Public API for handler patterns and message processing
    docs/python-module-organization.md: Internal module architecture

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from flext_core._mixins_base import _BaseTimingMixin
from flext_core.flext_types import FlextTypes, R, T, TAnyDict, TServiceName
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult


class _LoggerProtocol(Protocol):
    """Protocol for logger objects with standard logging methods."""

    def debug(self, message: str, **kwargs: object) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs: object) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs: object) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **kwargs: object) -> None:
        """Log error message."""
        ...

    def critical(self, message: str, **kwargs: object) -> None:
        """Log critical message."""
        ...

    def exception(self, message: str, **kwargs: object) -> None:
        """Log exception message."""
        ...


class _BaseHandler[T, R](_BaseTimingMixin, ABC):
    """Base handler interface for type-safe message processing.

    Foundation class providing comprehensive handler functionality
    through composition-based delegation patterns. Eliminates multiple inheritance
    complexity while maintaining full functionality.

    Architecture:
        - Single inheritance from ABC and Generic for interface contracts
        - Composition-based delegation for mixin functionality
        - Template method pattern with customizable lifecycle hooks
        - Type-safe message processing with generic constraints
    """

    def __init__(self, handler_name: TServiceName | None = None) -> None:
        """Initialize handler with logging and metrics through composition."""
        self._handler_name = handler_name or self.__class__.__name__
        # Initialize mixin functionality through composition
        self._logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._metrics = {
            "messages_handled": 0,
            "successes": 0,
            "failures": 0,
            "total_processing_time_ms": 0.0,
        }

        self._get_logger().debug(
            "Handler initialized",
            handler_name=self._handler_name,
        )

    # =========================================================================
    # LOGGING FUNCTIONALITY - Composition-based delegation
    # =========================================================================

    def _get_logger(self) -> _LoggerProtocol:
        """Get logger instance (lazy initialization)."""
        if not hasattr(self, "_logger"):
            self._logger = FlextLoggerFactory.get_logger(self._logger_name)
        return self._logger

    @property
    def logger(self) -> _LoggerProtocol:
        """Access to logger instance."""
        return self._get_logger()

    # =========================================================================
    # TIMING FUNCTIONALITY - Composition-based delegation
    # =========================================================================

    @abstractmethod
    def handle(self, message: T) -> FlextResult[R]:
        """Handle the message and return result.

        Args:
            message: Message to handle

        Returns:
            FlextResult with processing result or error

        """

    def can_handle(self, message: object) -> bool:
        """Check if handler can process this message."""
        self.logger.debug(
            "Checking if handler can process message",
            message_type=type(message).__name__,
        )

        # Get expected message type from Generic parameter
        if hasattr(self, "__orig_bases__"):
            for base in self.__orig_bases__:
                if hasattr(base, "__args__") and len(base.__args__) >= 1:
                    expected_type = base.__args__[0]
                    # Use FlextTypeGuards for validation
                    can_handle = FlextTypes.TypeGuards.is_instance_of(
                        message,
                        expected_type,
                    )

                    self.logger.debug(
                        "Handler check result",
                        can_handle=can_handle,
                        expected_type=getattr(
                            expected_type,
                            "__name__",
                            str(expected_type),
                        ),
                    )
                    return can_handle

        # Default to True if we can't determine type
        return True

    def pre_handle(self, message: T) -> FlextResult[T]:
        """Pre-processing hook with logging."""
        self.logger.debug(
            "Pre-processing message",
            message_type=type(message).__name__,
        )

        # Validate message if it has validation method
        if hasattr(message, "validate"):
            validation_result = message.validate()
            if (
                hasattr(validation_result, "is_failure")
                and validation_result.is_failure
            ):
                self.logger.warning(
                    "Message validation failed in pre-processing",
                    error=validation_result.error,
                )
                return FlextResult.fail(
                    validation_result.error or "Validation failed",
                )

        return FlextResult.ok(message)

    def post_handle(self, result: FlextResult[R]) -> FlextResult[R]:
        """Post-processing hook with metrics."""
        # Update metrics
        if result.is_success:
            self._metrics["successes"] += 1
            self.logger.debug(
                "Post-processing successful result",
                handler=self._handler_name,
            )
        else:
            self._metrics["failures"] += 1
            self.logger.warning(
                "Post-processing failed result",
                handler=self._handler_name,
                error=result.error,
            )

        return result

    def handle_with_hooks(self, message: T) -> FlextResult[R]:
        """Handle with pre/post processing hooks and full logging."""
        start_time = self._start_timing()

        self.logger.info(
            "Starting message handling",
            handler=self._handler_name,
            message_type=type(message).__name__,
        )

        # Check if can handle
        if not self.can_handle(message):
            error_msg = f"{self._handler_name} cannot handle {type(message).__name__}"
            self.logger.error(error_msg)
            return FlextResult.fail(error_msg)

        # Pre-process
        pre_result = self.pre_handle(message)
        if pre_result.is_failure:
            self.logger.error(
                "Pre-processing failed",
                error=pre_result.error,
            )
            return FlextResult.fail(pre_result.error or "Pre-processing failed")

        # Main handling
        processed_message = pre_result.unwrap()
        main_result = self.handle(processed_message)

        # Post-process
        final_result = self.post_handle(main_result)

        # Update metrics using timing mixin
        execution_time_ms = self._get_execution_time_ms(start_time)
        self._metrics["messages_handled"] += 1
        self._metrics["total_processing_time_ms"] += execution_time_ms

        self.logger.info(
            "Message handling completed",
            handler=self._handler_name,
            success=final_result.is_success,
            execution_time_ms=self._get_execution_time_ms_rounded(start_time),
            total_handled=self._metrics["messages_handled"],
        )

        return final_result

    def get_metrics(self) -> TAnyDict:
        """Get handler metrics."""
        avg_time = 0.0
        if self._metrics["messages_handled"] > 0:
            avg_time = (
                self._metrics["total_processing_time_ms"]
                / self._metrics["messages_handled"]
            )

        return {
            **self._metrics,
            "average_processing_time_ms": round(avg_time, 2),
            "success_rate": self._metrics["successes"]
            / max(1, self._metrics["messages_handled"]),
        }


class _BaseCommandHandler(_BaseHandler[T, R]):
    """Handler specifically for commands."""

    def validate_command(self, command: T) -> FlextResult[None]:
        """Validate command before processing."""
        _ = command  # Mark as used for linting
        return FlextResult.ok(None)

    def pre_handle(self, command: T) -> FlextResult[T]:
        """Pre-process command with validation."""
        validation_result = self.validate_command(command)
        if validation_result.is_failure:
            return FlextResult.fail(
                validation_result.error or "Command validation failed",
            )
        return FlextResult.ok(command)


class _BaseEventHandler(_BaseHandler[T, None]):
    """Handler specifically for events."""

    def handle(self, event: T) -> FlextResult[None]:
        """Handle event."""
        self.process_event(event)
        return FlextResult.ok(None)

    @abstractmethod
    def process_event(self, event: T) -> None:
        """Process the event."""


class _BaseQueryHandler(_BaseHandler[T, R]):
    """Handler specifically for queries."""

    def authorize_query(self, query: T) -> FlextResult[None]:
        """Check query authorization."""
        _ = query  # Mark as used for linting
        return FlextResult.ok(None)

    def pre_handle(self, query: T) -> FlextResult[T]:
        """Pre-process query with authorization."""
        auth_result = self.authorize_query(query)
        if auth_result.is_failure:
            return FlextResult.fail(
                auth_result.error or "Query authorization failed",
            )
        return FlextResult.ok(query)


__all__ = [
    "_BaseCommandHandler",
    "_BaseEventHandler",
    "_BaseHandler",
    "_BaseQueryHandler",
]
