"""Concrete handler implementations using handlers_base abstractions.

Provides domain-specific handler implementations for CQRS patterns using
handlers_base.py abstractions following SOLID principles.

Classes:
    FlextBaseHandler: Concrete base handler implementation.
    FlextValidatingHandler: Handler with validation logic.
    FlextAuthorizingHandler: Handler with authorization logic.
    FlextEventHandler: Handler for event processing.
    FlextMetricsHandler: Handler with metrics collection.
    FlextHandlerChain: Concrete chain of responsibility implementation.
    FlextHandlerRegistry: Concrete handler registry implementation.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from flext_core.commands import FlextCommands
from flext_core.handlers_base import (
    FlextAbstractHandler,
    FlextAbstractHandlerChain,
    FlextAbstractHandlerRegistry,
    FlextAbstractMetricsHandler,
    FlextAbstractValidatingHandler,
)
from flext_core.result import FlextResult

# Type variables for generic handlers
TQuery = TypeVar("TQuery")
TQueryResult = TypeVar("TQueryResult")

# =============================================================================
# HANDLER IMPLEMENTATIONS - Concrete implementations for CQRS patterns
# =============================================================================


class FlextBaseHandler(FlextAbstractHandler[object, object]):
    """Concrete base handler implementation using abstractions."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize base handler."""
        self._name = name or self.__class__.__name__

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return self._name

    def handle(self, request: object) -> FlextResult[object]:
        """Handle request."""
        return self.process_message(request)

    def validate_request(self, request: object) -> FlextResult[None]:
        """Validate request."""
        if request is None:
            return FlextResult.fail("Request cannot be None")
        return FlextResult.ok(None)

    def can_handle(self, message_type: object) -> bool:  # noqa: ARG002
        """Check if handler can process message type.

        Args:
            message_type: Type of message to check.

        Returns:
            True if handler can process message type.

        """
        # Default implementation - override in subclasses
        return True

    def pre_process(self, message: object) -> FlextResult[None]:
        """Pre-processing hook.

        Args:
            message: Message to pre-process.

        Returns:
            FlextResult indicating success or failure.

        """
        del message  # Unused argument
        return FlextResult.ok(None)

    def process_message(self, message: object) -> FlextResult[object]:
        """Process main message - override in subclasses."""
        return FlextResult.ok(message)

    def post_process(
        self,
        _message: object,
        result: FlextResult[object],  # noqa: ARG002
    ) -> FlextResult[None]:
        """Post-processing hook."""
        return FlextResult.ok(None)

    def get_handler_metadata(self) -> dict[str, object]:
        """Get handler metadata for monitoring."""
        return {
            "handler_name": self.handler_name,
            "handler_class": self.__class__.__name__,
            "can_handle_all": True,  # Default implementation handles all
        }


class FlextValidatingHandler(FlextAbstractValidatingHandler[object, object]):
    """Concrete handler with validation using abstractions."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize validating handler."""
        self._name = name or self.__class__.__name__

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return self._name

    def handle(self, request: object) -> FlextResult[object]:
        """Handle request with validation."""
        validation = self.validate_input(request)
        if validation.is_failure:
            return FlextResult.fail(validation.error or "Validation failed")
        result = self.process_message(request)
        return self.validate_output(result.unwrap() if result.is_success else None)

    def validate_request(self, request: object) -> FlextResult[None]:
        """Validate request."""
        return self.validate_input(request)

    def can_handle(self, request: object) -> bool:
        """Check if handler can handle request."""
        return request is not None

    def validate_input(self, request: object) -> FlextResult[None]:
        """Validate input."""
        if request is None:
            return FlextResult.fail("Request cannot be None")
        return FlextResult.ok(None)

    def validate_output(self, response: object) -> FlextResult[object]:
        """Validate output."""
        return FlextResult.ok(response)

    def get_validation_rules(self) -> list[object]:
        """Get validation rules."""
        return []

    def validate_message(self, message: object) -> FlextResult[None]:
        """Validate message - implements abstract method."""
        # Basic validation - check not None
        if message is None:
            return FlextResult.fail("Message cannot be None")
        return FlextResult.ok(None)

    def process_message(self, message: object) -> FlextResult[object]:
        """Process validated message."""
        return FlextResult.ok(message)

    def validate(self, message: object) -> FlextResult[object]:
        """Validate message before processing (compatibility method)."""
        validation_result = self.validate_message(message)
        if validation_result.is_failure:
            return FlextResult.fail(validation_result.error or "Validation failed")
        return FlextResult.ok(message)

    def post_process(
        self,
        message: object,
        result: FlextResult[object],  # noqa: ARG002
    ) -> FlextResult[None]:
        """Post-process the result."""
        del message
        return FlextResult.ok(None)

    def get_handler_metadata(self) -> dict[str, object]:
        """Get handler metadata for monitoring."""
        return {
            "handler_name": self.handler_name,
            "handler_class": self.__class__.__name__,
            "validation_enabled": True,
        }


class FlextAuthorizingHandler(FlextBaseHandler):
    """Handler with authorization implementing AuthorizingHandler protocol."""

    def __init__(
        self,
        name: str | None = None,
        required_permissions: list[str] | None = None,
    ) -> None:
        """Initialize with required permissions."""
        super().__init__()
        self._name = name or self.__class__.__name__
        self.required_permissions = required_permissions or []

    def authorize(
        self,
        message: object,
        context: dict[str, object],
    ) -> FlextResult[bool]:
        """Check authorization for message processing."""
        return self.authorize_message(message, context)

    def authorize_message(
        self,
        message: object,
        context: dict[str, object],
    ) -> FlextResult[bool]:
        """Override this method to implement specific authorization."""
        del message  # Unused argument
        # Basic authorization - check context has user
        user = context.get("user")
        if not user:
            return FlextResult.fail("No user in context")

        # Check permissions if required
        if self.required_permissions:
            user_permissions = context.get("permissions", [])
            if not isinstance(user_permissions, (list, set, tuple)):
                return FlextResult.fail("Invalid permissions format in context")
            for permission in self.required_permissions:
                if permission not in user_permissions:
                    return FlextResult.fail(f"Missing permission: {permission}")

        authorized = True
        return FlextResult.ok(authorized)

    def pre_process(self, message: object) -> FlextResult[None]:
        """Pre-process with authorization."""
        del message  # Unused argument
        # In a real implementation, context would be passed through
        # For now, we'll skip authorization in pre-process
        return FlextResult.ok(None)

    def get_handler_metadata(self) -> dict[str, object]:
        """Get handler metadata for monitoring."""
        return {
            "handler_name": self.handler_name,
            "handler_class": self.__class__.__name__,
            "authorization_enabled": True,
            "required_permissions": self.required_permissions,
        }


class FlextEventHandler(FlextBaseHandler):
    """Event handler implementing EventProcessor protocol."""

    def process_event(self, event: dict[str, object]) -> FlextResult[None]:
        """Process domain event."""
        try:
            event_type = event.get("event_type")
            if not event_type:
                return FlextResult.fail("Event missing event_type")

            # Process based on event type
            return self.handle_event_type(str(event_type), event)

        except Exception as e:
            return FlextResult.fail(f"Event processing failed: {e}")

    def can_process(self, event_type: str) -> bool:
        """Check if handler can process event type."""
        del event_type  # Unused argument
        # Override in subclasses for specific event types
        return True

    def handle_event_type(
        self,
        event_type: str,
        event: dict[str, object],
    ) -> FlextResult[None]:
        """Handle specific event type - override in subclasses."""
        del event_type, event  # Unused arguments
        return FlextResult.ok(None)


class FlextMetricsHandler(FlextAbstractMetricsHandler[object, object]):
    """Concrete handler with metrics collection using abstractions."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize metrics handler."""
        self._name = name or self.__class__.__name__
        self.metrics: dict[str, object] = {
            "total_requests": 0,
            "messages_processed": 0,
        }
        self._start_time: float | None = None

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return self._name

    @property
    def name(self) -> str:
        """Get handler name (compat)."""
        return self._name

    def handle(self, request: object) -> FlextResult[object]:
        """Handle request with metrics."""
        # Increment counters at start
        current_total = self.metrics.get("total_requests", 0)
        current_processed = self.metrics.get("messages_processed", 0)
        self.metrics["total_requests"] = (
            int(current_total) + 1 if isinstance(current_total, (int, float)) else 1
        )

        self.start_metrics(request)
        result = self.process_message(request)
        if result.is_success:
            self.stop_metrics(request, result.unwrap())
            # Increment processed count only on success
            self.metrics["messages_processed"] = (
                int(current_processed) + 1
                if isinstance(current_processed, (int, float))
                else 1
            )
        return result

    def validate_request(self, request: object) -> FlextResult[None]:
        """Validate request."""
        if request is None:
            return FlextResult.fail("Request cannot be None")
        return FlextResult.ok(None)

    def can_handle(self, request: object) -> bool:
        """Check if handler can handle request."""
        return request is not None

    def start_metrics(self, request: object) -> None:
        """Start metrics collection."""
        import time  # noqa: PLC0415

        self._start_time = time.time()
        self.metrics["request_type"] = type(request).__name__

    def stop_metrics(self, request: object, response: object) -> None:  # noqa: ARG002
        """Stop metrics collection."""
        import time  # noqa: PLC0415

        if self._start_time is not None:
            self.metrics["last_duration"] = time.time() - self._start_time
        self.metrics["response_type"] = type(response).__name__

    def get_metrics(self) -> dict[str, object]:
        """Get collected metrics."""
        return self.metrics.copy()

    def clear_metrics(self) -> None:
        """Clear metrics."""
        self.metrics.clear()
        self._start_time = None

    def process_message(self, message: object) -> FlextResult[object]:
        """Process message - basic implementation."""
        return FlextResult.ok(message)

    def collect_metrics(self, message: object, result: FlextResult[object]) -> None:  # noqa: ARG002
        """Collect custom metrics - implements abstract method."""
        # Collect basic processing metrics
        success = result.is_success

        # Initialize custom metrics storage inside metrics dict
        if "custom_metrics" not in self.metrics:
            self.metrics["custom_metrics"] = {"success_count": 0, "failure_count": 0}

        store_obj = self.metrics["custom_metrics"]
        if not isinstance(store_obj, dict):
            self.metrics["custom_metrics"] = {"success_count": 0, "failure_count": 0}
            store_obj = self.metrics["custom_metrics"]

        custom_metrics: dict[str, int] = store_obj  # type: ignore[assignment]

        # Update counts
        if success:
            count = custom_metrics.get("success_count", 0)
            custom_metrics["success_count"] = int(count) + 1
        else:
            count = custom_metrics.get("failure_count", 0)
            custom_metrics["failure_count"] = int(count) + 1

    def collect_operation_metrics(
        self,
        operation: str,
        duration: float,
    ) -> FlextResult[None]:
        """Collect performance metrics for specific operations."""
        try:
            # Initialize operations dict if needed
            if "operations" not in self.metrics:
                self.metrics["operations"] = {}

            # Get operations dict with type safety
            operations_obj = self.metrics["operations"]
            if not isinstance(operations_obj, dict):
                self.metrics["operations"] = {}
                operations_obj = self.metrics["operations"]

            operations_dict: dict[str, dict[str, int | float]] = operations_obj  # type: ignore[assignment]

            # Initialize operation metrics if needed
            if operation not in operations_dict:
                operations_dict[operation] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                }

            # Update metrics with type safety
            op_metrics = operations_dict[operation]
            current_count = op_metrics.get("count", 0)
            current_total = op_metrics.get("total_duration", 0.0)

            if isinstance(current_count, (int, float)) and isinstance(
                current_total,
                (int, float),
            ):
                new_count = int(current_count) + 1
                new_total = float(current_total) + duration
                op_metrics["count"] = new_count
                op_metrics["total_duration"] = new_total
                op_metrics["avg_duration"] = new_total / new_count

            return FlextResult.ok(None)

        except Exception as e:
            return FlextResult.fail(f"Metrics collection failed: {e}")

    def get_metrics_summary(self) -> dict[str, object]:
        """Get current metrics summary."""
        return dict(self.metrics)

    def post_process(
        self,
        message: object,
        result: FlextResult[object],
    ) -> FlextResult[None]:
        """Post-process with metrics collection."""
        _ = super().post_process(message, result)  # type: ignore[misc]

        # Record operation metrics
        duration_val = self.metrics.get("last_duration", 0.0)
        duration = (
            float(duration_val) if isinstance(duration_val, (int, float)) else 0.0
        )
        operation = f"{self.name}_handle"
        _ = self.collect_operation_metrics(operation, duration)
        return FlextResult.ok(None)

    def get_handler_metadata(self) -> dict[str, object]:
        """Get handler metadata for monitoring."""
        return {
            "handler_name": self.handler_name,
            "handler_class": self.__class__.__name__,
            "metrics_enabled": True,
            "current_metrics": self.get_metrics(),
        }


# =============================================================================
# HANDLER REGISTRY - Service location pattern
# =============================================================================


class FlextHandlerRegistry(
    FlextAbstractHandlerRegistry[FlextAbstractHandler[object, object]],
):
    """Concrete handler registry implementation using abstractions."""

    def __init__(self) -> None:
        """Initialize registry with abstractions."""
        self.registry: dict[str, FlextAbstractHandler[object, object]] = {}
        self._type_mappings: dict[type, list[str]] = {}

    def register_handler(  # type: ignore[override]
        self,
        name: str,
        handler: FlextAbstractHandler[object, object],
    ) -> FlextResult[None]:
        """Register handler - implements abstract method."""
        try:
            self.registry[name] = handler
            return FlextResult.ok(None)
        except Exception as e:
            return FlextResult.fail(f"Handler registration failed: {e}")

    def get_handler(
        self, name: str
    ) -> FlextResult[FlextAbstractHandler[object, object]]:
        """Get handler - implements abstract method."""
        handler = self.registry.get(name)
        if not handler:
            available = list(self.registry.keys())
            return FlextResult.fail(
                f"Handler '{name}' not found. Available: {available}",
            )
        return FlextResult.ok(handler)

    def register(self, name: str, handler: object) -> FlextResult[None]:
        """Register handler with name (compatibility method)."""
        if isinstance(handler, FlextAbstractHandler):
            return self.register_handler(name, handler)
        return FlextResult.fail(
            f"Handler must be FlextAbstractHandler, got {type(handler)}",
        )

    def get_handlers_for_type(
        self, message_type: type
    ) -> list[FlextAbstractHandler[object, object]]:
        """Get all handlers that can process message type."""
        return [
            handler
            for handler in self.registry.values()
            if handler.can_handle(message_type)
        ]

    # ------------------------------------------------------------------
    # Compatibility helpers expected by legacy tests
    # ------------------------------------------------------------------
    def register_for_type(
        self,
        message_type: type,
        name: str,
        handler: FlextAbstractHandler[object, object],
    ) -> FlextResult[None]:
        """Register a handler for a specific message type.

        Keeps an auxiliary mapping from type -> handler names to support
        quick lookup via get_handler_for_type.
        """
        result = self.register_handler(name, handler)
        if result.is_failure:
            return result
        # Maintain mapping list
        names = self._type_mappings.setdefault(message_type, [])
        if name not in names:
            names.append(name)
        return FlextResult.ok(None)

    def get_handler_for_type(
        self,
        message_type: type,
    ) -> FlextResult[FlextAbstractHandler[object, object]]:
        """Get first handler registered for a specific message type."""
        # Prefer explicit mapping first
        names = self._type_mappings.get(message_type)
        if names:
            for name in names:
                handler = self.registry.get(name)
                if handler is not None:
                    return FlextResult.ok(handler)
        # Fallback to capability-based scan
        handlers = self.get_handlers_for_type(message_type)
        if handlers:
            return FlextResult.ok(handlers[0])
        return FlextResult.fail(f"No handler registered for type: {message_type!r}")

    def unregister_handler(self, name: str) -> bool:
        """Unregister handler - implements abstract method."""
        if name in self.registry:
            del self.registry[name]
            return True
        return False

    def get_all_handlers(self) -> dict[str, FlextAbstractHandler[object, object]]:
        """Get all handlers - implements abstract method."""
        return dict(self.registry)

    def clear_handlers(self) -> None:
        """Clear all handlers - implements abstract method."""
        self.registry.clear()
        self._type_mappings.clear()

    def clear(self) -> None:
        """Clear all registered handlers."""
        self.clear_handlers()


# =============================================================================
# CHAIN OF RESPONSIBILITY - Multi-handler processing
# =============================================================================


class FlextHandlerChain(FlextAbstractHandlerChain[object, object]):
    """Concrete chain of responsibility implementation using abstractions."""

    def __init__(
        self,
        handlers: list[FlextAbstractHandler[object, object]] | None = None,
    ) -> None:
        """Initialize with optional handler list using abstractions."""
        self.handlers: list[FlextAbstractHandler[object, object]] = handlers or []

    @property
    def handler_name(self) -> str:
        """Get handler chain name."""
        return self.__class__.__name__

    def add_handler(self, handler: FlextAbstractHandler[object, object]) -> None:
        """Add handler to chain - implements abstract method."""
        self.handlers.append(handler)

    def remove_handler(self, handler: FlextAbstractHandler[object, object]) -> bool:
        """Remove handler from chain - implements abstract method."""
        if handler in self.handlers:
            self.handlers.remove(handler)
            return True
        return False

    def handle_request(self, request: object) -> FlextResult[object]:
        """Handle request through chain - implements abstract method."""
        for handler in self.handlers:
            if handler.can_handle(request):
                return handler.handle(request)
        return FlextResult.fail("No handler could process the request")

    def get_handlers(self) -> list[FlextAbstractHandler[object, object]]:
        """Get all handlers - implements abstract method."""
        return self.handlers.copy()

    def process_chain(self, message: object) -> FlextResult[object]:
        """Process message through handler chain - implements abstract method."""
        results = []
        last_successful_result = None

        for handler in self.handlers:
            try:
                result = handler.handle(message)
                results.append(result)

                if result.is_success:
                    last_successful_result = result.data
                else:
                    # Stop on first failure if desired
                    break

            except Exception as e:
                error_result: FlextResult[object] = FlextResult.fail(
                    f"Chain handler failed: {e}",
                )
                results.append(error_result)
                break

        # Return last successful result or failure
        if last_successful_result is not None:
            return FlextResult.ok(last_successful_result)
        if results:
            # Return first failure
            for result in results:
                if result.is_failure:
                    return FlextResult.fail(result.error or "Chain processing failed")

        return FlextResult.fail("No handlers processed the message")

    def handle(self, message: object) -> FlextResult[list[object]]:
        """Handle message through chain (compatibility method)."""
        results = []

        for handler in self.handlers:
            try:
                result = handler.handle(message)
                results.append(result)

                # Stop on first failure if desired
                if result.is_failure:
                    break

            except Exception as e:
                error_result: FlextResult[object] = FlextResult.fail(
                    f"Chain handler failed: {e}",
                )
                results.append(error_result)
                break

        # Convert list of results to list of objects
        result_objects: list[object] = []
        for result in results:
            if result.is_success and result.data is not None:
                result_objects.append(result.data)
            elif result.is_failure:
                result_objects.append(result.error or "Unknown error")
        return FlextResult.ok(result_objects)

    def can_handle(self, message_type: type) -> bool:
        """Check if any handler in chain can process message type."""
        return any(handler.can_handle(message_type) for handler in self.handlers)

    # ------------------------------------------------------------------
    # Compatibility helpers expected by legacy tests
    # ------------------------------------------------------------------
    def process(self, message: object) -> FlextResult[object]:
        """Compatibility alias for processing a single message."""
        return self.process_chain(message)

    def process_all(self, messages: list[object]) -> FlextResult[list[object]]:
        """Process a list of messages and collect successful results."""
        collected: list[object] = []
        for message in messages:
            result = self.process_chain(message)
            if result.is_success and result.data is not None:
                collected.append(result.data)
            else:
                # short-circuit on failure to align with conservative semantics
                return FlextResult.fail(result.error or "Chain processing failed")
        return FlextResult.ok(collected)


# =============================================================================
# HANDLER FACTORIES MOVED TO LEGACY.PY
# =============================================================================
# Factory functions create_base_handler, create_validating_handler, etc.
# have been moved to legacy.py with deprecation warnings.
#
# NEW USAGE: Use direct class instantiation
#   handler = FlextBaseHandler(name="my_handler")
#   validator = FlextValidatingHandler(name="validator")
#
# OLD USAGE (deprecated): Import from legacy.py
#   from flext_core.legacy import create_base_handler


# =============================================================================
# GLOBAL REGISTRY MOVED TO LEGACY.PY
# =============================================================================
# Global registry functions have been moved to legacy.py with deprecation warnings.
#
# NEW USAGE: Create and manage your own registry
#   registry = FlextHandlerRegistry()
#   registry.register("my_handler", my_handler)
#
# OLD USAGE (deprecated): Import from legacy.py
#   from flext_core.legacy import get_global_registry, reset_global_registry


# =============================================================================
# EXPORTS - Handler implementations only
# =============================================================================

__all__: list[str] = [
    # Clean Handler Implementations Only (alphabetically sorted)
    "FlextAuthorizingHandler",
    "FlextBaseHandler",
    "FlextCommandHandler",  # Backward-compat
    "FlextEventHandler",
    "FlextHandlerChain",
    "FlextHandlerRegistry",
    "FlextHandlers",  # Backward-compat facade expected by tests
    "FlextMetricsHandler",
    "FlextQueryHandler",  # Backward-compat
    "FlextValidatingHandler",
]


# Backward-compat handler classes used in historical tests
class FlextCommandHandler(FlextCommands.Handler[object, object]):
    """Test-friendly command handler with default pass-through behavior."""

    def __init__(
        self,
        handler_name: str | None = None,
        *,
        validator: object | None = None,
        authorizer: object | None = None,
        metrics_collector: object | None = None,
    ) -> None:
        """Initialize command handler with optional injected components."""
        super().__init__()
        self._name = handler_name or self.__class__.__name__
        self._validator = validator
        self._authorizer = authorizer
        self._metrics = metrics_collector

    def validate(self, message: object) -> FlextResult[object]:
        """Validate message using injected validator if available."""
        if self._validator:
            validate_method = getattr(self._validator, "validate_message", None)
            if callable(validate_method):
                validation_result = validate_method(message)
                if validation_result.is_failure:
                    return FlextResult.fail(
                        validation_result.error or "Validation failed"
                    )
        return FlextResult.ok(message)

    def handle(self, command: object) -> FlextResult[object]:
        """Handle command."""
        return FlextResult.ok(command)

    # Helpers expected in tests
    def handle_with_hooks(self, command: object) -> FlextResult[object]:
        """Handle command with hooks."""
        # Initialize metrics state if not present
        if not hasattr(self, "_metrics_state"):
            self._metrics_state = {"total": 0, "success": 0}

        # Process command - metrics are tracked inside process_command
        return self.process_command(command)

    def get_metrics(self) -> dict[str, object]:
        """Get metrics using injected collector combined with internal metrics."""
        # Start with default metrics structure
        metrics = {
            "handler_type": "CommandHandler",
            "handler_name": self._name,
            "commands_processed": 0,
            "successful_commands": 0,
            "success_rate": 0.0,
        }

        # Get environment info expected by tests
        from os import environ  # noqa: PLC0415

        metrics["environment"] = environ.get("ENVIRONMENT", "production")
        metrics["version"] = "1.0"

        # Apply internal metrics counters if available
        if hasattr(self, "_metrics_state"):
            state = self._metrics_state
            total = state.get("total", 0)
            success = state.get("success", 0)
            metrics["commands_processed"] = total
            metrics["successful_commands"] = success
            metrics["success_rate"] = (float(success) / float(total)) if total else 0.0

        # Merge with injected metrics collector if available (but preserve
        # internal counters)
        if self._metrics:
            get_metrics_method = getattr(self._metrics, "get_metrics", None)
            if callable(get_metrics_method):
                collector_metrics = get_metrics_method()
                if isinstance(collector_metrics, dict):
                    # Merge collector metrics but preserve our internal counters
                    excluded_keys = {
                        "commands_processed",
                        "successful_commands",
                        "success_rate",
                    }
                    metrics.update(
                        {
                            k: v
                            for k, v in collector_metrics.items()
                            if k not in excluded_keys
                        }
                    )

        return metrics


class FlextQueryHandler(FlextCommands.QueryHandler[object, object]):
    """Minimal query handler with optional authorizer injection."""

    def __init__(
        self,
        handler_name: str | None = None,
        authorizer: object | None = None,
    ) -> None:
        """Initialize query handler with optional authorizer."""
        self._name = handler_name or self.__class__.__name__
        self._authorizer = authorizer

    def handle(self, query: object) -> FlextResult[object]:
        """Handle query."""
        return FlextResult.ok(query)

    def authorize_query(self, query: object) -> FlextResult[None]:
        """Authorize query."""
        if self._authorizer and hasattr(self._authorizer, "authorize_query"):
            auth_func = self._authorizer.authorize_query
            return auth_func(query)  # type: ignore[no-any-return]
        return FlextResult.ok(None)

    def pre_handle(self, query: object) -> FlextResult[None]:
        """Pre-handle query."""
        return self.authorize_query(query)

    def validate_message(self, message: object) -> FlextResult[None]:
        """Validate message for test compatibility."""
        # Basic validation - check not None
        if message is None:
            return FlextResult.fail("Message cannot be None")
        return FlextResult.ok(None)


class FlextHandlers:
    """Legacy facade exposing command/query handler classes for tests."""

    class CommandHandler(FlextCommandHandler):
        """Concrete alias class to satisfy type usage in tests."""

    class QueryHandler(Generic[TQuery, TQueryResult]):
        """Generic query handler alias for test compatibility."""

        def __init__(
            self,
            handler_name: str | None = None,
            authorizer: object | None = None,
        ) -> None:
            """Initialize query handler with optional authorizer."""
            self._name = handler_name or self.__class__.__name__
            self._authorizer = authorizer

        def handle(self, query: TQuery) -> FlextResult[TQueryResult]:  # noqa: ARG002
            """Handle query - override in subclasses."""
            # Default implementation - subclasses must override with proper return type
            return FlextResult.fail("Query handler not implemented")

        def authorize_query(self, query: object) -> FlextResult[None]:
            """Authorize query."""
            if self._authorizer and hasattr(self._authorizer, "authorize_query"):
                auth_func = self._authorizer.authorize_query
                return auth_func(query)  # type: ignore[no-any-return]
            return FlextResult.ok(None)

        def pre_handle(self, query: object) -> FlextResult[None]:
            """Pre-handle query."""
            return self.authorize_query(query)

        def validate_message(self, message: object) -> FlextResult[None]:
            """Validate message for test compatibility."""
            # Basic validation - check not None
            if message is None:
                return FlextResult.fail("Message cannot be None")
            return FlextResult.ok(None)

    class Handler(FlextCommands.Handler[object, object]):
        """Generic handler base alias for compatibility."""

    class EventHandler(FlextEventHandler):
        """Concrete alias for event handler used in tests."""

    # Utility construction helpers expected by tests
    @staticmethod
    def flext_create_chain(
        handlers: list[FlextAbstractHandler[object, object]] | None = None,
    ) -> FlextHandlerChain:
        """Create a handler chain from a list of handlers."""
        return FlextHandlerChain(handlers or [])

    @staticmethod
    def create_registry() -> FlextHandlerRegistry:
        """Create a new handler registry instance."""
        return FlextHandlerRegistry()

    @staticmethod
    def register_for_type(
        registry: FlextHandlerRegistry,
        message_type: type,
        name: str,
        handler: FlextAbstractHandler[object, object],
    ) -> FlextResult[None]:
        """Register handler for a specific type in the given registry."""
        return registry.register_for_type(message_type, name, handler)

    @staticmethod
    def get_handler_for_type(
        registry: FlextHandlerRegistry,
        message_type: type,
    ) -> FlextResult[FlextAbstractHandler[object, object]]:
        """Get handler for a specific type from registry."""
        return registry.get_handler_for_type(message_type)

    @staticmethod
    def get_metrics(handler: object) -> dict[str, object]:
        """Get metrics from handler if available."""
        if hasattr(handler, "get_metrics") and callable(handler.get_metrics):
            return handler.get_metrics()  # type: ignore[no-any-return]
        return {}

    # Pipeline behavior for middleware pattern
    class PipelineBehavior:
        """Base pipeline behavior for middleware pattern."""

        def __init__(self) -> None:
            """Initialize pipeline behavior."""
            self.next_behavior: object | None = None

        def handle(self, message: object, next_handler: object) -> FlextResult[object]:
            """Handle message in pipeline."""
            # Default implementation just calls next handler
            if hasattr(next_handler, "handle"):
                return next_handler.handle(message)  # type: ignore[no-any-return]
            return FlextResult.ok(message)

        def set_next(self, behavior: object) -> None:
            """Set next behavior in chain."""
            self.next_behavior = behavior

    # Command and Query Bus implementations for test compatibility
    class CommandBus:
        """Command bus implementation for test compatibility."""

        def __init__(self) -> None:
            """Initialize command bus."""
            self._handlers: dict[type, object] = {}
            self._metrics = {
                "commands_processed": 0,
                "successful_commands": 0,
                "failed_commands": 0,
            }

        def register(self, command_type: type, handler: object) -> FlextResult[None]:
            """Register command handler."""
            self._handlers[command_type] = handler
            return FlextResult.ok(None)

        def send(self, command: object) -> FlextResult[object]:
            """Send command to handler."""
            command_type = type(command)
            handler = self._handlers.get(command_type)

            if not handler:
                return FlextResult.fail(
                    f"No handler registered for {command_type.__name__}"
                )

            # Update metrics
            self._metrics["commands_processed"] += 1

            # Call handler if it has handle method
            if hasattr(handler, "handle"):
                # Handler is expected to implement .handle(command) -> FlextResult
                result = handler.handle(command)
                if hasattr(result, "is_success") and result.is_success:
                    self._metrics["successful_commands"] += 1
                else:
                    self._metrics["failed_commands"] += 1
                return result  # type: ignore[no-any-return]

            return FlextResult.ok(command)

        def get_metrics(self) -> dict[str, object]:
            """Get command bus metrics."""
            return dict(self._metrics)

    class QueryBus:
        """Query bus implementation for test compatibility."""

        def __init__(self) -> None:
            """Initialize query bus."""
            self._handlers: dict[type, object] = {}
            self._metrics = {
                "queries_processed": 0,
                "successful_queries": 0,
                "failed_queries": 0,
            }

        def register(self, query_type: type, handler: object) -> FlextResult[None]:
            """Register query handler."""
            self._handlers[query_type] = handler
            return FlextResult.ok(None)

        def send(self, query: object) -> FlextResult[object]:
            """Send query to handler."""
            query_type = type(query)
            handler = self._handlers.get(query_type)

            if not handler:
                return FlextResult.fail(
                    f"No handler registered for {query_type.__name__}"
                )

            # Update metrics
            self._metrics["queries_processed"] += 1

            # Call handler if it has handle method
            if hasattr(handler, "handle"):
                result = handler.handle(query)
                if hasattr(result, "is_success") and result.is_success:
                    self._metrics["successful_queries"] += 1
                else:
                    self._metrics["failed_queries"] += 1
                return result  # type: ignore[no-any-return]

            return FlextResult.ok(query)

        def get_metrics(self) -> dict[str, object]:
            """Get query bus metrics."""
            return dict(self._metrics)


# =============================================================================
# LEGACY ALIASES MOVED TO LEGACY.PY
# =============================================================================
# Legacy aliases like FlextHandlers have been moved to legacy.py
# Import from legacy.py if needed for backward compatibility
