"""FlextProcessing - Simple unified processing for FLEXT.

Just what's needed, no over-engineering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from flext_core.result import FlextResult


class FlextProcessing:
    """Simple processing utilities."""

    class Handler(ABC):
        """Base handler - actually simple."""

        @abstractmethod
        def handle(self, request: object) -> FlextResult[object]:
            """Handle a request."""
            raise NotImplementedError

    class HandlerRegistry:
        """Simple registry for handlers."""

        def __init__(self) -> None:
            """Initialize handler registry."""
            self._handlers: dict[str, object] = {}

        def register(self, name: str, handler: object) -> FlextResult[None]:
            """Register a handler."""
            if name in self._handlers:
                return FlextResult[None].fail(f"Handler '{name}' already registered")
            self._handlers[name] = handler
            return FlextResult[None].ok(None)

        def get(self, name: str) -> FlextResult[object]:
            """Get a handler."""
            if name not in self._handlers:
                return FlextResult[object].fail(f"Handler '{name}' not found")
            return FlextResult[object].ok(self._handlers[name])

        def execute(self, name: str, request: object) -> FlextResult[object]:
            """Execute a handler by name."""
            handler_result = self.get(name)
            if handler_result.is_failure:
                return FlextResult[object].fail(handler_result.error or "Unknown error")

            handler = handler_result.unwrap()
            if hasattr(handler, "handle") and callable(getattr(handler, "handle")):
                result = handler.handle(request)
                return (
                    FlextResult[object].ok(result)
                    if not isinstance(result, FlextResult)
                    else result
                )
            if callable(handler):
                result = handler(request)
                return (
                    FlextResult[object].ok(result)
                    if not isinstance(result, FlextResult)
                    else result
                )
            return FlextResult[object].fail(f"Handler '{name}' is not callable")

    class Pipeline:
        """Simple processing pipeline."""

        def __init__(self) -> None:
            """Initialize processing pipeline."""
            self._steps: list[Callable[[object], FlextResult[object]]] = []

        def add_step(self, step: Callable[[object], FlextResult[object]]) -> None:
            """Add a processing step."""
            self._steps.append(step)

        def process(self, data: object) -> FlextResult[object]:
            """Process data through pipeline."""
            current = data
            for step in self._steps:
                result = step(current)
                if result.is_failure:
                    return result
                current = result.unwrap()
            return FlextResult[object].ok(current)

    # Factory methods for convenience
    @staticmethod
    def create_handler_registry() -> HandlerRegistry:
        """Create a new handler registry."""
        return FlextProcessing.HandlerRegistry()

    @staticmethod
    def create_pipeline() -> Pipeline:
        """Create a new processing pipeline."""
        return FlextProcessing.Pipeline()

    # =========================================================================
    # HANDLER CLASSES - For examples and demos
    # =========================================================================

    class Implementation:
        """Handler implementation utilities."""

        class BasicHandler:
            """Basic handler implementation."""

            def __init__(self, name: str) -> None:
                """Initialize basic handler with name."""
                self.name = name

            def handle(self, request: object) -> object:
                """Handle request."""
                return f"Handled by {self.name}: {request}"

    class Management:
        """Handler management utilities."""

        class HandlerRegistry:
            """Handler registry for examples."""

            def __init__(self) -> None:
                """Initialize handler registry."""
                self._handlers: dict[str, object] = {}

            def register(self, name: str, handler: object) -> None:
                """Register handler."""
                self._handlers[name] = handler

            def get(self, name: str) -> object | None:
                """Get handler by name."""
                return self._handlers.get(name)

    class Patterns:
        """Handler patterns for examples."""

        class HandlerChain:
            """Handler chain for examples."""

            def __init__(self, name: str) -> None:
                """Initialize handler chain with name."""
                self.name = name
                self._handlers: list[object] = []

            def add_handler(self, handler: object) -> None:
                """Add handler to chain."""
                self._handlers.append(handler)

    class Protocols:
        """Handler protocols for examples."""

        class ChainableHandler:
            """Chainable handler for examples."""

            def __init__(self, name: str) -> None:
                """Initialize chainable handler with name."""
                self.name = name

            def handle(self, request: object) -> object:
                """Handle request in chain."""
                return f"Chain handled by {self.name}: {request}"
