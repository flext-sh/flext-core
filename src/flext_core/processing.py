"""Processing helpers that complement the FLEXT-Core 1.0.0 dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

from flext_core.result import FlextResult


class FlextProcessing:
    """Processing convenience namespace aligned with dispatcher workflows.

    Registries, pipelines, and handler helpers mirror the ergonomics promoted in
    the modernization plan so supporting packages can compose around
    ``FlextDispatcher`` without bespoke glue code.
    """

    class Handler:
        """Minimal handler base returning modernization-compliant results."""

        def handle(self, request: object) -> FlextResult[object]:
            """Handle a request."""
            return FlextResult[object].ok(f"Base handler processed: {request}")

    class HandlerRegistry:
        """Registry managing named handler instances for dispatcher pilots."""

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
            try:
                if hasattr(handler, "handle") and callable(handler.handle):
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
                return FlextResult[object].fail(
                    f"Handler '{name}' does not implement handle method",
                )
            except Exception as e:
                return FlextResult[object].fail(f"Handler execution failed: {e}")

        def count(self) -> int:
            """Get the number of registered handlers."""
            return len(self._handlers)

        def exists(self, name: str) -> bool:
            """Check if a handler exists."""
            return name in self._handlers

        def get_optional(self, name: str) -> object | None:
            """Get a handler optionally, returning None if not found."""
            return self._handlers.get(name)

    class Pipeline:
        """Simple processing pipeline mirroring modernization samples."""

        def __init__(self) -> None:
            """Initialize processing pipeline."""
            self._steps: list[
                Callable[[object], FlextResult[object] | object]
                | dict[str, object]
                | object
            ] = []

        def add_step(
            self,
            step: Callable[[object], FlextResult[object] | object]
            | dict[str, object]
            | object,
        ) -> None:
            """Add a processing step."""
            self._steps.append(step)

        def process(self, data: object) -> FlextResult[object]:
            """Process data through pipeline."""
            current: object = data

            for step in self._steps:
                # Handle callable steps
                if callable(step):
                    result = step(current)
                    if isinstance(result, FlextResult):
                        if result.is_failure:
                            return result
                        current = result.unwrap()
                        continue

                    # Non-FlextResult return from callable
                    current = result

                # Handle non-callable steps
                elif isinstance(current, dict) and isinstance(step, dict):
                    # Merge dictionaries
                    current = {**current, **step}
                else:
                    # Replace current data
                    current = step

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

    @staticmethod
    def is_handler_safe(handler: object) -> bool:
        """Check if a handler is safe (has handle method or is callable)."""
        return (
            hasattr(handler, "handle") and callable(handler.handle)
        ) or callable(handler)

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

            @property
            def handler_name(self) -> str:
                """Get handler name."""
                return self.name

            def handle(self, request: object) -> FlextResult[str]:
                """Handle request."""
                result = f"Handled by {self.name}: {request}"
                return FlextResult[str].ok(result)

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

            def get_optional(self, name: str) -> object | None:
                """Get handler optionally, returning None if not found."""
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

            def handle(self, request: object) -> FlextResult[object]:
                """Handle request by executing all handlers in chain."""
                result = request
                for handler in self._handlers:
                    if hasattr(handler, "handle"):
                        handler_result = handler.handle(result)
                        if (
                            hasattr(handler_result, "success")
                            and not handler_result.success
                        ):
                            return FlextResult[object].fail(
                                f"Handler failed: {handler_result.error}",
                            )
                        result = (
                            handler_result.data
                            if hasattr(handler_result, "data")
                            else handler_result
                        )
                return FlextResult[object].ok(result)

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
