"""Processing helpers that complement the FLEXT-Core 1.0.0 dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

from flext_core.result import FlextResult
from flext_core.utilities import FlextUtilities


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
            """Execute a handler by name using railway pattern."""
            return self.get(name).flat_map(
                lambda handler: self._execute_handler_safely(handler, request, name)
            )

        def _execute_handler_safely(
            self, handler: object, request: object, name: str
        ) -> FlextResult[object]:
            """Execute handler with proper method resolution and error handling."""
            try:
                # Check for handle method first
                if hasattr(handler, "handle"):
                    handle_method = getattr(handler, "handle", None)
                    if handle_method is not None and callable(handle_method):
                        result = handle_method(request)
                        return (
                            FlextResult[object].ok(result)
                            if not isinstance(result, FlextResult)
                            else result
                        )

                # Check if handler itself is callable
                if callable(handler):
                    result = handler(request)
                    return (
                        FlextResult[object].ok(result)
                        if not isinstance(result, FlextResult)
                        else result
                    )

                return FlextResult[object].fail(
                    f"Handler '{name}' does not implement handle method"
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

        def execute_with_timeout(
            self, name: str, request: object, timeout_seconds: float = 30.0
        ) -> FlextResult[object]:
            """Execute handler with timeout using advanced railway patterns."""
            return FlextResult.ok(None).with_timeout(
                timeout_seconds, lambda _: self.execute(name, request)
            )

        def execute_with_fallback(
            self, primary_name: str, request: object, *fallback_names: str
        ) -> FlextResult[object]:
            """Execute handler with fallback handlers using railway patterns."""
            return FlextUtilities.Reliability.with_fallback(
                lambda: self.execute(primary_name, request),
                *[
                    lambda: self.execute(fallback, request)
                    for fallback in fallback_names
                ],
            )

        def execute_batch(
            self,
            handler_requests: list[tuple[str, object]],
            fail_fast: bool = True,
        ) -> FlextResult[list[object]]:
            """Execute multiple handlers using advanced railway patterns."""
            return FlextResult.parallel_map(
                handler_requests,
                lambda item: self.execute(item[0], item[1]),
                fail_fast=fail_fast,
            )

        def register_with_validation(
            self,
            name: str,
            handler: object,
            validator: Callable[[object], FlextResult[None]] | None = None,
        ) -> FlextResult[None]:
            """Register handler with optional validation using railway patterns."""
            if validator:
                return validator(handler) >> (lambda _: self.register(name, handler))
            return self.register(name, handler)

    class Pipeline:
        """Advanced processing pipeline using monadic composition."""

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
            """Process data through pipeline using advanced railway pattern."""
            return FlextResult.pipeline(
                data, *[self._process_step(step) for step in self._steps]
            )

        def process_conditionally(
            self,
            data: object,
            condition: Callable[[object], bool],
        ) -> FlextResult[object]:
            """Process data conditionally using railway patterns."""
            return FlextResult.ok(data).when(condition) >> (self.process)

        def process_with_timeout(
            self,
            data: object,
            timeout_seconds: float = 30.0,
        ) -> FlextResult[object]:
            """Process data with timeout using advanced railway patterns."""
            return FlextResult.ok(data).with_timeout(timeout_seconds, self.process)

        def process_with_fallback(
            self,
            data: object,
            *fallback_pipelines: FlextProcessing.Pipeline,
        ) -> FlextResult[object]:
            """Process with fallback pipelines using railway patterns."""
            return FlextUtilities.Reliability.with_fallback(
                lambda: self.process(data),
                *[lambda: pipeline.process(data) for pipeline in fallback_pipelines],
            )

        def process_batch(
            self,
            data_items: list[object],
            fail_fast: bool = True,
        ) -> FlextResult[list[object]]:
            """Process batch of data using advanced railway patterns."""
            return FlextResult.parallel_map(
                data_items, self.process, fail_fast=fail_fast
            )

        def process_with_validation(
            self,
            data: object,
            *validators: Callable[[object], FlextResult[None]],
        ) -> FlextResult[object]:
            """Process with comprehensive validation pipeline."""
            return FlextResult.validate_all(data, *validators) >> (self.process)

        def _process_step(
            self, step: object
        ) -> Callable[[object], FlextResult[object]]:
            """Convert pipeline step to FlextResult-returning function."""

            def step_processor(current: object) -> FlextResult[object]:
                return FlextResult.from_exception(
                    lambda: self._execute_step(step, current)
                )

            return step_processor

        def _execute_step(self, step: object, current: object) -> object:
            """Execute a single pipeline step."""
            # Handle callable steps
            if callable(step):
                result = step(current)
                if isinstance(result, FlextResult):
                    return result.unwrap()  # Will raise if failed
                return result

            # Handle dictionary merging
            if isinstance(current, dict) and isinstance(step, dict):
                return {**current, **step}

            # Replace current data
            return step

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
        if hasattr(handler, "handle"):
            handle_method = getattr(handler, "handle", None)
            if handle_method is not None and callable(handle_method):
                return True
        return callable(handler)

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
                        handle_method = getattr(handler, "handle", None)
                        if handle_method is not None:
                            handler_result = handle_method(result)
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
