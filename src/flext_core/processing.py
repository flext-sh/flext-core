"""Processing helpers that complement the FLEXT-Core 1.0.0 dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.utilities import FlextUtilities


class FlextProcessing:
    """Processing convenience namespace aligned with dispatcher workflows.

    Registries, pipelines, and handler helpers mirror the ergonomics promoted in
    the modernization plan so supporting packages can compose around
    ``FlextDispatcher`` without bespoke glue code.
    """

    class Config:
        """Configuration settings for FlextProcessing using FlextConfig defaults."""

        @classmethod
        def get_default_timeout(cls) -> float:
            """Get default timeout from configuration or constants.

            Returns:
                float: Default timeout in seconds, from configuration or
                fallback to constants.

            """
            try:
                config = FlextConfig.get_global_instance()
                return float(
                    getattr(config, "default_timeout", FlextConstants.Defaults.TIMEOUT)
                )
            except Exception:
                return float(FlextConstants.Defaults.TIMEOUT)

        @classmethod
        def get_max_batch_size(cls) -> int:
            """Get maximum batch size from configuration or constants.

            Returns:
                int: Maximum batch size configured or the default constant.

            """
            try:
                config = FlextConfig.get_global_instance()
                return int(
                    getattr(
                        config,
                        "max_batch_size",
                        FlextConstants.Performance.DEFAULT_BATCH_SIZE,
                    )
                )
            except Exception:
                return FlextConstants.Performance.DEFAULT_BATCH_SIZE

        @classmethod
        def get_max_handlers(cls) -> int:
            """Get maximum number of handlers from configuration or constants.

            Returns:
                int: Maximum handlers configured or the default constant.

            """
            try:
                config = FlextConfig.get_global_instance()
                return int(
                    getattr(
                        config, "max_handlers", FlextConstants.Container.MAX_SERVICES
                    )
                )
            except Exception:
                return FlextConstants.Container.MAX_SERVICES

    class Handler:
        """Minimal handler base returning modernization-compliant results."""

        def handle(self, request: object) -> FlextResult[object]:
            """Handle a request.

            Returns:
                FlextResult[object]: A successful FlextResult wrapping handler
                output.

            """
            return FlextResult[object].ok(f"Base handler processed: {request}")

    class HandlerRegistry:
        """Registry managing named handler instances for dispatcher pilots."""

        def __init__(self) -> None:
            """Initialize handler registry."""
            self._handlers: dict[str, object] = {}

        def register(
            self, registration: FlextModels.HandlerRegistration
        ) -> FlextResult[None]:
            """Register a handler using Pydantic model validation.

            Returns:
                FlextResult[None]: Success when registration is stored or a
                failed FlextResult with a validation/exists error.

            """
            if registration.name in self._handlers:
                return FlextResult[None].fail(
                    f"Handler '{registration.name}' already registered",
                    error_code=FlextConstants.Errors.ALREADY_EXISTS,
                )

            # Check handler registry size limits
            max_handlers = FlextProcessing.Config.get_max_handlers()
            if len(self._handlers) >= max_handlers:
                return FlextResult[None].fail(
                    f"Handler registry full: {len(self._handlers)}/{max_handlers} handlers registered",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate handler safety
            if not FlextProcessing.is_handler_safe(registration.handler):
                return FlextResult[None].fail(
                    f"Handler '{registration.name}' is not safe (must have handle method or be callable)",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate handler using the model's built-in validation
            self._handlers[registration.name] = registration.handler
            return FlextResult[None].ok(None)

        def get(self, name: str) -> FlextResult[object]:
            """Get a handler.

            Returns:
                FlextResult[object]: The handler instance wrapped in a
                successful FlextResult, or a failed result if not found.

            """
            if name not in self._handlers:
                return FlextResult[object].fail(
                    f"Handler '{name}' not found", error_code="NOT_FOUND_ERROR"
                )
            return FlextResult[object].ok(self._handlers[name])

        def execute(self, name: str, request: object) -> FlextResult[object]:
            """Execute a handler by name using railway pattern.

            Returns:
                FlextResult[object]: Result of handler execution or failure
                indicating handler not found or execution error.

            """
            return self.get(name).flat_map(
                lambda handler: self._execute_handler_safely(handler, request, name)
            )

        def _execute_handler_safely(
            self, handler: object, request: object, name: str
        ) -> FlextResult[object]:
            """Execute handler with proper method resolution and error handling.

            Returns:
                FlextResult[object]: The result returned by the handler, or a
                failed FlextResult with a ProcessingError on exception.

            """
            try:
                # Check for handle method first
                if hasattr(handler, FlextConstants.Mixins.METHOD_HANDLE):
                    handle_method = getattr(
                        handler, FlextConstants.Mixins.METHOD_HANDLE, None
                    )
                    if handle_method is not None and callable(handle_method):
                        result = handle_method(request)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[object]", result)
                        return FlextResult[object].ok(result)

                # Check if handler itself is callable
                if callable(handler):
                    result = handler(request)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

                return FlextResult[object].fail(
                    f"Handler '{name}' does not implement handle method",
                    error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
                )
            except Exception as e:
                return FlextResult[object].fail(
                    f"Handler execution failed: {e}",
                    error_code=FlextConstants.Errors.PROCESSING_ERROR,
                )

        def count(self) -> int:
            """Get the number of registered handlers.

            Returns:
                int: The number of handlers registered.

            """
            return len(self._handlers)

        def exists(self, name: str) -> bool:
            """Check if a handler exists.

            Returns:
                bool: True if a handler with `name` is registered.

            """
            return name in self._handlers

        def get_optional(self, name: str) -> object | None:
            """Get a handler optionally, returning None if not found.

            Returns:
                object | None: Handler instance or None when not registered.

            """
            return self._handlers.get(name)

        def execute_with_timeout(
            self, config: FlextModels.HandlerExecutionConfig
        ) -> FlextResult[object]:
            """Execute handler with timeout using HandlerExecutionConfig model.

            Returns:
                FlextResult[object]: The result of handler execution wrapped in
                a FlextResult, possibly a failure on timeout.

            """
            timeout_seconds = getattr(
                config, "timeout_seconds", FlextProcessing.Config.get_default_timeout()
            )
            return (
                FlextResult[object]
                .ok(None)
                .with_timeout(
                    timeout_seconds,
                    lambda _: self.execute(config.handler_name, config.input_data),
                )
            )

        def execute_with_fallback(
            self, config: FlextModels.HandlerExecutionConfig
        ) -> FlextResult[object]:
            """Execute handler with fallback handlers using HandlerExecutionConfig model.

            Returns:
                FlextResult[object]: The first successful handler result or the
                final failure if all fallbacks fail.

            """
            return FlextUtilities.Reliability.with_fallback(
                lambda: self.execute(config.handler_name, config.input_data),
                *[
                    (
                        lambda fallback=fallback: self.execute(
                            fallback, config.input_data
                        )
                    )
                    for fallback in config.fallback_handlers
                ],
            )

        def execute_batch(
            self, config: FlextModels.BatchProcessingConfig
        ) -> FlextResult[list[object]]:
            """Execute multiple handlers using BatchProcessingConfig model.

            Returns:
                FlextResult[list[object]]: List of handler results or a failed
                FlextResult if validation or batch processing fails.

            """
            # Pydantic validation is automatic when the model is created
            # No need to call validate_batch() manually

            # Validate batch size limits
            max_batch_size = FlextProcessing.Config.get_max_batch_size()
            if len(config.data_items) > max_batch_size:
                return FlextResult[list[object]].fail(
                    f"Batch size {len(config.data_items)} exceeds maximum {max_batch_size}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Convert data_items to handler execution tuples
            # Assume data_items contains tuples of (handler_name, request_data)
            handler_requests: list[tuple[str, object]] = []
            expected_tuple_length = FlextConstants.Performance.EXPECTED_TUPLE_LENGTH
            for item in config.data_items:
                if (
                    isinstance(item, tuple)
                    and len(cast("tuple[object, ...]", item)) == expected_tuple_length
                ):
                    # Type assertion for tuple elements
                    handler_name, request_data = cast("tuple[str, object]", item)
                    handler_requests.append((handler_name, request_data))
                else:
                    return FlextResult[list[object]].fail(
                        "Each data item must be a tuple of (handler_name, request_data)",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            return FlextResult.parallel_map(
                handler_requests,
                lambda item: self.execute(item[0], item[1]),
                fail_fast=not config.continue_on_error,
            )

        def register_with_validation(
            self,
            registration: FlextModels.HandlerRegistration,
            validator: Callable[[object], FlextResult[None]] | None = None,
        ) -> FlextResult[None]:
            """Register handler with optional validation using HandlerRegistration model.

            Returns:
                FlextResult[None]: Result of registration, success or failure.

            """
            if validator:
                return validator(registration.handler) >> (
                    lambda _: self.register(registration)
                )
            return self.register(registration)

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
            """Process data through pipeline using advanced railway pattern.

            Returns:
                FlextResult[object]: Result of pipeline processing.

            """
            return FlextResult.pipeline(
                data, *[self._process_step(step) for step in self._steps]
            )

        def process_conditionally(
            self,
            request: FlextModels.ProcessingRequest,
            condition: Callable[[object], bool],
        ) -> FlextResult[object]:
            """Process data conditionally using railway patterns.

            Returns:
                FlextResult[object]: Result of conditional processing.

            """
            return FlextResult[dict[str, object]].ok(request.data).when(condition) >> (
                cast(
                    "Callable[[dict[str, object]], FlextResult[object]]",
                    lambda data: self.process(
                        FlextModels.ProcessingRequest(
                            data=cast("dict[str, object]", data),
                            context=request.context,
                            timeout_seconds=request.timeout_seconds,
                        )
                    ),
                )
            )

        def process_with_timeout(
            self, request: FlextModels.ProcessingRequest
        ) -> FlextResult[object]:
            """Process data with timeout using ProcessingRequest model.

            Returns:
                FlextResult[object]: Result of processing or timeout error.

            """
            timeout_seconds = getattr(
                request, "timeout_seconds", FlextProcessing.Config.get_default_timeout()
            )

            # Validate timeout bounds
            if timeout_seconds < FlextConstants.Container.MIN_TIMEOUT_SECONDS:
                return FlextResult[object].fail(
                    f"Timeout {timeout_seconds} is below minimum {FlextConstants.Container.MIN_TIMEOUT_SECONDS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            if timeout_seconds > FlextConstants.Container.MAX_TIMEOUT_SECONDS:
                return FlextResult[object].fail(
                    f"Timeout {timeout_seconds} exceeds maximum {FlextConstants.Container.MAX_TIMEOUT_SECONDS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return (
                FlextResult[object]
                .ok(request.data)
                .with_timeout(timeout_seconds, self.process)
            )

        def process_with_fallback(
            self,
            request: FlextModels.ProcessingRequest,
            *fallback_pipelines: FlextProcessing.Pipeline,
        ) -> FlextResult[object]:
            """Process with fallback pipelines using ProcessingRequest model.

            Returns:
                FlextResult[object]: Result from the first successful pipeline or
                the final failure.

            """
            return FlextUtilities.Reliability.with_fallback(
                lambda: self.process(request.data),
                *[
                    (lambda pipeline=pipeline: pipeline.process(request.data))
                    for pipeline in fallback_pipelines
                ],
            )

        def process_batch(
            self, config: FlextModels.BatchProcessingConfig
        ) -> FlextResult[list[object]]:
            """Process batch of data using validated BatchProcessingConfig model.

            Args:
                config: BatchProcessingConfig model with data items and processing options

            Returns:
                FlextResult[list[object]]: List of processed data items or a failure.

            """
            # Pydantic validation is automatic when the model is created
            # No need to call validate_batch() manually

            # Validate batch size limits
            max_batch_size = FlextProcessing.Config.get_max_batch_size()
            if len(config.data_items) > max_batch_size:
                return FlextResult[list[object]].fail(
                    f"Batch size {len(config.data_items)} exceeds maximum {max_batch_size}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Process all data items using parallel processing
            return FlextResult.parallel_map(
                config.data_items, self.process, fail_fast=not config.continue_on_error
            )

        def process_with_validation(
            self,
            request: FlextModels.ProcessingRequest,
            *validators: Callable[[object], FlextResult[None]],
        ) -> FlextResult[object]:
            """Process with comprehensive validation pipeline using ProcessingRequest model.

            Returns:
                FlextResult[object]: Result of validation-then-processing or processing directly.

            """
            # Apply validation if enabled in the request
            if request.enable_validation:
                return FlextResult.validate_all(request.data, *validators) >> (
                    self.process
                )
            return self.process(request.data)

        def _process_step(
            self, step: object
        ) -> Callable[[object], FlextResult[object]]:
            """Convert pipeline step to FlextResult-returning function.

            Returns:
                Callable[[object], FlextResult[object]]: Adapter that wraps step execution.

            """

            def step_processor(current: object) -> FlextResult[object]:
                return cast(
                    "FlextResult[object]",
                    FlextResult.from_exception(
                        lambda: self._execute_step(step, current)
                    ),
                )

            return step_processor

        def _execute_step(self, step: object, current: object) -> object:
            """Execute a single pipeline step.

            Returns:
                object: Result of step execution; may be a FlextResult unwrapped.

            """
            # Handle callable steps
            if callable(step):
                result = step(current)
                if isinstance(result, FlextResult):
                    if result.is_failure:
                        # Explicitly raise the error to be caught by from_exception wrapper
                        msg = f"Pipeline step failed: {result.error}"
                        raise RuntimeError(msg)
                    step_result = result.value_or_none
                    if step_result is None:
                        msg = "Pipeline step returned None despite success"
                        raise RuntimeError(
                            msg
                        )
                    return cast("object", step_result)
                return result

            # Handle dictionary merging
            if isinstance(current, dict) and isinstance(step, dict):
                merged_dict: dict[str, object] = {**current, **step}
                return merged_dict

            # Replace current data
            return step

    # Factory methods for convenience
    @staticmethod
    def create_handler_registry() -> HandlerRegistry:
        """Create a new handler registry.

        Returns:
            HandlerRegistry: A fresh handler registry instance.

        """
        return FlextProcessing.HandlerRegistry()

    @staticmethod
    def create_pipeline() -> Pipeline:
        """Create a new processing pipeline.

        Returns:
            Pipeline: A new processing pipeline instance.

        """
        return FlextProcessing.Pipeline()

    @staticmethod
    def is_handler_safe(handler: object) -> bool:
        """Check if a handler is safe (has handle method or is callable).

        Returns:
            bool: True if handler is safe to execute.

        """
        if hasattr(handler, FlextConstants.Mixins.METHOD_HANDLE):
            handle_method = getattr(handler, FlextConstants.Mixins.METHOD_HANDLE, None)
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
                """Handle request.

                Returns:
                    FlextResult[str]: Successful result wrapping a string message.

                """
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
                """Get handler by name.

                Returns:
                    object | None: The handler instance or None if not found.

                """
                return self._handlers.get(name)

            def get_optional(self, name: str) -> object | None:
                """Get handler optionally, returning None if not found.

                Returns:
                    object | None: The handler instance or None when not present.

                """
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
                """Handle request by executing all handlers in chain.

                Returns:
                    FlextResult[object]: Result after processing through the chain.

                """
                result = request
                for handler in self._handlers:
                    handle_method_name = FlextConstants.Mixins.METHOD_HANDLE
                    if hasattr(handler, handle_method_name):
                        handle_method = getattr(handler, handle_method_name, None)
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
            """Chainable handler for examples - Application.Handler protocol implementation."""

            def __init__(self, name: str) -> None:
                """Initialize chainable handler with name."""
                self.name = name

            def handle(self, request: object) -> FlextResult[object]:
                """Handle request in chain.

                Returns:
                    FlextResult[object]: Handler output wrapped in FlextResult.

                """
                result = f"Chain handled by {self.name}: {request}"
                return FlextResult[object].ok(result)

            def can_handle(self, message_type: object) -> bool:
                """Check if handler can process this message type.

                Args:
                    message_type: The message type to check

                Returns:
                    bool: True since this is a generic example handler

                """
                return True

            def execute(self, message: object) -> FlextResult[object]:
                """Execute the handler with the given message.

                Args:
                    message: The input message to execute

                Returns:
                    FlextResult[object]: Execution result

                """
                return self.handle(message)

            def validate_command(self, command: object) -> FlextResult[None]:
                """Validate a command message.

                Args:
                    command: The command to validate

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                if command is None:
                    return FlextResult[None].fail("Command cannot be None")
                return FlextResult[None].ok(None)

            def validate_query(self, query: object) -> FlextResult[None]:
                """Validate a query message.

                Args:
                    query: The query to validate

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                if query is None:
                    return FlextResult[None].fail("Query cannot be None")
                return FlextResult[None].ok(None)

            @property
            def handler_name(self) -> str:
                """Get the handler name.

                Returns:
                    str: Handler name

                """
                return self.name

            @property
            def mode(self) -> str:
                """Get the handler mode (command/query).

                Returns:
                    str: Handler mode

                """
                return "command"
