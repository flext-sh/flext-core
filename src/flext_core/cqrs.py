"""CQRS utilities aligned with the 1.0.0 unified dispatcher initiative.

These helpers sit on top of ``FlextBus``/``FlextDispatcher`` so downstream
packages can standardise command/query orchestration without custom buses.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


HandlerTypeLiteral = Literal[
    FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
    FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
]


class FlextCqrs:
    """Unified CQRS infrastructure with FlextModels integration.

    Provides result helpers, command/query operations, and decorators for CQRS patterns.
    Integrates with FlextModels for validation, FlextConfig for configuration,
    FlextUtilities for processing, and FlextConstants for defaults.
    """

    class Results:
        """FlextResult factory methods with FlextModels configuration support."""

        @staticmethod
        def success(
            data: object, config: FlextModels.CqrsConfig.Handler | None = None
        ) -> FlextResult[object]:
            """Create a success result with optional handler configuration.

            Args:
                data: The success data to wrap
                config: Optional handler configuration for metadata

            Returns:
                FlextResult[object]: Success result with the provided data

            """
            result = FlextResult[object].ok(data)

            # Add configuration metadata if provided
            if config:
                # Use proper metadata access pattern
                # Note: FlextResult doesn't support metadata on success results
                # Metadata would only be available on failure results via error_data
                # For success results, we skip metadata to maintain type safety
                pass

            return result

        @staticmethod
        def failure(
            message: str,
            *,
            error_code: str | None = None,
            error_data: FlextTypes.Core.Dict | None = None,
            config: FlextModels.CqrsConfig.Handler | None = None,
        ) -> FlextResult[object]:
            """Create a failure result with structured error handling.

            Args:
                message: Error message
                error_code: Optional specific error code
                error_data: Optional additional error context
                config: Optional handler configuration for metadata

            Returns:
                FlextResult[object]: Failure result with error details

            """
            final_error_code = error_code or FlextConstants.Cqrs.CQRS_OPERATION_FAILED

            # Enhance error data with configuration context
            enhanced_error_data = dict(error_data) if error_data else {}
            if config:
                enhanced_error_data.update({
                    "handler_id": config.handler_id,
                    "handler_name": config.handler_name,
                    "handler_type": config.handler_type,
                    "handler_metadata": config.metadata,
                })

            return FlextResult[object].fail(
                message,
                error_code=final_error_code,
                error_data=enhanced_error_data,
            )

    class Operations:
        """Command/Query factory methods using FlextModels validation."""

        @staticmethod
        def create_command(
            command_data: FlextTypes.Core.Dict,
            config: FlextModels.CqrsConfig.Handler | None = None,
        ) -> FlextResult[FlextModels.Command]:
            """Create a validated CQRS command with configuration.

            Args:
                command_data: Raw command data to validate
                config: Optional handler configuration

            Returns:
                FlextResult containing validated CqrsCommand

            """
            try:
                # Use FlextUtilities for ID generation if needed
                if "command_id" not in command_data:
                    command_data = {
                        **command_data,
                        "command_id": FlextUtilities.Generators.generate_id(),
                    }

                # Create validated command using FlextModels
                command = FlextModels.Command.model_validate(command_data)

                return FlextResult[FlextModels.Command].ok(command)

            except Exception as e:
                return FlextResult[FlextModels.Command].fail(
                    f"Command validation failed: {e!s}",
                    error_code=FlextConstants.Cqrs.COMMAND_VALIDATION_FAILED,
                    error_data={
                        "command_data": command_data,
                        "config": config.model_dump() if config else None,
                    },
                )

        @staticmethod
        def create_query(
            query_data: FlextTypes.Core.Dict,
            config: FlextModels.CqrsConfig.Handler | None = None,
        ) -> FlextResult[FlextModels.Query]:
            """Create a validated CQRS query with configuration.

            Args:
                query_data: Raw query data to validate
                config: Optional handler configuration

            Returns:
                FlextResult containing validated CqrsQuery

            """
            try:
                # Use FlextUtilities for ID generation if needed
                if "query_id" not in query_data:
                    query_data = {
                        **query_data,
                        "query_id": FlextUtilities.Generators.generate_id(),
                    }

                # Create validated query using FlextModels
                query = FlextModels.Query.model_validate(query_data)

                return FlextResult[FlextModels.Query].ok(query)

            except Exception as e:
                return FlextResult[FlextModels.Query].fail(
                    f"Query validation failed: {e!s}",
                    error_code=FlextConstants.Cqrs.QUERY_VALIDATION_FAILED,
                    error_data={
                        "query_data": query_data,
                        "config": config.model_dump() if config else None,
                    },
                )

        @staticmethod
        def create_handler_config(
            handler_type: HandlerTypeLiteral,
            *,
            handler_name: str | None = None,
            handler_id: str | None = None,
            config_overrides: FlextTypes.Core.Dict | None = None,
        ) -> FlextResult[FlextModels.CqrsConfig.Handler]:
            """Create a validated handler configuration.

            Args:
                handler_type: Type of handler (command or query)
                handler_name: Optional custom handler name
                handler_id: Optional custom handler ID
                config_overrides: Optional configuration overrides

            Returns:
                FlextResult containing validated handler configuration

            """
            try:
                # Use FlextUtilities for name/ID generation
                default_name = handler_name or f"{handler_type}_handler"
                default_id = (
                    handler_id
                    or f"handler_{handler_type}_{FlextUtilities.Generators.generate_id()}"
                )

                # Use FlextModels factory method with FlextConstants defaults
                handler_config = FlextModels.CqrsConfig.Handler.create_handler_config(
                    handler_type=handler_type,
                    default_name=default_name,
                    default_id=default_id,
                    handler_config=config_overrides,
                    command_timeout=FlextConstants.Cqrs.DEFAULT_TIMEOUT,
                    max_command_retries=FlextConstants.Cqrs.DEFAULT_RETRIES,
                )

                return FlextResult[FlextModels.CqrsConfig.Handler].ok(handler_config)

            except Exception as e:
                return FlextResult[FlextModels.CqrsConfig.Handler].fail(
                    f"Handler configuration creation failed: {e!s}",
                    error_code=FlextConstants.Cqrs.HANDLER_CONFIG_INVALID,
                    error_data={
                        "handler_type": handler_type,
                        "handler_name": handler_name,
                        "config_overrides": config_overrides,
                    },
                )

    class Decorators:
        """Enhanced decorators using FlextModels.CqrsConfig."""

        @staticmethod
        def command_handler[TCmd, TResult](
            command_type: type[TCmd],
            *,
            config: FlextModels.CqrsConfig.Handler | FlextTypes.Core.Dict | None = None,
        ) -> Callable[[Callable[[TCmd], TResult]], Callable[[TCmd], TResult]]:
            """Mark function as command handler with FlextModels configuration.

            Args:
                command_type: The command type this handler processes
                config: Optional handler configuration

            Returns:
                Decorator function that wraps the handler

            """

            def decorator(
                func: Callable[[TCmd], TResult],
            ) -> Callable[[TCmd], TResult]:
                # Create validated handler configuration
                handler_config_result = FlextCqrs.Operations.create_handler_config(
                    handler_type=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
                    handler_name=getattr(
                        func, "__name__", f"{command_type.__name__}Handler"
                    ),
                    config_overrides=config
                    if isinstance(config, dict)
                    else (config.model_dump() if config else None),
                )

                if handler_config_result.is_failure:
                    # Log the configuration error but continue with basic setup
                    handler_config = FlextModels.CqrsConfig.Handler(
                        handler_id=FlextUtilities.Generators.generate_id(),
                        handler_name=getattr(
                            func, "__name__", f"{command_type.__name__}Handler"
                        ),
                        handler_type=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
                        handler_mode=FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                    )
                else:
                    handler_config = handler_config_result.value

                # Create handler class from function using FlextHandlers
                class FunctionHandler(FlextHandlers[TCmd, TResult]):
                    def __init__(self) -> None:
                        super().__init__(
                            handler_mode=FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                            handler_name=handler_config.handler_name,
                            handler_config={"metadata": handler_config.metadata},
                        )

                    def handle(self, message: TCmd) -> FlextResult[TResult]:
                        """Handle command with enhanced error context.

                        Args:
                            message: The command to handle

                        Returns:
                            FlextResult containing the handler execution result

                        """
                        try:
                            result = func(message)
                            if isinstance(result, FlextResult):
                                # Result is already a FlextResult, just return it
                                return result
                            return FlextResult[TResult].ok(result)
                        except Exception as e:
                            return FlextResult[TResult].fail(
                                f"Command handler execution failed: {e!s}",
                                error_code=FlextConstants.Cqrs.COMMAND_PROCESSING_FAILED,
                                error_data={
                                    "command_type": command_type.__name__,
                                    "handler_id": handler_config.handler_id,
                                    "handler_name": handler_config.handler_name,
                                },
                            )

                # Create wrapper function with metadata
                def wrapper(command: TCmd) -> TResult:
                    return func(command)

                # Preserve the original function's metadata
                wrapper.__annotations__ = func.__annotations__
                wrapper.__name__ = func.__name__
                wrapper.__doc__ = func.__doc__

                # Store enhanced metadata in wrapper's __dict__ for type safety
                wrapper.__dict__.update({
                    "command_type": command_type,
                    "handler_instance": FunctionHandler(),
                    "handler_config": handler_config,
                    "flext_cqrs_decorator": True,
                })

                return wrapper

            return decorator

    class _ConfigurationHelper:
        """Private helper for FlextConfig integration."""

        @staticmethod
        def get_default_cqrs_config() -> FlextResult[FlextModels.CqrsConfig.Bus]:
            """Get CQRS bus configuration from FlextConfig.

            Returns:
                FlextResult containing validated bus configuration

            """
            try:
                config_instance = FlextConfig.get_global_instance()
                # Create CQRS config from FlextConfig fields
                bus_config = FlextModels.CqrsConfig.Bus.create_bus_config({
                    "timeout_seconds": config_instance.dispatcher_timeout_seconds,
                    "enable_metrics": config_instance.dispatcher_enable_metrics,
                    "enable_logging": config_instance.dispatcher_enable_logging,
                })
                return FlextResult[FlextModels.CqrsConfig.Bus].ok(bus_config)
            except Exception:
                # Fallback to default configuration
                default_config = FlextModels.CqrsConfig.Bus.create_bus_config(None)
                return FlextResult[FlextModels.CqrsConfig.Bus].ok(default_config)

    class _ProcessingHelper:
        """Private helper for FlextUtilities integration."""

        @staticmethod
        def generate_handler_id(base_name: str) -> str:
            """Generate a unique handler ID using FlextUtilities.

            Args:
                base_name: Base name for the handler

            Returns:
                Generated unique handler ID

            """
            # Manual concatenation since generate_id() doesn't accept prefix
            unique_id = FlextUtilities.Generators.generate_id()
            return f"handler_{base_name}_{unique_id}"

        @staticmethod
        def derive_command_type(class_name: str) -> FlextResult[str]:
            """Derive command type from class name using manual conversion.

            Args:
                class_name: The class name to derive from

            Returns:
                FlextResult containing derived command type

            """
            try:
                # Remove 'Command' suffix and convert to snake_case manually
                base_name = class_name.removesuffix("Command")

                # Manual conversion from PascalCase to snake_case
                snake_case = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base_name)
                snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake_case).lower()

                return FlextResult[str].ok(snake_case)
            except Exception as e:
                return FlextResult[str].fail(
                    f"Failed to derive command type from {class_name}: {e!s}",
                    error_code=FlextConstants.Errors.PROCESSING_ERROR,
                )

    class _ErrorHelper:
        """Private helper for FlextExceptions handling."""

        @staticmethod
        def handle_validation_error(
            error: Exception, context: FlextTypes.Core.Dict
        ) -> FlextResult[object]:
            """Handle validation errors with structured context.

            Args:
                error: The validation exception
                context: Additional error context

            Returns:
                FlextResult with structured error information

            """
            # Check if it's a Pydantic validation error
            if hasattr(error, "errors") and callable(getattr(error, "errors", None)):
                validation_errors = getattr(error, "errors")()
                return FlextResult[object].fail(
                    f"Validation failed: {error!s}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    error_data={
                        "context": context,
                        "validation_errors": validation_errors,
                        "error_type": type(error).__name__,
                    },
                )

            # Generic error handling
            return FlextResult[object].fail(
                f"Processing error: {error!s}",
                error_code=FlextConstants.Errors.PROCESSING_ERROR,
                error_data={
                    "context": context,
                    "error_type": type(error).__name__,
                },
            )


__all__: list[str] = [
    "FlextCqrs",
]
