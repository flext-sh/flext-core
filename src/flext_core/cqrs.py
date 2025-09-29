"""CQRS utilities aligned with the 1.0.0 unified dispatcher initiative.

These helpers sit on top of ``FlextBus``/``FlextDispatcher`` so downstream
packages can standardise command/query orchestration without custom buses.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from flext_core.constants import FlextConstants
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

HandlerTypeLiteral = Literal["command", "query"]


class FlextCqrs:
    """Unified CQRS infrastructure using existing flext-core functionality.

    Provides simplified access to CQRS operations by leveraging the existing
    flext-core infrastructure: FlextModels for validation, FlextUtilities for
    processing, FlextHandlers for handler creation, FlextConfig for configuration,
    and FlextResult for railway-oriented error handling.
    """

    class Results:
        """CQRS Results using existing FlextResult functionality."""

        @staticmethod
        def success(data: object, config: object | None = None) -> FlextResult[object]:
            """Create a success result with optional handler configuration metadata.

            Args:
                data: The success data to wrap
                config: Optional handler configuration for metadata

            Returns:
                FlextResult[object]: Success result with the provided data

            """
            result: FlextResult[object] = FlextResult[object].ok(data)

            # Add configuration metadata if provided for test compatibility
            if config and hasattr(config, "handler_id"):
                metadata = {
                    "handler_id": getattr(config, "handler_id", None),
                    "handler_name": getattr(config, "handler_name", None),
                    "handler_type": getattr(config, "handler_type", None),
                }
                setattr(result, "_metadata", metadata)

            return result

        @staticmethod
        def failure(
            message: str,
            *,
            error_code: str | None = None,
            error_data: FlextTypes.Core.Dict | None = None,
            config: object | None = None,
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
            enhanced_error_data: dict[str, object] = (
                dict(error_data) if error_data else {}
            )
            if config and hasattr(config, "handler_id"):
                enhanced_error_data.update({
                    "handler_id": getattr(config, "handler_id", None),
                    "handler_name": getattr(config, "handler_name", None),
                    "handler_type": getattr(config, "handler_type", None),
                    "handler_metadata": getattr(config, "metadata", None),
                })

            return FlextResult[object].fail(
                message,
                error_code=final_error_code,
                error_data=enhanced_error_data,
            )

    class Operations:
        """CQRS Operations using existing flext-core functionality."""

        @staticmethod
        def create_command(
            command_data: FlextTypes.Core.Dict,
            config: object | None = None,
        ) -> FlextResult[FlextModels.Command]:
            """Create a validated CQRS command using FlextModels.

            Args:
                command_data: Raw command data to validate
                config: Optional handler configuration (for metadata)

            Returns:
                FlextResult containing validated Command

            """
            try:
                # Use FlextUtilities for ID generation if needed
                if "command_id" not in command_data:
                    command_data = {
                        **command_data,
                        "command_id": FlextUtilities.generate_id(),
                    }

                # Use FlextModels for validation
                command = FlextModels.Command.model_validate(command_data)
                return FlextResult[FlextModels.Command].ok(command)

            except Exception as e:
                return FlextResult[FlextModels.Command].fail(
                    f"{FlextConstants.Messages.VALIDATION_FAILED}: Command validation failed: {e!s}",
                    error_code=FlextConstants.Cqrs.COMMAND_VALIDATION_FAILED,
                    error_data={
                        "command_data": command_data,
                        "config": getattr(config, "model_dump", lambda: None)()
                        if config
                        else None,
                    },
                )

        @staticmethod
        def create_query(
            query_data: FlextTypes.Core.Dict,
            config: object | None = None,
        ) -> FlextResult[FlextModels.Query]:
            """Create a validated CQRS query using FlextModels.

            Args:
                query_data: Raw query data to validate
                config: Optional handler configuration (for metadata)

            Returns:
                FlextResult containing validated Query

            """
            # 🚨 DUPLICATION: FlextModels.Query.validate_query already performs this validation and ID generation.
            try:
                # Use FlextUtilities for ID generation if needed
                if "query_id" not in query_data:
                    query_data = {
                        **query_data,
                        "query_id": FlextUtilities.generate_id(),
                    }

                # Use FlextModels for validation
                query = FlextModels.Query.model_validate(query_data)
                return FlextResult[FlextModels.Query].ok(query)

            except Exception as e:
                return FlextResult[FlextModels.Query].fail(
                    f"{FlextConstants.Messages.VALIDATION_FAILED} (query): {e!s}",
                    error_code=FlextConstants.Cqrs.QUERY_VALIDATION_FAILED,
                    error_data={
                        "query_data": query_data,
                        "config": getattr(config, "model_dump", lambda: None)()
                        if config
                        else None,
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
            """Create a validated handler configuration using FlextModels.

            Args:
                handler_type: Type of handler (command or query)
                handler_name: Optional custom handler name
                handler_id: Optional custom handler ID
                config_overrides: Optional configuration overrides

            Returns:
                FlextResult containing validated handler configuration

            """
            # 🚨 DUPLICATION: This method is a thin wrapper over FlextModels.CqrsConfig.Handler.create_handler_config.
            try:
                # Use FlextUtilities for name/ID generation
                default_name = handler_name or f"{handler_type}_handler"
                default_id = (
                    handler_id
                    or f"handler_{handler_type}_{FlextUtilities.generate_id()}"
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
        """CQRS Decorators using existing FlextHandlers functionality."""

        @staticmethod
        def command_handler[TCmd, TResult](
            command_type: type[TCmd],
            *,
            config: object | FlextTypes.Core.Dict | None = None,
        ) -> Callable[[Callable[[TCmd], TResult]], Callable[[TCmd], TResult]]:
            """Mark function as command handler using FlextHandlers.

            Args:
                command_type: The command type this handler processes
                config: Optional handler configuration

            Returns:
                Decorator function that wraps the handler

            """

            def decorator(
                func: Callable[[TCmd], TResult],
            ) -> Callable[[TCmd], TResult]:
                # 🚨 BEHAVIOUR GAP: Despite the docstring, this decorator never produces a FlextHandlers instance.
                # Create validated handler configuration
                handler_config_result = FlextCqrs.Operations.create_handler_config(
                    handler_type=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
                    handler_name=getattr(
                        func,
                        "__name__",
                        f"{getattr(command_type, '__name__', str(command_type))}Handler",
                    ),
                    config_overrides=config
                    if isinstance(config, dict)
                    else (
                        getattr(config, "model_dump", lambda: None)()
                        if config
                        else None
                    ),
                )

                if handler_config_result.is_failure:
                    # Create basic configuration if validation failed
                    handler_config = FlextModels.CqrsConfig.Handler(
                        handler_id=FlextUtilities.generate_id(),
                        handler_name=getattr(
                            func,
                            "__name__",
                            f"{getattr(command_type, '__name__', str(command_type))}Handler",
                        ),
                        handler_type="command",
                        handler_mode="command",
                    )
                else:
                    handler_config = handler_config_result.value

                # Create wrapper function with metadata
                def wrapper(command: TCmd) -> TResult:
                    # 🚨 BEHAVIOUR GAP: No FlextHandlers plumbing or validation is invoked here; this is a plain passthrough.
                    return func(command)

                # Preserve the original function's metadata
                wrapper.__annotations__ = func.__annotations__
                wrapper.__name__ = func.__name__
                wrapper.__doc__ = func.__doc__

                # Store enhanced metadata in wrapper's __dict__ for type safety
                wrapper.__dict__.update({
                    "command_type": command_type,
                    "handler_config": handler_config,
                    "flext_cqrs_decorator": True,
                })

                return wrapper

            return decorator


__all__: list[str] = [
    "FlextCqrs",
]
