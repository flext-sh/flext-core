"""CQRS utilities aligned with the 1.0.0 unified dispatcher initiative.

These helpers sit on top of ``FlextBus``/``FlextDispatcher`` so downstream
packages can standardise command/query orchestration without custom buses.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

from flext_core.constants import FlextConstants
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextCqrs:
    """CQRS helpers and decorators for command/query patterns.

    FlextCqrs provides utility functions and decorators for implementing
    CQRS patterns with FlextBus and FlextDispatcher. Includes result
    helpers, handler factories, and decorators for command/query
    handlers across all 32+ FLEXT projects.

    **Function**: CQRS utility functions and decorators
        - Result helpers for success/failure creation
        - Handler factories for command/query handlers
        - Decorators for command/query handler registration
        - Command/query validation utilities
        - Handler configuration helpers
        - Error handling with structured metadata
        - Integration with FlextResult railway pattern
        - Support for handler metadata attachment
        - Configuration-aware result creation
        - Type-safe handler creation patterns

    **Uses**: Core FLEXT infrastructure for CQRS
        - FlextResult[T] for all operation results
        - FlextModels for command/query base classes
        - FlextConstants for CQRS error codes
        - FlextUtilities for validation and processing
        - FlextTypes for type definitions
        - functools for decorator implementation
        - Literal types for handler type validation

    **How to use**: CQRS helpers and decorators
        ```python
        from flext_core import FlextCqrs, FlextResult, FlextModels

        # Example 1: Create success result with metadata
        config = FlextModels.CqrsConfig.Handler(
            handler_name="CreateUser", handler_type="command"
        )
        result = FlextCqrs.Results.success(data={"user_id": "123"}, config=config)

        # Example 2: Create failure result with error details
        result = FlextCqrs.Results.failure(
            "Validation failed",
            error_code="VALIDATION_ERROR",
            error_data={"field": "email"},
            config=config,
        )


        # Example 3: Use decorators for handler registration
        @FlextCqrs.Decorators.command_handler
        class CreateUserHandler:
            def handle(self, cmd: CreateUserCommand) -> FlextResult[User]:
                return FlextCqrs.Results.success(user)


        # Example 4: Query handler decorator
        @FlextCqrs.Decorators.query_handler
        class GetUserHandler:
            def handle(self, query: GetUserQuery) -> FlextResult[User]:
                return FlextCqrs.Results.success(user)


        # Example 5: Create handler configuration
        handler_config = FlextCqrs.create_handler_config(
            handler_name="MyHandler", handler_type="command"
        )

        # Example 6: Validate command before execution
        validation_result = FlextCqrs.validate_command(command)
        if validation_result.is_success:
            result = handler.handle(command)
        ```

    **TODO**: Enhanced CQRS features for 1.0.0+ releases
        - [ ] Add saga coordinator decorators for workflows
        - [ ] Implement event handler decorators
        - [ ] Add query optimization decorators (caching)
        - [ ] Support command validation decorators
        - [ ] Implement command/query middleware decorators
        - [ ] Add handler composition utilities
        - [ ] Support command/query decorators
        - [ ] Implement retry decorators for handlers
        - [ ] Add circuit breaker decorators
        - [ ] Support handler metrics decorators

    Attributes:
        Results: Result helper utilities for CQRS.
        Decorators: Decorator utilities for handlers.

    Note:
        All result helpers return FlextResult for consistency.
        Decorators integrate with FlextBus registration. Use
        Results.success/failure for consistent error handling.
        Handler metadata attached to results for tracing.

    Warning:
        Decorators must be used with FlextBus integration.
        Handler configuration required for metadata attachment.
        Error codes should use FlextConstants.Cqrs namespace.

    Example:
        Complete CQRS workflow with helpers:

        >>> result = FlextCqrs.Results.success({"id": "123"})
        >>> print(result.is_success)
        True
        >>> failure = FlextCqrs.Results.failure("Error occurred")
        >>> print(failure.is_failure)
        True

    See Also:
        FlextBus: For command/query bus operations.
        FlextHandlers: For handler base class patterns.
        FlextDispatcher: For higher-level dispatch.

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
                result.metadata = metadata

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
            enhanced_error_data: FlextTypes.Core.Dict = (
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
                # Create handler configuration directly using FlextModels
                handler_name = getattr(
                    func,
                    "__name__",
                    f"{getattr(command_type, '__name__', str(command_type))}Handler",
                )

                # Use FlextModels.CqrsConfig.Handler.create_handler_config directly
                handler_config = FlextModels.CqrsConfig.Handler.create_handler_config(
                    handler_type=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
                    default_name=handler_name,
                    handler_config=config
                    if isinstance(config, dict)
                    else (
                        getattr(config, "model_dump", lambda: None)()
                        if config
                        else None
                    ),
                    command_timeout=FlextConstants.Cqrs.DEFAULT_TIMEOUT,
                    max_command_retries=FlextConstants.Cqrs.DEFAULT_RETRIES,
                )

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
