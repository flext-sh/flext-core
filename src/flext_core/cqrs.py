"""CQRS utilities aligned with the 1.0.0 unified dispatcher initiative.

These helpers sit on top of ``FlextBus``/``FlextDispatcher`` so downstream
packages can standardise command/query orchestration without custom buses.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextCqrs:
    """CQRS utilities for Command Query Responsibility Segregation patterns.

    Provides result helpers and decorators for CQRS patterns. Factory methods
    for command buses and handlers are available directly from FlextBus.
    """

    class Results:
        """Result helper methods for FlextResult patterns in CQRS context."""

        @staticmethod
        def success(data: object) -> FlextResult[object]:
            """Create a success result."""
            return FlextResult[object].ok(data)

        @staticmethod
        def failure(
            error: str,
            error_code: str | None = None,
            error_data: FlextTypes.Core.Dict | None = None,
        ) -> FlextResult[object]:
            """Create a failure result."""
            return FlextResult[object].fail(
                error,
                error_code=error_code
                or FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                error_data=error_data,
            )

    class Decorators:
        """Decorator utilities for CQRS components."""

        @staticmethod
        def command_handler[TCmd, TResult](
            command_type: type[TCmd],
        ) -> Callable[[Callable[[TCmd], TResult]], Callable[[TCmd], TResult]]:
            """Mark function as command handler."""

            def decorator(
                func: Callable[[TCmd], TResult],
            ) -> Callable[[TCmd], TResult]:
                metadata_payload = {
                    "command_type": getattr(
                        command_type,
                        "__name__",
                        str(command_type),
                    ),
                }

                # Create handler class from function
                class FunctionHandler(FlextHandlers[TCmd, TResult]):
                    def __init__(self) -> None:
                        super().__init__(
                            handler_mode="command",
                            handler_name=getattr(
                                func,
                                "__name__",
                                self.__class__.__name__,
                            ),
                            handler_config={"metadata": metadata_payload},
                        )

                    def handle(self, message: TCmd) -> FlextResult[TResult]:
                        result = func(message)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[TResult]", result)
                        return FlextResult[TResult].ok(result)

                # Create wrapper function with metadata
                def wrapper(command: TCmd) -> TResult:
                    return func(command)

                # Preserve the original function's return type
                wrapper.__annotations__ = func.__annotations__

                # Store metadata in wrapper's __dict__ for type safety
                wrapper.__dict__["command_type"] = command_type
                wrapper.__dict__["handler_instance"] = FunctionHandler()

                return wrapper

            return decorator


__all__: list[str] = [
    "FlextCqrs",
]
