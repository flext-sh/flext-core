"""CQRS utilities aligned with the 1.0.0 unified dispatcher initiative.

These helpers sit on top of ``FlextBus``/``FlextDispatcher`` so downstream
packages can standardise command/query orchestration without custom buses.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from flext_core.bus import FlextBus
from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextCqrs:
    """CQRS utilities for Command Query Responsibility Segregation patterns."""

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

    class Factories:
        """Factory methods for creating CQRS components."""

        @staticmethod
        def create_command_bus(
            bus_config: FlextModels.CqrsConfig.Bus | dict[str, object] | None = None,
            *,
            enable_middleware: bool = True,
            enable_metrics: bool = True,
            enable_caching: bool = True,
            execution_timeout: int = 30,
            max_cache_size: int = 1000,
            implementation_path: str = "flext_core.bus:FlextBus",
        ) -> FlextBus:
            """Create a new command bus instance with validated configuration."""
            config_model = FlextModels.CqrsConfig.create_bus_config(
                bus_config,
                enable_middleware=enable_middleware,
                enable_metrics=enable_metrics,
                enable_caching=enable_caching,
                execution_timeout=execution_timeout,
                max_cache_size=max_cache_size,
                implementation_path=implementation_path,
            )

            if config_model.implementation_path != "flext_core.bus:FlextBus":
                message = "Unsupported command bus implementation"
                raise ValueError(message)

            config_payload = config_model.model_dump(exclude={"implementation_path"})
            return FlextBus.create_command_bus(bus_config=config_payload)

        @staticmethod
        def create_simple_handler(
            handler_func: Callable[[object], object],
            handler_config: FlextModels.CqrsConfig.Handler
            | dict[str, object]
            | None = None,
        ) -> FlextHandlers[object, object]:
            """Create a simple command handler from a function."""

            class SimpleHandler(FlextHandlers[object, object]):
                def __init__(self) -> None:
                    super().__init__(
                        handler_mode="command",
                        handler_name=getattr(
                            handler_func,
                            "__name__",
                            self.__class__.__name__,
                        ),
                        handler_config=handler_config,
                    )

                def handle(self, message: object) -> FlextResult[object]:
                    result = handler_func(message)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

                def __call__(self, command: object) -> FlextResult[object]:
                    """Make the handler callable."""
                    return self.handle(command)

            return SimpleHandler()

        @staticmethod
        def create_query_handler(
            handler_func: Callable[[object], object],
            handler_config: FlextModels.CqrsConfig.Handler
            | dict[str, object]
            | None = None,
        ) -> FlextHandlers[object, object]:
            """Create a simple query handler from a function."""

            class SimpleQueryHandler(FlextHandlers[object, object]):
                def __init__(self) -> None:
                    super().__init__(
                        handler_mode="query",
                        handler_name=getattr(
                            handler_func,
                            "__name__",
                            self.__class__.__name__,
                        ),
                        handler_config=handler_config,
                    )

                def handle(self, message: object) -> FlextResult[object]:
                    result = handler_func(message)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

                def __call__(self, query: object) -> FlextResult[object]:
                    """Make the query handler callable."""
                    return self.handle(query)

            return SimpleQueryHandler()

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
                        command_type, "__name__", str(command_type)
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
