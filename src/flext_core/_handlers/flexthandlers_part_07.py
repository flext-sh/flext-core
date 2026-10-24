"""CQRS handler discovery helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import TypeForm

from flext_core import p, t, u
from flext_core._handlers import flexthandlers_part_06

if TYPE_CHECKING:
    from types import ModuleType


class FlextHandlersPart07[MessageT_contra, ResultT](
    flexthandlers_part_06.FlextHandlersPart06[MessageT_contra, ResultT],
):
    """Final CQRS handler facade with discovery utilities composed by MRO."""

    def __init__(self, *, settings: p.HandlerConfig | None = None) -> None:
        """Initialize the facade, forwarding configuration to the base.

        Declared on the public facade so pyright's pydantic model synthesis for
        this class keeps exposing the base ``settings`` keyword to consumers.

        Args:
            settings: Optional handler configuration model.

        """
        super().__init__(settings=settings)

    class Discovery:
        """Auto-discovery mechanism for handler decorators.

        Scans classes for methods decorated with @handler() and provides
        utilities for finding and analyzing handler configurations.

        Discovery does not construct services or register dependencies. The API
        composition root chooses the module/class and registers the returned data.
        """

        @staticmethod
        def has_handlers(target_class: type) -> bool:
            """Check if class has any handler-decorated methods.

            Efficiently checks if a class contains any methods marked with
            the @handler() decorator without scanning all methods.

            Args:
                target_class: Class to check for handlers

            Returns:
                True if class has at least one handler, False otherwise

            """
            return bool(FlextHandlersPart07.Discovery.scan_class(target_class))

        @staticmethod
        def scan_class(
            target_class: type,
        ) -> t.SequenceOf[tuple[str, p.DecoratorConfig]]:
            """Scan class for methods decorated with @handler().

            Introspects the class to find all methods with handler configuration
            metadata, returning them sorted by priority (highest first).

            Args:
                target_class: Class to scan for handler decorators

            Returns:
                List of tuples (method_name, DecoratorConfig) sorted by priority

            Example:
                >>> handlers = FlextHandlers.Discovery.scan_class(MyService)
                >>> for method_name, settings in handlers:
                ...     u.Cli.print(f"{method_name}: {settings.command.__name__}")

            """
            handlers: t.MutableSequenceOf[tuple[str, p.DecoratorConfig]] = []
            callable_adapter = u.type_adapter(TypeForm(p.ResolvedHandlerCallable))
            for name in dir(target_class):
                method = getattr(target_class, name, None)
                if callable_adapter.validator.isinstance_python(method):
                    candidate = callable_adapter.validate_python(method)
                    metadata = u.handler_metadata(
                        candidate, name=f"{target_class.__qualname__}.{name}"
                    )
                    if metadata is not None:
                        handlers.append((name, metadata))
            return sorted(handlers, key=lambda entry: (-entry[1].priority, entry[0]))

        @staticmethod
        def scan_module(
            module: ModuleType,
        ) -> t.SequenceOf[tuple[str, p.ResolvedHandlerCallable, p.DecoratorConfig]]:
            """Scan module for functions decorated with @handler().

            Introspects the module to find all functions with handler configuration
            metadata, returning them sorted by priority for consistent ordering.
            SDK admission preserves callable identity; it checks callability only,
            not the signature or return domain checked at execution.

            Args:
                module: Module to scan for handler decorators

            Returns:
                List of tuples (function_name, function, DecoratorConfig)
                sorted by priority

            Example:
                >>> handlers = FlextHandlers.Discovery.scan_module(my_module)
                >>> for func_name, func, settings in handlers:
                ...     u.Cli.print(f"{func_name}: {settings.command.__name__}")

            """
            handlers: list[
                tuple[str, p.ResolvedHandlerCallable, p.DecoratorConfig]
            ] = []
            callable_adapter = u.type_adapter(TypeForm(p.ResolvedHandlerCallable))
            for name, func in vars(module).items():
                if name.startswith(
                    "_"
                ) or not callable_adapter.validator.isinstance_python(func):
                    continue
                candidate = callable_adapter.validate_python(func)
                settings = u.handler_metadata(
                    candidate, name=f"{module.__name__}.{name}"
                )
                if settings is None:
                    continue
                handlers.append((name, candidate, settings))
            return sorted(handlers, key=lambda entry: (-entry[2].priority, entry[0]))


__all__: list[str] = ["FlextHandlersPart07"]
