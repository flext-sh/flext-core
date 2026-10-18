"""CQRS handler discovery helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c, m, t
from flext_core._handlers_parts import flexthandlers_part_06

if TYPE_CHECKING:
    from types import ModuleType

    from flext_core import p


class FlextHandlers[MessageT_contra, ResultT](
    flexthandlers_part_06.FlextHandlers[MessageT_contra, ResultT],
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

        This class enables zero-settings handler registration in FlextService
        by automatically discovering decorated methods at initialization time.
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

            Example:
                >>> if FlextHandlers.Discovery.has_handlers(MyService):
                ...     # Auto-setup dispatcher/registry
                ...     service._setup_dispatcher()

            """
            return any(
                hasattr(getattr(target_class, name, None), c.HANDLER_ATTR)
                for name in dir(target_class)
            )

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
            handlers: t.SequenceOf[tuple[str, p.DecoratorConfig]] = [
                (name, getattr(method, c.HANDLER_ATTR))
                for name in dir(target_class)
                if hasattr(method := getattr(target_class, name, None), c.HANDLER_ATTR)
            ]

            def _priority(entry: tuple[str, p.DecoratorConfig]) -> int:
                return entry[1].priority

            return sorted(handlers, key=_priority, reverse=True)

        @staticmethod
        def scan_module(
            module: ModuleType,
        ) -> t.SequenceOf[tuple[str, t.ResolvedHandlerCallable, p.DecoratorConfig]]:
            """Scan module for functions decorated with @handler().

            Introspects the module to find all functions with handler configuration
            metadata, returning them sorted by priority for consistent ordering.

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
                tuple[str, t.ResolvedHandlerCallable, p.DecoratorConfig]
            ] = []
            for name, func in vars(module).items():
                if name.startswith("_") or not callable(func):
                    continue
                settings = vars(func).get(c.HANDLER_ATTR)
                if settings is None:
                    continue
                if not isinstance(settings, m.DecoratorConfig):
                    msg = f"Invalid handler metadata for {module.__name__}.{name}"
                    raise TypeError(msg)
                if settings.timeout is not None or settings.middleware:
                    msg = (
                        "Unsupported handler execution policy for "
                        f"{module.__name__}.{name}"
                    )
                    raise ValueError(msg)
                handlers.append((name, func, settings))
            return sorted(handlers, key=lambda entry: (-entry[2].priority, entry[0]))


__all__: list[str] = ["FlextHandlers"]
