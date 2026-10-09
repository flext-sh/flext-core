"""CQRS handler discovery helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c, t
from flext_core._handlers_parts.flexthandlers_part_06 import (
    FlextHandlers as FlextHandlersPart06,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from flext_core import p


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersPart06[MessageT_contra, ResultT],
):
    """Final CQRS handler facade with discovery utilities composed by MRO."""

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

        @classmethod
        def scan_module(
            cls,
            module: ModuleType,
        ) -> t.SequenceOf[
            tuple[str, Callable[..., t.Scalar | None], p.DecoratorConfig]
        ]:
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
            handlers = [
                _narrowed_entry(module, name)
                for name in dir(module)
                if not name.startswith("_")
                and _is_decorated_handler(getattr(module, name, None))
            ]

            def _priority_then_name(
                entry: tuple[str, Callable[..., t.Scalar | None], p.DecoratorConfig],
            ) -> tuple[int, str]:
                return (-entry[2].priority, entry[0])

            return sorted(handlers, key=_priority_then_name)


def _is_decorated_handler(func: object) -> bool:
    """Whether one module attribute is a callable @handler function.

    Returns:
        The resulting ``bool``.

    """
    return func is not None and callable(func) and hasattr(func, c.HANDLER_ATTR)


def _narrowed_entry(
    module: ModuleType,
    name: str,
) -> tuple[str, Callable[..., t.Scalar | None], p.DecoratorConfig]:
    """Build one sorted-scan entry wrapping the function for scalar calls.

    Returns:
        The resulting ``(name, narrowed_func, settings)`` entry.

    """
    func = getattr(module, name)
    settings: p.DecoratorConfig = getattr(func, c.HANDLER_ATTR)

    def narrowed_func(
        message: t.JsonPayload,
        function_name: str = name,
    ) -> t.Scalar | None:
        resolved_callable = getattr(module, function_name, None)
        if not callable(resolved_callable):
            return None
        result = resolved_callable(message)
        if result is None:
            return None
        if isinstance(result, c.SCALAR_TYPES):
            return result
        return str(result)

    setattr(narrowed_func, c.HANDLER_ATTR, settings)
    return (name, narrowed_func, settings)


__all__: list[str] = ["FlextHandlers"]
