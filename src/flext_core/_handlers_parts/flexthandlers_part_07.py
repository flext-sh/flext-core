"""CQRS handler discovery helpers."""

from __future__ import annotations


from flext_core import c, p, t

from .flexthandlers_part_06 import FlextHandlers as FlextHandlersPart06

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    from collections.abc import Callable, MutableSequence


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersPart06[MessageT_contra, ResultT]
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
                ...     print(f"{method_name}: {settings.command.__name__}")

            """
            handlers: t.SequenceOf[tuple[str, p.DecoratorConfig]] = [
                (name, getattr(method, c.HANDLER_ATTR))
                for name in dir(target_class)
                if hasattr(method := getattr(target_class, name, None), c.HANDLER_ATTR)
            ]
            return sorted(handlers, key=lambda x: x[1].priority, reverse=True)

        @staticmethod
        def scan_module(
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
                List of tuples (function_name, function, DecoratorConfig) sorted by priority

            Example:
                >>> handlers = FlextHandlers.Discovery.scan_module(my_module)
                >>> for func_name, func, settings in handlers:
                ...     print(f"{func_name}: {settings.command.__name__}")

            """
            handlers: MutableSequence[
                tuple[str, Callable[..., t.Scalar | None], p.DecoratorConfig]
            ] = []
            for name in dir(module):
                if name.startswith("_"):
                    continue
                func = getattr(module, name, None)
                if func is None:
                    continue
                if not callable(func):
                    continue
                if not hasattr(func, c.HANDLER_ATTR):
                    continue
                settings: p.DecoratorConfig = getattr(func, c.HANDLER_ATTR)

                def narrowed_func(
                    message: t.JsonPayload, function_name: str = name
                ) -> t.Scalar | None:
                    resolved_callable = getattr(module, function_name, None)
                    if not callable(resolved_callable):
                        return None
                    result = resolved_callable(message)
                    if result is None:
                        return None
                    if isinstance(result, t.SCALAR_TYPES):
                        return result
                    return str(result)

                setattr(narrowed_func, c.HANDLER_ATTR, settings)
                handlers.append((name, narrowed_func, settings))
            return sorted(handlers, key=lambda x: (-x[2].priority, x[0]))


__all__: list[str] = ["FlextHandlers"]
