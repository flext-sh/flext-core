"""Deprecation warnings pattern for backward compatibility.

FlextUtilitiesDeprecation provides utilities for marking deprecated functions,
parameters, and classes with migration guidance while maintaining backward
compatibility.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import functools
import pathlib
import warnings
from collections.abc import Callable
from typing import ClassVar

from flext_core.typings import P, R


class FlextUtilitiesDeprecation:
    """Deprecation utilities for marking deprecated code."""

    # Class-level set to track warnings that have already been issued
    _warned_once: ClassVar[set[str]] = set()

    @staticmethod
    def deprecated(
        replacement: str | None = None,
        version: str | None = None,
        reason: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Mark function as deprecated.

        Args:
            replacement: Name of replacement function/class.
            version: Version when deprecation was introduced.
            reason: Reason for deprecation.

        Returns:
            Decorator that wraps function with deprecation warning.

        Example:
            >>> @FlextUtilitiesDeprecation.deprecated(
            ...     replacement="new_function",
            ...     version="2.0.0",
            ...     reason="Use new_function instead"
            ... )
            >>> def old_function(): ...

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                message_parts = [f"{func.__name__} is deprecated"]
                if version:
                    message_parts.append(f"since version {version}")
                if replacement:
                    message_parts.append(f"Use {replacement} instead")
                if reason:
                    message_parts.append(f"Reason: {reason}")
                warnings.warn(
                    ". ".join(message_parts),
                    DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def deprecated_parameter(
        param_name: str,
        replacement: str | None = None,
        version: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Mark function parameter as deprecated.

        Args:
            param_name: Name of deprecated parameter.
            replacement: Name of replacement parameter.
            version: Version when deprecation was introduced.

        Returns:
            Decorator that warns when deprecated parameter is used.

        Example:
            >>> @FlextUtilitiesDeprecation.deprecated_parameter(
            ...     "old_param",
            ...     replacement="new_param",
            ...     version="2.0.0"
            ... )
            >>> def my_function(new_param: str, old_param: str | None = None): ...

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if param_name in kwargs:
                    message_parts = [f"Parameter '{param_name}' is deprecated"]
                    if version:
                        message_parts.append(f"since version {version}")
                    if replacement:
                        message_parts.append(f"Use '{replacement}' instead")
                    warnings.warn(
                        ". ".join(message_parts),
                        DeprecationWarning,
                        stacklevel=2,
                    )
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def deprecated_class[T: type](
        replacement: str | None = None,
        version: str | None = None,
    ) -> Callable[[T], T]:
        """Mark class as deprecated.

        Args:
            replacement: Name of replacement class.
            version: Version when deprecation was introduced.

        Returns:
            Class decorator that warns when class is instantiated.

        Example:
            >>> @FlextUtilitiesDeprecation.deprecated_class(
            ...     replacement="NewClass",
            ...     version="2.0.0"
            ... )
            >>> class OldClass: ...

        """

        def decorator(cls: T) -> T:
            # Access __init__ from the class type using getattr to avoid mypy error
            # This avoids accessing __init__ on an instance which is unsound
            original_init = getattr(cls, "__init__", None)
            if original_init is None:
                # If no __init__, create a no-op one
                def noop_init(self: object, *args: object, **kwargs: object) -> None:
                    pass

                original_init = noop_init

            @functools.wraps(original_init)
            def new_init(self: object, *args: object, **kwargs: object) -> None:
                message_parts = [f"{cls.__name__} is deprecated"]
                if version:
                    message_parts.append(f"since version {version}")
                if replacement:
                    message_parts.append(f"Use {replacement} instead")
                warnings.warn(
                    ". ".join(message_parts),
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Call original __init__ method - original_init is bound to the class
                # Type narrowing: original_init is guaranteed to be callable after None check
                init_func = original_init
                if init_func is not None:
                    init_func(self, *args, **kwargs)

            # Set __init__ on the class for decorator pattern
            # Accessing __init__ on class is necessary for decorator pattern
            # Use setattr with variable to avoid mypy error about accessing __init__ on instance
            init_attr = "__init__"
            setattr(cls, init_attr, new_init)
            return cls

        return decorator

    @staticmethod
    def generate_migration_report(
        module_name: str,
        output_file: str | None = None,
    ) -> str:
        """Generate migration report for deprecated code.

        Args:
            module_name: Name of module to scan for deprecations.
            output_file: Optional file path to write report.

        Returns:
            Migration report as string.

        Note:
            This is a placeholder implementation. Full implementation would
            scan the module for deprecated functions/classes and generate
            a comprehensive migration guide.

        """
        report = f"Migration Report for {module_name}\n"
        report += "=" * 50 + "\n\n"
        report += "This is a placeholder implementation.\n"
        report += "Full implementation would scan for deprecations and generate migration guide.\n"
        if output_file:
            with pathlib.Path(output_file).open("w", encoding="utf-8") as f:
                f.write(report)
        return report

    @classmethod
    def warn_once(cls, identifier: str, message: str) -> None:
        """Emit a deprecation warning only once per unique identifier.

        Args:
            identifier: Unique identifier for this warning (used to prevent duplicates).
            message: Warning message to display.

        Example:
            >>> FlextUtilitiesDeprecation.warn_once(
            ...     "old_api_v1", "This API is deprecated. Use v2 instead."
            ... )

        """
        if identifier not in cls._warned_once:
            cls._warned_once.add(identifier)
            warnings.warn(message, DeprecationWarning, stacklevel=2)


__all__ = [
    "FlextUtilitiesDeprecation",
]
