"""Deprecation warnings pattern for backward compatibility.

FlextUtilitiesDeprecation provides utilities for marking deprecated functions,
parameters, and classes with migration guidance while maintaining backward
compatibility.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import ClassVar
from warnings import deprecated as _stdlib_deprecated

from pydantic import BaseModel

from flext_core import P, R, t


class FlextUtilitiesDeprecation:
    """Deprecation utilities for marking deprecated code."""

    _warned_once: ClassVar[set[str]] = set()

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

    @classmethod
    def warn_polymorphic_input(
        cls,
        value: t.NormalizedValue | BaseModel | None,
        context: str,
        preferred: str,
        *,
        removal_version: str = "0.14.0",
    ) -> None:
        """Emit deprecation warning for overly polymorphic inputs.

        Use at system boundaries where broad union types (NormalizedValue,
        t.RuntimeData, RegisterableService) accept inputs that should migrate
        to narrower types (Container, Scalar, ConfigMap).

        Args:
            value: The actual value received.
            context: Name of the function/parameter accepting the value.
            preferred: Preferred narrower type name.
            removal_version: Version when broad acceptance will be removed.

        """
        type_name = type(value).__name__
        identifier = f"polymorphic_{context}_{type_name}"
        cls.warn_once(
            identifier,
            f"Passing {type_name} to {context} is deprecated. "
            f"Prefer {preferred}. Broad union acceptance will be removed in {removal_version}.",
        )

    @staticmethod
    def deprecated(
        replacement: str | None = None,
        version: str | None = None,
        reason: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        has_replacement = replacement is not None
        has_version = version is not None
        has_reason = reason is not None

        if has_replacement and has_version and has_reason:
            return _stdlib_deprecated(
                "This callable is deprecated. It has a replacement and deprecation reason.",
                category=DeprecationWarning,
            )
        if has_replacement and has_version:
            return _stdlib_deprecated(
                "This callable is deprecated and has a replacement.",
                category=DeprecationWarning,
            )
        if has_replacement and has_reason:
            return _stdlib_deprecated(
                "This callable is deprecated. Use the replacement callable.",
                category=DeprecationWarning,
            )
        if has_version and has_reason:
            return _stdlib_deprecated(
                "This callable is deprecated with deprecation metadata.",
                category=DeprecationWarning,
            )
        if has_replacement:
            return _stdlib_deprecated(
                "This callable is deprecated. Use the replacement callable.",
                category=DeprecationWarning,
            )
        if has_version:
            return _stdlib_deprecated(
                "This callable is deprecated.",
                category=DeprecationWarning,
            )
        if has_reason:
            return _stdlib_deprecated(
                "This callable is deprecated.",
                category=DeprecationWarning,
            )
        return _stdlib_deprecated(
            "This callable is deprecated.", category=DeprecationWarning
        )

    @staticmethod
    def deprecated_class[TClass: type](
        replacement: str | None = None, version: str | None = None
    ) -> Callable[[TClass], TClass]:
        has_replacement = replacement is not None
        has_version = version is not None

        if has_replacement and has_version:
            return _stdlib_deprecated(
                "This class is deprecated and has a replacement.",
                category=DeprecationWarning,
            )
        if has_replacement:
            return _stdlib_deprecated(
                "This class is deprecated. Use the replacement class.",
                category=DeprecationWarning,
            )
        if has_version:
            return _stdlib_deprecated(
                "This class is deprecated.",
                category=DeprecationWarning,
            )
        return _stdlib_deprecated(
            "This class is deprecated.", category=DeprecationWarning
        )

    @staticmethod
    def deprecated_parameter(
        param_name: str, replacement: str | None = None, version: str | None = None
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
                        ". ".join(message_parts), DeprecationWarning, stacklevel=2
                    )
                return func(*args, **kwargs)

            return wrapper

        return decorator


__all__ = ["FlextUtilitiesDeprecation"]
