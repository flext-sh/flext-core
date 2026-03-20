"""Deprecation utilities for flext-core."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar
from warnings import deprecated as _stdlib_deprecated

from flext_core import t

if TYPE_CHECKING:
    from flext_core import P, R


class FlextUtilitiesDeprecation:
    """Deprecation utilities for marking deprecated code."""

    _warned_once: ClassVar[set[str]] = set()

    @classmethod
    def warn_once(cls, identifier: str, message: str) -> None:
        """Emit one deprecation warning per identifier."""
        if identifier not in cls._warned_once:
            cls._warned_once.add(identifier)
            warnings.warn(message, DeprecationWarning, stacklevel=2)

    @classmethod
    def warn_polymorphic_input(
        cls,
        value: t.ValueOrModel | None,
        context: str,
        preferred: str,
        *,
        removal_version: str = "0.14.0",
    ) -> None:
        """Warn on broad input unions that should be narrowed."""
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
            "This callable is deprecated.",
            category=DeprecationWarning,
        )

    @staticmethod
    def deprecated_class[TClass: type](
        replacement: str | None = None,
        version: str | None = None,
    ) -> Callable[[TClass], TClass]:
        has_replacement = replacement is not None
        has_version = version is not None

        if has_replacement and has_version:
            msg = "This class is deprecated and has a replacement."
        elif has_replacement:
            msg = "This class is deprecated. Use the replacement class."
        elif has_version:
            msg = "This class is deprecated."
        else:
            msg = "This class is deprecated."

        stdlib_decorator = _stdlib_deprecated(msg, category=DeprecationWarning)

        def _wrapper(cls: TClass) -> TClass:
            return stdlib_decorator(cls)

        return _wrapper

    @staticmethod
    def deprecated_parameter(
        param_name: str,
        replacement: str | None = None,
        version: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Warn when a deprecated parameter is passed by keyword."""

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


__all__ = ["FlextUtilitiesDeprecation"]
