"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``h``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeIs

from flext_core._typings.services import FlextTypesServices as ts

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core._typings.base import FlextTypingBase as tb


class FlextUtilitiesChecker:
    """Handler type checking utilities for h complexity reduction.

    Extracts type introspection and compatibility logic from h
    to simplify handler initialization and provide reusable type checking.
    """

    @staticmethod
    def _is_module_export_callable(
        value: Callable[..., ts.ModuleExport] | ts.GuardInput | None,
    ) -> TypeIs[Callable[..., ts.ModuleExport]]:
        """Narrow value to a callable returning module exports.

        Excludes ``type`` objects (classes are callable but are not the
        bound/free functions we expect as handle methods).
        """
        return callable(value) and not isinstance(value, type)

    @staticmethod
    def _is_subclass_of(candidate: tb.TypeHintSpecifier, parent: type) -> bool:
        """Safe subclass check that never raises TypeError."""
        return isinstance(candidate, type) and issubclass(candidate, parent)

    @classmethod
    def _is_dict_type(cls, candidate: tb.TypeHintSpecifier) -> bool:
        """Check if candidate is dict or a subclass of dict."""
        return cls._is_subclass_of(candidate, dict)

    @classmethod
    def _check_dict_compatibility(
        cls,
        expected_type: tb.TypeHintSpecifier,
        message_type: ts.MessageTypeSpecifier,
        origin_type: tb.TypeHintSpecifier,
        message_origin: tb.TypeHintSpecifier,
    ) -> bool:
        """Check dict type compatibility between expected and message types."""
        origin_is_dict = cls._is_dict_type(origin_type)
        message_origin_is_dict = cls._is_dict_type(message_origin)
        if origin_is_dict and (
            message_origin_is_dict or cls._is_dict_type(message_type)
        ):
            return True
        return cls._is_dict_type(message_type) and (
            origin_is_dict or cls._is_dict_type(expected_type)
        )

    @classmethod
    def _check_object_type_compatibility(
        cls, expected_type: tb.TypeHintSpecifier
    ) -> bool:
        """Check if expected type is a canonical catch-all value contract."""
        return expected_type is ts.JsonPayload


__all__: list[str] = ["FlextUtilitiesChecker"]
