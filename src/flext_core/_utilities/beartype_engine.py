"""Annotation inspection engine for enforcement checks.

Provides recursive inspection helpers for type annotations used by
enforcement.py checks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from typing import get_args

from flext_core import FlextTypesServices as t


class FlextUtilitiesBeartypeEngine:
    """Annotation inspection utilities.

    All methods are static — called from enforcement check methods.
    Provides recursive annotation analysis via ``get_args`` walking.
    """

    @staticmethod
    def contains_any(hint: t.TypeHintSpecifier | None) -> bool:
        """Recursively detect Any anywhere in a type hint.

        Catches Any and object at the top level, then walks nested args
        recursively (for example Mapping[str, Any]).
        """
        if hint is typing.Any:
            return True
        if hint is object:
            return True
        for arg in get_args(hint):
            if arg is typing.Any:
                return True
            if FlextUtilitiesBeartypeEngine.contains_any(arg):
                return True
        return False

    @staticmethod
    def has_forbidden_collection_origin(
        hint: t.TypeHintSpecifier | None,
        forbidden: frozenset[str],
    ) -> tuple[bool, str]:
        """Detect dict[...]/list[...]/set[...] as annotation origin.

        Returns (is_forbidden, origin_name).
        """
        origin = getattr(hint, "__origin__", None)
        if origin is not None and hasattr(origin, "__name__"):
            name: str = origin.__name__
            if name in forbidden:
                return True, name
        return False, ""

    @staticmethod
    def count_union_members(
        hint: t.TypeHintSpecifier | None,
    ) -> int:
        """Count non-None members in a union type."""
        args = get_args(hint)
        if not args:
            return 0
        return sum(1 for a in args if a is not type(None))

    @staticmethod
    def is_str_none_union(
        hint: t.TypeHintSpecifier | None,
    ) -> bool:
        """Detect str | None union pattern."""
        args = get_args(hint)
        if not args:
            return False
        return str in args and type(None) in args

    @staticmethod
    def alias_contains_any(
        alias_value: t.TypeHintSpecifier | None,
    ) -> bool:
        """Check PEP 695 type alias __value__ for Any references.

        Walks recursively through the alias value and falls back to string
        inspection for recursive aliases.
        """
        try:
            return FlextUtilitiesBeartypeEngine.contains_any(alias_value)
        except Exception:
            return "Any" in str(alias_value)
