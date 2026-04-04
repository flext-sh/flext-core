"""beartype.door-powered annotation inspection engine.

Replaces hand-rolled get_origin/get_args/is typing.Any inspection with
beartype's recursive TypeHint walker. Used by enforcement.py checks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from typing import get_args


class FlextUtilitiesBeartypeEngine:
    """beartype.door annotation inspection utilities.

    All methods are static — called from enforcement check methods.
    Provides robust recursive annotation analysis via TypeHint.
    """

    @staticmethod
    def contains_any(hint: object) -> bool:
        """Recursively detect Any anywhere in a type hint.

        Uses TypeHint.is_ignorable (catches Any, object at top level)
        then walks children for nested Any (e.g., Mapping[str, Any]).
        """
        from beartype.door import TypeHint

        if hint is typing.Any:
            return True
        try:
            th = TypeHint(hint)
        except Exception:
            return False
        if th.is_ignorable:
            return True
        return any(
            FlextUtilitiesBeartypeEngine.contains_any(child)
            for child in th
        )

    @staticmethod
    def has_forbidden_collection_origin(
        hint: object,
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
    def count_union_members(hint: object) -> int:
        """Count non-None members in a union type."""
        args = get_args(hint)
        if not args:
            return 0
        return sum(1 for a in args if a is not type(None))

    @staticmethod
    def is_str_none_union(hint: object) -> bool:
        """Detect str | None union pattern."""
        args = get_args(hint)
        if not args:
            return False
        return str in args and type(None) in args

    @staticmethod
    def alias_contains_any(alias_value: object) -> bool:
        """Check PEP 695 type alias __value__ for Any references.

        Unwraps alias.__value__ and walks via TypeHint.
        Falls back to string inspection for recursive aliases.
        """
        try:
            return FlextUtilitiesBeartypeEngine.contains_any(alias_value)
        except Exception:
            return "Any" in str(alias_value)
