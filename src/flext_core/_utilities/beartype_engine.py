"""Annotation inspection engine for enforcement checks.

Uses beartype DOOR API (TypeHint class hierarchy) for all type hint
inspection. Never uses raw typing.get_args/get_origin — those are
stdlib bypasses.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beartype.door import AnyTypeHint, TypeHint, UnionTypeHint

if TYPE_CHECKING:
    from flext_core import t


class FlextUtilitiesBeartypeEngine:
    """Annotation inspection utilities via beartype DOOR API.

    All methods are static — called from enforcement check methods.
    Uses ``beartype.door.TypeHint`` for recursive annotation analysis.
    """

    @staticmethod
    def contains_any(hint: t.TypeHintSpecifier | None) -> bool:
        """Recursively detect Any or object anywhere in a type hint.

        Uses beartype DOOR ``AnyTypeHint`` detection instead of raw
        ``typing.Any`` identity checks.
        """
        if hint is None:
            return False
        try:
            th = TypeHint(hint)
        except Exception:
            return False
        if isinstance(th, AnyTypeHint):
            return True
        if hint is object:
            return True
        return any(FlextUtilitiesBeartypeEngine.contains_any(arg) for arg in th.args)

    @staticmethod
    def has_forbidden_collection_origin(
        hint: t.TypeHintSpecifier | None,
        forbidden: frozenset[str],
    ) -> tuple[bool, str]:
        """Detect dict[...]/list[...]/set[...] as annotation origin.

        Uses beartype DOOR to extract the origin type name.
        """
        if hint is None:
            return False, ""
        try:
            th = TypeHint(hint)
        except Exception:
            return False, ""
        origin = getattr(th.hint, "__origin__", None)
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
        if hint is None:
            return 0
        try:
            th = TypeHint(hint)
        except Exception:
            return 0
        if not isinstance(th, UnionTypeHint):
            return 0
        return sum(1 for a in th.args if a is not type(None))

    @staticmethod
    def is_str_none_union(
        hint: t.TypeHintSpecifier | None,
    ) -> bool:
        """Detect str | None union pattern."""
        if hint is None:
            return False
        try:
            th = TypeHint(hint)
        except Exception:
            return False
        if not isinstance(th, UnionTypeHint):
            return False
        raw_args = [
            a if isinstance(a, type) else getattr(a, "hint", a) for a in th.args
        ]
        return str in raw_args and type(None) in raw_args

    @staticmethod
    def alias_contains_any(
        alias_value: t.TypeHintSpecifier | None,
    ) -> bool:
        """Check PEP 695 type alias __value__ for Any references.

        Uses beartype DOOR for recursive inspection. Falls back to
        string check only for genuinely recursive aliases that crash
        beartype's TypeHint constructor.
        """
        try:
            return FlextUtilitiesBeartypeEngine.contains_any(alias_value)
        except (TypeError, AttributeError, RuntimeError, RecursionError):
            return "Any" in str(alias_value)
