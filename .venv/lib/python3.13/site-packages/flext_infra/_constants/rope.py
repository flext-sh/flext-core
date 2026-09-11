"""Rope Project configuration constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextInfraConstantsRope:
    """Rope Project configuration constants — accessed via c.Infra.*."""

    @unique
    class RopeScopeKind(StrEnum):
        """Semantic scope kinds returned by rope's ``PyScope.get_kind()``.

        Rope emits ``"Module"``/``"Function"``/``"Class"`` for those scopes and
        ``None`` for comprehension (temporary) scopes; ``UNKNOWN`` is the
        canonical fallback for that ``None`` so the contract stays typed.
        """

        MODULE = "Module"
        "Module (global) scope."
        FUNCTION = "Function"
        "Function or method scope."
        CLASS = "Class"
        "Class scope."
        UNKNOWN = "Unknown"
        "Fallback for rope comprehension/temporary scopes (get_kind() is None)."

    @unique
    class StatementCategory(StrEnum):
        """Lexical category of one logical statement from the rope structure boundary.

        Derived from the leading token of a rope ``LogicalLineFinder`` region;
        the classification is lexical (rope-owned source slice), never ``ast``.
        """

        IMPORT = "import"
        "Plain ``import x`` statement."
        FROM_IMPORT = "from_import"
        "``from m import x`` statement."
        TYPE_ALIAS = "type_alias"
        "PEP 695 ``type X = ...`` alias declaration."
        ANN_ASSIGN = "ann_assign"
        "Annotated assignment ``x: T = v`` or ``x: T``."
        ASSIGN = "assign"
        "Plain assignment ``x = v``."
        IF_GUARD = "if_guard"
        "``if ...:`` block header (e.g. ``if TYPE_CHECKING:``)."
        CLASS_DEF = "class_def"
        "``class X...:`` header."
        FUNC_DEF = "func_def"
        "``def``/``async def`` header."
        CALL = "call"
        "Expression statement whose head is a call."
        OTHER = "other"
        "Any statement not matched by a more specific category."

    # NOTE (flext-0ftd.3.10.2.4): runtime engine types and exception-boundary
    # tuples live in u.Infra (FlextInfraUtilitiesRopeRuntimeTypes), not in the
    # constants layer, to keep c.Infra declarative and cycle-free.

    ROPE_IGNORED_RESOURCES: Final[tuple[str, ...]] = (
        ".venv",
        "venv",
        "node_modules",
        "*.pyc",
        "dist/",
        "build/",
        ".tox/",
        ".cache/",
        "__pycache__",
        ".mypy_cache",
        ".pyrefly_cache",
        ".pytest_cache",
        ".git",
    )
    "Resources rope should ignore when scanning the project tree."

    PROPERTY_DECORATORS: Final[frozenset[str]] = frozenset({
        "property",
        "cached_property",
        "computed_field",
    })
    "Decorator names that mark Python descriptors / Pydantic computed fields."


__all__: list[str] = ["FlextInfraConstantsRope"]
