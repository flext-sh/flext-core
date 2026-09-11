"""Guard 5 — tier / abstraction-boundary enforcer.

Enforces AGENTS.md §2.7 abstraction-boundary rules: each external library
listed in ``c.ENFORCEMENT_LIBRARY_OWNERS`` (flext-core SSOT) has exactly
one owning FLEXT project. Bare runtime imports of those libs outside the
canonical wrapper sites are violations.

Uses rope's semantic import resolution so ``if TYPE_CHECKING:`` imports
are automatically exempt (they live in conditional blocks that rope
skips when collecting runtime imports).

Mandate: 100% ROPE-based per the flext-infra detector mandate — no raw
``ast``/``libcst`` source analysis. Uses
``u.Infra`` Rope boundary helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c
from flext_infra.validate._rope_import_boundary import FlextInfraRopeImportBoundaryBase

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraValidateTierWhitelist(FlextInfraRopeImportBoundaryBase):
    """Enforce the §2.7 abstraction boundary at runtime-import level.

    Banned-lib set + per-library ownership are both derived from
    ``c.ENFORCEMENT_LIBRARY_OWNERS`` (flext-core SSOT): each banned library's
    owning project tree is the only place that library may be imported.
    """

    _BANNED: ClassVar[frozenset[str]] = frozenset(c.ENFORCEMENT_LIBRARY_OWNERS)
    _OK_SUMMARY: ClassVar[str] = (
        "abstraction boundary respected (flext-core-only libs not imported elsewhere)"
    )
    _VIOLATION_KIND: ClassVar[str] = "abstraction-boundary"
    _SCAN_KIND: ClassVar[str] = "tier-whitelist"

    @override
    def _is_allowlisted(self, _file_path: Path, _module_name: str) -> bool:
        """Return True iff ``file_path`` owns ``module_name`` per OWNERS SSOT.

        Ownership comes directly from ``c.ENFORCEMENT_LIBRARY_OWNERS``
        (flext-core SSOT): each banned library has exactly one owning project,
        and the entire ``<owner>/src/<package>/`` tree is allowed to import
        that library. ``tests/``, ``examples/``, ``scripts/`` are runtime-
        exempt globally.

        Settings modules (``*/settings.py``) are additionally allowed to
        import ``pydantic_settings`` — the canonical pattern for project
        configuration is ``class Foo(FlextSettings, BaseSettings)`` per
        ``flext_core._settings.base`` docstring, and that base name only
        lives in ``pydantic_settings``.
        """
        if any(
            part in c.Infra.TIER_WHITELIST_NON_RUNTIME_DIR_PARTS
            for part in _file_path.parts
        ):
            return True
        top = self._top_module(_module_name)
        if (
            top in c.Infra.TIER_WHITELIST_SETTINGS_MODULE_LIBRARIES
            and _file_path.name == "settings.py"
        ):
            return True
        owner = c.ENFORCEMENT_LIBRARY_OWNERS.get(top)
        if owner is None:
            return False
        return f"/{owner}/src/" in _file_path.as_posix()

    @override
    def _format_violation(self, file_path: Path, module_name: str) -> str:
        """Format the abstraction-boundary violation message."""
        return (
            f"{file_path}: bare import of {module_name!r} "
            "— use flext_core facades (c/m/p/t/u)"
        )


__all__: t.StrSequence = ("FlextInfraValidateTierWhitelist",)
