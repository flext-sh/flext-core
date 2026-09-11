"""Guard 8 — metadata-discipline validator.

Enforces that pyproject metadata parsing is centralized. Runtime imports of
``tomllib`` are forbidden outside allowlisted metadata readers/writers.

This validator is ROPE-backed (no raw AST/CST parsing) and plugs into the
existing ``flext-infra validate`` command hierarchy.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c
from flext_infra.validate._rope_import_boundary import FlextInfraRopeImportBoundaryBase

if TYPE_CHECKING:
    from pathlib import Path


class FlextInfraValidateMetadataDiscipline(FlextInfraRopeImportBoundaryBase):
    """Detect rogue ``tomllib`` imports outside canonical metadata modules."""

    _BANNED: ClassVar[frozenset[str]] = c.Infra.METADATA_TOMLLIB_MODULES
    _OK_SUMMARY: ClassVar[str] = (
        "metadata-discipline respected (tomllib imports centralized)"
    )
    _VIOLATION_KIND: ClassVar[str] = "metadata-discipline"
    _SCAN_KIND: ClassVar[str] = "metadata-discipline"

    @override
    def _is_in_scope(self, _file_path: Path) -> bool:
        """Return True when path belongs to metadata-discipline enforcement scope."""
        posix = _file_path.as_posix()
        return any(marker in posix for marker in c.Infra.METADATA_TARGET_SCOPE_MARKERS)

    @override
    def _is_allowlisted(self, _file_path: Path, _module_name: str) -> bool:
        """Return True when file path is in canonical metadata reader set."""
        posix = _file_path.as_posix()
        return any(
            marker in posix for marker in c.Infra.METADATA_ALLOWLIST_PATH_MARKERS
        )

    @override
    def _format_violation(self, file_path: Path, module_name: str) -> str:
        """Format the metadata-discipline violation message."""
        return (
            f"{file_path}: direct metadata parser import {module_name!r} "
            "— route pyproject reads through canonical metadata utilities"
        )


__all__: tuple[str, ...] = ("FlextInfraValidateMetadataDiscipline",)
