"""PEP 695 type aliases for project metadata (Tier 0, no flext-core deps).

Every alias is non-nullable; ``| None`` is added inline at usage sites
only (per AGENTS.md §3.2 / flext-type-system line 30).

Architecture: Tier 0 — zero flext-core imports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar


class FlextTypingProjectMetadata:
    """PEP 695 aliases for project-metadata SSOT."""

    _flext_enforcement_exempt: ClassVar[bool] = True

    type AliasName = str
    type TierName = str
    type ProjectName = str
    type PackageName = str
    type ClassStem = str
    type LibraryName = str
    type AliasToSuffixMap = Mapping[AliasName, str]
    type TierFacadePrefixMap = Mapping[TierName, str]
    type AliasParentSourceMap = Mapping[AliasName, str]
    type SpecialNameOverrideMap = Mapping[ProjectName, ClassStem]
    type ManagedKeyName = str
    type ManagedKeyTuple = tuple[ManagedKeyName, ...]
