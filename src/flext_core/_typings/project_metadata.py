"""PEP 695 type aliases for project metadata (Tier 0, no flext-core deps).

Every alias is non-nullable; ``| None`` is added inline at usage sites
only (per AGENTS.md §3.2 / flext-type-system line 30). Inherited flat
into ``FlextTypes`` so consumers access each alias as ``t.AliasName``
etc. — never via a sub-namespace.

Architecture: Tier 0 — zero flext-core imports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar


class FlextTypingProjectMetadata:
    """PEP 695 aliases for project-metadata SSOT (flat on ``t.*``)."""

    _flext_enforcement_exempt: ClassVar[bool] = True

    type ProjectAliasName = str
    type ProjectTierName = str
    type ProjectName = str
    type ProjectPackageName = str
    type ProjectClassStem = str
    type ProjectLibraryName = str
    type ProjectAliasToSuffixMap = Mapping[ProjectAliasName, str]
    type ProjectTierFacadePrefixMap = Mapping[ProjectTierName, str]
    type ProjectAliasParentSourceMap = Mapping[ProjectAliasName, str]
    type ProjectSpecialNameOverrideMap = Mapping[ProjectName, ProjectClassStem]
    type ProjectManagedKeyName = str
    type ProjectManagedKeyTuple = tuple[ProjectManagedKeyName, ...]
