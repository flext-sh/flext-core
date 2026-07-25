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


class FlextTypingProjectMetadata:
    """PEP 695 aliases for project-metadata SSOT (flat on ``t.*``)."""
