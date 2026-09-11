"""Composed workspace cleanup utilities for flext-tests.

Git-aware, fail-loud cleanup of exact ignored development residues. The facade
composes the focused sibling mixins (git → paths → inspect → plan) by MRO, in
line with the flext-law §1B.2 thin-facade pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests._utilities.workspace_cleanup_plan import (
    FlextTestsWorkspaceCleanupPlanUtilitiesMixin,
)


class FlextTestsWorkspaceCleanupUtilitiesMixin(
    FlextTestsWorkspaceCleanupPlanUtilitiesMixin
):
    """Deterministic workspace cleanup plan/apply namespace."""


__all__: tuple[str, ...] = ("FlextTestsWorkspaceCleanupUtilitiesMixin",)
