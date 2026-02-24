"""Type aliases for flext-infra.

Re-exports and extends flext_core typings for infrastructure services.
Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.typings import FlextTypes


class FlextInfraTypes(FlextTypes):
    """Type namespace for flext-infra; extends FlextTypes."""


t = FlextInfraTypes

__all__ = ["FlextInfraTypes", "t"]
