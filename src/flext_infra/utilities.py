"""Utilities facade for flext-infra.

Re-exports flext_core utilities for infrastructure services.
Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities


class FlextInfraUtilities(FlextUtilities):
    """Utility namespace for flext-infra; extends FlextUtilities."""


u = FlextInfraUtilities

__all__ = ["FlextInfraUtilities", "u"]
