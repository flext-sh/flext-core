"""Exceptions utility wrapper for flext-infra.

Provides FlextInfraUtilitiesExceptions wrapper class that extends
FlextInfraExceptions with utility-specific namespace organization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextExceptions


class FlextInfraExceptions(FlextExceptions):
    """FLEXT infrastructure exception hierarchy."""

    class Infra:
        """Infrastructure-specific exceptions."""

        pass


class FlextInfraUtilitiesExceptions(FlextInfraExceptions):
    """FLEXT infrastructure utilities exception wrapper.

    Extends FlextInfraExceptions with utility-specific namespace organization.
    Provides canonical `e` alias accessor pattern for backward compatibility.
    """

    pass


e = FlextInfraExceptions

__all__ = ["FlextInfraExceptions", "FlextInfraUtilitiesExceptions", "e"]
