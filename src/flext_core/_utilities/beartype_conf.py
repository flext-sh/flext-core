"""Centralized BeartypeConf factory for FLEXT ecosystem.

Provides a single configuration point that downstream projects inherit.
Matches ENFORCEMENT_MODE: "warn" -> UserWarning, "strict" -> TypeError.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from beartype import BeartypeConf, BeartypeStrategy

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeConf:
    """Centralized beartype configuration for the FLEXT ecosystem.

    All methods are static. Exposed via u.build_beartype_conf() for
    downstream projects to use in their beartype_this_package() calls.
    """

    CLAW_SKIP_PACKAGES: ClassVar[t.VariadicTuple[str]] = c.BEARTYPE_CLAW_SKIP_PACKAGES
    """Packages skipped by the centralized flext_core beartype bootstrap."""

    @staticmethod
    def build_beartype_conf() -> BeartypeConf:
        """Build BeartypeConf matching current FLEXT beartype mode.

        - "warn" -> violation_type=UserWarning (default)
        - "strict" -> violation_type=TypeError (raises on violation)
        - "off" -> returns conf with O0 strategy (no checking)
        """
        mode = c.BEARTYPE_MODE
        if mode is c.EnforcementMode.OFF:
            return BeartypeConf(strategy=BeartypeStrategy.O0)
        return BeartypeConf(
            violation_type=UserWarning if mode is c.EnforcementMode.WARN else TypeError,
            strategy=BeartypeStrategy.O1,
            claw_skip_package_names=FlextUtilitiesBeartypeConf.CLAW_SKIP_PACKAGES,
        )
