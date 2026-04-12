"""Centralized BeartypeConf factory for FLEXT ecosystem.

Provides a single configuration point that downstream projects inherit.
Matches ENFORCEMENT_MODE: "warn" -> UserWarning, "strict" -> TypeError.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from beartype import BeartypeConf, BeartypeStrategy

from flext_core import c


class FlextUtilitiesBeartypeConf:
    """Centralized beartype configuration for the FLEXT ecosystem.

    All methods are static. Exposed via u.build_beartype_conf() for
    downstream projects to use in their beartype_this_package() calls.
    """

    CLAW_SKIP_PACKAGES: tuple[str, ...] = ("flext_core",)
    """Packages to skip in beartype.claw due to Pydantic recursive type conflicts."""

    @staticmethod
    def build_beartype_conf() -> BeartypeConf:
        """Build BeartypeConf matching current FLEXT enforcement mode.

        - "warn" -> violation_type=UserWarning (default)
        - "strict" -> violation_type=TypeError (raises on violation)
        - "off" -> returns conf with O0 strategy (no checking)

        Skips flext_core unresolved PEP 695 aliases and
        recursive container schemas are still incompatible with beartype.claw.
        """
        mode = c.ENFORCEMENT_MODE
        if mode == "off":
            return BeartypeConf(strategy=BeartypeStrategy.O0)
        return BeartypeConf(
            violation_type=UserWarning if mode == "warn" else TypeError,
            strategy=BeartypeStrategy.O1,
            claw_skip_package_names=FlextUtilitiesBeartypeConf.CLAW_SKIP_PACKAGES,
        )
