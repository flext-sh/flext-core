"""Centralized BeartypeConf factory for FLEXT ecosystem.

Provides a single configuration point that downstream projects inherit. The
mode is the operational ``FlextSettings.beartype_mode``: ``warn`` raises
``UserWarning``, ``strict`` raises ``TypeError``, ``off`` checks nothing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beartype import BeartypeConf, BeartypeStrategy

from flext_core import c

if TYPE_CHECKING:
    from flext_core import FlextSettings


class FlextUtilitiesBeartypeConf:
    """Centralized beartype configuration for the FLEXT ecosystem.

    All methods are static. Exposed via u.build_beartype_conf() for
    downstream projects to use in their beartype_this_package() calls.
    """

    @staticmethod
    def build_beartype_conf(mode: FlextSettings.BeartypeMode) -> BeartypeConf:
        """Build the BeartypeConf of one beartype mode.

        Args:
            mode: ``off`` (O0, no checking), ``warn`` or ``strict``.

        Returns:
            The resulting ``BeartypeConf``.

        """
        if mode == "off":
            return BeartypeConf(strategy=BeartypeStrategy.O0)
        return BeartypeConf(
            violation_type=UserWarning if mode == "warn" else TypeError,
            strategy=BeartypeStrategy.O1,
            claw_skip_package_names=c.BEARTYPE_CLAW_SKIP_PACKAGES,
        )


__all__: tuple[str, ...] = ("FlextUtilitiesBeartypeConf",)
