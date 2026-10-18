"""FlextServiceConfig - config/settings access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, p, s


class FlextServiceConfig(s[p.Base]):
    """Provide config and settings access through the public facade."""

    @property
    def config(self) -> p.Settings:
        """Return the config from this component's container."""
        result = self.container.resolve(c.Directory.CONFIG)
        if result.failure:
            msg = f"Failed to resolve config: {result.error}"
            raise RuntimeError(msg)
        return result.value

    @property
    def settings(self) -> p.Settings:
        """Return the settings from this component's runtime."""
        return self._get_runtime().settings


__all__: list[str] = ["FlextServiceConfig"]
