"""FlextServiceConfig - config/settings access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, p, r, s


class FlextServiceConfig(s[p.Base]):
    """Provide config and settings access through the public facade."""

    @property
    def config(self) -> p.Settings:
        """Config from this component's container."""
        result = self.container.resolve(c.Directory.CONFIG)
        if result.failure:
            return r[p.Settings].from_failure(result)
        return r[p.Settings].ok(result.value)

    @property
    def settings(self) -> p.Settings:
        """Settings from this component's runtime."""
        return self._get_runtime().settings


__all__: list[str] = ["FlextServiceConfig"]
