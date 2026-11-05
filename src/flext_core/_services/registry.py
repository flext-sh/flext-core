"""FlextServiceRegistry - registry access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import flext_core


class FlextServiceRegistry(
    flext_core.s[flext_core.p.Base], flext_core.m.RegistryServiceState
):
    """Provide registry access through the public facade."""

    @property
    def registry(self) -> flext_core.p.Registry:
        """Persistent registry bound to this component's exact runtime."""
        with self._registry_lock:
            registry = self._registry
            if registry is None:
                registry = flext_core.FlextRegistry(
                    registry_runtime=self._get_runtime()
                )
                self._registry = registry
            return registry


__all__: list[str] = ["FlextServiceRegistry"]
