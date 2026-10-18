"""FlextServiceRegistry - registry access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import m, p, r, s, u


class FlextServiceRegistry(s[p.Base]):
    """Provide registry access through the public facade."""

    @property
    def registry(self) -> p.Registry:
        """Return the registry from this component's container."""
        from flext_core import registry as registry_module

        runtime = self._get_runtime()
        registry_instance = registry_module.FlextRegistry(
            registry_runtime=u.build_service_runtime(
                m.RuntimeBootstrapOptions(dispatcher=runtime.dispatcher)
            )
        )
        if not isinstance(registry_instance, p.Registry):
            return r[p.Registry].fail(
                u.TypeError(
                    "Resolved registry implementation does not satisfy p.Registry"
                )
            )
        return r[p.Registry].ok(registry_instance)


__all__: list[str] = ["FlextServiceRegistry"]
