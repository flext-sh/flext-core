"""FlextServiceRegistry - registry access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import p, s


class FlextServiceRegistry(s[p.Base]):
    """Provide registry access through the public facade."""

    @property
    def registry(self) -> p.Registry:
        """Registry from this component's container."""
        return self.container.registry()


__all__: list[str] = ["FlextServiceRegistry"]
