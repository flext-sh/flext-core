"""FlextServiceContainer - container access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import p, s


class FlextServiceContainer(s[p.Base]):
    """Provide container access through the public facade."""

    @property
    def container(self) -> p.Container:
        """Return the container from this component's runtime."""
        return self._get_runtime().container


__all__: list[str] = ["FlextServiceContainer"]
