"""FlextServiceLazy - lazy export access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import p, s


class FlextServiceLazy(s[p.Base]):
    """Provide lazy export map access through the public facade."""

    @property
    def lazy(self) -> p.Lazy:
        """Lazy export map from this component's runtime."""
        return self._get_runtime().lazy


__all__: list[str] = ["FlextServiceLazy"]
