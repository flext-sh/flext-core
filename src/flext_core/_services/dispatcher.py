"""FlextServiceDispatcher - dispatcher access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import flext_core


class FlextServiceDispatcher(flext_core.s[flext_core.p.Base]):
    """Provide dispatcher access through the public facade."""

    @property
    def dispatcher(self) -> flext_core.p.Dispatcher:
        """Dispatcher from this component's container."""
        return self.container.dispatcher().unwrap()


__all__: list[str] = ["FlextServiceDispatcher"]
