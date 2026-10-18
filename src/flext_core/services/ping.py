"""FlextServicePing - base service for flext-core facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import p, s


class FlextServicePing(s[p.Base]):
    """Base service for flext-core facade composition."""


__all__: list[str] = ["FlextServicePing"]
