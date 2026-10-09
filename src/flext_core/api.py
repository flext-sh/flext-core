"""API facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextApi:
    """API facade for flext-core."""


core: FlextApi = FlextApi()
"""Process-wide ``FlextApi`` facade singleton."""


__all__: list[str] = ["FlextApi", "core"]
