"""API facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Self


class FlextApi:
    """API facade for flext-core."""

    @classmethod
    def fetch_global(cls) -> Self:
        """Return the process-wide API facade instance."""
        return cls()


core: FlextApi = FlextApi.fetch_global()
"""Process-wide ``FlextApi`` facade singleton."""


__all__: list[str] = ["FlextApi", "core"]
