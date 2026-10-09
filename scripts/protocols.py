"""Protocol definitions for flext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextProtocols


class ScriptsFlextProtocols(FlextProtocols):
    """Protocol definitions for flext."""


p = ScriptsFlextProtocols

__all__: list[str] = ["ScriptsFlextProtocols", "p"]
