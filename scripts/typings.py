"""Type aliases for flext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextTypes


class ScriptsFlextTypes(FlextTypes):
    """Type aliases for flext."""


t = ScriptsFlextTypes

__all__: list[str] = ["ScriptsFlextTypes", "t"]
