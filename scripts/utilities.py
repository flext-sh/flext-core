"""Utility functions for flext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities
from scripts import t


class ScriptsFlextUtilities(FlextUtilities):
    """Utility functions for flext."""


u = ScriptsFlextUtilities

__all__: t.MutableSequenceOf[str] = ["ScriptsFlextUtilities", "u"]
