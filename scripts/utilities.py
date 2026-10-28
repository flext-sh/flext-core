"""Utility functions for flext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence

from flext_core import FlextUtilities


class ScriptsFlextUtilities(FlextUtilities):
    """Utility functions for flext."""


u = ScriptsFlextUtilities

__all__: MutableSequence[str] = ["ScriptsFlextUtilities", "u"]
