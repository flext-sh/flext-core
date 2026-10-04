"""Constants for flext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextConstants
from scripts import t


class ScriptsFlextConstants(FlextConstants):
    """Constants for flext."""


c = ScriptsFlextConstants

__all__: t.MutableSequenceOf[str] = ["ScriptsFlextConstants", "c"]
