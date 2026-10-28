"""Domain models for flext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence

from flext_core import FlextModels


class ScriptsFlextModels(FlextModels):
    """Domain models for flext."""


m = ScriptsFlextModels

__all__: MutableSequence[str] = ["ScriptsFlextModels", "m"]
