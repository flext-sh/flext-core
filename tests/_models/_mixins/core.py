"""Core shared model helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .core_errors import TestsFlextModelsCoreErrorsMixin
from .core_public import TestsFlextModelsCorePublicMixin
from .core_state import TestsFlextModelsCoreStateMixin


class TestsFlextModelsCoreMixin(
    TestsFlextModelsCoreStateMixin,
    TestsFlextModelsCoreErrorsMixin,
    TestsFlextModelsCorePublicMixin,
):
    """Core shared model helpers."""


__all__: list[str] = ["TestsFlextModelsCoreMixin"]
