"""Service case factory helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._utilities import (
    TestsFlextUtilitiesCaseGeneratorsMixin,
    TestsFlextUtilitiesCaseServiceFactoriesMixin,
)


class TestsFlextUtilitiesCaseFactoriesMixin(
    TestsFlextUtilitiesCaseGeneratorsMixin,
    TestsFlextUtilitiesCaseServiceFactoriesMixin,
):
    """Service case factory helpers."""


u = TestsFlextUtilitiesCaseFactoriesMixin

__all__: list[str] = ["TestsFlextUtilitiesCaseFactoriesMixin", "u"]
