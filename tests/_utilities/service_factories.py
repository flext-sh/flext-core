"""Service factory helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._utilities import (
    TestsFlextUtilitiesUserFactoriesMixin,
    TestsFlextUtilitiesValidationFactoriesMixin,
)


class TestsFlextUtilitiesServiceFactoriesMixin(
    TestsFlextUtilitiesValidationFactoriesMixin,
    TestsFlextUtilitiesUserFactoriesMixin,
):
    """Service factory helpers."""


__all__: list[str] = ["TestsFlextUtilitiesServiceFactoriesMixin"]
