"""Service factory helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._utilities.user_factories import TestsFlextUtilitiesUserFactoriesMixin
from tests._utilities.validation_factories import (
    TestsFlextUtilitiesValidationFactoriesMixin,
)


class TestsFlextUtilitiesServiceFactoriesMixin(
    TestsFlextUtilitiesValidationFactoriesMixin,
    TestsFlextUtilitiesUserFactoriesMixin,
):
    """Service factory helpers."""


__all__: list[str] = ["TestsFlextUtilitiesServiceFactoriesMixin"]
