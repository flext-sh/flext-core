"""Service case factory helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._utilities.case_generators import TestsFlextUtilitiesCaseGeneratorsMixin
from tests._utilities.case_service_factories import (
    TestsFlextUtilitiesCaseServiceFactoriesMixin,
)


class TestsFlextUtilitiesCaseFactoriesMixin(
    TestsFlextUtilitiesCaseGeneratorsMixin,
    TestsFlextUtilitiesCaseServiceFactoriesMixin,
):
    """Service case factory helpers."""


u = TestsFlextUtilitiesCaseFactoriesMixin

__all__: list[str] = ["TestsFlextUtilitiesCaseFactoriesMixin", "u"]
