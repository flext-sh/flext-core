"""Service case generator helpers for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._utilities import TestsFlextUtilitiesCaseServiceFactoriesMixin


class TestsFlextUtilitiesCaseGeneratorsMixin(
    TestsFlextUtilitiesCaseServiceFactoriesMixin,
):
    """Service case generator helpers."""

    @staticmethod
    def reset_all_factories() -> None:
        """Reset all factory states for test isolation."""
        TestsFlextUtilitiesCaseGeneratorsMixin.GetUserServiceFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ValidatingServiceFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.GetUserServiceAutoFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ValidatingServiceAutoFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ServiceTestCaseFactory.reset()


__all__: list[str] = ["TestsFlextUtilitiesCaseGeneratorsMixin"]
