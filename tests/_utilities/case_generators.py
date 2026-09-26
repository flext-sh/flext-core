"""Service case generator helpers for flext-core tests."""

from __future__ import annotations

from .case_service_factories import TestsFlextUtilitiesCaseServiceFactoriesMixin


class TestsFlextUtilitiesCaseGeneratorsMixin(
    TestsFlextUtilitiesCaseServiceFactoriesMixin
):
    """Service case generator helpers."""

    @staticmethod
    def reset_all_factories() -> None:
        """Reset all factory states for test isolation."""
        TestsFlextUtilitiesCaseGeneratorsMixin.UserFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.GetUserServiceFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ValidatingServiceFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.GetUserServiceAutoFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ValidatingServiceAutoFactory.reset()
        TestsFlextUtilitiesCaseGeneratorsMixin.ServiceTestCaseFactory.reset()


__all__: list[str] = ["TestsFlextUtilitiesCaseGeneratorsMixin"]
