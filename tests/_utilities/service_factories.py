"""Service factory helper namespace."""

from __future__ import annotations

from .user_factories import TestsFlextUtilitiesUserFactoriesMixin
from .validation_factories import TestsFlextUtilitiesValidationFactoriesMixin


class TestsFlextUtilitiesServiceFactoriesMixin(
    TestsFlextUtilitiesValidationFactoriesMixin, TestsFlextUtilitiesUserFactoriesMixin
):
    """Service factory helpers."""


__all__: list[str] = ["TestsFlextUtilitiesServiceFactoriesMixin"]
