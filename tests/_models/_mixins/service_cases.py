"""Service and validation case model helper namespace."""

from __future__ import annotations

from tests._models._mixins.service_case_core import TestsFlextModelsServiceCaseCoreMixin
from tests._models._mixins.service_case_reliability import (
    TestsFlextModelsServiceCaseReliabilityMixin,
)
from tests._models._mixins.service_case_validation import (
    TestsFlextModelsServiceCaseValidationMixin,
)


class TestsFlextModelsServiceCasesMixin(
    TestsFlextModelsServiceCaseReliabilityMixin,
    TestsFlextModelsServiceCaseValidationMixin,
    TestsFlextModelsServiceCaseCoreMixin,
):
    """Service and validation case model helpers."""


__all__: list[str] = ["TestsFlextModelsServiceCasesMixin"]
