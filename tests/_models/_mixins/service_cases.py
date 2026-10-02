"""Service and validation case model helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .service_case_core import TestsFlextModelsServiceCaseCoreMixin
from .service_case_reliability import TestsFlextModelsServiceCaseReliabilityMixin
from .service_case_validation import TestsFlextModelsServiceCaseValidationMixin


class TestsFlextModelsServiceCasesMixin(
    TestsFlextModelsServiceCaseReliabilityMixin,
    TestsFlextModelsServiceCaseValidationMixin,
    TestsFlextModelsServiceCaseCoreMixin,
):
    """Service and validation case model helpers."""


__all__: list[str] = ["TestsFlextModelsServiceCasesMixin"]
