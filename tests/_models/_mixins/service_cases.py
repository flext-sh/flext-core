"""Service and validation case model helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._models._mixins import (
    TestsFlextModelsServiceCaseCore,
    TestsFlextModelsServiceCaseReliability,
    TestsFlextModelsServiceCaseValidation,
)


class TestsFlextModelsServiceCases(
    TestsFlextModelsServiceCaseReliability,
    TestsFlextModelsServiceCaseValidation,
    TestsFlextModelsServiceCaseCore,
):
    """Service and validation case model helpers."""


__all__: list[str] = ["TestsFlextModelsServiceCases"]
