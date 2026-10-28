"""Service and validation case model helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from tests._models._mixins import (
    TestsFlextModelsServiceCaseCoreMixin,
    TestsFlextModelsServiceCaseReliabilityMixin,
    TestsFlextModelsServiceCaseValidationMixin,
)


class TestsFlextModelsServiceCasesMixin(
    TestsFlextModelsServiceCaseReliabilityMixin,
    TestsFlextModelsServiceCaseValidationMixin,
    TestsFlextModelsServiceCaseCoreMixin,
):
    """Service and validation case model helpers."""


__all__: list[str] = ["TestsFlextModelsServiceCasesMixin"]
