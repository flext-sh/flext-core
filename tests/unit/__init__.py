# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests.unit package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests.unit.test_lazy_composition_parts import (
        TestsFlextCoreLazyCompositionParts,
    )
    from tests.unit.test_lazy_dotted_bases import TestsFlextCoreLazyDottedBases
    from tests.unit.test_lazy_dynamic_members import TestsFlextCoreLazyDynamicMembers
    from tests.unit.test_utilities_models_family import (
        TestsFlextCoreUtilitiesModelsFamily,
    )


__all__: tuple[str, ...] = (
    "TestsFlextCoreLazyCompositionParts",
    "TestsFlextCoreLazyDottedBases",
    "TestsFlextCoreLazyDynamicMembers",
    "TestsFlextCoreUtilitiesModelsFamily",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "TestsFlextCoreLazyCompositionParts": ".test_lazy_composition_parts",
        "TestsFlextCoreLazyDottedBases": ".test_lazy_dotted_bases",
        "TestsFlextCoreLazyDynamicMembers": ".test_lazy_dynamic_members",
        "TestsFlextCoreUtilitiesModelsFamily": ".test_utilities_models_family",
    },
    public_exports=__all__,
)
