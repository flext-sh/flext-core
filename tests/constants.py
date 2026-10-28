# AUTO-GENERATED FILE — Regenerate with: make gen
"""Constants for flext-core tests.

MRO-composed constants facade — all constants defined in _constants/ mixins.

Copyright (c) 2025 FLEXT Team. All rights reserved.
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

from flext_tests import c as flext_tests_c

from flext_core import c as flext_core_c
from tests._constants import (
    TestsFlextConstantsDomain,
    TestsFlextConstantsEnums,
    TestsFlextConstantsErrors,
    TestsFlextConstantsFixtures,
    TestsFlextConstantsOther,
    TestsFlextConstantsResult,
    TestsFlextConstantsServices,
)


class TestsFlextConstants(flext_tests_c, flext_core_c):
    """Layer 0 constants facade for flext-core tests."""

    class Tests(
        TestsFlextConstantsDomain,
        TestsFlextConstantsEnums,
        TestsFlextConstantsErrors,
        TestsFlextConstantsFixtures,
        TestsFlextConstantsOther,
        TestsFlextConstantsResult,
        TestsFlextConstantsServices,
        flext_tests_c.Tests,
    ):
        """Flat constants for test composition and parametrization.

        All constants come from MRO-composed mixins in _constants/.
        """


c = TestsFlextConstants
__all__: list[str] = ["TestsFlextConstants", "c"]
