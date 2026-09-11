"""Test matchers and assertions for FLEXT ecosystem tests.

Provides unified assertion API with powerful generalist methods.
Short alias: tm (test matchers)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_tests._utilities._matchers._assertions import (
    FlextTestsMatchersAssertionsMixin,
)
from flext_tests._utilities._matchers._containment import (
    FlextTestsMatchersContainmentMixin,
)
from flext_tests._utilities._matchers._immutability import (
    FlextTestsMatchersImmutabilityMixin,
)
from flext_tests._utilities._matchers._result import FlextTestsMatchersResultMixin
from flext_tests._utilities._matchers._scope import FlextTestsMatchersScopeMixin
from flext_tests._utilities._matchers._that import FlextTestsMatchersThatMixin
from flext_tests._utilities._matchers._typeguards import (
    FlextTestsMatchersTypeGuardsMixin,
)


class FlextTestsMatchersUtilities:
    """Namespace for test matcher utilities used in flext-tests."""

    class Tests:
        """Container for test utility storages and aliases."""

        class Matchers(
            FlextTestsMatchersContainmentMixin,
            FlextTestsMatchersImmutabilityMixin.Tests.Matchers,
            FlextTestsMatchersResultMixin.Tests.Matchers,
            FlextTestsMatchersScopeMixin.Tests.Matchers,
            FlextTestsMatchersThatMixin.Tests.Matchers,
            FlextTestsMatchersTypeGuardsMixin,
            FlextTestsMatchersAssertionsMixin,
        ):
            """Test matchers with powerful generalist methods."""


tm = FlextTestsMatchersUtilities.Tests.Matchers
__all__: list[str] = ["FlextTestsMatchersUtilities", "tm"]
