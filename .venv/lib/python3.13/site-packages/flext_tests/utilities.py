"""Test utilities for FLEXT ecosystem tests.

Provides essential test utilities extending u with test-specific
helpers for result validation, context management, and test data creation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra import u
from flext_tests._utilities.container import FlextTestsContainerHelpersUtilitiesMixin
from flext_tests._utilities.files import FlextTestsFilesUtilitiesMixin
from flext_tests._utilities.fixtures_dsl import FlextTestsFixturesDSLMixin
from flext_tests._utilities.generic import FlextTestsGenericHelpersUtilitiesMixin
from flext_tests._utilities.handler import FlextTestsHandlerHelpersUtilitiesMixin
from flext_tests._utilities.make import FlextTestsMakeUtilitiesMixin
from flext_tests._utilities.matchers import FlextTestsMatchersUtilities
from flext_tests._utilities.result import FlextTestsResultUtilitiesMixin
from flext_tests._utilities.settings import FlextTestsConfigHelpersUtilitiesMixin
from flext_tests._utilities.testcontext import FlextTestsTestContextUtilitiesMixin
from flext_tests._utilities.validator import FlextTestsValidatorUtilitiesMixin
from flext_tests._utilities.workspace_cleanup import (
    FlextTestsWorkspaceCleanupUtilitiesMixin,
)


class FlextTestsUtilities(u):
    """Test utilities for FLEXT ecosystem - extends u.

    Provides essential test helpers that complement u.
    All u functionality is available via inheritance.
    """

    class Tests(
        FlextTestsResultUtilitiesMixin,
        FlextTestsTestContextUtilitiesMixin,
        FlextTestsGenericHelpersUtilitiesMixin,
        FlextTestsConfigHelpersUtilitiesMixin,
        FlextTestsContainerHelpersUtilitiesMixin,
        FlextTestsHandlerHelpersUtilitiesMixin,
        FlextTestsFilesUtilitiesMixin,
        FlextTestsMakeUtilitiesMixin,
        FlextTestsValidatorUtilitiesMixin,
        FlextTestsMatchersUtilities.Tests,
        FlextTestsFixturesDSLMixin,
        # NOTE (multi-agent): compose guarded cleanup planning/apply into u.Tests.
        FlextTestsWorkspaceCleanupUtilitiesMixin,
    ):
        """Test utilities namespace."""


u = FlextTestsUtilities

__all__: list[str] = ["FlextTestsFixturesDSLMixin", "FlextTestsUtilities", "u"]
