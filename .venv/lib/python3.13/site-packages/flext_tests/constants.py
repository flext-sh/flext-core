"""Constants for FLEXT tests.

Provides FlextTestsConstants, extending FlextCliConstants with test-specific constants
for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra import c as infra_c
from flext_tests._constants.data_cases import FlextTestsConstantsDataCases
from flext_tests._constants.docker import FlextTestsConstantsDocker
from flext_tests._constants.files import FlextTestsConstantsFiles
from flext_tests._constants.make import FlextTestsConstantsMake
from flext_tests._constants.matcher import FlextTestsConstantsMatcher
from flext_tests._constants.validator import FlextTestsConstantsValidator


class FlextTestsConstants(infra_c):
    """Constants for FLEXT tests - extends FlextCliConstants.

    Architecture layer: Layer 0 foundation constants with test extensions.
    All base constants from FlextCliConstants are available through inheritance.
    """

    class Tests(
        FlextTestsConstantsDataCases,
        FlextTestsConstantsDocker,
        FlextTestsConstantsFiles,
        FlextTestsConstantsMake,
        FlextTestsConstantsMatcher,
        FlextTestsConstantsValidator,
    ):
        """Test-specific constants namespace.

        All test-specific constants are organized under this namespace to clearly
        distinguish them from base FlextCliConstants. Access via c.Tests.*
        """


c = FlextTestsConstants

__all__: tuple[str, ...] = ("FlextTestsConstants", "c")
