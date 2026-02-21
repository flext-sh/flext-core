"""FLEXT Tests - Shared test utilities and fixtures package.

Provides comprehensive test infrastructure for the FLEXT ecosystem including
common test utilities, matchers, domain objects, factories, builders, Docker
container management, file operations, and integration with core FLEXT components.

Scope: Public API exports for all flext_tests modules including test utilities,
factories, builders, matchers, domain objects, Docker container management,
file operations, and re-exports of core FLEXT components for testing purposes.
All classes and utilities are designed for reuse across FLEXT test suites with
consistent patterns and comprehensive functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests.base import FlextTestsServiceBase, FlextTestsUtilityBase, s, su
from flext_tests.builders import FlextTestsBuilders, tb
from flext_tests.constants import FlextTestsConstants, c
from flext_tests.docker import FlextTestsDocker
from flext_tests.domains import FlextTestsDomains
from flext_tests.factories import FlextTestsFactories, f, tt
from flext_tests.files import FlextTestsFiles, tf
from flext_tests.matchers import FlextTestsMatchers, tm
from flext_tests.models import FlextTestsModels, m
from flext_tests.protocols import FlextTestsProtocols, p
from flext_tests.typings import FlextTestsTypes, t
from flext_tests.utilities import FlextTestsUtilities, u
from flext_tests.validator import FlextTestsValidator, tv

__all__ = [
    "FlextTestsBuilders",
    "FlextTestsConstants",
    "FlextTestsDocker",
    "FlextTestsDomains",
    "FlextTestsFactories",
    "FlextTestsFiles",
    "FlextTestsMatchers",
    "FlextTestsModels",
    "FlextTestsProtocols",
    "FlextTestsServiceBase",
    "FlextTestsTypes",
    "FlextTestsUtilities",
    "FlextTestsUtilityBase",
    "FlextTestsValidator",
    "c",
    "f",
    "m",
    "p",
    "s",
    "su",
    "t",
    "tb",
    "tf",
    "tm",
    "tt",
    "tv",
    "u",
]
