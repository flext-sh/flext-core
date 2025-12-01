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

from flext_tests.builders import FlextTestsBuilders
from flext_tests.constants import FlextTestConstants
from flext_tests.docker import FlextTestDocker
from flext_tests.domains import FlextTestsDomains
from flext_tests.factories import FlextTestsFactories
from flext_tests.files import FlextTestsFileManager
from flext_tests.matchers import FlextTestsMatchers
from flext_tests.models import FlextTestModels
from flext_tests.protocols import FlextTestProtocols
from flext_tests.typings import FlextTestsTypings
from flext_tests.utilities import FlextTestsUtilities, ModelFactory

__all__ = [
    "FlextTestConstants",
    "FlextTestDocker",
    "FlextTestModels",
    "FlextTestProtocols",
    "FlextTestsBuilders",
    "FlextTestsDomains",
    "FlextTestsFactories",
    "FlextTestsFileManager",
    "FlextTestsMatchers",
    "FlextTestsTypings",
    "FlextTestsUtilities",
    "ModelFactory",
]
