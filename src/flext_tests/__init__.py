"""FLEXT Core Test Support - ABSOLUTE USAGE OF FLEXT_TESTS.

COMPLETE flext_tests library with ALL working classes.
NO wrappers, NO aliases, ONLY direct class access.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests.asyncs import FlextTestsAsyncs
from flext_tests.builders import FlextTestsBuilders
from flext_tests.docker import (
    ContainerInfo,
    ContainerStatus,
    FlextTestDocker,
)
from flext_tests.domains import FlextTestsDomains
from flext_tests.factories import FlextTestsFactories
from flext_tests.fixtures import (
    flext_docker,
    ldap_container,
    oracle_container,
    postgres_container,
    redis_container,
)
from flext_tests.http_support import FlextTestsHttp
from flext_tests.hypothesis import FlextTestsHypothesis
from flext_tests.matchers import FlextTestsMatchers
from flext_tests.performance import FlextTestsPerformance
from flext_tests.utilities import FlextTestsUtilities

__all__ = [
    "ContainerInfo",
    "ContainerStatus",
    "FlextTestDocker",
    "FlextTestsAsyncs",
    "FlextTestsBuilders",
    "FlextTestsDomains",
    "FlextTestsFactories",
    "FlextTestsHttp",
    "FlextTestsHypothesis",
    "FlextTestsMatchers",
    "FlextTestsPerformance",
    "FlextTestsUtilities",
    "flext_docker",
    "ldap_container",
    "oracle_container",
    "postgres_container",
    "redis_container",
]
