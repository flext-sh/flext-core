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
from flext_tests.fixtures import FlextTestsFixtures
from flext_tests.http_support import FlextTestsHttp
from flext_tests.hypothesis import FlextTestsHypothesis
from flext_tests.matchers import FlextTestsMatchers
from flext_tests.parallel_docker import (
    ContainerSpec,
    ParallelDockerManager,
    get_client-a_oud_container,
    get_shared_openldap_container,
    release_client-a_oud_container,
    release_shared_openldap_container,
)
from flext_tests.performance import FlextTestsPerformance
from flext_tests.utilities import FlextTestsUtilities

__all__ = [
    "ContainerInfo",
    "ContainerSpec",
    "ContainerStatus",
    "FlextTestDocker",
    "FlextTestsAsyncs",
    "FlextTestsBuilders",
    "FlextTestsDomains",
    "FlextTestsFactories",
    "FlextTestsFixtures",
    "FlextTestsHttp",
    "FlextTestsHypothesis",
    "FlextTestsMatchers",
    "FlextTestsPerformance",
    "FlextTestsUtilities",
    "ParallelDockerManager",
    "get_client-a_oud_container",
    "get_shared_openldap_container",
    "release_client-a_oud_container",
    "release_shared_openldap_container",
]
