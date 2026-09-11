"""Docker test infrastructure constants for flext_tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flext_infra import t


class FlextTestsConstantsDocker:
    """Docker test infrastructure constants mixin."""

    # Exact Make CI token (flext-infra config.codegen.make.ci); never treat
    # GitHub's CI=true as docker-disable.
    ENV_CI: Final[str] = "CI"
    CI_MAKE_VALUE: Final[str] = "Y"
    DOCKER_CI_SKIP_REASON: Final[str] = "docker disabled under CI=Y"
    DOCKER_CONNECTIVITY_MARKER: Final[str] = "docker"
    DOCKER_UNREACHABLE_SKIP_REASON: Final[str] = (
        "Docker daemon unreachable; start it to run Docker integration tests"
    )
    # Default probe ceiling for callers that omit max_wait. Under CI=Y the
    # Docker lifecycle skips before probing. Outside CI, shared-container
    # startup_timeout remains the SSOT for long boots (Oracle/kind).
    DOCKER_PROBE_MAX_WAIT_SECONDS: Final[int] = 8

    # Connectivity markers auto-skip when their service is unreachable
    # (AGENTS.md: "tests that need external/docker services skip when
    # unreachable"). Each marker maps to the shared container whose declared
    # host/port is probed once per session. A marker absent from this map is
    # never skipped, so adding one is a deliberate data change.
    CONNECTIVITY_MARKER_CONTAINERS: Final[Mapping[str, str]] = {
        "oracle": "flext-oracle-db-test",
        "ldap": "flext-openldap-test",
        "kubernetes": "flext-kind-test",
    }
    CONNECTIVITY_MARKERS: Final[tuple[str, ...]] = (
        DOCKER_CONNECTIVITY_MARKER,
        *CONNECTIVITY_MARKER_CONTAINERS,
    )
    UNREACHABLE_SKIP_REASON: Final[str] = (
        "{marker} service unreachable at {host}:{port}; start it to run these tests"
    )
    CONNECTIVITY_PROBE_TIMEOUT_SECONDS: Final[float] = 1.5

    SHARED_CONTAINERS: Final[Mapping[str, t.HeaderMapping]] = {
        "flext-openldap-test": {
            "compose_file": "docker/docker-compose.openldap.yml",
            "service": "openldap",
            "port": 3390,
            "host": "localhost",
        },
        "flext-oracle-db-test": {
            "compose_file": "docker/docker-compose.oracle-db.yml",
            "service": "oracle-db",
            "port": 1521,
            "host": "localhost",
            "startup_timeout": 900,
        },
        "flext-kind-test": {
            "compose_file": "docker/docker-compose.kubernetes.yml",
            "service": "kind",
            "port": 6443,
            "host": "localhost",
            "startup_timeout": 120,
        },
    }

    @unique
    class ContainerStatus(StrEnum):
        """Container status enumeration for test infrastructure."""

        CREATED = "created"
        RUNNING = "running"
        EXITED = "exited"
        PAUSED = "paused"
        REMOVING = "removing"
        DEAD = "dead"
        STOPPED = "stopped"
        NOT_FOUND = "not_found"
        ERROR = "error"
        STARTING = "starting"
        STOPPING = "stopping"
        RESTARTING = "restarting"
