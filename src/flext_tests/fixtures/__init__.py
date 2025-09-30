"""FLEXT Test Fixtures - Unified Docker Management Integration.

This module provides centralized Docker fixtures for all FLEXT projects.
All Docker operations should use these fixtures to ensure consistency.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests.fixtures.docker_fixtures import (
    client-a_oud_container,
    flext_docker,
    ldap_container,
    oracle_container,
    postgres_container,
    redis_container,
)

__all__ = [
    "client-a_oud_container",
    "flext_docker",
    "ldap_container",
    "oracle_container",
    "postgres_container",
    "redis_container",
]
