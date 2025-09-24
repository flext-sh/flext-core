"""Pytest fixtures for FLEXT Docker container management.

Provides pytest integration for automatic container lifecycle management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_tests.docker import FlextTestDocker

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session")
def docker_control() -> FlextTestDocker:
    """Session-scoped Docker control instance."""
    return FlextTestDocker()


@pytest.fixture(scope="session")
def flext_ldap_container(docker_control: FlextTestDocker) -> Generator[str]:
    """Session-scoped LDAP container fixture.

    Starts the shared LDAP container for the session and ensures it's stopped at the end.
    """
    container_name = "flext-shared-ldap"

    status_result = docker_control.get_container_status(container_name)
    if status_result.is_success and status_result.value.status.value == "running":
        yield container_name
        return

    result = docker_control.start_container(container_name)
    if result.is_failure:
        pytest.skip(f"Failed to start LDAP container: {result.error}")

    yield container_name

    docker_control.stop_container(container_name, remove=False)


@pytest.fixture(scope="session")
def flext_postgres_container(docker_control: FlextTestDocker) -> Generator[str]:
    """Session-scoped PostgreSQL container fixture."""
    container_name = "flext-postgres"

    status_result = docker_control.get_container_status(container_name)
    if status_result.is_success and status_result.value.status.value == "running":
        yield container_name
        return

    result = docker_control.start_container(container_name)
    if result.is_failure:
        pytest.skip(f"Failed to start PostgreSQL container: {result.error}")

    yield container_name

    docker_control.stop_container(container_name, remove=False)


@pytest.fixture(scope="session")
def flext_redis_container(docker_control: FlextTestDocker) -> Generator[str]:
    """Session-scoped Redis container fixture."""
    container_name = "flext-redis"

    status_result = docker_control.get_container_status(container_name)
    if status_result.is_success and status_result.value.status.value == "running":
        yield container_name
        return

    result = docker_control.start_container(container_name)
    if result.is_failure:
        pytest.skip(f"Failed to start Redis container: {result.error}")

    yield container_name

    docker_control.stop_container(container_name, remove=False)


@pytest.fixture(scope="session")
def flext_oracle_container(docker_control: FlextTestDocker) -> Generator[str]:
    """Session-scoped Oracle DB container fixture."""
    container_name = "flext-oracle"

    status_result = docker_control.get_container_status(container_name)
    if status_result.is_success and status_result.value.status.value == "running":
        yield container_name
        return

    result = docker_control.start_container(container_name)
    if result.is_failure:
        pytest.skip(f"Failed to start Oracle container: {result.error}")

    yield container_name

    docker_control.stop_container(container_name, remove=False)


@pytest.fixture
def reset_ldap_container(docker_control: FlextTestDocker) -> str:
    """Function-scoped LDAP container with reset before each test."""
    container_name = "flext-shared-ldap"

    result = docker_control.reset_container(container_name)
    if result.is_failure:
        pytest.skip(f"Failed to reset LDAP container: {result.error}")

    return container_name


@pytest.fixture
def reset_postgres_container(docker_control: FlextTestDocker) -> str:
    """Function-scoped PostgreSQL container with reset before each test."""
    container_name = "flext-postgres"

    result = docker_control.reset_container(container_name)
    if result.is_failure:
        pytest.skip(f"Failed to reset PostgreSQL container: {result.error}")

    return container_name


@pytest.fixture
def all_containers_running(
    docker_control: FlextTestDocker,
) -> Generator[dict[str, str]]:
    """Ensure all FLEXT test containers are running."""
    result = docker_control.start_all()
    if result.is_failure:
        pytest.skip(f"Failed to start all containers: {result.error}")

    yield result.value

    docker_control.stop_all(remove=False)
