"""FLEXT Docker Test Fixtures - Unified Docker Management Integration.

This module provides Docker-specific fixtures for integration testing using the
unified FlextTestDocker class. This eliminates direct docker module usage and
provides consistent Docker management across all FLEXT projects.

ARCHITECTURAL PRINCIPLE: All Docker operations go through FlextTestDocker
to maintain consistency and eliminate duplicate Docker management logic.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from flext_core import FlextConstants
from flext_tests import FlextTestDocker


@pytest.fixture(scope="session")
def flext_docker() -> Generator[FlextTestDocker]:
    """FlextTestDocker unified management fixture for all Docker operations."""
    docker_manager = FlextTestDocker()
    try:
        yield docker_manager
    finally:
        # Cleanup handled by FlextTestDocker automatically
        pass


@pytest.fixture
def postgres_container(flext_docker: FlextTestDocker) -> Generator[str]:
    """PostgreSQL container fixture using unified FlextTestDocker management."""
    # Start the shared PostgreSQL container
    start_result = flext_docker.start_container("flext-postgres-test")
    if start_result.is_failure:
        pytest.skip(f"PostgreSQL container failed to start: {start_result.error}")

    # Get container status to extract connection info
    status_result = flext_docker.get_container_status("flext-postgres-test")
    if status_result.is_failure:
        pytest.skip(f"Failed to get PostgreSQL container status: {status_result.error}")

    container_info = status_result.value
    if "5433" not in container_info.ports:
        pytest.skip("PostgreSQL container port not available")

    host_port = container_info.ports["5433"]

    # Return connection string compatible with existing tests
    connection_string = f"postgresql://flext:flext_password@{FlextConstants.Platform.DEFAULT_HOST}:{host_port}/flext_db"

    try:
        yield connection_string
    finally:
        # Container cleanup is managed by FlextTestDocker lifecycle
        # No explicit stop needed - shared containers persist across tests
        pass


@pytest.fixture
def ldap_container(flext_docker: FlextTestDocker) -> Generator[str]:
    """OpenLDAP container fixture using unified FlextTestDocker management."""
    # Start the shared OpenLDAP container
    start_result = flext_docker.start_container("flext-openldap-test")
    if start_result.is_failure:
        pytest.skip(f"OpenLDAP container failed to start: {start_result.error}")

    # Get container status to extract connection info
    status_result = flext_docker.get_container_status("flext-openldap-test")
    if status_result.is_failure:
        pytest.skip(f"Failed to get OpenLDAP container status: {status_result.error}")

    container_info = status_result.value
    if "3390" not in container_info.ports:
        pytest.skip("OpenLDAP container port not available")

    host_port = container_info.ports["3390"]

    # Return connection string compatible with existing tests
    connection_string = f"ldap://{FlextConstants.Platform.DEFAULT_HOST}:{host_port}"

    try:
        yield connection_string
    finally:
        # Container cleanup is managed by FlextTestDocker lifecycle
        # No explicit stop needed - shared containers persist across tests
        pass


@pytest.fixture
def algar_oud_container(flext_docker: FlextTestDocker) -> Generator[str]:
    """Oracle Unified Directory container fixture for ALGAR testing.

    This fixture provides a real Oracle Unified Directory container configured
    for ALGAR Telecom testing with production-compatible port mapping:
    - Base DN: dc=example,dc=com (generic test data)
    - Port: 3389 (ALGAR production-compatible port mapping)
    - Admin: cn=Directory Manager
    - Password: TestPassword123
    - Single structural objectclass behavior enabled

    NOTE: Uses generic test credentials only - no production data.
    Port 3389 is mapped to container's internal 1389 for ALGAR compatibility.
    """
    # Start the ALGAR OUD test container
    start_result = flext_docker.start_container("flext-algar-oud-test")
    if start_result.is_failure:
        pytest.skip(f"ALGAR OUD container failed to start: {start_result.error}")

    # Get container status to extract connection info
    status_result = flext_docker.get_container_status("flext-algar-oud-test")
    if status_result.is_failure:
        pytest.skip(f"Failed to get ALGAR OUD container status: {status_result.error}")

    container_info = status_result.value
    if "3389" not in container_info.ports:
        pytest.skip("ALGAR OUD container port 3389 not available")

    host_port = container_info.ports["3389"]

    # Return connection string for ALGAR tests
    connection_string = f"ldap://{FlextConstants.Platform.DEFAULT_HOST}:{host_port}"

    try:
        yield connection_string
    finally:
        # Container cleanup is managed by FlextTestDocker lifecycle
        # No explicit stop needed - shared containers persist across tests
        pass


@pytest.fixture
def redis_container(flext_docker: FlextTestDocker) -> Generator[str]:
    """Redis container fixture using unified FlextTestDocker management."""
    # Start the shared Redis container
    start_result = flext_docker.start_container("flext-redis-test")
    if start_result.is_failure:
        pytest.skip(f"Redis container failed to start: {start_result.error}")

    # Get container status to extract connection info
    status_result = flext_docker.get_container_status("flext-redis-test")
    if status_result.is_failure:
        pytest.skip(f"Failed to get Redis container status: {status_result.error}")

    container_info = status_result.value
    if "6380" not in container_info.ports:
        pytest.skip("Redis container port not available")

    host_port = container_info.ports["6380"]

    # Return connection string compatible with existing tests
    connection_string = f"redis://{FlextConstants.Platform.DEFAULT_HOST}:{host_port}/0"

    try:
        yield connection_string
    finally:
        # Container cleanup is managed by FlextTestDocker lifecycle
        # No explicit stop needed - shared containers persist across tests
        pass


@pytest.fixture
def oracle_container(flext_docker: FlextTestDocker) -> Generator[str]:
    """Oracle Database container fixture using unified FlextTestDocker management."""
    # Start the shared Oracle container
    start_result = flext_docker.start_container("flext-oracle-db-test")
    if start_result.is_failure:
        pytest.skip(f"Oracle container failed to start: {start_result.error}")

    # Get container status to extract connection info
    status_result = flext_docker.get_container_status("flext-oracle-db-test")
    if status_result.is_failure:
        pytest.skip(f"Failed to get Oracle container status: {status_result.error}")

    container_info = status_result.value
    if "1522" not in container_info.ports:
        pytest.skip("Oracle container port not available")

    host_port = container_info.ports["1522"]

    # Return connection string compatible with existing tests
    connection_string = f"oracle://flext:flext_password@{FlextConstants.Platform.DEFAULT_HOST}:{host_port}/FLEXT"

    try:
        yield connection_string
    finally:
        # Container cleanup is managed by FlextTestDocker lifecycle
        # No explicit stop needed - shared containers persist across tests
        pass


# Legacy compatibility fixtures - deprecated but maintained for backward compatibility
@pytest.fixture(scope="session")
def docker_client(flext_docker: FlextTestDocker) -> object:
    """Deprecated: Direct Docker client access. Use flext_docker fixture instead.

    This fixture is maintained for backward compatibility but is deprecated.
    New tests should use the flext_docker fixture directly.
    """
    # Return the FlextTestDocker client for legacy compatibility
    return flext_docker.client
