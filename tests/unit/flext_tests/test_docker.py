"""Unit tests for simplified flext_tests.docker module.

Tests essential Docker container management functionality:
- Container status/info retrieval
- Dirty state tracking
- Docker-compose operations
- Port readiness checking

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from docker import DockerClient
from flext_core import FlextConstants, FlextResult
from flext_tests.constants import c as tests_c
from flext_tests.docker import FlextTestsDocker

from tests.test_utils import assertion_helpers

# Access nested classes
ContainerInfo = FlextTestsDocker.ContainerInfo
ContainerStatus = tests_c.Tests.Docker.ContainerStatus


class TestContainerStatus:
    """Test suite for ContainerStatus enum."""

    def test_container_status_values(self) -> None:
        """Test ContainerStatus enum values."""
        assert ContainerStatus.RUNNING.value == "running"
        assert ContainerStatus.STOPPED.value == "stopped"
        assert ContainerStatus.NOT_FOUND.value == "not_found"
        assert ContainerStatus.ERROR.value == "error"


class TestContainerInfo:
    """Test suite for ContainerInfo model."""

    def test_container_info_creation(self) -> None:
        """Test ContainerInfo creation with required fields."""
        info = ContainerInfo(
            name="test_container",
            status=ContainerStatus.RUNNING,
            ports={"8080/tcp": "8080"},
            image="nginx:latest",
        )

        assert info.name == "test_container"
        assert info.status == ContainerStatus.RUNNING.value
        assert info.ports == {"8080/tcp": "8080"}
        assert info.image == "nginx:latest"
        assert not info.container_id

    def test_container_info_with_container_id(self) -> None:
        """Test ContainerInfo with container_id."""
        info = ContainerInfo(
            name="test_container",
            status=ContainerStatus.RUNNING,
            ports={},
            image="nginx:latest",
            container_id="abc123",
        )

        assert info.container_id == "abc123"


class TestFlextTestsDocker:
    """Test suite for FlextTestsDocker class."""

    @pytest.fixture
    def docker_manager(self, tmp_path: Path) -> FlextTestsDocker:
        """Create a FlextTestsDocker instance for testing."""
        # Get fixtures directory
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures"

        # Create manager with isolated state
        manager = FlextTestsDocker(workspace_root=fixtures_dir)
        # Override state file to temporary location for test isolation
        manager._state_file = tmp_path / "test_docker_state.json"
        # Clear any loaded state
        manager._dirty_containers.clear()

        return manager

    def test_init(self, docker_manager: FlextTestsDocker) -> None:
        """Test FlextTestsDocker initialization."""
        assert isinstance(docker_manager, FlextTestsDocker)
        assert docker_manager.workspace_root is not None
        assert isinstance(docker_manager._dirty_containers, set)

    def test_get_client_initialization(self) -> None:
        """Test Docker client lazy initialization."""
        manager = FlextTestsDocker()
        manager._dirty_containers.clear()
        manager._client = None

        client = manager.get_client()

        assert client is not None
        assert manager._client is not None
        assert isinstance(client, DockerClient)

    def test_get_client_cached(self) -> None:
        """Test Docker client caching."""
        manager = FlextTestsDocker()
        manager._dirty_containers.clear()
        manager._client = None

        client1 = manager.get_client()
        client2 = manager.get_client()

        assert client1 is client2

    def test_load_dirty_state_file_not_exists(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test loading dirty state when file doesn't exist."""
        docker_manager._state_file = Path("/tmp/nonexistent_state.json")
        docker_manager._load_dirty_state()
        assert docker_manager._dirty_containers == set()

    def test_load_dirty_state_file_exists(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test loading dirty state from existing file."""
        test_data = {"dirty_containers": ["container1", "container2"]}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            delete=False,
            suffix=".json",
        ) as f:
            json.dump(test_data, f)
            temp_file = Path(f.name)

        try:
            docker_manager._state_file = temp_file
            docker_manager._load_dirty_state()

            assert "container1" in docker_manager._dirty_containers
            assert "container2" in docker_manager._dirty_containers
        finally:
            temp_file.unlink()

    def test_save_dirty_state(
        self,
        docker_manager: FlextTestsDocker,
        tmp_path: Path,
    ) -> None:
        """Test saving dirty state to file.

        Validates:
        1. Dirty state is saved to file correctly
        2. File is created in tmp_path (not permanent location)
        3. Saved data matches expected content
        4. File can be read back correctly
        """
        docker_manager._dirty_containers = {"test_container"}

        # Validate state file is in tmp_path (not permanent location)
        # The fixture sets _state_file to tmp_path / "test_docker_state.json"
        assert docker_manager._state_file.parent == tmp_path
        assert docker_manager._state_file.name == "test_docker_state.json"

        docker_manager._save_dirty_state()

        # Validate file was created
        assert docker_manager._state_file.exists()

        # Validate file content
        with docker_manager._state_file.open("r") as f:
            data = json.load(f)

        assert "dirty_containers" in data
        assert isinstance(data["dirty_containers"], list)
        assert "test_container" in data["dirty_containers"]

        # Validate data structure
        assert len(data["dirty_containers"]) == 1

    def test_mark_container_dirty(self, docker_manager: FlextTestsDocker) -> None:
        """Test marking container as dirty."""
        result = docker_manager.mark_container_dirty("test_container")

        assertion_helpers.assert_flext_result_success(result)
        assert "test_container" in docker_manager._dirty_containers

    def test_mark_container_clean(self, docker_manager: FlextTestsDocker) -> None:
        """Test marking container as clean."""
        docker_manager._dirty_containers.add("test_container")

        result = docker_manager.mark_container_clean("test_container")

        assertion_helpers.assert_flext_result_success(result)
        assert "test_container" not in docker_manager._dirty_containers

    def test_is_container_dirty(self, docker_manager: FlextTestsDocker) -> None:
        """Test checking if container is dirty."""
        docker_manager._dirty_containers.add("dirty_container")

        assert docker_manager.is_container_dirty("dirty_container")
        assert not docker_manager.is_container_dirty("clean_container")

    def test_get_dirty_containers(self, docker_manager: FlextTestsDocker) -> None:
        """Test getting list of dirty containers."""
        docker_manager._dirty_containers = {"container1", "container2"}

        dirty = docker_manager.get_dirty_containers()

        assert len(dirty) == 2
        assert "container1" in dirty
        assert "container2" in dirty

    def test_shared_containers_attribute(self) -> None:
        """Test SHARED_CONTAINERS class attribute."""
        assert FlextTestsDocker.SHARED_CONTAINERS is not None
        assert isinstance(FlextTestsDocker.SHARED_CONTAINERS, dict)

    def test_shared_containers_property(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test shared_containers property."""
        containers = docker_manager.shared_containers

        assert containers is not None
        assert isinstance(containers, dict)

    def test_compose_up_returns_flext_result(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test compose_up returns FlextResult."""
        result = docker_manager.compose_up("docker-compose.yml")

        assert isinstance(result, FlextResult)

    def test_compose_down_returns_flext_result(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test compose_down returns FlextResult."""
        result = docker_manager.compose_down("docker-compose.yml")

        assert isinstance(result, FlextResult)

    def test_start_existing_container_not_found(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test starting non-existent container."""
        result = docker_manager.start_existing_container("nonexistent_container")

        assertion_helpers.assert_flext_result_failure(result)
        assert "not found" in str(result.error).lower()

    def test_get_container_info_not_found(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test getting info for non-existent container."""
        result = docker_manager.get_container_info("nonexistent_container")

        assertion_helpers.assert_flext_result_failure(result)
        assert "not found" in str(result.error).lower()

    def test_get_container_status_alias(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test get_container_status is alias for get_container_info."""
        result = docker_manager.get_container_status("nonexistent")

        assertion_helpers.assert_flext_result_failure(result)

    def test_wait_for_port_ready_immediate(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test wait_for_port_ready returns quickly for unavailable port."""
        result = docker_manager.wait_for_port_ready(
            FlextConstants.Network.LOOPBACK_IP, 59999, max_wait=1,
        )

        assertion_helpers.assert_flext_result_success(result)
        assert result.value is False

    def test_start_compose_stack_returns_result(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test start_compose_stack returns FlextResult."""
        result = docker_manager.start_compose_stack("docker-compose.yml")

        assert isinstance(result, FlextResult)

    def test_cleanup_dirty_containers_empty(
        self,
        docker_manager: FlextTestsDocker,
    ) -> None:
        """Test cleanup with no dirty containers."""
        docker_manager._dirty_containers.clear()

        result = docker_manager.cleanup_dirty_containers()

        assertion_helpers.assert_flext_result_success(result)
        assert result.value == []


class TestFlextTestsDockerWorkerId:
    """Test worker_id functionality."""

    def test_default_worker_id(self) -> None:
        """Test default worker_id is 'master'."""
        manager = FlextTestsDocker()
        manager._dirty_containers.clear()

        assert manager.worker_id == "master"

    def test_custom_worker_id(self) -> None:
        """Test custom worker_id."""
        manager = FlextTestsDocker(worker_id="worker_1")
        manager._dirty_containers.clear()

        assert manager.worker_id == "worker_1"

    def test_state_file_includes_worker_id(self) -> None:
        """Test state file path includes worker_id."""
        manager = FlextTestsDocker(worker_id="test_worker")
        manager._dirty_containers.clear()

        assert "test_worker" in str(manager._state_file)


class TestFlextTestsDockerWorkspaceRoot:
    """Test workspace_root functionality."""

    def test_default_workspace_root(self) -> None:
        """Test default workspace_root is cwd."""
        manager = FlextTestsDocker()
        manager._dirty_containers.clear()

        assert manager.workspace_root == Path.cwd()

    def test_custom_workspace_root(self, tmp_path: Path) -> None:
        """Test custom workspace_root."""
        manager = FlextTestsDocker(workspace_root=tmp_path)
        manager._dirty_containers.clear()

        assert manager.workspace_root == tmp_path
