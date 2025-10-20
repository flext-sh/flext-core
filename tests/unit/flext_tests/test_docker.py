"""Unit tests for flext_tests.docker module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flext_tests.docker import FlextTestDocker

# Access nested classes
ContainerInfo = FlextTestDocker.ContainerInfo
ContainerStatus = FlextTestDocker.ContainerStatus


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
        assert (
            info.status == ContainerStatus.RUNNING.value
        )  # Pydantic stores enum as value
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


class TestFlextTestDocker:
    """Test suite for FlextTestDocker class."""

    @pytest.fixture
    def docker_manager(self, tmp_path: Path) -> FlextTestDocker:
        """Create a FlextTestDocker instance for testing."""
        with (
            patch("flext_tests.docker.docker.from_env"),
            patch.object(FlextTestDocker, "_load_dirty_state"),
        ):
            # Create manager with isolated state
            manager = FlextTestDocker()
            # Override state file to temporary location for test isolation
            manager._state_file = tmp_path / "test_docker_state.json"
            # Clear any loaded dirty state for test isolation
            manager._dirty_containers.clear()
            manager._registered_services.clear()
            manager._service_dependencies.clear()
            return manager

    @pytest.fixture
    def mock_docker_client(self) -> MagicMock:
        """Create a mock Docker client."""
        return MagicMock()

    def test_init(self, docker_manager: FlextTestDocker) -> None:
        """Test FlextTestDocker initialization."""
        assert isinstance(docker_manager, FlextTestDocker)
        assert docker_manager.workspace_root is not None
        assert isinstance(docker_manager._dirty_containers, set)
        assert isinstance(docker_manager._registered_services, set)

    def test_get_client_initialization(self) -> None:
        """Test Docker client lazy initialization."""
        with patch("flext_tests.docker.docker.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_from_env.return_value = mock_client

            manager = FlextTestDocker()
            # Clear loaded state
            manager._dirty_containers.clear()

            client = manager.get_client()

            assert client is mock_client
            assert manager._client is mock_client
            mock_from_env.assert_called_once()

    def test_get_client_cached(self) -> None:
        """Test Docker client caching."""
        with patch("flext_tests.docker.docker.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_from_env.return_value = mock_client

            manager = FlextTestDocker()
            # Clear loaded state
            manager._dirty_containers.clear()

            # First call
            client1 = manager.get_client()
            # Second call should use cached client
            client2 = manager.get_client()

            assert client1 is client2
            mock_from_env.assert_called_once()

    def test_load_dirty_state_file_not_exists(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test loading dirty state when file doesn't exist."""
        # Set state file to non-existent path and reload
        docker_manager._state_file = Path("/tmp/definitely_non_existent_state.json")
        docker_manager._load_dirty_state()
        assert docker_manager._dirty_containers == set()

    def test_load_dirty_state_file_exists(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test loading dirty state from existing file."""
        test_data = {"dirty_containers": ["container1", "container2"]}

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, suffix=".json"
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

    def test_save_dirty_state(self, docker_manager: FlextTestDocker) -> None:
        """Test saving dirty state to file."""
        docker_manager._dirty_containers.add("test_container")

        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "test_state.json"
            docker_manager._state_file = state_file

            docker_manager._save_dirty_state()

            assert state_file.exists()
            with state_file.open("r") as f:
                data = json.load(f)
                assert "test_container" in data["dirty_containers"]

    def test_mark_container_dirty_success(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test marking container as dirty successfully."""
        with patch.object(docker_manager, "_save_dirty_state") as mock_save:
            result = docker_manager.mark_container_dirty("test_container")

            assert result.is_success
            assert "test_container" in docker_manager._dirty_containers
            mock_save.assert_called_once()

    def test_mark_container_dirty_failure(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test marking container as dirty with failure."""
        with patch.object(docker_manager, "_save_dirty_state") as mock_save:
            mock_save.side_effect = Exception("Save failed")

            result = docker_manager.mark_container_dirty("test_container")

            assert result.is_failure
            assert result.error is not None and "Save failed" in result.error

    def test_mark_container_clean_success(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test marking container as clean successfully."""
        docker_manager._dirty_containers.add("test_container")

        with patch.object(docker_manager, "_save_dirty_state") as mock_save:
            result = docker_manager.mark_container_clean("test_container")

            assert result.is_success
            assert "test_container" not in docker_manager._dirty_containers
            mock_save.assert_called_once()

    def test_mark_container_clean_failure(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test marking container as clean with failure."""
        with patch.object(docker_manager, "_save_dirty_state") as mock_save:
            mock_save.side_effect = Exception("Save failed")

            result = docker_manager.mark_container_clean("test_container")

            assert result.is_failure
            assert result.error is not None and "Save failed" in result.error

    def test_is_container_dirty(self, docker_manager: FlextTestDocker) -> None:
        """Test checking if container is dirty."""
        docker_manager._dirty_containers.add("dirty_container")

        assert docker_manager.is_container_dirty("dirty_container") is True
        assert docker_manager.is_container_dirty("clean_container") is False

    def test_get_dirty_containers(self, docker_manager: FlextTestDocker) -> None:
        """Test getting list of dirty containers."""
        docker_manager._dirty_containers.clear()
        docker_manager._dirty_containers.update(["container1", "container2"])

        dirty = docker_manager.get_dirty_containers()

        assert isinstance(dirty, list)
        assert set(dirty) == {"container1", "container2"}

    def test_start_all(self, docker_manager: FlextTestDocker) -> None:
        """Test start_all method."""
        result = docker_manager.start_all()

        assert result.is_success
        assert result.value == {"message": "All containers started"}

    def test_stop_all(self, docker_manager: FlextTestDocker) -> None:
        """Test stop_all method."""
        result = docker_manager.stop_all()

        assert result.is_success
        assert result.value == {"message": "All containers stopped"}

    def test_reset_all(self, docker_manager: FlextTestDocker) -> None:
        """Test reset_all method."""
        result = docker_manager.reset_all()

        assert result.is_success
        assert result.value == {"message": "All containers reset"}

    def test_reset_container(self, docker_manager: FlextTestDocker) -> None:
        """Test reset_container method."""
        result = docker_manager.reset_container("test_container")

        assert result.is_success
        assert result.value == "Container test_container reset"

    def test_get_all_status(self, docker_manager: FlextTestDocker) -> None:
        """Test get_all_status method."""
        result = docker_manager.get_all_status()

        assert result.is_success
        assert result.value == {}

    def test_register_service(self, docker_manager: FlextTestDocker) -> None:
        """Test register_service method."""
        result = docker_manager.register_service(
            service_name="test_service",
            container_name="test_container",
        )

        assert result.is_success
        assert "test_service" in docker_manager._registered_services
        assert docker_manager._service_dependencies["test_service"] == []

    def test_register_service_with_dependencies(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test register_service with dependencies."""
        result = docker_manager.register_service(
            service_name="test_service",
            container_name="test_container",
            depends_on=["dep1", "dep2"],
        )

        assert result.is_success
        assert docker_manager._service_dependencies["test_service"] == [
            "dep1",
            "dep2",
        ]

    def test_get_running_services(self, docker_manager: FlextTestDocker) -> None:
        """Test get_running_services method."""
        result = docker_manager.get_running_services()

        assert result.is_success
        assert result.value == []

    def test_compose_up(self, docker_manager: FlextTestDocker) -> None:
        """Test compose_up method."""
        result = docker_manager.compose_up("docker-compose.yml", "web")

        assert result.is_success
        assert "Compose stack started" in result.value

    def test_compose_down(self, docker_manager: FlextTestDocker) -> None:
        """Test compose_down method."""
        result = docker_manager.compose_down("docker-compose.yml")

        assert result.is_success
        assert "Compose stack stopped" in result.value

    def test_compose_logs(self, docker_manager: FlextTestDocker) -> None:
        """Test compose_logs method."""
        result = docker_manager.compose_logs("docker-compose.yml")

        assert result.is_success
        assert "Compose logs retrieved" in result.value

    def test_build_image_advanced(self, docker_manager: FlextTestDocker) -> None:
        """Test build_image_advanced method."""
        result = docker_manager.build_image_advanced(
            path="/app",
            tag="myapp:latest",
        )

        assert result.is_success
        assert "Image myapp:latest built successfully" in result.value

    def test_cleanup_volumes(self, docker_manager: FlextTestDocker) -> None:
        """Test cleanup_volumes method."""
        result = docker_manager.cleanup_volumes()

        assert result.is_success
        assert result.value["removed"] == 0
        assert result.value["volumes"] == []

    def test_cleanup_images(self, docker_manager: FlextTestDocker) -> None:
        """Test cleanup_images method."""
        result = docker_manager.cleanup_images()

        assert result.is_success
        assert result.value["removed"] == 0
        assert result.value["images"] == []

    def test_cleanup_networks(self, docker_manager: FlextTestDocker) -> None:
        """Test cleanup_networks method."""
        result = docker_manager.cleanup_networks()

        assert result.is_success
        assert result.value == []

    def test_cleanup_all_test_containers(self, docker_manager: FlextTestDocker) -> None:
        """Test cleanup_all_test_containers method."""
        result = docker_manager.cleanup_all_test_containers()

        assert result.is_success
        assert result.value["message"] == "All test containers cleaned up"

    def test_stop_services_for_test(self, docker_manager: FlextTestDocker) -> None:
        """Test stop_services_for_test method."""
        result = docker_manager.stop_services_for_test("test_session")

        assert result.is_success
        assert "Services stopped for test test_session" in result.value["message"]

    def test_get_service_dependency_graph(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test get_service_dependency_graph method."""
        docker_manager._service_dependencies = {
            "service1": ["dep1"],
            "service2": ["dep2"],
        }

        graph = docker_manager.get_service_dependency_graph()

        assert graph == {"service1": ["dep1"], "service2": ["dep2"]}

    def test_images_formatted(self, docker_manager: FlextTestDocker) -> None:
        """Test images_formatted method."""
        result = docker_manager.images_formatted()

        assert result.is_success
        assert result.value == ["test:latest"]

    def test_list_containers_formatted(self, docker_manager: FlextTestDocker) -> None:
        """Test list_containers_formatted method."""
        result = docker_manager.list_containers_formatted()

        assert result.is_success
        assert result.value == ["test_container_1", "test_container_2"]

    def test_list_networks(self, docker_manager: FlextTestDocker) -> None:
        """Test list_networks method."""
        result = docker_manager.list_networks()

        assert result.is_success
        assert result.value == []

    def test_list_volumes(self, docker_manager: FlextTestDocker) -> None:
        """Test list_volumes method."""
        result = docker_manager.list_volumes()

        assert result.is_success
        assert result.value == []

    def test_shared_containers_attribute(self, docker_manager: FlextTestDocker) -> None:
        """Test SHARED_CONTAINERS class attribute."""
        shared = FlextTestDocker.SHARED_CONTAINERS

        assert isinstance(shared, dict)
        assert "flext-openldap-test" in shared
        assert "flext-postgres-test" in shared
        assert "flext-redis-test" in shared
        assert "flext-oracle-db-test" in shared

        # Check structure of one container config
        ldap_config = shared["flext-openldap-test"]
        assert ldap_config["compose_file"] == "docker/docker-compose.openldap.yml"
        assert ldap_config["service"] == "openldap"
        assert ldap_config["port"] == 3390

    def test_start_container_with_image(self, docker_manager: FlextTestDocker) -> None:
        """Test start_container with image parameter."""
        with patch.object(docker_manager, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            result = docker_manager.start_container(
                "test_container", image="nginx:latest"
            )

            assert result.is_success
            assert "Container test_container started" in result.value
            mock_client.containers.run.assert_called_once()

    def test_start_container_docker_exception(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test start_container with Docker exception."""
        from docker.errors import DockerException

        with patch.object(docker_manager, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.containers.run.side_effect = DockerException("Docker error")
            mock_get_client.return_value = mock_client

            result = docker_manager.start_container("test_container")

            assert result.is_failure
            assert result.error is not None and "Failed to start container" in result.error

    def test_build_image(self, docker_manager: FlextTestDocker) -> None:
        """Test build_image method."""
        result = docker_manager.build_image(path="/app", tag="myapp:latest")

        assert result.is_success
        assert "Image myapp:latest built successfully" in result.value

    def test_remove_container_success(self, docker_manager: FlextTestDocker) -> None:
        """Test remove_container successfully."""
        with patch.object(docker_manager, "get_client") as mock_get_client:
            mock_container = MagicMock()
            mock_client = MagicMock()
            mock_client.containers.get.return_value = mock_container
            mock_get_client.return_value = mock_client

            result = docker_manager.remove_container("test_container")

            assert result.is_success
            assert "Container test_container removed" in result.value

    def test_remove_container_not_found(self, docker_manager: FlextTestDocker) -> None:
        """Test remove_container when container not found."""
        with patch.object(docker_manager, "get_client") as mock_get_client:
            from docker.errors import NotFound

            mock_client = MagicMock()
            mock_client.containers.get.side_effect = NotFound("Container not found")
            mock_get_client.return_value = mock_client

            result = docker_manager.remove_container("test_container")

            assert result.is_failure
            assert result.error is not None and "Container test_container not found" in result.error

    def test_remove_image_success(self, docker_manager: FlextTestDocker) -> None:
        """Test remove_image successfully."""
        with patch.object(docker_manager, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            result = docker_manager.remove_image("test_image:latest")

            assert result.is_success
            assert "Image test_image:latest removed" in result.value

    def test_container_logs_formatted_success(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test container_logs_formatted successfully."""
        with patch.object(docker_manager, "get_client") as mock_get_client:
            mock_container = MagicMock()
            mock_container.logs.return_value = b"test logs"
            mock_client = MagicMock()
            mock_client.containers.get.return_value = mock_container
            mock_get_client.return_value = mock_client

            result = docker_manager.container_logs_formatted("test_container")

            assert result.is_success
            assert result.value == "test logs"

    def test_execute_command_in_container_success(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test execute_command_in_container successfully."""
        with patch.object(docker_manager, "get_client") as mock_get_client:
            mock_container = MagicMock()
            mock_exec_result = MagicMock()
            mock_exec_result.output = b"command output"
            mock_container.exec_run.return_value = mock_exec_result
            mock_client = MagicMock()
            mock_client.containers.get.return_value = mock_container
            mock_get_client.return_value = mock_client

            result = docker_manager.execute_command_in_container(
                "test_container", "echo hello"
            )

            assert result.is_success
            assert result.value == "command output"
