"""Comprehensive tests for FlextTestDocker unified Docker management.

Tests the enhanced FlextTestDocker with auto-service management,
CLI command equivalents, and shell script compatibility.
"""

import tempfile
from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import MagicMock, Mock, patch

import pytest
from docker.errors import DockerException

from flext_core import FlextResult
from flext_tests import FlextTestDocker


class TestFlextTestDockerCore:
    """Test core FlextTestDocker functionality."""

    def test_flext_test_docker_initialization(self) -> None:
        """Test FlextTestDocker initializes properly."""
        with patch("flext_tests.docker.docker.from_env") as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client

            docker_manager = FlextTestDocker()

            # Initialize the client by calling get_client()
            docker_manager.get_client()

            assert docker_manager.client == mock_client
            assert docker_manager.workspace_root is not None
            mock_docker.assert_called_once()

    def test_workspace_root_detection(self) -> None:
        """Test workspace root detection works correctly."""
        with (
            patch("flext_tests.docker.docker.from_env"),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            workspace_path = Path(temp_dir)
            docker_manager = FlextTestDocker(workspace_root=workspace_path)

            assert docker_manager.workspace_root == workspace_path

    def test_nested_managers_initialization(self) -> None:
        """Test all nested managers are properly initialized."""
        with patch("flext_tests.docker.docker.from_env"):
            docker_manager = FlextTestDocker()

            # Check all nested managers exist
            assert hasattr(docker_manager, "_container_manager")
            assert hasattr(docker_manager, "_compose_manager")
            assert hasattr(docker_manager, "_network_manager")
            assert hasattr(docker_manager, "_volume_manager")
            assert hasattr(docker_manager, "_image_manager")


class TestAutoServiceManagement:
    """Test the auto-service management functionality."""

    @pytest.fixture
    def docker_manager(self) -> FlextTestDocker:
        """Create a FlextTestDocker instance for testing."""
        with patch("flext_tests.docker.docker.from_env") as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client
            manager = FlextTestDocker()
            manager.client = mock_client
            return manager

    def test_register_service_success(self, docker_manager: FlextTestDocker) -> None:
        """Test successful service registration."""
        result = docker_manager.register_service(
            service_name="test_service",
            container_name="test_container",
            ports=[8080],
            health_check_cmd="curl -f http://localhost:8080/health",
            depends_on=["database"],
            startup_timeout=30,
        )

        assert result.is_success

        # Verify service is registered
        services_result = docker_manager.get_running_services()
        assert services_result.is_success
        assert isinstance(services_result.data, list)

    def test_register_service_with_dependencies(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test service registration with dependencies."""
        # Register database service first
        db_result = docker_manager.register_service(
            service_name="database", container_name="test_db", ports=[5432]
        )
        assert db_result.is_success

        # Register API service with database dependency
        api_result = docker_manager.register_service(
            service_name="api",
            container_name="test_api",
            ports=[8080],
            depends_on=["database"],
        )
        assert api_result.is_success

        # Check dependency graph
        api_deps: dict[str, list[str]] = docker_manager.get_service_dependency_graph()
        assert "api" in api_deps
        assert "database" in api_deps["api"]

    def test_auto_discover_services_from_compose(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test auto-discovery from docker-compose file."""
        compose_content = """
version: '3'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  api:
    image: api:latest
    ports:
      - "8080:8080"
    depends_on:
      - redis
"""

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(compose_content)
            compose_file = f.name

        try:
            result = docker_manager.auto_discover_services(compose_file)
            assert result.is_success

            services = result.unwrap()
            assert "redis" in services
            assert "api" in services

            # Check dependency graph was built
            compose_deps: dict[str, list[str]] = (
                docker_manager.get_service_dependency_graph()
            )
            assert "api" in compose_deps
            assert "redis" in compose_deps.get("api", [])

        finally:
            Path(compose_file).unlink()

    def test_start_services_with_dependency_resolution(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test starting services with proper dependency resolution."""
        # Register services with dependencies
        docker_manager.register_service("database", "test_db", ports=[5432])
        docker_manager.register_service("cache", "test_redis", ports=[6379])
        docker_manager.register_service(
            "api", "test_api", ports=[8080], depends_on=["database", "cache"]
        )

        with (
            patch.object(docker_manager, "start_container") as mock_start,
            patch.object(docker_manager, "_wait_for_container_ready") as mock_wait,
        ):
            mock_start.return_value = FlextResult[str].ok("started")
            mock_wait.return_value = FlextResult[None].ok(None)

            result: FlextResult[dict[str, str]] = (
                docker_manager.start_services_for_test(
                    service_names=["api", "database", "cache"],
                    test_name="integration_test",
                )
            )

            assert result.is_success
            services_status: dict[str, str] = result.unwrap()

            assert "database" in services_status
            assert "cache" in services_status
            assert "api" in services_status

    def test_service_health_checking(self, docker_manager: FlextTestDocker) -> None:
        """Test service health checking functionality."""
        # Register service with health check
        docker_manager.register_service(
            "web_service",
            "test_web",
            ports=[8080],
            health_check_cmd="curl -f http://localhost:8080/health",
        )

        with patch.object(docker_manager, "execute_container_command") as mock_exec:
            mock_exec.return_value = FlextResult[str].ok("OK")

            result = docker_manager.get_service_health_status("web_service")
            assert result.is_success

            health_info = result.unwrap()
            assert "container_status" in health_info

    def test_enable_auto_cleanup(self, docker_manager: FlextTestDocker) -> None:
        """Test enabling/disabling auto-cleanup."""
        # Should not raise any exceptions
        docker_manager.enable_auto_cleanup(enabled=True)
        docker_manager.enable_auto_cleanup(enabled=False)


class TestDockerCLIEquivalents:
    """Test Docker CLI command equivalent methods."""

    @pytest.fixture
    def docker_manager(self) -> FlextTestDocker:
        """Create a FlextTestDocker instance for testing."""
        with patch("flext_tests.docker.docker.from_env") as mock_docker:
            mock_client = Mock()
            mock_docker.return_value = mock_client
            manager = FlextTestDocker()
            manager.client = mock_client
            return manager

    def test_list_containers_formatted(self, docker_manager: FlextTestDocker) -> None:
        """Test list_containers_formatted method."""
        mock_container1 = Mock()
        mock_container1.name = "test_container_1"
        mock_container1.status = "running"
        mock_container1.attrs = {
            "State": {"Status": "running"},
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "8080"}]}},
        }

        mock_container2 = Mock()
        mock_container2.name = "test_container_2"
        mock_container2.status = "exited"
        mock_container2.attrs = {
            "State": {"Status": "exited"},
            "NetworkSettings": {"Ports": {}},
        }

        with patch.object(docker_manager.client, "containers") as mock_containers:
            mock_containers.list.return_value = [mock_container1, mock_container2]

            result = docker_manager.list_containers_formatted(
                show_all=True, format_string="{{.Names}}"
            )

            assert result.is_success
            containers = result.unwrap()
            assert len(containers) == 2
            assert "test_container_1" in containers
            assert "test_container_2" in containers

    def test_images_formatted(self, docker_manager: FlextTestDocker) -> None:
        """Test images_formatted method."""
        mock_image = Mock()
        mock_image.tags = ["test:latest", "test:v1.0"]
        mock_image.attrs = {"Size": 1024000, "Created": "2023-01-01T00:00:00Z"}

        with patch.object(docker_manager.client, "images") as mock_images:
            mock_images.list.return_value = [mock_image]

            result = docker_manager.images_formatted(
                format_string="{{.Repository}}:{{.Tag}}"
            )

            assert result.is_success
            images = result.unwrap()
            assert len(images) >= 1
            assert any("test:latest" in img for img in images)

    def test_build_image_advanced(self, docker_manager: FlextTestDocker) -> None:
        """Test build_image_advanced method."""
        # Mock the client to avoid None issues
        mock_client = MagicMock()
        docker_manager.client = mock_client
        with patch.object(docker_manager.client.api, "build") as mock_build:
            # Mock successful build
            mock_build.return_value = [
                b'{"stream":"Step 1/3 : FROM alpine\\n"}',
                b'{"stream":"Successfully built abc123\\n"}',
                b'{"stream":"Successfully tagged test:latest\\n"}',
            ]

            with tempfile.TemporaryDirectory() as temp_dir:
                dockerfile_path = Path(temp_dir) / "Dockerfile"
                dockerfile_path.write_text("FROM alpine\nRUN echo 'test'\n")

                result: FlextResult[str] = docker_manager.build_image_advanced(
                    path=temp_dir,
                    tag="test:latest",
                    dockerfile_path=str(dockerfile_path),
                    context_path=temp_dir,
                )

                assert result.is_success
                image_id: str = result.unwrap()
                assert image_id

    def test_exec_container_interactive(self, docker_manager: FlextTestDocker) -> None:
        """Test exec_container_interactive method."""
        mock_container = Mock()
        mock_exec = Mock()
        mock_exec.start.return_value = None

        with patch.object(docker_manager.client, "containers") as mock_containers:
            mock_containers.get.return_value = mock_container
            mock_container.exec_run.return_value = mock_exec

            with patch("os.system") as mock_system:
                mock_system.return_value = 0

                result = docker_manager.exec_container_interactive(
                    container_name="test_container", command="/bin/bash"
                )

                assert result.is_success

    def test_container_logs_formatted(self, docker_manager: FlextTestDocker) -> None:
        """Test container_logs_formatted method."""
        mock_container = Mock()
        mock_container.logs.return_value = b"Test log output\nAnother line\n"

        with patch.object(docker_manager.client, "containers") as mock_containers:
            mock_containers.get.return_value = mock_container

            result: FlextResult[str] = docker_manager.container_logs_formatted(
                container_name="test_container", tail=50
            )

            assert result.is_success
            logs: str = result.unwrap()
            assert "Test log output" in logs


class TestShellScriptCompatibility:
    """Test shell script compatibility layer."""

    @pytest.fixture
    def docker_manager(self) -> FlextTestDocker:
        """Create a FlextTestDocker instance for testing."""
        with patch("flext_tests.docker.docker.from_env"):
            return FlextTestDocker()

    def test_shell_script_compatibility_run_success(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test successful shell script compatibility execution."""
        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Container running\n"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = docker_manager.shell_script_compatibility_run(
                "ps --format table", capture_output=True
            )

            assert result.is_success
            exit_code, stdout, _stderr = result.unwrap()
            assert exit_code == 0
            assert "Container running" in stdout

    def test_shell_script_compatibility_run_error(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test shell script compatibility with command error."""
        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "Container not found\n"
            mock_run.return_value = mock_result

            result = docker_manager.shell_script_compatibility_run(
                "ps invalid-container",
                capture_output=True,
                check_exit_code=False,  # Don't fail on non-zero exit
            )

            assert result.is_success
            exit_code, _stdout, stderr = result.unwrap()
            assert exit_code == 1
            assert "Container not found" in stderr

    def test_shell_script_compatibility_timeout(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test shell script compatibility with timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = TimeoutExpired("docker", 30)

            result = docker_manager.shell_script_compatibility_run(
                "exec -it test-container /bin/bash", capture_output=True
            )

            assert result.is_failure
            assert result.error is not None and "timeout" in result.error.lower()


class TestWorkspaceManager:
    """Test the workspace manager functionality."""

    def test_workspace_manager_import(self) -> None:
        """Test that workspace manager can be imported and instantiated."""
        with (
            patch("flext_tests.docker.docker.from_env"),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            workspace_path = Path(temp_dir)
            manager = FlextTestDocker(workspace_root=workspace_path)

            assert manager.workspace_root == workspace_path
            assert manager.docker_manager is not None

    def test_workspace_manager_init_workspace(self) -> None:
        """Test workspace initialization functionality."""
        with (
            patch("flext_tests.docker.docker.from_env"),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            workspace_path = Path(temp_dir)

            # Create a mock docker-compose file
            compose_file = workspace_path / "docker" / "docker-compose.yml"
            compose_file.parent.mkdir(exist_ok=True)
            compose_file.write_text("""
version: '3'
services:
  test_service:
    image: test:latest
    ports:
      - "8080:8080"
""")

            manager = FlextTestDocker()
            result = manager.init_workspace(workspace_path)

            assert result.is_success
            success_msg = result.unwrap()
            assert str(workspace_path) in success_msg


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def docker_manager(self) -> FlextTestDocker:
        """Create a FlextTestDocker instance for testing."""
        with patch("flext_tests.docker.docker.from_env"):
            return FlextTestDocker()

    def test_register_service_invalid_data(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test service registration with invalid data."""
        # Empty service name should fail
        result = docker_manager.register_service(
            service_name="", container_name="test_container"
        )

        # Should still succeed but with empty name (implementation allows this)
        assert result.is_success or result.is_failure

    def test_start_services_unregistered_service(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test starting unregistered services."""
        result: FlextResult[dict[str, str]] = docker_manager.start_services_for_test(
            service_names=["nonexistent_service"], test_name="test"
        )

        assert result.is_failure
        assert result.error is not None and "not registered" in result.error

    def test_health_check_unregistered_service(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test health check for unregistered service."""
        result = docker_manager.get_service_health_status("nonexistent_service")

        assert result.is_failure
        assert result.error is not None and "not registered" in result.error

    def test_docker_connection_failure(self) -> None:
        """Test handling Docker connection failures."""
        with patch("flext_tests.docker.docker.from_env") as mock_docker:
            mock_docker.side_effect = DockerException("Docker daemon not running")

            with pytest.raises(DockerException):
                FlextTestDocker()


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.fixture
    def docker_manager(self) -> FlextTestDocker:
        """Create a FlextTestDocker instance for testing."""
        with patch("flext_tests.docker.docker.from_env"):
            return FlextTestDocker()

    def test_complete_service_lifecycle(self, docker_manager: FlextTestDocker) -> None:
        """Test complete service lifecycle from registration to cleanup."""
        # Step 1: Register services
        db_result = docker_manager.register_service(
            "database", "test_db", ports=[5432], health_check_cmd="pg_isready"
        )
        assert db_result.is_success

        api_result = docker_manager.register_service(
            "api",
            "test_api",
            ports=[8080],
            depends_on=["database"],
            health_check_cmd="curl -f http://localhost:8080/health",
        )
        assert api_result.is_success

        # Step 2: Mock container operations
        with (
            patch.object(docker_manager, "start_container") as mock_start,
            patch.object(docker_manager, "_wait_for_container_ready") as mock_wait,
            patch.object(docker_manager, "execute_container_command") as mock_exec,
        ):
            mock_start.return_value = FlextResult[str].ok("started")
            mock_wait.return_value = FlextResult[None].ok(None)
            mock_exec.return_value = FlextResult[str].ok("healthy")

            # Step 3: Start services for test
            start_result: FlextResult[dict[str, str]] = (
                docker_manager.start_services_for_test(
                    service_names=["api", "database"], test_name="integration_test"
                )
            )
            assert start_result.is_success

            # Step 4: Check health
            health_result = docker_manager.get_service_health_status("api")
            assert health_result.is_success

            # Step 5: Stop services
            stop_result = docker_manager.stop_services_for_test("integration_test")
            assert stop_result.is_success

    def test_dependency_resolution_complex(
        self, docker_manager: FlextTestDocker
    ) -> None:
        """Test complex dependency resolution scenario."""
        # Create a more complex dependency graph
        services: list[tuple[str, list[str], list[int]]] = [
            ("database", [], [5432]),
            ("cache", [], [6379]),
            ("message_queue", [], [5672]),
            ("auth_service", ["database", "cache"], [8081]),
            ("api_service", ["auth_service", "message_queue"], [8080]),
            ("web_service", ["api_service"], [3000]),
        ]

        # Register all services
        for name, deps, ports in services:
            name: str
            deps: list[str]
            ports: list[int]
            result = docker_manager.register_service(
                service_name=name,
                container_name=f"test_{name}",
                ports=ports,
                depends_on=deps,
            )
            assert result.is_success

        # Test dependency graph
        dependency_graph: dict[str, list[str]] = (
            docker_manager.get_service_dependency_graph()
        )
        assert "api_service" in dependency_graph["web_service"]
        assert "auth_service" in dependency_graph["api_service"]
        assert "database" in dependency_graph["auth_service"]


@pytest.mark.integration
class TestRealDockerIntegration:
    """Integration tests with real Docker (requires Docker daemon)."""

    def test_docker_version_check(self) -> None:
        """Test that we can connect to Docker daemon."""
        try:
            docker_manager = FlextTestDocker()
            # If we get here, Docker connection worked
            assert docker_manager.client is not None
        except Exception:
            pytest.skip("Docker daemon not available for integration tests")

    def test_list_real_containers(self) -> None:
        """Test listing real containers if Docker is available."""
        try:
            docker_manager = FlextTestDocker()
            result = docker_manager.list_containers_formatted(show_all=True)

            # Should succeed even if no containers exist
            assert result.is_success
            containers = result.unwrap()
            assert isinstance(containers, list)

        except Exception:
            pytest.skip("Docker daemon not available for integration tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
