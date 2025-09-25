"""Docker container control for FLEXT test infrastructure.

Provides unified start/stop/reset functionality for all FLEXT Docker test containers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container

from flext_core import FlextLogger, FlextResult

if TYPE_CHECKING:
    from docker import DockerClient


# Lazy logger initialization to avoid configuration issues
class _LoggerSingleton:
    """Singleton logger instance."""

    _instance: FlextLogger | None = None

    @classmethod
    def get_logger(cls) -> FlextLogger:
        """Get logger instance with lazy initialization."""
        if cls._instance is None:
            cls._instance = FlextLogger(__name__)
        return cls._instance


def get_logger() -> FlextLogger:
    """Get logger instance with lazy initialization."""
    return _LoggerSingleton.get_logger()


class ContainerStatus(Enum):
    """Container status enumeration."""

    RUNNING = "running"
    STOPPED = "stopped"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass(frozen=True)
class ContainerInfo:
    """Container information."""

    name: str
    status: ContainerStatus
    ports: dict[str, str]
    image: str
    container_id: str = ""


class FlextTestDocker:
    """Unified Docker container control for FLEXT test infrastructure.

    ARCHITECTURAL PRINCIPLE: Single unified class for ALL Docker operations
    across the entire FLEXT ecosystem. This eliminates the need for direct
    docker module imports in any other FLEXT project.

    Manages comprehensive Docker operations including:
    - Container lifecycle (start, stop, reset, health checks)
    - Docker Compose orchestration (up, down, logs, status)
    - Network management (create, connect, disconnect, cleanup)
    - Volume management (create, mount, cleanup, backup)
    - Image management (pull, build, cleanup, registry operations)
    - Advanced health checks and monitoring
    """

    SHARED_CONTAINERS: ClassVar[dict[str, dict[str, str | int]]] = {
        "flext-openldap-test": {
            "compose_file": "docker/docker-compose.openldap.yml",
            "service": "openldap",
            "port": 3390,
        },
        "flext-postgres-test": {
            "compose_file": "docker/docker-compose.postgres.yml",
            "service": "postgres",
            "port": 5433,
        },
        "flext-redis-test": {
            "compose_file": "docker/docker-compose.redis.yml",
            "service": "redis",
            "port": 6380,
        },
        "flext-oracle-db-test": {
            "compose_file": "docker/docker-compose.oracle-db.yml",
            "service": "oracle-db",
            "port": 1522,
        },
        "flext-test": {
            "compose_file": "docker/docker-compose.flext.yml",
            "service": "flext",
            "port": 8000,
        },
        "flext-flexcore-test": {
            "compose_file": "docker/docker-compose.flexcore.yml",
            "service": "flexcore",
            "port": 8090,
        },
    }

    class _ContainerManager:
        """Nested container lifecycle management."""

        def __init__(self, client: DockerClient, logger: FlextLogger) -> None:
            self.client = client
            self._logger = logger

        def find_by_port(self, port: int) -> Container | None:
            """Find running container using specified port."""
            try:
                containers: list[Container] = self.client.containers.list(
                    filters={"status": "running"}
                )
                for container in containers:
                    if container.ports:
                        for bindings in container.ports.values():
                            if bindings:
                                for binding in bindings:
                                    if binding.get("HostPort") == str(port):
                                        return container
                return None
            except DockerException:
                self._logger.exception("Failed to list containers")
                return None

        def find_by_name(self, name: str) -> Container | None:
            """Find container by exact name match."""
            try:
                return self.client.containers.get(name)
            except NotFound:
                return None
            except DockerException:
                self._logger.exception("Failed to get container %s", name)
                return None

        def get_logs(self, container_name: str, tail: int = 100) -> FlextResult[str]:
            """Get container logs."""
            container = self.find_by_name(container_name)
            if not container:
                return FlextResult[str].fail(f"Container {container_name} not found")

            try:
                logs = container.logs(tail=tail, timestamps=True).decode("utf-8")
                return FlextResult[str].ok(logs)
            except DockerException as e:
                return FlextResult[str].fail(f"Failed to get logs: {e}")

        def execute_command(
            self, container_name: str, command: str
        ) -> FlextResult[str]:
            """Execute command in running container."""
            container = self.find_by_name(container_name)
            if not container:
                return FlextResult[str].fail(f"Container {container_name} not found")

            try:
                result = container.exec_run(command)
                output = result.output.decode("utf-8")
                if result.exit_code != 0:
                    return FlextResult[str].fail(
                        f"Command failed with exit code {result.exit_code}: {output}"
                    )
                return FlextResult[str].ok(output)
            except DockerException as e:
                return FlextResult[str].fail(f"Failed to execute command: {e}")

    class _ComposeManager:
        """Nested docker-compose orchestration."""

        def __init__(self, workspace_root: Path, logger: FlextLogger) -> None:
            self.workspace_root = workspace_root
            self._logger = logger
            self._running_services: dict[str, dict[str, str]] = {}

        def compose_up(
            self, compose_file: str, service: str | None = None
        ) -> FlextResult[str]:
            """Start services using docker-compose."""
            compose_path = self.workspace_root / compose_file
            if not compose_path.exists():
                return FlextResult[str].fail(f"Compose file not found: {compose_path}")

            try:
                cmd = ["docker-compose", "-f", str(compose_path), "up", "-d"]
                if service:
                    cmd.append(service)

                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=300
                )
                if result.returncode != 0:
                    return FlextResult[str].fail(f"Compose up failed: {result.stderr}")

                self._logger.info(
                    "Successfully started compose services: %s", compose_file
                )
                return FlextResult[str].ok(f"Compose up successful: {result.stdout}")

            except subprocess.TimeoutExpired:
                return FlextResult[str].fail("Compose up timed out after 5 minutes")
            except Exception as e:
                return FlextResult[str].fail(f"Compose up failed: {e}")

        def compose_down(
            self, compose_file: str, *, remove_volumes: bool = False
        ) -> FlextResult[str]:
            """Stop services using docker-compose."""
            compose_path = self.workspace_root / compose_file
            if not compose_path.exists():
                return FlextResult[str].fail(f"Compose file not found: {compose_path}")

            try:
                cmd = ["docker-compose", "-f", str(compose_path), "down"]
                if remove_volumes:
                    cmd.append("-v")

                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=180
                )
                if result.returncode != 0:
                    return FlextResult[str].fail(
                        f"Compose down failed: {result.stderr}"
                    )

                self._logger.info(
                    "Successfully stopped compose services: %s", compose_file
                )
                return FlextResult[str].ok(f"Compose down successful: {result.stdout}")

            except subprocess.TimeoutExpired:
                return FlextResult[str].fail("Compose down timed out after 3 minutes")
            except Exception as e:
                return FlextResult[str].fail(f"Compose down failed: {e}")

        def compose_logs(
            self, compose_file: str, service: str | None = None, tail: int = 100
        ) -> FlextResult[str]:
            """Get logs from docker-compose services."""
            compose_path = self.workspace_root / compose_file
            if not compose_path.exists():
                return FlextResult[str].fail(f"Compose file not found: {compose_path}")

            try:
                cmd = [
                    "docker-compose",
                    "-f",
                    str(compose_path),
                    "logs",
                    "--tail",
                    str(tail),
                ]
                if service:
                    cmd.append(service)

                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=60
                )
                if result.returncode != 0:
                    return FlextResult[str].fail(
                        f"Compose logs failed: {result.stderr}"
                    )

                return FlextResult[str].ok(result.stdout)

            except subprocess.TimeoutExpired:
                return FlextResult[str].fail("Compose logs timed out after 1 minute")
            except Exception as e:
                return FlextResult[str].fail(f"Compose logs failed: {e}")

    class _NetworkManager:
        """Nested network and port management."""

        def __init__(self, client: DockerClient, logger: FlextLogger) -> None:
            self.client = client
            self._logger = logger

        def create_network(self, name: str, driver: str = "bridge") -> FlextResult[str]:
            """Create Docker network."""
            try:
                # Check if network already exists
                existing_networks = self.client.networks.list(names=[name])
                if existing_networks:
                    self._logger.info("Network %s already exists", name)
                    return FlextResult[str].ok(f"Network {name} already exists")

                network = self.client.networks.create(name, driver=driver)
                self._logger.info("Created network: %s", name)
                return FlextResult[str].ok(f"Network {name} created: {network.id}")

            except DockerException as e:
                return FlextResult[str].fail(f"Failed to create network: {e}")

        def connect_container(
            self, network_name: str, container_name: str
        ) -> FlextResult[str]:
            """Connect container to network."""
            try:
                network = self.client.networks.get(network_name)
                container = self.client.containers.get(container_name)

                network.connect(container)
                self._logger.info(
                    "Connected %s to network %s", container_name, network_name
                )
                return FlextResult[str].ok(
                    f"Container {container_name} connected to {network_name}"
                )

            except NotFound as e:
                return FlextResult[str].fail(f"Network or container not found: {e}")
            except DockerException as e:
                return FlextResult[str].fail(f"Failed to connect container: {e}")

        def list_networks(self) -> FlextResult[list[str]]:
            """List all Docker networks."""
            try:
                networks = self.client.networks.list()
                network_names = [net.name for net in networks if net.name]
                return FlextResult[list[str]].ok(network_names)
            except DockerException as e:
                return FlextResult[list[str]].fail(f"Failed to list networks: {e}")

        def cleanup_networks(self) -> FlextResult[list[str]]:
            """Remove unused networks."""
            try:
                pruned = self.client.networks.prune()
                removed_networks = pruned.get("NetworksDeleted", [])
                self._logger.info("Cleaned up %d networks", len(removed_networks))
                return FlextResult[list[str]].ok(removed_networks)
            except DockerException as e:
                return FlextResult[list[str]].fail(f"Failed to cleanup networks: {e}")

    class _VolumeManager:
        """Nested volume management."""

        def __init__(self, client: DockerClient, logger: FlextLogger) -> None:
            self.client = client
            self._logger = logger

        def create_volume(self, name: str, driver: str = "local") -> FlextResult[str]:
            """Create Docker volume."""
            try:
                # Check if volume already exists
                try:
                    self.client.volumes.get(name)
                    self._logger.info("Volume %s already exists", name)
                    return FlextResult[str].ok(f"Volume {name} already exists")
                except NotFound:
                    pass

                volume = self.client.volumes.create(name, driver=driver)
                self._logger.info("Created volume: %s", name)
                return FlextResult[str].ok(f"Volume {name} created: {volume.id}")

            except DockerException as e:
                return FlextResult[str].fail(f"Failed to create volume: {e}")

        def list_volumes(self) -> FlextResult[list[str]]:
            """List all Docker volumes."""
            try:
                volumes = self.client.volumes.list()
                volume_names = [vol.name for vol in volumes if vol.name]
                return FlextResult[list[str]].ok(volume_names)
            except DockerException as e:
                return FlextResult[list[str]].fail(f"Failed to list volumes: {e}")

        def cleanup_volumes(self) -> FlextResult[dict[str, int | list[str]]]:
            """Remove unused volumes."""
            try:
                pruned = self.client.volumes.prune()
                removed_volumes = pruned.get("VolumesDeleted", [])
                space_reclaimed = pruned.get("SpaceReclaimed", 0)

                self._logger.info(
                    "Cleaned up %d volumes, reclaimed %d bytes",
                    len(removed_volumes),
                    space_reclaimed,
                )

                return FlextResult[dict[str, int | list[str]]].ok({
                    "volumes_deleted": removed_volumes,
                    "space_reclaimed": space_reclaimed,
                })
            except DockerException as e:
                return FlextResult[dict[str, int | list[str]]].fail(
                    f"Failed to cleanup volumes: {e}"
                )

    class _ImageManager:
        """Nested image management."""

        def __init__(self, client: DockerClient, logger: FlextLogger) -> None:
            self.client = client
            self._logger = logger

        def pull_image(self, image_name: str, tag: str = "latest") -> FlextResult[str]:
            """Pull image from registry."""
            try:
                full_name = f"{image_name}:{tag}"
                self._logger.info("Pulling image: %s", full_name)

                image = self.client.images.pull(image_name, tag=tag)
                self._logger.info("Successfully pulled image: %s", full_name)
                return FlextResult[str].ok(f"Image pulled: {full_name} ({image.id})")

            except DockerException as e:
                return FlextResult[str].fail(f"Failed to pull image: {e}")

        def build_image(
            self,
            dockerfile_path: Path,
            tag: str,
            build_args: dict[str, str] | None = None,
        ) -> FlextResult[str]:
            """Build image from Dockerfile."""
            try:
                if not dockerfile_path.parent.exists():
                    return FlextResult[str].fail(
                        f"Build context not found: {dockerfile_path.parent}"
                    )

                self._logger.info("Building image: %s from %s", tag, dockerfile_path)

                # Use explicit parameters to avoid type checking issues
                image, _logs = self.client.images.build(
                    path=str(dockerfile_path.parent),
                    dockerfile=dockerfile_path.name,
                    tag=tag,
                    rm=True,
                    buildargs=build_args,
                )
                self._logger.info("Successfully built image: %s", tag)
                return FlextResult[str].ok(f"Image built: {tag} ({image.id})")

            except DockerException as e:
                return FlextResult[str].fail(f"Failed to build image: {e}")

        def list_images(self) -> FlextResult[list[str]]:
            """List all Docker images."""
            try:
                images = self.client.images.list()
                image_tags = []
                for image in images:
                    if image.tags:
                        image_tags.extend(image.tags)
                    elif image.id:
                        image_tags.append(f"<none>:{image.id[:12]}")
                return FlextResult[list[str]].ok(image_tags)
            except DockerException as e:
                return FlextResult[list[str]].fail(f"Failed to list images: {e}")

        def cleanup_images(
            self, *, dangling_only: bool = True
        ) -> FlextResult[dict[str, int | list[str]]]:
            """Remove unused images."""
            try:
                filters = {"dangling": True} if dangling_only else {}
                pruned = self.client.images.prune(filters=filters)

                removed_images = pruned.get("ImagesDeleted", [])
                space_reclaimed = pruned.get("SpaceReclaimed", 0)

                self._logger.info(
                    "Cleaned up %d images, reclaimed %d bytes",
                    len(removed_images),
                    space_reclaimed,
                )

                return FlextResult[dict[str, int | list[str]]].ok({
                    "images_deleted": removed_images,
                    "space_reclaimed": space_reclaimed,
                })
            except DockerException as e:
                return FlextResult[dict[str, int | list[str]]].fail(
                    f"Failed to cleanup images: {e}"
                )

    class _TestLifecycleManager:
        """Nested test lifecycle management for Docker containers."""

        def __init__(self, docker_manager: FlextTestDocker, logger: FlextLogger) -> None:
            self._docker = docker_manager
            self._logger = logger
            self._test_containers: dict[str, str] = {}  # test_name -> container_name
            self._cleanup_registry: list[tuple[str, dict[str, str]]] = []  # (action_type, details)

        def register_test_container(self, test_name: str, container_name: str) -> FlextResult[None]:
            """Register a container for test lifecycle management."""
            try:
                self._test_containers[test_name] = container_name
                self._cleanup_registry.append(("stop_container", {"container": container_name}))
                self._logger.debug("Registered test container %s for test %s", container_name, test_name)
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to register test container: {e}")

        def start_test_containers(self, test_name: str, containers: list[str]) -> FlextResult[dict[str, str]]:
            """Start containers needed for a specific test."""
            results = {}

            for container_name in containers:
                start_result = self._docker.start_container(container_name)
                if start_result.is_failure:
                    # Stop any already started containers on failure
                    for started_container in results:
                        self._docker.stop_container(started_container)
                    return FlextResult[dict[str, str]].fail(
                        f"Failed to start container {container_name}: {start_result.error}"
                    )

                results[container_name] = "started"
                self.register_test_container(test_name, container_name)

                # Wait for container to be ready
                wait_result = self._docker._wait_for_container_ready(container_name, timeout=60)
                if wait_result.is_failure:
                    self._logger.warning(
                        "Container %s started but not ready: %s",
                        container_name,
                        wait_result.error
                    )

            return FlextResult[dict[str, str]].ok(results)

        def stop_test_containers(self, test_name: str) -> FlextResult[dict[str, str]]:
            """Stop containers associated with a specific test."""
            results = {}

            if test_name not in self._test_containers:
                return FlextResult[dict[str, str]].ok(results)

            container_name = self._test_containers[test_name]
            stop_result = self._docker.stop_container(container_name, remove=False)

            if stop_result.is_success:
                results[container_name] = "stopped"
                del self._test_containers[test_name]
            else:
                results[container_name] = f"stop_failed: {stop_result.error}"

            return FlextResult[dict[str, str]].ok(results)

        def cleanup_all_test_containers(self) -> FlextResult[dict[str, str]]:
            """Clean up all registered test containers."""
            results = {}

            for test_name in list(self._test_containers.keys()):
                cleanup_result = self.stop_test_containers(test_name)
                if cleanup_result.is_success:
                    results.update(cleanup_result.unwrap())
                else:
                    self._logger.error("Failed to cleanup test %s: %s", test_name, cleanup_result.error)
                    results[f"test_{test_name}"] = f"cleanup_failed: {cleanup_result.error}"

            # Execute registered cleanup actions
            for action_type, details in self._cleanup_registry:
                if action_type == "stop_container":
                    container = details["container"]
                    if container not in results:  # Avoid duplicate stops
                        stop_result = self._docker.stop_container(container, remove=True)
                        results[container] = "cleaned" if stop_result.is_success else f"cleanup_failed: {stop_result.error}"

            self._cleanup_registry.clear()
            self._test_containers.clear()

            return FlextResult[dict[str, str]].ok(results)

        def ensure_container_ready_for_test(
            self, container_name: str, health_check_command: str | None = None
        ) -> FlextResult[None]:
            """Ensure a container is ready for testing with optional custom health check."""
            # Check if container is running
            status_result = self._docker.get_container_status(container_name)
            if status_result.is_failure:
                return FlextResult[None].fail(f"Cannot get container status: {status_result.error}")

            container_info = status_result.unwrap()
            if container_info.status != ContainerStatus.RUNNING:
                # Try to start the container
                start_result = self._docker.start_container(container_name)
                if start_result.is_failure:
                    return FlextResult[None].fail(f"Failed to start container: {start_result.error}")

            # Wait for container to be ready
            wait_result = self._docker._wait_for_container_ready(container_name, timeout=60)
            if wait_result.is_failure:
                return FlextResult[None].fail(f"Container not ready: {wait_result.error}")

            # Execute custom health check if provided
            if health_check_command:
                health_result = self._docker.execute_container_command(container_name, health_check_command)
                if health_result.is_failure:
                    return FlextResult[None].fail(f"Health check failed: {health_result.error}")

            return FlextResult[None].ok(None)

        def setup_test_session(self, session_name: str, required_containers: list[str]) -> FlextResult[None]:
            """Setup a test session with required containers."""
            self._logger.info("Setting up test session: %s", session_name)

            # Start all required containers
            start_result = self.start_test_containers(session_name, required_containers)
            if start_result.is_failure:
                return FlextResult[None].fail(f"Test session setup failed: {start_result.error}")

            # Verify all containers are ready
            for container_name in required_containers:
                ready_result = self.ensure_container_ready_for_test(container_name)
                if ready_result.is_failure:
                    # Cleanup partially started session
                    self.stop_test_containers(session_name)
                    return FlextResult[None].fail(f"Container {container_name} not ready: {ready_result.error}")

            self._logger.info("Test session %s setup completed successfully", session_name)
            return FlextResult[None].ok(None)

        def teardown_test_session(self, session_name: str) -> FlextResult[None]:
            """Teardown a test session and cleanup resources."""
            self._logger.info("Tearing down test session: %s", session_name)

            cleanup_result = self.stop_test_containers(session_name)
            if cleanup_result.is_failure:
                self._logger.warning("Test session teardown had issues: %s", cleanup_result.error)
                return FlextResult[None].fail(f"Session teardown failed: {cleanup_result.error}")

            self._logger.info("Test session %s teardown completed", session_name)
            return FlextResult[None].ok(None)

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize unified Docker management."""
        self.workspace_root = workspace_root or self._find_workspace_root()

        try:
            self.client: DockerClient = docker.from_env()
            get_logger().debug("Initialized FlextTestDocker at %s", self.workspace_root)
        except DockerException as e:
            get_logger().exception("Failed to connect to Docker: %s", e)
            raise

        # Initialize nested managers
        logger = get_logger()
        self._container_manager = self._ContainerManager(self.client, logger)
        self._compose_manager = self._ComposeManager(self.workspace_root, logger)
        self._network_manager = self._NetworkManager(self.client, logger)
        self._volume_manager = self._VolumeManager(self.client, logger)
        self._image_manager = self._ImageManager(self.client, logger)
        self._lifecycle_manager = self._TestLifecycleManager(self, logger)

    def _find_workspace_root(self) -> Path:
        """Find FLEXT workspace root directory."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "docker").is_dir() and (current / "CLAUDE.md").is_file():
                return current
            current = current.parent
        return Path.cwd()

    # Container Management (existing methods enhanced)
    def _find_container_by_port(self, port: int) -> Container | None:
        """Find any running container using the specified port."""
        return self._container_manager.find_by_port(port)

    def _find_container_by_name(self, name: str) -> Container | None:
        """Find container by exact name match."""
        return self._container_manager.find_by_name(name)

    def get_container_logs(
        self, container_name: str, tail: int = 100
    ) -> FlextResult[str]:
        """Get container logs."""
        return self._container_manager.get_logs(container_name, tail)

    def execute_container_command(
        self, container_name: str, command: str
    ) -> FlextResult[str]:
        """Execute command in running container."""
        return self._container_manager.execute_command(container_name, command)

    # Compose Management (new functionality)
    def compose_up(
        self, compose_file: str, service: str | None = None
    ) -> FlextResult[str]:
        """Start services using docker-compose."""
        return self._compose_manager.compose_up(compose_file, service)

    def compose_down(
        self, compose_file: str, *, remove_volumes: bool = False
    ) -> FlextResult[str]:
        """Stop services using docker-compose."""
        return self._compose_manager.compose_down(
            compose_file, remove_volumes=remove_volumes
        )

    def compose_logs(
        self, compose_file: str, service: str | None = None, tail: int = 100
    ) -> FlextResult[str]:
        """Get logs from docker-compose services."""
        return self._compose_manager.compose_logs(compose_file, service, tail)

    # Network Management (new functionality)
    def create_network(self, name: str, driver: str = "bridge") -> FlextResult[str]:
        """Create Docker network."""
        return self._network_manager.create_network(name, driver)

    def connect_container_to_network(
        self, network_name: str, container_name: str
    ) -> FlextResult[str]:
        """Connect container to network."""
        return self._network_manager.connect_container(network_name, container_name)

    def list_networks(self) -> FlextResult[list[str]]:
        """List all Docker networks."""
        return self._network_manager.list_networks()

    def cleanup_networks(self) -> FlextResult[list[str]]:
        """Remove unused networks."""
        return self._network_manager.cleanup_networks()

    # Volume Management (new functionality)
    def create_volume(self, name: str, driver: str = "local") -> FlextResult[str]:
        """Create Docker volume."""
        return self._volume_manager.create_volume(name, driver)

    def list_volumes(self) -> FlextResult[list[str]]:
        """List all Docker volumes."""
        return self._volume_manager.list_volumes()

    def cleanup_volumes(self) -> FlextResult[dict[str, int | list[str]]]:
        """Remove unused volumes."""
        return self._volume_manager.cleanup_volumes()

    # Image Management (new functionality)
    def pull_image(self, image_name: str, tag: str = "latest") -> FlextResult[str]:
        """Pull image from registry."""
        return self._image_manager.pull_image(image_name, tag)

    def build_image(
        self, dockerfile_path: Path, tag: str, build_args: dict[str, str] | None = None
    ) -> FlextResult[str]:
        """Build image from Dockerfile."""
        return self._image_manager.build_image(dockerfile_path, tag, build_args)

    def list_images(self) -> FlextResult[list[str]]:
        """List all Docker images."""
        return self._image_manager.list_images()

    def cleanup_images(
        self, *, dangling_only: bool = True
    ) -> FlextResult[dict[str, int | list[str]]]:
        """Remove unused images."""
        return self._image_manager.cleanup_images(dangling_only=dangling_only)

    # Test Lifecycle Management (new functionality)
    def setup_test_session(self, session_name: str, required_containers: list[str]) -> FlextResult[None]:
        """Setup a test session with required containers."""
        return self._lifecycle_manager.setup_test_session(session_name, required_containers)

    def teardown_test_session(self, session_name: str) -> FlextResult[None]:
        """Teardown a test session and cleanup resources."""
        return self._lifecycle_manager.teardown_test_session(session_name)

    def ensure_container_ready_for_test(
        self, container_name: str, health_check_command: str | None = None
    ) -> FlextResult[None]:
        """Ensure a container is ready for testing with optional custom health check."""
        return self._lifecycle_manager.ensure_container_ready_for_test(container_name, health_check_command)

    def register_test_container(self, test_name: str, container_name: str) -> FlextResult[None]:
        """Register a container for test lifecycle management."""
        return self._lifecycle_manager.register_test_container(test_name, container_name)

    def cleanup_all_test_containers(self) -> FlextResult[dict[str, str]]:
        """Clean up all registered test containers."""
        return self._lifecycle_manager.cleanup_all_test_containers()

    # Existing Container Lifecycle Methods (preserved for compatibility)
    def start_container(self, container_name: str) -> FlextResult[str]:
        """Start a specific FLEXT test container (idempotent - reuses existing if possible)."""
        if container_name not in self.SHARED_CONTAINERS:
            return FlextResult[str].fail(
                f"Unknown container: {container_name}. "
                f"Available: {', '.join(self.SHARED_CONTAINERS.keys())}"
            )

        config = self.SHARED_CONTAINERS[container_name]
        port = int(config["port"])

        # First check if another container is using the same port
        port_container = self._find_container_by_port(port)
        if port_container:
            get_logger().info(
                "Found %s using port %s (requested %s) - reusing it",
                port_container.name,
                port,
                container_name,
            )
            return FlextResult[str].ok(
                f"Reusing container {port_container.name} on port {port}"
            )

        # Check if requested container exists
        container = self._find_container_by_name(container_name)
        if container:
            if container.status == "running":
                get_logger().info("Container %s already running", container_name)
                return FlextResult[str].ok(
                    f"Container {container_name} already running"
                )
            # Container exists but stopped - start it
            try:
                get_logger().info("Starting existing container: %s", container_name)
                container.start()
                return FlextResult[str].ok(
                    f"Container {container_name} started successfully"
                )
            except DockerException as e:
                return FlextResult[str].fail(f"Failed to start container: {e}")

        # No existing container - try compose up
        compose_file = str(config["compose_file"])
        service = str(config["service"])

        get_logger().info("Starting container via compose: %s", container_name)
        compose_result = self.compose_up(compose_file, service)
        if compose_result.is_failure:
            return FlextResult[str].fail(
                f"Container {container_name} not found and compose failed: {compose_result.error}"
            )

        return FlextResult[str].ok(f"Container {container_name} started via compose")

    def stop_container(
        self, container_name: str, *, remove: bool = False
    ) -> FlextResult[str]:
        """Stop a specific FLEXT test container."""
        if container_name not in self.SHARED_CONTAINERS:
            return FlextResult[str].fail(f"Unknown container: {container_name}")

        container = self._find_container_by_name(container_name)
        if not container:
            get_logger().warning(
                "Container %s not found - nothing to stop", container_name
            )
            return FlextResult[str].ok(f"Container {container_name} not found")

        try:
            get_logger().info("Stopping container: %s", container_name)
            container.stop()

            if remove:
                get_logger().info("Removing container: %s", container_name)
                container.remove()

            action = "stopped and removed" if remove else "stopped"
            return FlextResult[str].ok(
                f"Container {container_name} {action} successfully"
            )
        except DockerException as e:
            return FlextResult[str].fail(f"Failed to stop container: {e}")

    def reset_container(self, container_name: str) -> FlextResult[str]:
        """Reset a container (stop, remove, and start fresh)."""
        get_logger().info("Resetting container: %s", container_name)

        stop_result = self.stop_container(container_name, remove=True)
        if stop_result.is_failure:
            get_logger().warning("Stop failed during reset: %s", stop_result.error)

        start_result = self.start_container(container_name)
        if start_result.is_failure:
            return FlextResult[str].fail(
                f"Failed to restart container: {start_result.error}"
            )

        return FlextResult[str].ok(f"Container {container_name} reset successfully")

    def start_all(self) -> FlextResult[dict[str, str]]:
        """Start all FLEXT test containers."""
        get_logger().info("Starting all FLEXT test containers")
        results = {}

        for container_name in self.SHARED_CONTAINERS:
            result = self.start_container(container_name)
            results[container_name] = (
                "success" if result.is_success else result.error or "failed"
            )

        failed: list[str] = [k for k, v in results.items() if v != "success"]
        if failed:
            return FlextResult[dict[str, str]].fail(
                f"Failed to start: {', '.join(failed)}"
            )

        return FlextResult[dict[str, str]].ok(results)

    def stop_all(self, *, remove: bool = False) -> FlextResult[dict[str, str]]:
        """Stop all FLEXT test containers."""
        get_logger().info("Stopping all FLEXT test containers")
        results = {}

        for container_name in self.SHARED_CONTAINERS:
            result = self.stop_container(container_name, remove=remove)
            results[container_name] = (
                "success" if result.is_success else result.error or "failed"
            )

        return FlextResult[dict[str, str]].ok(results)

    def reset_all(self) -> FlextResult[dict[str, str]]:
        """Reset all FLEXT test containers."""
        get_logger().info("Resetting all FLEXT test containers")
        results = {}

        for container_name in self.SHARED_CONTAINERS:
            result = self.reset_container(container_name)
            results[container_name] = (
                "success" if result.is_success else result.error or "failed"
            )

        failed: list[str] = [k for k, v in results.items() if v != "success"]
        if failed:
            return FlextResult[dict[str, str]].fail(
                f"Failed to reset: {', '.join(failed)}"
            )

        return FlextResult[dict[str, str]].ok(results)

    def is_container_running(self, container_name: str | None = None) -> bool:
        """Check if a container is running.

        Args:
            container_name: Optional container name. If None, checks if any container is running.

        Returns:
            True if container is running, False otherwise.

        """
        if container_name is None:
            # Check if any container is running
            try:
                containers: list[Container] = self.client.containers.list(
                    filters={"status": "running"}
                )
                return len(containers) > 0
            except DockerException:
                return False

        status_result = self.get_container_status(container_name)
        if status_result.is_failure:
            return False
        return status_result.value.status == ContainerStatus.RUNNING

    def get_container_status(self, container_name: str) -> FlextResult[ContainerInfo]:
        """Get status of a specific container.

        Checks both by exact container name and by port to handle cases where
        a different container is using the same port (e.g., flext-ldap-test-server
        running on port 3390 when looking for flext-openldap-test).
        """
        if container_name not in self.SHARED_CONTAINERS:
            return FlextResult[ContainerInfo].fail(
                f"Unknown container: {container_name}"
            )

        config = self.SHARED_CONTAINERS[container_name]
        port = int(config["port"])

        # First try exact container name match
        container = self._find_container_by_name(container_name)

        # If not found by exact name, check for any container using the same port
        if not container:
            container = self._find_container_by_port(port)
            if container:
                get_logger().info(
                    "Container %s not found, but found %s using port %s",
                    container_name,
                    container.name,
                    port,
                )

        if not container:
            return FlextResult[ContainerInfo].ok(
                ContainerInfo(
                    name=container_name,
                    status=ContainerStatus.NOT_FOUND,
                    ports={},
                    image="",
                )
            )

        # Extract port mappings
        ports: dict[str, str] = {}
        if container.ports:
            for container_port, bindings in container.ports.items():
                if bindings:
                    for binding in bindings:
                        host_port = binding.get("HostPort", "")
                        if host_port:
                            ports[container_port.split("/")[0]] = host_port

        # Determine status
        status_str = container.status
        status = (
            ContainerStatus.RUNNING
            if status_str == "running"
            else ContainerStatus.STOPPED
            if status_str in {"exited", "stopped", "created"}
            else ContainerStatus.ERROR
        )

        return FlextResult[ContainerInfo].ok(
            ContainerInfo(
                name=container.name or "unknown",
                status=status,
                ports=ports,
                image=container.image.tags[0]
                if container.image and container.image.tags
                else "",
                container_id=container.id[:12] if container.id else "unknown",
            )
        )

    def get_all_status(self) -> FlextResult[dict[str, ContainerInfo]]:
        """Get status of all FLEXT test containers."""
        results: dict[str, ContainerInfo] = {}

        for container_name in self.SHARED_CONTAINERS:
            status_result = self.get_container_status(container_name)
            if status_result.is_success:
                results[container_name] = status_result.value

        return FlextResult[dict[str, ContainerInfo]].ok(results)

    def _wait_for_container_ready(
        self, container_name: str, timeout: int = 60, port: int | None = None
    ) -> FlextResult[None]:
        """Wait for container to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status_result = self.get_container_status(container_name)
            if status_result.is_failure:
                return FlextResult[None].fail(
                    f"Failed to get status: {status_result.error}"
                )

            status = status_result.value
            if status.status == ContainerStatus.RUNNING:
                if port and str(port) in status.ports.values():
                    get_logger().debug(
                        "Container %s is ready on port %s", container_name, port
                    )
                    return FlextResult[None].ok(None)
                if not port:
                    get_logger().debug("Container %s is running", container_name)
                    return FlextResult[None].ok(None)

            time.sleep(1)

        return FlextResult[None].fail(
            f"Container {container_name} not ready after {timeout} seconds"
        )

    # Comprehensive Cleanup Operations
    def cleanup_all_resources(self) -> FlextResult[dict[str, str]]:
        """Comprehensive cleanup of all Docker resources."""
        get_logger().info("Starting comprehensive Docker cleanup")
        results = {}

        # Stop all containers
        stop_result = self.stop_all(remove=True)
        results["containers"] = (
            "success" if stop_result.is_success else stop_result.error or "failed"
        )

        # Cleanup networks
        network_result = self.cleanup_networks()
        results["networks"] = (
            "success" if network_result.is_success else network_result.error or "failed"
        )

        # Cleanup volumes
        volume_result = self.cleanup_volumes()
        results["volumes"] = (
            "success" if volume_result.is_success else volume_result.error or "failed"
        )

        # Cleanup images
        image_result = self.cleanup_images(dangling_only=False)
        results["images"] = (
            "success" if image_result.is_success else image_result.error or "failed"
        )

        failed = [k for k, v in results.items() if v != "success"]
        if failed:
            return FlextResult[dict[str, str]].fail(
                f"Cleanup failed for: {', '.join(failed)}"
            )

        get_logger().info("Docker cleanup completed successfully")
        return FlextResult[dict[str, str]].ok(results)

    def start_containers_for_test(
        self, test_name: str, containers: list[str], *, wait_ready: bool = True
    ) -> FlextResult[dict[str, str]]:
        """Start containers needed for a specific test with lifecycle management.

        This method provides automated container lifecycle for tests:
        - Starts requested containers
        - Waits for them to be ready (if wait_ready=True)
        - Registers them for automatic cleanup
        - Returns status of each container

        Args:
            test_name: Name of the test (for lifecycle tracking)
            containers: List of container names to start
            wait_ready: Whether to wait for containers to be ready

        Returns:
            FlextResult with dict mapping container names to their status

        """
        get_logger().info("Starting containers for test %s: %s", test_name, containers)

        # Use the lifecycle manager to start containers
        start_result = self._lifecycle_manager.start_test_containers(test_name, containers)
        if start_result.is_failure:
            return start_result

        results = start_result.unwrap()

        # Additional readiness checks if requested
        if wait_ready:
            for container_name in containers:
                if results.get(container_name) == "started":
                    ready_result = self.ensure_container_ready_for_test(container_name)
                    if ready_result.is_failure:
                        # Mark as not ready but don't fail the entire operation
                        results[container_name] = f"started_but_not_ready: {ready_result.error}"
                        get_logger().warning(
                            "Container %s started but not ready: %s",
                            container_name,
                            ready_result.error
                        )
                    else:
                        results[container_name] = "ready"

        return FlextResult[dict[str, str]].ok(results)

    def stop_containers_for_test(self, test_name: str) -> FlextResult[dict[str, str]]:
        """Stop containers associated with a specific test.

        This method stops containers that were started for a test and removes
        them from the lifecycle tracking.

        Args:
            test_name: Name of the test whose containers should be stopped

        Returns:
            FlextResult with dict mapping container names to their stop status

        """
        get_logger().info("Stopping containers for test: %s", test_name)
        return self._lifecycle_manager.stop_test_containers(test_name)
