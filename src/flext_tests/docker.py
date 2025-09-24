"""Docker container control for FLEXT test infrastructure.

Provides unified start/stop/reset functionality for all FLEXT Docker test containers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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

    Manages shared test containers across all FLEXT projects including:
    - OpenLDAP (port 3390)
    - PostgreSQL (port 5432)
    - Redis (port 6379)
    - Oracle DB (port 1521)
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

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize Docker control."""
        self.workspace_root = workspace_root or self._find_workspace_root()
        try:
            self.client: DockerClient = docker.from_env()
            get_logger().debug(
                "Initialized FlextDockerControl at %s", self.workspace_root
            )
        except DockerException as e:
            get_logger().exception("Failed to connect to Docker: %s", e)
            raise

    def _find_workspace_root(self) -> Path:
        """Find FLEXT workspace root directory."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "docker").is_dir() and (current / "CLAUDE.md").is_file():
                return current
            current = current.parent
        return Path.cwd()

    def _find_container_by_port(self, port: int) -> Container | None:
        """Find any running container using the specified port."""
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
        except DockerException as e:
            get_logger().exception("Failed to list containers: %s", e)
            return None

    def _find_container_by_name(self, name: str) -> Container | None:
        """Find container by exact name match."""
        try:
            return self.client.containers.get(name)
        except NotFound:
            return None
        except DockerException as e:
            get_logger().exception("Failed to get container %s: %s", name, e)
            return None

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

        # No existing container - need to create new one
        return FlextResult[str].fail(
            f"Container {container_name} not found and cannot be auto-created. "
            f"Please start it manually using docker-compose."
        )

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
