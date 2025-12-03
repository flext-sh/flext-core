"""Simplified Docker container control for FLEXT test infrastructure.

Essential container management using Python libraries only:
- docker SDK for container operations
- python_on_whales for docker-compose operations
- NO shell commands ever

Core functionality:
- Create/start containers via docker-compose
- Dirty state tracking for container recreation
- Port readiness checking

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import docker
from docker import DockerClient
from docker.errors import DockerException, NotFound
from python_on_whales import DockerClient as PowDockerClient, docker as pow_docker

from flext_core.loggings import FlextLogger
from flext_core.result import r
from flext_core.typings import t
from flext_tests.constants import FlextTestConstants
from flext_tests.models import FlextTestModels

logger: FlextLogger = FlextLogger(__name__)


class FlextTestDocker:
    """Simplified Docker container management for FLEXT tests.

    Essential functionality only:
    - Container creation via docker-compose (python_on_whales)
    - Container status/start/stop (docker SDK)
    - Dirty state tracking for container recreation
    - Port readiness checking
    """

    ContainerStatus = FlextTestConstants.Docker.ContainerStatus
    ContainerInfo = FlextTestModels.Docker.ContainerInfo

    SHARED_CONTAINERS: ClassVar[Mapping[str, t.Types.ContainerConfigDict]] = (
        FlextTestConstants.Docker.SHARED_CONTAINERS
    )

    def __init__(
        self,
        workspace_root: Path | None = None,
        worker_id: str | None = None,
    ) -> None:
        """Initialize Docker client with dirty state tracking."""
        self._client: DockerClient | None = None
        self.logger: FlextLogger = FlextLogger(__name__)
        self.workspace_root = workspace_root or Path.cwd()
        self.worker_id = worker_id or "master"
        self._dirty_containers: set[str] = set()
        self._state_file = (
            Path.home() / ".flext" / f"docker_state_{self.worker_id}.json"
        )
        self._load_dirty_state()
        self.get_client()

    @property
    def shared_containers(
        self,
    ) -> Mapping[str, t.Types.ContainerConfigDict]:
        """Get shared container configurations."""
        return FlextTestConstants.Docker.SHARED_CONTAINERS

    def get_client(self) -> DockerClient:
        """Get Docker client with lazy initialization."""
        if self._client is None:
            try:
                self._client = docker.from_env()
            except DockerException:
                self.logger.exception("Failed to initialize Docker client")
                raise
        return self._client

    def _load_dirty_state(self) -> None:
        """Load dirty container state from persistent storage."""
        try:
            if self._state_file.exists():
                with self._state_file.open("r") as f:
                    state = json.load(f)
                    self._dirty_containers = set(state.get("dirty_containers", []))
        except Exception as e:
            self.logger.warning("Failed to load dirty state", error=str(e))
            self._dirty_containers = set()

    def _save_dirty_state(self) -> None:
        """Save dirty container state to persistent storage."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with self._state_file.open("w") as f:
                json.dump({"dirty_containers": list(self._dirty_containers)}, f)
        except Exception as e:
            self.logger.warning("Failed to save dirty state", error=str(e))

    def mark_container_dirty(self, container_name: str) -> r[bool]:
        """Mark a container as dirty for recreation on next use."""
        try:
            self._dirty_containers.add(container_name)
            self._save_dirty_state()
            self.logger.info("Container marked dirty", container=container_name)
            return r[bool].ok(True)
        except Exception as e:
            return r[bool].fail(f"Failed to mark dirty: {e}")

    def mark_container_clean(self, container_name: str) -> r[bool]:
        """Mark a container as clean after successful recreation."""
        try:
            self._dirty_containers.discard(container_name)
            self._save_dirty_state()
            self.logger.info("Container marked clean", container=container_name)
            return r[bool].ok(True)
        except Exception as e:
            return r[bool].fail(f"Failed to mark clean: {e}")

    def is_container_dirty(self, container_name: str) -> bool:
        """Check if a container is marked as dirty."""
        return container_name in self._dirty_containers

    def get_dirty_containers(self) -> list[str]:
        """Get list of all dirty containers."""
        return list(self._dirty_containers)

    def cleanup_dirty_containers(
        self,
    ) -> r[list[str]]:
        """Clean up all dirty containers by recreating them with fresh volumes."""
        cleaned: list[str] = []

        for container_name in list(self._dirty_containers):
            config = self.shared_containers.get(container_name)
            if not config:
                continue

            compose_file = str(config.get("compose_file", ""))
            if not compose_file:
                continue

            if not Path(compose_file).is_absolute():
                compose_file = str(self.workspace_root / compose_file)

            service = str(config.get("service", ""))

            self.logger.info("Recreating dirty container", container=container_name)
            self.compose_down(compose_file)
            result = self.compose_up(compose_file, service, force_recreate=True)

            if result.is_success:
                self.mark_container_clean(container_name)
                cleaned.append(container_name)

        return r[list[str]].ok(cleaned)

    def get_container_info(
        self,
        container_name: str,
    ) -> r[FlextTestModels.Docker.ContainerInfo]:
        """Get container information."""
        try:
            client = self.get_client()
            container = client.containers.get(container_name)
            ports_raw = getattr(container, "ports", {}) or {}
            ports: dict[str, str] = {}
            for k, v in ports_raw.items():
                if v:
                    ports[str(k)] = str(v[0].get("HostPort", "")) if v else ""

            return r[FlextTestModels.Docker.ContainerInfo].ok(
                FlextTestModels.Docker.ContainerInfo(
                    name=container_name,
                    status=self.ContainerStatus(container.status),
                    ports=ports,
                    image=(
                        str(container.image.tags[0])
                        if container.image and container.image.tags
                        else ""
                    ),
                    container_id=container.id or "",
                )
            )
        except NotFound:
            return r[FlextTestModels.Docker.ContainerInfo].fail(
                f"Container {container_name} not found"
            )
        except Exception as e:
            return r[FlextTestModels.Docker.ContainerInfo].fail(str(e))

    def get_container_status(
        self,
        container_name: str,
    ) -> r[FlextTestModels.Docker.ContainerInfo]:
        """Get container status (alias for get_container_info)."""
        return self.get_container_info(container_name)

    def start_existing_container(self, name: str) -> r[str]:
        """Start an existing stopped container without recreating it."""
        try:
            client = self.get_client()
            container = client.containers.get(name)

            if container.status == "running":
                return r[str].ok(f"Container {name} already running")

            if container.status in {"exited", "created", "paused"}:
                container.start()
                return r[str].ok(f"Container {name} started")

            return r[str].ok(f"Container {name} in state: {container.status}")

        except NotFound:
            return r[str].fail(f"Container {name} not found")
        except Exception as e:
            return r[str].fail(f"Failed to start container: {e}")

    def compose_up(
        self,
        compose_file: str,
        service: str | None = None,
        *,
        force_recreate: bool = False,
    ) -> r[str]:
        """Start services using docker-compose via python_on_whales."""
        try:
            compose_path = Path(compose_file)
            if not compose_path.is_absolute():
                compose_path = self.workspace_root / compose_file

            docker_client: PowDockerClient = pow_docker

            original_files = docker_client.client_config.compose_files
            try:
                docker_client.client_config.compose_files = [compose_path]

                if force_recreate:
                    try:
                        docker_client.compose.down(remove_orphans=True, volumes=True)
                    except Exception:
                        pass

                services = [service] if service else []
                docker_client.compose.up(
                    services=services,
                    detach=True,
                    remove_orphans=True,
                )
            finally:
                docker_client.client_config.compose_files = original_files

            return r[str].ok("Compose up successful")

        except Exception as e:
            self.logger.exception("Compose up failed")
            return r[str].fail(f"Compose up failed: {e}")

    def compose_down(self, compose_file: str) -> r[str]:
        """Stop services using docker-compose via python_on_whales."""
        try:
            compose_path = Path(compose_file)
            if not compose_path.is_absolute():
                compose_path = self.workspace_root / compose_file

            docker_client: PowDockerClient = pow_docker

            original_files = docker_client.client_config.compose_files
            try:
                docker_client.client_config.compose_files = [compose_path]
                docker_client.compose.down(volumes=True, remove_orphans=True)
            finally:
                docker_client.client_config.compose_files = original_files

            return r[str].ok("Compose down successful")

        except Exception as e:
            self.logger.warning("Compose down failed", error=str(e))
            return r[str].fail(f"Compose down failed: {e}")

    def start_compose_stack(
        self,
        compose_file: str,
        network_name: str | None = None,
    ) -> r[str]:
        """Start a Docker Compose stack."""
        result = self.compose_up(compose_file)
        if result.is_failure:
            return r[str].fail(f"Stack start failed: {result.error}")
        return r[str].ok("Stack started successfully")

    def wait_for_port_ready(
        self,
        host: str,
        port: int,
        max_wait: int = 60,
    ) -> r[bool]:
        """Wait for network port to become available."""
        try:
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    sock.close()

                    if result == 0:
                        self.logger.info("Port ready", host=host, port=port)
                        return r[bool].ok(True)
                except OSError:
                    pass

                time.sleep(2)

            self.logger.warning(
                "Port not ready", host=host, port=port, max_wait=max_wait
            )
            return r[bool].ok(False)

        except Exception as e:
            return r[bool].fail(f"Failed to wait for port: {e}")


__all__ = ["FlextTestDocker"]
