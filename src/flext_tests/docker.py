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

import contextlib
import json
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from docker import DockerClient as DockerSDKClient, from_env as docker_from_env
from docker.errors import DockerException, NotFound
from docker.models.containers import Container
from flext_core.loggings import FlextLogger
from flext_core.result import r

try:
    from python_on_whales import (  # pyright: ignore[reportMissingImports]
        DockerClient as WhalesDockerClient,
        docker,
    )
except ImportError:
    WhalesDockerClient = Any  # type: ignore[assignment]
    docker = None

from flext_tests.constants import c
from flext_tests.models import m

logger: FlextLogger = FlextLogger(__name__)


class FlextTestsDocker:
    """Simplified Docker container management for FLEXT tests.

    Essential functionality only:
    - Container creation via docker-compose (python_on_whales)
    - Container status/start/stop (docker SDK)
    - Dirty state tracking for container recreation
    - Port readiness checking
    """

    # ContainerStatus is StrEnum - alias for type hints and runtime use
    ContainerStatus: type = c.Tests.Docker.ContainerStatus

    class ContainerInfo(m.Tests.Docker.ContainerInfo):
        """Container information model for tests - real inheritance from m."""

    SHARED_CONTAINERS: ClassVar[Mapping[str, m.ConfigMap]] = (
        c.Tests.Docker.SHARED_CONTAINERS
    )

    class _OfflineContainers:
        """Minimal container manager that always reports not found.

        Does NOT inherit from ContainerCollection (which is untyped in docker SDK).
        Instead, implements the same interface via composition and protocol.
        """

        def get(self, container_id: str) -> Container:
            """Raise NotFound for any container lookup.

            Business Rule: Always raises NotFound.
            This is intentional for offline mode - containers are not available.
            """
            msg = f"Container {container_id} not found (offline client)"
            raise NotFound(msg)

    class _OfflineDockerClient:
        """Offline Docker client used when the daemon is unavailable.

        Does NOT inherit from DockerClient (which is untyped in docker SDK).
        Instead, wraps the containers interface via composition.
        """

        def __init__(self) -> None:
            """Initialize offline Docker client without contacting daemon."""
            super().__init__()
            self._offline_containers = FlextTestsDocker._OfflineContainers()

        @property
        def containers(self) -> FlextTestsDocker._OfflineContainers:
            """Return offline container manager."""
            return self._offline_containers

    def __init__(
        self,
        workspace_root: Path | None = None,
        worker_id: str | None = None,
    ) -> None:
        """Initialize Docker client with dirty state tracking."""
        super().__init__()
        self._client: DockerSDKClient | FlextTestsDocker._OfflineDockerClient | None = (
            None
        )
        self.logger: FlextLogger = FlextLogger(__name__)
        self.workspace_root = workspace_root or Path.cwd()
        self.worker_id = worker_id or "master"
        self._dirty_containers: set[str] = set()
        self._state_file = (
            Path.home() / ".flext" / f"docker_state_{self.worker_id}.json"
        )
        self._load_dirty_state()
        _ = self.get_client()  # Initialize client

    @property
    def shared_containers(
        self,
    ) -> Mapping[str, m.ConfigMap]:
        """Get shared container configurations."""
        return c.Tests.Docker.SHARED_CONTAINERS

    def get_client(self) -> DockerSDKClient | FlextTestsDocker._OfflineDockerClient:
        """Get Docker client with lazy initialization.

        Returns either a real DockerClient connected to daemon, or an offline stub
        if the daemon is unavailable.
        """
        if self._client is None:
            try:
                self._client = docker_from_env()
            except (DockerException, OSError, TypeError, ValueError) as error:
                self.logger.exception(
                    "Failed to initialize Docker client",
                    error=str(error),
                )
                self._client = self._OfflineDockerClient()
        return self._client

    def _load_dirty_state(self) -> None:
        """Load dirty container state from persistent storage."""
        try:
            if self._state_file.exists():
                with self._state_file.open("r") as f:
                    state = json.load(f)
                    self._dirty_containers = set(state.get("dirty_containers", []))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            self.logger.warning("Failed to load dirty state", error=str(exc))
            self._dirty_containers = set()

    def _save_dirty_state(self) -> None:
        """Save dirty container state to persistent storage."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with self._state_file.open("w") as f:
                json.dump({"dirty_containers": list(self._dirty_containers)}, f)
        except (OSError, TypeError) as exc:
            self.logger.warning("Failed to save dirty state", error=str(exc))

    def mark_container_dirty(self, container_name: str) -> r[bool]:
        """Mark a container as dirty for recreation on next use."""
        try:
            self._dirty_containers.add(container_name)
            self._save_dirty_state()
            self.logger.info("Container marked dirty", container=container_name)
            return r[bool].ok(value=True)
        except (OSError, TypeError) as exc:
            return r[bool].fail(f"Failed to mark dirty: {exc}")

    def mark_container_clean(self, container_name: str) -> r[bool]:
        """Mark a container as clean after successful recreation."""
        try:
            self._dirty_containers.discard(container_name)
            self._save_dirty_state()
            self.logger.info("Container marked clean", container=container_name)
            return r[bool].ok(value=True)
        except (OSError, TypeError) as exc:
            return r[bool].fail(f"Failed to mark clean: {exc}")

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
            _ = self.compose_down(compose_file)  # Result not needed, just cleanup
            result = self.compose_up(compose_file, service, force_recreate=True)

            if result.is_success:
                _ = self.mark_container_clean(container_name)  # Result not needed
                cleaned.append(container_name)

        return r[list[str]].ok(cleaned)

    def get_container_info(
        self,
        container_name: str,
    ) -> r[m.Tests.Docker.ContainerInfo]:
        """Get container information."""
        try:
            client = self.get_client()
            container = client.containers.get(container_name)
            ports_raw = getattr(container, "ports", {}) or {}
            ports: dict[str, str] = {}
            for k, v in ports_raw.items():
                if v:
                    ports[str(k)] = str(v[0].get("HostPort", "")) if v else ""

            # Access Container attributes via getattr for type safety
            status_val = str(getattr(container, "status", "unknown"))
            image_obj = getattr(container, "image", None)
            image_tags = getattr(image_obj, "tags", []) if image_obj else []
            container_id = str(getattr(container, "id", ""))
            return r[m.Tests.Docker.ContainerInfo].ok(
                m.Tests.Docker.ContainerInfo(
                    name=container_name,
                    status=self.ContainerStatus(status_val),
                    ports=ports,
                    image=(str(image_tags[0]) if image_tags else ""),
                    container_id=container_id,
                ),
            )
        except NotFound:
            return r[m.Tests.Docker.ContainerInfo].fail(
                f"Container {container_name} not found",
            )
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            return r[m.Tests.Docker.ContainerInfo].fail(str(exc))

    def get_container_status(
        self,
        container_name: str,
    ) -> r[m.Tests.Docker.ContainerInfo]:
        """Get container status (alias for get_container_info)."""
        return self.get_container_info(container_name)

    def start_existing_container(self, name: str) -> r[str]:
        """Start an existing stopped container without recreating it."""
        try:
            client = self.get_client()
            container = client.containers.get(name)
            # Access Container attributes via getattr for type safety
            status_val = str(getattr(container, "status", "unknown"))

            if status_val == "running":
                return r[str].ok(f"Container {name} already running")

            if status_val in {"exited", "created", "paused"}:
                start_fn = getattr(container, "start", None)
                if callable(start_fn):
                    _ = start_fn()
                return r[str].ok(f"Container {name} started")

            return r[str].ok(f"Container {name} in state: {status_val}")

        except NotFound:
            return r[str].fail(f"Container {name} not found")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return r[str].fail(f"Failed to start container: {exc}")

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

            if docker is None:
                return r[str].fail("python_on_whales is not available")
            docker_client = docker

            original_files = docker_client.client_config.compose_files
            try:
                docker_client.client_config.compose_files = [str(compose_path)]

                if force_recreate:
                    with contextlib.suppress(Exception):
                        docker_client.compose.down(remove_orphans=True, volumes=True)

                services = [service] if service else []
                docker_client.compose.up(
                    services=services,
                    detach=True,
                    remove_orphans=True,
                )
            finally:
                docker_client.client_config.compose_files = original_files

            return r[str].ok("Compose up successful")

        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.logger.exception("Compose up failed")
            return r[str].fail(f"Compose up failed: {exc}")

    def compose_down(self, compose_file: str) -> r[str]:
        """Stop services using docker-compose via python_on_whales."""
        try:
            compose_path = Path(compose_file)
            if not compose_path.is_absolute():
                compose_path = self.workspace_root / compose_file

            if docker is None:
                return r[str].fail("python_on_whales is not available")
            docker_client = docker

            original_files = docker_client.client_config.compose_files
            try:
                docker_client.client_config.compose_files = [str(compose_path)]
                docker_client.compose.down(volumes=True, remove_orphans=True)
            finally:
                docker_client.client_config.compose_files = original_files

            return r[str].ok("Compose down successful")

        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.logger.warning("Compose down failed", error=str(exc))
            return r[str].fail(f"Compose down failed: {exc}")

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
                        return r[bool].ok(value=True)
                except OSError:
                    pass

                time.sleep(2)

            self.logger.warning(
                "Port not ready",
                host=host,
                port=port,
                max_wait=max_wait,
            )
            return r[bool].ok(False)

        except OSError as exc:
            return r[bool].fail(f"Failed to wait for port: {exc}")


__all__ = ["FlextTestsDocker"]
