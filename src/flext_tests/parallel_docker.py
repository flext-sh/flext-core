"""Parallel Docker container management for FLEXT ecosystem.

This module provides centralized Docker container management to enable
parallel test execution across flext-ldap, flext-ldif, and client-a-oud-mig
without port conflicts or resource contention.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import ClassVar, Self

from flext_core import FlextLogger, FlextResult
from flext_tests.docker import FlextTestDocker


@dataclass(frozen=True)
class ContainerSpec:
    """Specification for a Docker container."""

    name: str
    port: int
    compose_file: str | None = None
    service_name: str | None = None
    health_check_url: str | None = None


class ParallelDockerManager:
    """Centralized Docker container manager for parallel test execution.

    Manages container sharing and prevents port conflicts across multiple
    test suites running in parallel.
    """

    # Centralized container specifications
    CONTAINER_SPECS: ClassVar[dict[str, ContainerSpec]] = {
        "flext-openldap-test": ContainerSpec(
            name="flext-openldap-test",
            port=3390,
            compose_file="/home/marlonsc/flext/docker/docker-compose.openldap.yml",
            service_name="openldap",
            health_check_url="ldap://localhost:3390",
        ),
        "flext-client-a-oud-test": ContainerSpec(
            name="flext-client-a-oud-test",
            port=3389,
            compose_file="/home/marlonsc/flext/docker/docker-compose.client-a-oud.yml",
            service_name="client-a-oud",
            health_check_url="ldap://localhost:3389",
        ),
    }

    # Class-level tracking
    _instance_lock = threading.Lock()

    # Instance tracking
    _initialized: bool
    _instance: Self | None = None
    _active_containers: ClassVar[
        dict[str, int]
    ] = {}  # container_name -> reference_count
    _container_locks: ClassVar[dict[str, threading.Lock]] = {}

    def __new__(cls) -> Self:
        """Singleton pattern for centralized management."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the parallel Docker manager."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._logger = FlextLogger(__name__)
        self._docker = FlextTestDocker()
        self._initialized = True

        # Initialize locks for each container
        for container_name in self.CONTAINER_SPECS:
            if container_name not in self._container_locks:
                self._container_locks[container_name] = threading.Lock()

    def request_container(self, container_name: str) -> FlextResult[ContainerSpec]:
        """Request access to a shared container.

        Args:
            container_name: Name of the container to request

        Returns:
            FlextResult containing container specification if successful

        """
        if container_name not in self.CONTAINER_SPECS:
            return FlextResult[ContainerSpec].fail(
                f"Unknown container: {container_name}"
            )

        spec = self.CONTAINER_SPECS[container_name]

        with self._container_locks[container_name]:
            # Check if container is already running
            status_result = self._docker.get_container_status(container_name)
            if status_result.is_success:
                container_info = status_result.unwrap()
                if (
                    hasattr(container_info, "status")
                    and "running" in str(container_info.status).lower()
                ):
                    # Container is running, increment reference count
                    self._active_containers[container_name] = (
                        self._active_containers.get(container_name, 0) + 1
                    )
                    self._logger.info(
                        f"Sharing existing container {container_name} (refs: {self._active_containers[container_name]})"
                    )
                    return FlextResult[ContainerSpec].ok(spec)

            # Start the container
            self._logger.info(
                f"Starting container {container_name} on port {spec.port}"
            )
            start_result = self._start_container_with_compose(spec)
            if start_result.is_failure:
                return FlextResult[ContainerSpec].fail(
                    start_result.error or "Failed to start container"
                )

            # Increment reference count
            self._active_containers[container_name] = (
                self._active_containers.get(container_name, 0) + 1
            )
            self._logger.info(
                f"Started container {container_name} (refs: {self._active_containers[container_name]})"
            )

            return FlextResult[ContainerSpec].ok(spec)

    def release_container(
        self, container_name: str, *, force_stop: bool = False
    ) -> FlextResult[None]:
        """Release a shared container.

        Args:
            container_name: Name of the container to release
            force_stop: Whether to force stop the container regardless of references

        Returns:
            FlextResult indicating success or failure

        """
        if container_name not in self.CONTAINER_SPECS:
            return FlextResult[None].fail(f"Unknown container: {container_name}")

        with self._container_locks[container_name]:
            current_refs = self._active_containers.get(container_name, 0)

            if current_refs <= 0 and not force_stop:
                self._logger.warning(
                    f"Container {container_name} not currently managed"
                )
                return FlextResult[None].ok(None)

            if force_stop or current_refs <= 1:
                # Last reference or forced stop - actually stop the container
                self._logger.info(f"Stopping container {container_name}")
                stop_result = self._docker.stop_container(container_name, remove=False)

                self._active_containers[container_name] = 0

                if stop_result.is_failure:
                    self._logger.warning(
                        f"Failed to stop container {container_name}: {stop_result.error}"
                    )
                    return FlextResult[None].fail(
                        stop_result.error or "Failed to stop container"
                    )

                self._logger.info(f"Stopped container {container_name}")
            else:
                # Decrement reference count
                self._active_containers[container_name] = current_refs - 1
                self._logger.info(
                    f"Released container {container_name} (refs: {self._active_containers[container_name]})"
                )

            return FlextResult[None].ok(None)

    def _start_container_with_compose(self, spec: ContainerSpec) -> FlextResult[None]:
        """Start container using docker-compose if available, fallback to direct start."""
        if spec.compose_file and spec.service_name:
            # Try docker-compose first
            compose_result = self._docker.compose_up(spec.compose_file)
            if compose_result.is_success:
                # Wait for container to be healthy
                self._wait_for_container_health(spec.name, timeout=60)
                return FlextResult[None].ok(None)

            self._logger.warning(
                f"Compose start failed for {spec.name}: {compose_result.error}"
            )

        # Fallback to direct container start
        start_result = self._docker.start_container(spec.name)
        if start_result.is_success:
            self._wait_for_container_health(spec.name, timeout=60)
            return FlextResult[None].ok(None)

        return FlextResult[None].fail(start_result.error or "Failed to start container")

    def _wait_for_container_health(
        self, container_name: str, timeout: int = 60
    ) -> None:
        """Wait for container to become healthy."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_result = self._docker.get_container_status(container_name)
            if status_result.is_success:
                container_info = status_result.unwrap()
                if (
                    hasattr(container_info, "status")
                    and "running" in str(container_info.status).lower()
                ):
                    # Additional health check could be added here
                    time.sleep(2)  # Give container a moment to fully initialize
                    return

            time.sleep(1)

        self._logger.warning(
            f"Container {container_name} may not be fully healthy after {timeout}s"
        )

    def get_active_containers(self) -> dict[str, int]:
        """Get currently active containers and their reference counts."""
        return self._active_containers.copy()

    def check_port_conflicts(self) -> FlextResult[set[int]]:
        """Check for potential port conflicts."""
        used_ports = {spec.port for spec in self.CONTAINER_SPECS.values()}

        if len(used_ports) != len(self.CONTAINER_SPECS):
            conflicting_ports = set()
            port_counts: dict[int, int] = {}
            for spec in self.CONTAINER_SPECS.values():
                port_counts[spec.port] = port_counts.get(spec.port, 0) + 1
                if port_counts[spec.port] > 1:
                    conflicting_ports.add(spec.port)

            return FlextResult[set[int]].fail(
                f"Port conflicts detected: {conflicting_ports}"
            )

        return FlextResult[set[int]].ok(used_ports)


# Convenience functions for common use cases
def get_shared_openldap_container() -> FlextResult[ContainerSpec]:
    """Get the shared OpenLDAP container for flext-ldap/flext-ldif tests."""
    manager = ParallelDockerManager()
    return manager.request_container("flext-openldap-test")


def get_client-a_oud_container() -> FlextResult[ContainerSpec]:
    """Get the client-a OUD container for client-a-oud-mig tests."""
    manager = ParallelDockerManager()
    return manager.request_container("flext-client-a-oud-test")


def release_shared_openldap_container() -> FlextResult[None]:
    """Release the shared OpenLDAP container."""
    manager = ParallelDockerManager()
    return manager.release_container("flext-openldap-test")


def release_client-a_oud_container() -> FlextResult[None]:
    """Release the client-a OUD container."""
    manager = ParallelDockerManager()
    return manager.release_container("flext-client-a-oud-test")
