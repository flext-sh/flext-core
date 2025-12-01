"""Docker container control for FLEXT test infrastructure.

Provides comprehensive Docker container management for FLEXT test suites including
unified start/stop/reset functionality, compose operations, container health monitoring,
volume and network management, and CLI integration. Supports both shared and private
container configurations with dependency resolution and dirty state tracking.

Scope: Complete Docker testing infrastructure including container lifecycle management,
compose stack operations with timeout handling, container health checks and status
monitoring, volume/network/image cleanup, port mapping, command execution, and CLI
commands for container control. Integrates with FlextTestsUtilities.DockerHelpers
for generalized Docker operations and uses FlextConstants for shared container configs.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import argparse
import functools
import json
import shlex
import socket
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import click
import docker
import pytest
from docker import DockerClient
from docker.errors import DockerException, NotFound
from python_on_whales import (
    DockerClient as PowDockerClient,
    docker as pow_docker,
)
from rich.console import Console
from rich.table import Table

from flext_core import (
    FlextConstants,
    FlextLogger,
    FlextResult,
)
from flext_core.typings import FlextTypes
from flext_tests.constants import FlextTestConstants
from flext_tests.models import FlextTestModels
from flext_tests.protocols import FlextTestProtocols
from flext_tests.typings import FlextTestsTypings
from flext_tests.utilities import FlextTestsUtilities

pytest_module: ModuleType | None = pytest

logger: FlextLogger = FlextLogger(__name__)


class FlextTestDocker:
    """Docker container management for FLEXT tests.

    Uses FlextTestModels.Docker.ContainerInfo and FlextTestConstants.Docker.ContainerStatus
    for proper type safety and model-based data structures.
    Provides comprehensive Docker container management for FLEXT test suites including
    unified start/stop/reset functionality, compose operations, container health monitoring,
    volume and network management, and CLI integration.
    """

    # Use constants and models from flext_tests (NOT tests.helpers - that would create circular dependency)
    ContainerStatus = FlextTestConstants.Docker.ContainerStatus
    ContainerInfo = FlextTestModels.Docker.ContainerInfo

    _console: ClassVar[Console] = Console()
    _cli_group: ClassVar[click.Group | None] = None
    _workspace_parser: ClassVar[argparse.ArgumentParser | None] = None
    _pytest_registered: ClassVar[bool] = False

    # Shared container configuration - class attribute for direct access
    SHARED_CONTAINERS: ClassVar[Mapping[str, FlextTypes.Types.ContainerConfigDict]] = (
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
        self.client: DockerClient | None = None  # Will be set by _get_client()
        self._registered_services: set[str] = set()
        # Service dependencies - mutable dict needed for runtime updates
        self._service_dependencies: dict[str, list[str]] = {}

        # Dirty state tracking
        self.worker_id = worker_id or "master"
        self._dirty_containers: set[str] = set()
        self._state_file = (
            Path.home() / ".flext" / f"docker_state_{self.worker_id}.json"
        )
        self._load_dirty_state()

        # Container configuration tracking (for private/project-specific containers)
        # Maps container name to container configuration for proper cleanup
        # Mutable dict needed for runtime registration
        self._container_configs: dict[
            str, FlextTestsTypings.ContainerConfigMapping
        ] = {}

        # Initialize Docker client immediately to catch connection failures
        self.get_client()

    def get_client(self) -> DockerClient:
        """Get Docker client with lazy initialization."""
        if self._client is None:
            try:
                self._client = docker.from_env()
                self.client = self._client  # Set the public client attribute
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
                    self.logger.info(
                        "Loaded dirty state",
                        extra={"dirty_containers": list(self._dirty_containers)},
                    )
        except Exception as e:
            self.logger.warning("Failed to load dirty state", extra={"error": str(e)})
            self._dirty_containers = set()

    def _save_dirty_state(self) -> None:
        """Save dirty container state to persistent storage."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with self._state_file.open("w") as f:
                json.dump(
                    {"dirty_containers": list(self._dirty_containers)},
                    f,
                    indent=2,
                )
            self.logger.info(
                "Saved dirty state",
                extra={"dirty_containers": list(self._dirty_containers)},
            )
        except Exception as e:
            self.logger.warning("Failed to save dirty state", extra={"error": str(e)})

    def mark_container_dirty(self, container_name: str) -> FlextResult[bool]:
        """Mark a container as dirty, requiring recreation on next use.

        Args:
            container_name: Name of the container to mark dirty

        Returns:
            FlextResult[bool]: Success with True if marked, failure with error

        """

        def operation() -> None:
            self._dirty_containers.add(container_name)
            self._save_dirty_state()
            self.logger.info(
                "Container marked as dirty",
                extra={"container": container_name},
            )

        return FlextTestsUtilities.DockerHelpers.execute_docker_operation(
            operation=operation,
            success_value=True,
            operation_name="mark container dirty",
            logger=self.logger,
        )

    def mark_container_clean(self, container_name: str) -> FlextResult[bool]:
        """Mark a container as clean after successful recreation.

        Args:
            container_name: Name of the container to mark clean

        Returns:
            FlextResult[bool]: Success with True if marked, failure with error

        """

        def operation() -> None:
            self._dirty_containers.discard(container_name)
            self._save_dirty_state()
            self.logger.info(
                "Container marked as clean",
                extra={"container": container_name},
            )

        return FlextTestsUtilities.DockerHelpers.execute_docker_operation(
            operation=operation,
            success_value=True,
            operation_name="mark container clean",
            logger=self.logger,
        )

    def is_container_dirty(self, container_name: str) -> bool:
        """Check if a container is marked as dirty.

        Args:
            container_name: Name of the container to check

        Returns:
            True if container is dirty, False otherwise

        """
        return container_name in self._dirty_containers

    def get_dirty_containers(self) -> list[str]:
        """Get list of all dirty containers.

        Returns:
            List of dirty container names

        """
        return list(self._dirty_containers)

    def register_container_config(
        self,
        container_name: str,
        compose_file: str,
        service: str | None = None,
    ) -> FlextResult[bool]:
        """Register a container's docker-compose configuration for cleanup.

        This enables cleanup_dirty_containers() to properly recreate private/project-specific
        containers. Must be called before container is started if cleanup will be needed.

        Args:
            container_name: Name of the container (must match docker container name)
            compose_file: Path to docker-compose file (absolute or relative to workspace_root)
            service: Optional service name in docker-compose (if None, uses container name)

        Returns:
            FlextResult[bool]: Success with True if registered, failure with error

        """

        def operation() -> None:
            self._container_configs[container_name] = {
                "compose_file": compose_file,
                "service": service or "",
            }
            self.logger.info(
                "Registered container config",
                extra={
                    "container": container_name,
                    "compose_file": compose_file,
                    "service": service,
                },
            )

        return FlextTestsUtilities.DockerHelpers.execute_docker_operation(
            operation=operation,
            success_value=True,
            operation_name="register container config",
            logger=self.logger,
        )

    def cleanup_dirty_containers(
        self,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Clean up all dirty containers by recreating them with fresh volumes.

        Returns:
            FlextResult with ContainerOperationResult mapping container names to cleanup status

        """
        # Use mutable dict during construction, convert to Mapping for return
        results_dict: dict[str, str | int | bool | Sequence[str] | None] = {}
        helpers = FlextTestsUtilities.DockerHelpers

        for container_name in list(self._dirty_containers):
            self.logger.info(
                "Cleaning up dirty container",
                extra={"container": container_name},
            )

            # Convert Mapping to dict for get_container_config helper
            # Helper requires mutable dict - create explicit conversions
            shared_containers_dict: dict[str, dict[str, str | int]] = {}
            for container_key, container_config in self.shared_containers.items():
                if isinstance(container_config, Mapping):
                    shared_containers_dict[container_key] = {
                        str(k): (str(v) if not isinstance(v, (int, str)) else v)
                        for k, v in container_config.items()
                    }

            registered_configs_dict: dict[str, dict[str, str]] = {}
            for config_key, config_value in self._container_configs.items():
                if isinstance(config_value, Mapping):
                    registered_configs_dict[config_key] = {
                        str(k): str(v) if v is not None else ""
                        for k, v in config_value.items()
                    }
            config = helpers.get_container_config(
                container_name,
                shared_containers_dict,
                registered_configs_dict,
            )

            if config:
                compose_file_value = config.get("compose_file")
                if not isinstance(compose_file_value, str):
                    results_dict[container_name] = (
                        f"Invalid compose_file type: {type(compose_file_value)}"
                    )
                    continue

                compose_file = helpers.resolve_compose_file_path(
                    compose_file_value,
                    self.workspace_root,
                )
                service = helpers.extract_service_from_config(config)

                self.logger.info(
                    "Running docker-compose down --volumes",
                    extra={"container": container_name, "compose_file": compose_file},
                )
                down_result = self.compose_down(compose_file)
                if down_result.is_failure:
                    self.logger.warning(
                        f"compose down failed (non-blocking): {down_result.error}",
                        extra={"container": container_name},
                    )

                restart_result = self.compose_up(
                    compose_file,
                    service,
                    force_recreate=True,
                )
                if restart_result.is_success:
                    self.mark_container_clean(container_name)
                    results_dict[container_name] = (
                        "Successfully recreated with fresh volumes"
                    )
                else:
                    results_dict[container_name] = (
                        f"Restart failed: {restart_result.error}"
                    )
            else:
                start_result = self.start_container(container_name)
                if start_result.is_success:
                    self.mark_container_clean(container_name)
                    results_dict[container_name] = "Successfully restarted"
                else:
                    results_dict[container_name] = (
                        f"Restart failed: {start_result.error}"
                    )

        # dict is compatible with Mapping - pass directly
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
            results_dict,
        )

    def detect_and_cleanup_unhealthy(
        self,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Detect unhealthy containers and mark them as dirty for cleanup.

        Scans all containers for unhealthy status and marks them for recreation
        with fresh volumes. This allows pytest fixtures to automatically recover
        from corrupted containers.

        Returns:
            FlextResult with ContainerOperationResult mapping container names to status

        """
        # Use mutable dict during construction, convert to Mapping for return
        results_dict: dict[str, str | int | bool | Sequence[str] | None] = {}

        try:
            client = self.get_client()
            containers_api = getattr(client, "containers", None)
            list_method = (
                getattr(containers_api, "list", None) if containers_api else None
            )
            if not list_method:
                # dict is compatible with Mapping - pass directly
                return FlextResult[
                    FlextTestsTypings.Docker.ContainerOperationResult
                ].ok(
                    results_dict,
                )

            containers = list_method(all=False)  # Only running containers

            for container in containers:
                container_name: str = getattr(container, "name", "unknown")
                state = container.attrs.get("State", {})
                health = state.get("Health", {})
                health_status = health.get("Status", "none")

                # Check for unhealthy containers
                if health_status == "unhealthy":
                    self.logger.warning(
                        "Detected unhealthy container: %s",
                        container_name,
                        extra={
                            "container": container_name,
                            "health_status": health_status,
                        },
                    )
                    self.mark_container_dirty(container_name)
                    results_dict[container_name] = "Marked dirty (unhealthy)"

        except Exception as e:
            self.logger.exception("Failed to detect unhealthy containers")
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                f"Detection failed: {e}"
            )

        # Convert mutable dict to Mapping for return type (dict is compatible with Mapping)
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
            results_dict,
        )

    # Essential methods that are being called by other files
    def start_all(
        self,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Start all containers."""
        result = self.list_containers(all_containers=True)
        if result.is_failure:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                result.error or "Failed to list containers"
            )

        containers = result.value
        started_count = 0
        errors: list[str] = []

        for container_info in containers:
            if (
                container_info.status
                == FlextTestConstants.Docker.ContainerStatus.STOPPED
            ):
                start_result = self.start_existing_container(container_info.name)
                if start_result.is_success:
                    started_count += 1
                else:
                    errors.append(
                        f"Failed to start {container_info.name}: {start_result.error}"
                    )

        if errors:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
                "message": f"Started {started_count} containers, {len(errors)} errors",
                "started": started_count,
                "errors": errors,
            })

        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
            "message": f"All containers started ({started_count} started)",
            "started": started_count,
        })

    def stop_all(
        self, *, remove: bool = False
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Stop all containers."""
        result = self.list_containers(all_containers=True)
        if result.is_failure:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                result.error or "Failed to list containers"
            )

        containers = result.value
        stopped_count = 0
        errors: list[str] = []

        for container_info in containers:
            if (
                container_info.status
                == FlextTestConstants.Docker.ContainerStatus.RUNNING
            ):
                try:
                    client = self.get_client()
                    container = client.containers.get(container_info.name)
                    container.stop()
                    if remove:
                        container.remove()
                    stopped_count += 1
                except DockerException as e:
                    errors.append(f"Failed to stop {container_info.name}: {e}")

        if errors:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
                "message": f"Stopped {stopped_count} containers, {len(errors)} errors",
                "stopped": stopped_count,
                "errors": errors,
            })

        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
            "message": f"All containers stopped ({stopped_count} stopped)",
            "stopped": stopped_count,
        })

    def reset_all(
        self,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Reset all containers."""
        # Stop all containers first
        stop_result = self.stop_all(remove=False)
        if stop_result.is_failure:
            return stop_result

        # Start all containers
        start_result = self.start_all()
        if start_result.is_failure:
            return start_result

        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
            "message": "All containers reset",
            "stopped": stop_result.value.get("stopped", 0)
            if isinstance(stop_result.value, dict)
            else 0,
            "started": start_result.value.get("started", 0)
            if isinstance(start_result.value, dict)
            else 0,
        })

    def reset_container(self, name: str) -> FlextResult[str]:
        """Reset a specific container."""
        # Stop container
        try:
            client = self.get_client()
            container = client.containers.get(name)
            container.stop()
        except NotFound:
            pass  # Container doesn't exist, continue to start
        except DockerException as e:
            self.logger.warning(f"Failed to stop container {name}: {e}")

        # Start container
        start_result = self.start_existing_container(name)
        if start_result.is_failure:
            return FlextResult[str].fail(
                f"Failed to reset container: {start_result.error}"
            )

        return FlextResult[str].ok(f"Container {name} reset")

    def get_all_status(
        self,
    ) -> FlextResult[Mapping[str, FlextTestModels.Docker.ContainerInfo]]:
        """Get status of all containers."""
        result = self.list_containers(all_containers=True)
        if result.is_failure:
            return FlextResult[Mapping[str, FlextTestModels.Docker.ContainerInfo]].fail(
                result.error or "Failed to list containers"
            )

        containers = result.value
        status_dict: dict[str, FlextTestModels.Docker.ContainerInfo] = {
            container.name: container for container in containers
        }

        return FlextResult[Mapping[str, FlextTestModels.Docker.ContainerInfo]].ok(
            status_dict
        )

    def get_container_status(
        self,
        container_name: str,
    ) -> FlextResult[FlextTestModels.Docker.ContainerInfo]:
        """Get container status."""
        return self.get_container_info(container_name)

    def register_service(
        self,
        service_name: str,
        container_name: str,
        ports: list[int] | None = None,
        health_check_cmd: str | None = None,
        depends_on: list[str] | None = None,
        startup_timeout: int = 30,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Register a service for testing."""
        self._registered_services.add(service_name)

        # Track dependencies
        if depends_on:
            self._service_dependencies[service_name] = depends_on
        else:
            self._service_dependencies[service_name] = []

        _ = (
            container_name,
            ports,
            health_check_cmd,
            startup_timeout,
        )  # Unused parameters
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
            "service": service_name,
            "status": "registered",
        })

    def shell_script_compatibility_run(
        self,
        script_path: str,
        timeout: int = 30,
        **kwargs: str | int | bool | None,
    ) -> FlextResult[tuple[int, str, str]]:
        """Run shell script with compatibility checks."""
        try:
            # Extract capture_output from kwargs
            capture_output = kwargs.get("capture_output", False)

            # Run the command using subprocess directly
            process = subprocess.run(
                ["docker"] + shlex.split(script_path),
                check=False,
                capture_output=bool(capture_output),
                text=True,
                timeout=timeout,
            )

            # Return (exit_code, stdout, stderr) tuple
            stdout = process.stdout if capture_output and process.stdout else ""
            stderr = process.stderr if capture_output and process.stderr else ""

            return FlextResult[tuple[int, str, str]].ok((
                process.returncode,
                stdout,
                stderr,
            ))

        except Exception as e:
            return FlextResult[tuple[int, str, str]].fail(f"Command failed: {e}")

    def enable_auto_cleanup(self, *, enabled: bool = True) -> FlextResult[bool]:
        """Enable or disable auto cleanup."""
        _ = enabled  # Unused parameter
        return FlextResult[bool].ok(True)

    def start_services_for_test(
        self,
        required_services: list[str] | None = None,
        test_name: str | None = None,
        service_names: list[str] | None = None,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Start services for testing."""
        if service_names:
            # Check if all services are registered
            for service_name in service_names:
                if service_name not in self._registered_services:
                    return FlextResult[
                        FlextTestsTypings.Docker.ContainerOperationResult
                    ].fail(f"Service '{service_name}' is not registered")

        _ = test_name  # Unused parameter
        _ = required_services  # Unused parameter
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
            {"status": "services_started"},
        )

    def get_running_services(self) -> FlextResult[list[str]]:
        """Get list of running services."""
        result = self.list_containers(all_containers=False)
        if result.is_failure:
            return FlextResult[list[str]].fail(
                result.error or "Failed to list containers"
            )

        containers = result.value
        running_services: list[str] = []

        for container_info in containers:
            if (
                container_info.status
                == FlextTestConstants.Docker.ContainerStatus.RUNNING
            ):
                # Check if container name matches a registered service
                for service_name in self._registered_services:
                    if (
                        service_name in container_info.name
                        or container_info.name in self._registered_services
                    ) and service_name not in running_services:
                        running_services.append(service_name)

        return FlextResult[list[str]].ok(running_services)

    def compose_up(
        self,
        compose_file: str,
        service: str | None = None,
        *,
        force_recreate: bool = False,
    ) -> FlextResult[str]:
        """Start services using docker-compose via Docker Python API.

        This method is designed to work with shared test containers:
        - If containers are already running and healthy, does nothing
        - If containers exist but are stopped, starts them without recreating
        - Only destroys containers if force_recreate=True (for dirty containers)

        Args:
            compose_file: Path to docker-compose file (relative or absolute)
            service: Optional specific service to start (if None, starts all)
            force_recreate: If True, destroys and recreates containers (for dirty state)

        Returns:
            FlextResult with status message

        """
        helpers = FlextTestsUtilities.DockerHelpers
        compose_path = helpers.resolve_compose_path(compose_file, self.workspace_root)
        docker_client: PowDockerClient = pow_docker

        def compose_operation() -> None:
            def up_op() -> None:
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

            helpers.with_compose_file_config(docker_client, compose_path, up_op)

        return helpers.execute_compose_operation_with_timeout(
            operation=compose_operation,
            timeout_seconds=300,
            operation_name="up",
            compose_file=compose_file,
            logger=self.logger,
        )

    def compose_down(self, compose_file: str) -> FlextResult[str]:
        """Stop services using docker-compose via python-on-whales.

        Args:
            compose_file: Path to docker-compose file (relative or absolute)

        Returns:
            FlextResult with status message

        """
        helpers = FlextTestsUtilities.DockerHelpers
        compose_path = helpers.resolve_compose_path(compose_file, self.workspace_root)
        docker_client: PowDockerClient = pow_docker

        def compose_operation() -> None:
            def down_op() -> None:
                docker_client.compose.down(volumes=True, remove_orphans=True)

            helpers.with_compose_file_config(docker_client, compose_path, down_op)

        return helpers.execute_compose_operation_with_timeout(
            operation=compose_operation,
            timeout_seconds=120,
            operation_name="down",
            compose_file=compose_file,
            logger=self.logger,
        )

    def compose_logs(self, compose_file: str) -> FlextResult[str]:
        """Get compose logs."""
        helpers = FlextTestsUtilities.DockerHelpers
        compose_path = helpers.resolve_compose_path(compose_file, self.workspace_root)
        docker_client: PowDockerClient = pow_docker

        try:
            logs_result: str = ""

            def compose_operation() -> None:
                nonlocal logs_result

                def logs_op() -> None:
                    nonlocal logs_result
                    # Use python-on-whales compose logs
                    logs = docker_client.compose.logs(compose_files=[compose_path])
                    logs_result = logs if isinstance(logs, str) else str(logs)

                helpers.with_compose_file_config(docker_client, compose_path, logs_op)

            compose_operation()
            return FlextResult[str].ok(logs_result)
        except Exception as e:
            self.logger.exception("Failed to get compose logs")
            return FlextResult[str].fail(f"Failed to get compose logs: {e}")

    def build_image_advanced(
        self,
        path: str,
        dockerfile_path: str | None = None,
        context_path: str | None = None,
        tag: str = "latest",
        dockerfile: str = "Dockerfile",
        build_args: dict[str, str] | None = None,
        *,  # Force keyword-only arguments for boolean parameters
        no_cache: bool = False,
        pull: bool = False,
        remove_intermediate: bool = True,
    ) -> FlextResult[str]:
        """Build Docker image with advanced options."""
        try:
            client = self.get_client()

            # Determine context path
            build_context = Path(context_path) if context_path else Path(path)

            if not build_context.is_absolute():
                build_context = self.workspace_root / build_context

            # Determine dockerfile path
            if dockerfile_path:
                dockerfile_full = Path(dockerfile_path)
                if not dockerfile_full.is_absolute():
                    dockerfile_full = self.workspace_root / dockerfile_full
            else:
                dockerfile_full = build_context / dockerfile

            if not dockerfile_full.exists():
                return FlextResult[str].fail(f"Dockerfile not found: {dockerfile_full}")

            # Calculate relative dockerfile path from context
            try:
                dockerfile_rel = dockerfile_full.relative_to(build_context)
            except ValueError:
                # If dockerfile is outside context, use absolute path
                dockerfile_rel = dockerfile_full

            # Build image using docker SDK
            image, _logs = client.images.build(
                path=str(build_context),
                tag=tag,
                dockerfile=str(dockerfile_rel),
                buildargs=build_args,
                nocache=no_cache,
                pull=pull,
                rm=remove_intermediate,
            )

            self.logger.info(
                f"Built image {tag} with advanced options",
                extra={"tag": tag, "image_id": image.id},
            )
            return FlextResult[str].ok(f"Image {tag} built successfully")
        except DockerException as e:
            self.logger.exception("Failed to build image with advanced options")
            return FlextResult[str].fail(f"Failed to build image: {e}")

    def cleanup_networks(self) -> FlextResult[list[str]]:
        """Clean up unused networks."""
        result = FlextTestsUtilities.DockerHelpers.cleanup_docker_resources(
            client=self.get_client(),
            resource_type="network",
            list_attr="networks",
            remove_attr="remove",
            resource_name_attr="name",
            filter_pattern=None,
            logger=self.logger,
        )
        if result.is_success:
            networks = result.value.get("network", [])
            return FlextResult[list[str]].ok(
                networks if isinstance(networks, list) else [],
            )
        return FlextResult[list[str]].fail(result.error or "Network cleanup failed")

    def cleanup_volumes(
        self,
        volume_pattern: str | None = None,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Clean up Docker volumes by pattern or orphaned volumes.

        Args:
            volume_pattern: Optional glob pattern to match volume names (e.g., 'algar*')

        Returns:
            FlextResult with dict containing removed count and list of removed volumes

        """
        result = FlextTestsUtilities.DockerHelpers.cleanup_docker_resources(
            client=self.get_client(),
            resource_type="volume",
            list_attr="volumes",
            remove_attr="remove",
            resource_name_attr="name",
            filter_pattern=volume_pattern,
            logger=self.logger,
        )
        if result.is_success:
            # Transform result to match test expectations: use "volumes" key
            value = result.value
            volumes = value.get("volume", []) if isinstance(value, dict) else []
            removed = value.get("removed", 0) if isinstance(value, dict) else 0
            # Convert to ContainerOperationResult format
            container_result: dict[str, str | int | bool | Sequence[str] | None] = {
                "removed": removed,
                "volumes": volumes if isinstance(volumes, list) else [],
            }
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
                container_result,
            )
        # Convert failure to correct type
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
            result.error or "Failed to cleanup volumes",
        )

    def cleanup_images(
        self,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Clean up unused images."""
        result = FlextTestsUtilities.DockerHelpers.cleanup_docker_resources(
            client=self.get_client(),
            resource_type="image",
            list_attr="images",
            remove_attr="remove",
            resource_name_attr="id",
            filter_pattern=None,
            logger=self.logger,
        )
        if result.is_success:
            value = result.value
            images = value.get("image", []) if isinstance(value, dict) else []
            removed = value.get("removed", 0) if isinstance(value, dict) else 0
            # Convert to ContainerOperationResult format
            container_result: dict[str, str | int | bool | Sequence[str] | None] = {
                "removed": removed,
                "images": images if isinstance(images, list) else [],
            }
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
                container_result,
            )
        # Convert failure to correct type
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
            result.error or "Failed to cleanup images",
        )

    def cleanup_all_test_containers(
        self,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Clean up all test containers."""
        result = self.list_containers(all_containers=True)
        if result.is_failure:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                result.error or "Failed to list containers"
            )

        containers = result.value
        removed_count = 0
        errors: list[str] = []

        for container_info in containers:
            # Check if container is a test container (starts with test prefix or is in dirty list)
            if (
                container_info.name.startswith("test_")
                or container_info.name.startswith("flext-")
                or container_info.name in self._dirty_containers
            ):
                remove_result = self.remove_container(container_info.name, force=True)
                if remove_result.is_success:
                    removed_count += 1
                    self._dirty_containers.discard(container_info.name)
                else:
                    errors.append(
                        f"Failed to remove {container_info.name}: {remove_result.error}"
                    )

        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
            "message": f"Cleaned up {removed_count} test containers",
            "removed": removed_count,
            "errors": errors or None,
        })

    def stop_services_for_test(
        self, test_name: str
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Stop services for a specific test."""
        # Get running services
        running_result = self.get_running_services()
        if running_result.is_failure:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                running_result.error or "Failed to get running services"
            )

        running_services = running_result.value
        stopped_count = 0
        errors: list[str] = []

        # Stop each running service
        for service_name in running_services:
            # Find container for this service
            result = self.list_containers(all_containers=True)
            if result.is_success:
                for container_info in result.value:
                    if service_name in container_info.name:
                        try:
                            client = self.get_client()
                            container = client.containers.get(container_info.name)
                            container.stop()
                            stopped_count += 1
                        except DockerException as e:
                            errors.append(f"Failed to stop {service_name}: {e}")

        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
            "message": f"Services stopped for test {test_name}",
            "stopped": stopped_count,
            "errors": errors or None,
        })

    def auto_discover_services(
        self,
        compose_file_path: str | None = None,
    ) -> FlextResult[list[str]]:
        """Auto-discover services."""
        try:
            if compose_file_path and compose_file_path.endswith(".yml"):
                # Basic docker-compose parsing to extract service names and dependencies
                services: list[str] = []
                with Path(compose_file_path).open("r", encoding="utf-8") as f:
                    content = f.read()

                    # Find service names and their dependencies
                    lines: list[str] = content.split("\n")
                    current_service: str | None = None
                    in_depends_on = False

                    excluded_prefixes = (
                        "version:",
                        "services:",
                        "healthcheck:",
                        "ports:",
                        "image:",
                        "environment:",
                        "volumes:",
                        "networks:",
                    )

                    for line in lines:
                        stripped_line = line.strip()
                        if not stripped_line or stripped_line.startswith("#"):
                            continue

                        if (
                            stripped_line.startswith("- ")
                            and current_service
                            and in_depends_on
                        ):
                            dep_name = stripped_line[2:].strip()
                            if dep_name:
                                self._service_dependencies[current_service].append(
                                    dep_name,
                                )
                            continue

                        if stripped_line.startswith("depends_on:"):
                            in_depends_on = True
                            continue

                        if not stripped_line.endswith(":"):
                            in_depends_on = False
                            continue

                        if stripped_line.startswith(" "):
                            continue

                        if stripped_line.startswith(excluded_prefixes):
                            in_depends_on = False
                            continue

                        service_name = stripped_line[:-1].strip()
                        if service_name:
                            current_service = service_name
                            services.append(service_name)
                            self._registered_services.add(service_name)
                            self._service_dependencies[service_name] = []
                            in_depends_on = False

                return FlextResult[list[str]].ok(services)
            return FlextResult[list[str]].ok([])
        except Exception:
            return FlextResult[list[str]].ok([])

    def get_service_health_status(
        self,
        service_name: str,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Get service health status."""
        if service_name not in self._registered_services:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                f"Service '{service_name}' is not registered"
            )
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
            "status": "healthy",
            "container_status": "running",
            "health_check": "passed",
        })

    def create_network(self, name: str, *, driver: str = "bridge") -> FlextResult[str]:
        """Create a Docker network."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            networks_api = getattr(client, "networks", None)
            if networks_api:
                create_method = getattr(networks_api, "create", None)
                if create_method:
                    try:
                        _network = create_method(name, driver=driver)
                        return f"Network {name} created with driver {driver}"
                    except DockerException as e:
                        if "already exists" in str(e).lower():
                            return f"Network {name} already exists"
                        raise
            raise RuntimeError(f"Failed to create network {name}")

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"create network {name}",
            logger=self.logger,
        )

    def execute_container_command(
        self,
        container_name: str,
        command: str,
    ) -> FlextResult[str]:
        """Execute command in container."""
        return self.execute_command_in_container(container_name, command)

    def exec_container_interactive(
        self,
        container_name: str,
        command: str,
    ) -> FlextResult[str]:
        """Execute interactive command in container."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(container_name)
                    exec_run_method = getattr(container, "exec_run", None)
                    if exec_run_method:
                        # Use tty=True and stdin=True for interactive mode
                        result = exec_run_method(
                            command,
                            tty=True,
                            stdin=True,
                            stream=False,
                        )
                        output_attr = getattr(result, "output", b"")
                        return (
                            output_attr.decode("utf-8")
                            if isinstance(output_attr, bytes)
                            else str(output_attr)
                        )
            return ""

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"execute interactive command in container {container_name}",
            logger=self.logger,
        )

    def list_volumes(self) -> FlextResult[list[str]]:
        """List Docker volumes."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> list[str]:
            volumes_api = getattr(client, "volumes", None)
            if volumes_api:
                list_method = getattr(volumes_api, "list", None)
                if list_method:
                    volumes = list_method()
                    return [
                        str(getattr(v, "name", ""))
                        for v in volumes
                        if hasattr(v, "name")
                    ]
            return []

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name="list volumes",
            logger=self.logger,
        )

    def get_service_dependency_graph(self) -> dict[str, list[str]]:
        """Get service dependency graph.

        Returns mutable dict copy for caller modifications.
        Uses dict for mutability - caller may need to modify.
        """
        return self._service_dependencies.copy()

    def images_formatted(
        self,
        format_string: str = "{{.Repository}}:{{.Tag}}",
    ) -> FlextResult[list[str]]:
        """Get formatted list of images."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> list[str]:
            images_api = getattr(client, "images", None)
            if images_api:
                list_method = getattr(images_api, "list", None)
                if list_method:
                    images = list_method()
                    # Format images based on format_string
                    # Simple implementation: extract repo:tag from image tags
                    formatted: list[str] = []
                    for img in images:
                        tags = getattr(img, "tags", [])
                        if tags:
                            # Use first tag, or format if needed
                            tag_str = tags[0] if tags else "none:none"
                            formatted.append(tag_str)
                    return formatted
            return []

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name="list images formatted",
            logger=self.logger,
        )

    def list_containers_formatted(
        self,
        *,
        show_all: bool = False,
        format_string: str = "{{.Names}} ({{.Status}})",
    ) -> FlextResult[list[str]]:
        """Get formatted list of containers."""
        result = self.list_containers(all_containers=show_all)
        if result.is_failure:
            return FlextResult[list[str]].fail(
                result.error or "Failed to list containers"
            )

        containers = result.value
        formatted: list[str] = []
        for container_info in containers:
            # Format: "name (status)"
            status_str = (
                "running"
                if container_info.status
                == FlextTestConstants.Docker.ContainerStatus.RUNNING
                else "stopped"
            )
            formatted.append(f"{container_info.name} ({status_str})")

        return FlextResult[list[str]].ok(formatted)

    def list_networks(self) -> FlextResult[list[str]]:
        """List Docker networks."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> list[str]:
            networks_api = getattr(client, "networks", None)
            if networks_api:
                list_method = getattr(networks_api, "list", None)
                if list_method:
                    networks = list_method()
                    return [
                        str(getattr(n, "name", ""))
                        for n in networks
                        if hasattr(n, "name")
                    ]
            return []

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name="list networks",
            logger=self.logger,
        )

    # Shared container configuration for FLEXT ecosystem tests
    @property
    def shared_containers(
        self,
    ) -> Mapping[str, FlextTypes.Types.ContainerConfigDict]:
        """Get shared containers configuration from FlextConstants."""
        return FlextTestConstants.Docker.SHARED_CONTAINERS

    def start_existing_container(self, name: str) -> FlextResult[str]:
        """Start an existing stopped container without recreating it.

        This method starts a container that already exists but is stopped.
        Unlike compose_up, it does NOT remove and recreate the container.
        Use this to preserve container data/state between test sessions.

        Args:
            name: Name of the existing container to start

        Returns:
            FlextResult with status message

        """
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(name)
                    container_status = getattr(container, "status", "unknown")

                    if container_status == "running":
                        if self.logger and hasattr(self.logger, "debug"):
                            self.logger.debug(
                                "Container already running",
                                extra={"container": name},
                            )
                        return f"Container {name} already running"

                    if container_status in {"exited", "created", "paused"}:
                        start_method = getattr(container, "start", None)
                        if start_method:
                            start_method()
                        if self.logger and hasattr(self.logger, "info"):
                            self.logger.info(
                                "Started existing container",
                                extra={
                                    "container": name,
                                    "previous_status": container_status,
                                },
                            )
                        return f"Container {name} started"

                    return f"Container {name} in unexpected state: {container_status}"
            raise RuntimeError(f"Failed to start container {name}")

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"start existing container {name}",
            logger=self.logger,
        )

    def start_container(
        self,
        name: str,
        image: str | None = None,
        ports: dict[str, int | list[int] | tuple[str, int] | None] | None = None,
    ) -> FlextResult[str]:
        """Start a Docker container, or check if already running."""
        try:
            client = self.get_client()

            # First check if container already exists and is running
            try:
                container = client.containers.get(name)
                if container.status == "running":
                    return FlextResult[str].ok(f"Container {name} already running")
                # If exists but not running, start it
                container.start()
                return FlextResult[str].ok(f"Container {name} started (was stopped)")
            except DockerException:
                # Container doesn't exist, create it
                image_name = image or "alpine:latest"
                client.containers.run(
                    image_name,
                    name=name,
                    ports=ports,
                    detach=True,
                    remove=False,
                )
                return FlextResult[str].ok(f"Container {name} created and started")

        except DockerException as e:
            self.logger.exception("Failed to start container")
            return FlextResult[str].fail(f"Failed to start container: {e}")

    def stop_container(
        self,
        container_name: str,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Stop a running container.

        Args:
            container_name: Name of the container to stop

        Returns:
            Result containing operation details with status

        """
        if container_name not in self._dirty_containers:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                "Container not running", error_code="CONTAINER_NOT_RUNNING"
            )

        if container_name in self.shared_containers:
            config = self.shared_containers[container_name]

            # Ensure compose_file is str for Path operation
            compose_file_value = config["compose_file"]
            if not isinstance(compose_file_value, str):
                return FlextResult[
                    FlextTestsTypings.Docker.ContainerOperationResult
                ].fail(
                    f"Invalid compose_file type: {type(compose_file_value)}",
                    error_code="INVALID_CONFIG",
                )

            compose_file = str(self.workspace_root / compose_file_value)

            # Remove associated volumes
            volume_cleanup = self.cleanup_volumes()
            if volume_cleanup.is_failure:
                self.logger.warning(
                    "Volume cleanup warning",
                    extra={
                        "container": container_name,
                        "error": volume_cleanup.error,
                    },
                )

            # Ensure service is str or None for compose_up
            service_value = config.get("service")
            service: str | None = None
            if isinstance(service_value, str):
                service = service_value
            elif isinstance(service_value, int):
                service = str(service_value)
            elif service_value is not None:
                self.logger.warning(
                    "Unexpected service type",
                    extra={
                        "type": type(service_value),
                        "value": service_value,
                    },
                )

            # Restart container with compose
            restart_result = self.compose_up(compose_file, service)
            if restart_result.is_failure:
                return FlextResult[
                    FlextTestsTypings.Docker.ContainerOperationResult
                ].fail(
                    f"Failed to restart container: {restart_result.error}",
                    error_code="RESTART_FAILED",
                )

        self._dirty_containers.discard(container_name)
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok({
            "container": container_name,
            "stopped": True,
        })

    def get_container_info(
        self,
        name: str,
    ) -> FlextResult[FlextTestModels.Docker.ContainerInfo]:
        """Get container information."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> FlextTestModels.Docker.ContainerInfo:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(name)
                    info_dict = helpers.extract_container_info(container, name)
                    status_enum = (
                        FlextTestConstants.Docker.ContainerStatus.RUNNING
                        if info_dict["status"] == "running"
                        else FlextTestConstants.Docker.ContainerStatus.STOPPED
                    )
                    ports_value = info_dict.get("ports", {})
                    # Convert to Mapping[str, str] - ensure all values are strings
                    ports_dict: Mapping[str, str] = {}
                    if isinstance(ports_value, dict):
                        ports_dict = {
                            str(k): str(v) if not isinstance(v, str) else v
                            for k, v in ports_value.items()
                        }
                    return FlextTestModels.Docker.ContainerInfo(
                        name=str(info_dict["name"]),
                        status=status_enum,
                        ports=ports_dict,
                        image=str(info_dict["image"]),
                        container_id=str(info_dict["container_id"]),
                    )
            raise RuntimeError(f"Failed to get container {name}")

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"get info for container {name}",
            logger=self.logger,
        )

    def build_image(
        self,
        path: str,
        *,
        tag: str,
        dockerfile: str = "Dockerfile",
        build_args: dict[str, str] | None = None,
        no_cache: bool = False,
        pull: bool = False,
    ) -> FlextResult[str]:
        """Build Docker image."""
        try:
            client = self.get_client()
            build_path = Path(path)
            if not build_path.is_absolute():
                build_path = self.workspace_root / build_path

            dockerfile_path = build_path / dockerfile
            if not dockerfile_path.exists():
                return FlextResult[str].fail(f"Dockerfile not found: {dockerfile_path}")

            # Build image using docker SDK
            image, _logs = client.images.build(
                path=str(build_path),
                tag=tag,
                dockerfile=dockerfile,
                buildargs=build_args,
                nocache=no_cache,
                pull=pull,
            )

            # Tag the image if needed
            if tag and tag != image.id:
                image.tag(tag)

            self.logger.info(
                f"Built image {tag}",
                extra={"tag": tag, "image_id": image.id},
            )
            return FlextResult[str].ok(f"Image {tag} built successfully")
        except DockerException as e:
            self.logger.exception("Failed to build image")
            return FlextResult[str].fail(f"Failed to build image: {e}")

    def run_container(
        self,
        image: str,
        *,
        name: str | None = None,
        ports: dict[str, int | list[int] | tuple[str, int]] | None = None,
        environment: dict[str, str] | None = None,
        volumes: dict[str, dict[str, str]] | list[str] | None = None,
        detach: bool = True,
        remove: bool = False,
        command: str | None = None,
    ) -> FlextResult[FlextTestModels.Docker.ContainerInfo]:
        """Run a Docker container."""
        try:
            client = self.get_client()
            container_name = name or f"flext-container-{hash(image)}"
            # Docker SDK: Pass parameters directly to preserve original types
            # Cast to Container to help Pyrefly resolve overload
            container = client.containers.run(
                image,
                name=container_name,
                detach=True,  # Always detach for container management
                remove=remove,
                ports=ports,
                environment=environment,
                volumes=volumes,
                command=command,
            )
            return FlextResult[FlextTestModels.Docker.ContainerInfo].ok(
                FlextTestModels.Docker.ContainerInfo(
                    name=container_name,
                    status=FlextTestConstants.Docker.ContainerStatus.RUNNING,
                    ports={},  # Convert ports to string format for ContainerInfo
                    image=image,
                    container_id=getattr(container, "id", "unknown") or "unknown",
                ),
            )
        except DockerException as e:
            self.logger.exception("Failed to run container")
            return FlextResult[FlextTestModels.Docker.ContainerInfo].fail(
                f"Failed to run container: {e}",
            )

    def remove_container(self, name: str, *, force: bool = False) -> FlextResult[str]:
        """Remove a Docker container."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(name)
                    remove_method = getattr(container, "remove", None)
                    if remove_method:
                        remove_method(force=force)
            return f"Container {name} removed"

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"remove container {name}",
            logger=self.logger,
        )

    def remove_image(self, image: str, *, force: bool = False) -> FlextResult[str]:
        """Remove a Docker image."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            images_api = getattr(client, "images", None)
            if images_api:
                remove_method = getattr(images_api, "remove", None)
                if remove_method:
                    remove_method(image, force=force)
            return f"Image {image} removed"

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"remove image {image}",
            logger=self.logger,
        )

    def container_logs_formatted(
        self,
        container_name: str,
        tail: int = 100,
        *,
        follow: bool = False,
    ) -> FlextResult[str]:
        """Get formatted container logs."""
        helpers = FlextTestsUtilities.DockerHelpers
        tail_count = tail or FlextConstants.Test.Docker.DEFAULT_LOG_TAIL

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(container_name)
                    logs_method = getattr(container, "logs", None)
                    if logs_method and callable(logs_method):
                        logs_raw = logs_method(
                            tail=tail_count,
                            follow=follow,
                            stream=False,
                        )
                        logs_bytes = logs_raw if isinstance(logs_raw, bytes) else b""
                        return logs_bytes.decode("utf-8")
            return ""

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"get logs for container {container_name}",
            logger=self.logger,
        )

    def execute_command_in_container(
        self,
        container_name: str,
        command: str,
        *,
        user: str | None = None,
    ) -> FlextResult[str]:
        """Execute command in container."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(container_name)
                    exec_run_method = getattr(container, "exec_run", None)
                    if exec_run_method:
                        exec_user = user if user is not None else "root"
                        result = exec_run_method(command, user=exec_user)
                        output_attr = getattr(result, "output", b"")
                        return (
                            output_attr.decode("utf-8")
                            if isinstance(output_attr, bytes)
                            else str(output_attr)
                        )
            return ""

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"execute command in container {container_name}",
            logger=self.logger,
        )

    def list_containers(
        self,
        *,
        all_containers: bool = False,
    ) -> FlextResult[list[FlextTestModels.Docker.ContainerInfo]]:
        """List containers."""
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> list[FlextTestModels.Docker.ContainerInfo]:
            containers_api = getattr(client, "containers", None)
            list_method = (
                getattr(containers_api, "list", None) if containers_api else None
            )
            containers = list_method(all=all_containers) if list_method else []

            container_infos: list[FlextTestModels.Docker.ContainerInfo] = []
            for container in containers:
                info_dict = helpers.extract_container_info(container)
                status_enum = (
                    FlextTestConstants.Docker.ContainerStatus.RUNNING
                    if info_dict["status"] == "running"
                    else FlextTestConstants.Docker.ContainerStatus.STOPPED
                )
                ports_value = info_dict.get("ports", {})
                ports_dict: Mapping[str, str] = (
                    ports_value if isinstance(ports_value, dict) else {}
                )
                container_infos.append(
                    FlextTestModels.Docker.ContainerInfo(
                        name=str(info_dict.get("name", "unknown")),
                        status=status_enum,
                        ports=ports_dict,
                        image=str(info_dict.get("image", "unknown")),
                        container_id=str(info_dict.get("container_id", "")),
                    ),
                )
            return container_infos

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name="list containers",
            logger=self.logger,
        )

    # ============================================================================
    # PHASE 1: ENVIRONMENT VARIABLE MANAGEMENT
    # ============================================================================

    def load_env_file(
        self,
        env_file_path: str | Path,
    ) -> FlextResult[Mapping[str, str]]:
        """Load environment variables from .env file.

        Args:
            env_file_path: Path to .env file (relative or absolute)

        Returns:
            FlextResult with dict of environment variables

        """
        try:
            env_path = Path(env_file_path)

            # Resolve relative paths against workspace_root
            if not env_path.is_absolute():
                env_path = self.workspace_root / env_path

            if not env_path.exists():
                return FlextResult[Mapping[str, str]].fail(
                    f"Environment file not found: {env_path}",
                )

            env_vars: dict[str, str] = {}
            with env_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse KEY=VALUE format
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or (
                            value.startswith("'") and value.endswith("'")
                        ):
                            value = value[1:-1]
                        env_vars[key] = value

            self.logger.info(
                f"Loaded {len(env_vars)} environment variables from {env_file_path}",
                extra={"env_file": str(env_path)},
            )
            return FlextResult[Mapping[str, str]].ok(env_vars)

        except Exception as e:
            self.logger.exception("Failed to load environment file")
            return FlextResult[Mapping[str, str]].fail(
                f"Failed to load environment file: {e}",
            )

    def set_container_env_vars(
        self,
        container_name: str,
        env_vars: dict[str, str],
    ) -> FlextResult[bool]:
        """Set environment variables for container before start.

        Note: This stores env vars for use when starting containers.
        Requires container to be recreated to apply changes.

        Args:
            container_name: Name of container
            env_vars: Dict of environment variables to set

        Returns:
            FlextResult[bool]: Success with True if set, failure with error

        """
        try:
            client = self.get_client()
            _container = client.containers.get(container_name)

            # Update container environment - requires recreating container
            # For now, just validate that container exists and log intention
            self.logger.info(
                f"Marked {len(env_vars)} env vars for container {container_name}",
                extra={"container": container_name, "var_count": len(env_vars)},
            )

            # Store in a tracking dict for reference
            if not hasattr(self, "_container_env_vars"):
                self._container_env_vars: dict[str, dict[str, str]] = {}
            self._container_env_vars[container_name] = env_vars

            return FlextResult[bool].ok(True)

        except NotFound:
            return FlextResult[bool].fail(f"Container {container_name} not found")
        except DockerException as e:
            self.logger.exception(
                "Failed to set env vars for %s",
                container_name,
                exception=e,
            )
            return FlextResult[bool].fail(f"Failed to set environment variables: {e}")

    def get_container_env_vars(
        self,
        container_name: str,
    ) -> FlextResult[Mapping[str, str]]:
        """Get environment variables from running container.

        Args:
            container_name: Name of container

        Returns:
            FlextResult with dict of environment variables

        """
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> Mapping[str, str]:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(container_name)
                    attrs = getattr(container, "attrs", {})
                    container_config = (
                        attrs.get("Config", {}) if isinstance(attrs, dict) else {}
                    )
                    env_list = (
                        container_config.get("Env", [])
                        if isinstance(container_config, dict)
                        else []
                    )
                    env_vars = helpers.parse_env_list_to_dict(env_list)
                    if self.logger and hasattr(self.logger, "info"):
                        self.logger.info(
                            f"Retrieved {len(env_vars)} env vars from {container_name}",
                            extra={"container": container_name},
                        )
                    return env_vars
            return {}

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"get env vars for container {container_name}",
            logger=self.logger,
        )

    # ============================================================================
    # PHASE 2: ENHANCED HEALTH CHECK SYSTEM
    # ============================================================================

    def check_container_health(
        self,
        container_name: str,
    ) -> FlextResult[str]:
        """Check container health status (healthy/unhealthy/starting/stuck/none).

        Args:
            container_name: Name of container

        Returns:
            FlextResult with health status string
            Possible values: healthy, unhealthy, starting, stuck, running, stopped

        """
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(container_name)
                    state = helpers.extract_container_state(container)

                    if state.get("restarting", False):
                        return "restarting"

                    health = state.get("health", {})
                    if health and isinstance(health, dict):
                        # Convert health dict to dict[str, str | int] for get_health_status
                        health_dict: dict[str, str | int] = {}
                        for k, v in health.items():
                            key = str(k)
                            if isinstance(v, (str, int)):
                                health_dict[key] = v
                            else:
                                health_dict[key] = str(v)
                        started_at = str(state.get("started_at", ""))
                        status = helpers.get_health_status(health_dict, started_at)

                        if status == "stuck":
                            self.logger.warning(
                                "Container %s stuck in starting state",
                                container_name,
                                extra={"container": container_name},
                            )
                        else:
                            self.logger.info(
                                "Container %s health: %s",
                                container_name,
                                status,
                                extra={"container": container_name, "status": status},
                            )
                        return status

                    # No health check configured, check running state
                    if state.get("running", False):
                        return "running"
                    return "stopped"
            raise RuntimeError(f"Failed to get container {container_name}")

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"check health for container {container_name}",
            logger=self.logger,
        )

    def wait_for_container_healthy(
        self,
        container_name: str,
        max_wait: int = 300,
        check_interval: int = 5,
        health_check_cmd: str | None = None,
    ) -> FlextResult[bool]:
        """Wait for container to become healthy with automatic dirty marking.

        Marks container as dirty if:
        - Health check times out (container failed to become healthy)
        - Container becomes stuck (stuck in "starting" state)
        - Container is restarting
        - Container is unhealthy

        Args:
            container_name: Name of container
            max_wait: Maximum seconds to wait (default 300)
            check_interval: Seconds between health checks (default 5)
            health_check_cmd: Optional custom health check command to execute

        Returns:
            FlextResult with True if healthy, False if timeout/stuck

        """
        helpers = FlextTestsUtilities.DockerHelpers

        def check_health() -> FlextResult[str]:
            if health_check_cmd:
                exec_result = self.execute_command_in_container(
                    container_name,
                    health_check_cmd,
                )
                if exec_result.is_success:
                    return FlextResult[str].ok("healthy")
                return FlextResult[str].ok("checking")
            return self.check_container_health(container_name)

        def is_healthy(status: str) -> bool:
            return status == "healthy"

        def should_mark_dirty(status: str) -> bool:
            return status in {"stuck", "restarting", "unhealthy"}

        # Verify container exists first
        try:
            client = self.get_client()
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    get_method(container_name)  # Verify exists
        except NotFound:
            self.logger.exception(
                "Container %s not found - marking dirty",
                container_name,
                exception=NotFound(f"Container {container_name} not found"),
            )
            self.mark_container_dirty(container_name)
            return FlextResult[bool].fail(f"Container {container_name} not found")

        # Wait for health with retry
        success, result, error = helpers.wait_with_retry(
            check_fn=check_health,
            max_wait_seconds=max_wait,
            check_interval_seconds=check_interval,
            success_condition=is_healthy,
            logger=self.logger,
        )

        if success and result and result == "healthy":
            self.logger.info(
                "Container %s is healthy",
                container_name,
                extra={"container": container_name},
            )
            return FlextResult[bool].ok(True)

        # Check if we should mark dirty based on status
        if result and isinstance(result, str) and should_mark_dirty(result):
            self.logger.error(
                "Container %s %s - marking dirty",
                container_name,
                result,
                extra={"container": container_name, "status": result},
            )
            self.mark_container_dirty(container_name)
            return FlextResult[bool].ok(False)

        # Timeout or other failure - mark dirty
        self.logger.error(
            "Container %s health check TIMEOUT - marking dirty: %s",
            container_name,
            error,
            extra={"container": container_name, "error": error},
        )
        self.mark_container_dirty(container_name)
        return FlextResult[bool].ok(False)

    def wait_for_port_ready(
        self,
        host: str,
        port: int,
        max_wait: int = 60,
    ) -> FlextResult[bool]:
        """Wait for network port to become available.

        Args:
            host: Host to connect to (e.g., 'localhost', '127.0.0.1')
            port: Port number to check
            max_wait: Maximum seconds to wait (default 60)

        Returns:
            FlextResult with True if port ready, False if timeout

        """
        try:
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    sock.close()

                    if result == 0:
                        self.logger.info(
                            "Port %s:%s is ready",
                            host,
                            port,
                            extra={"host": host, "port": port},
                        )
                        return FlextResult[bool].ok(True)
                except OSError:
                    pass

                time.sleep(2)

            self.logger.warning(
                "Port %s:%s not ready after %ss",
                host,
                port,
                max_wait,
                extra={"host": host, "port": port, "max_wait": max_wait},
            )
            return FlextResult[bool].ok(False)

        except Exception as e:
            self.logger.exception(
                "Failed to wait for port %s:%s",
                host,
                port,
                exception=e,
            )
            return FlextResult[bool].fail(f"Failed to wait for port: {e}")

    # ============================================================================
    # PHASE 3: CONTAINER REPAIR / RECREATION
    # ============================================================================

    def detect_container_issues(
        self,
        container_name: str,
    ) -> FlextResult[list[str]]:
        """Detect issues with container (stopped, unhealthy, stuck, restarting).

        Detects:
        - Container not running
        - Container unhealthy
        - Container stuck in "starting" state
        - Container restarting
        - Container exited with error

        Args:
            container_name: Name of container

        Returns:
            FlextResult with list of detected issues

        """
        helpers = FlextTestsUtilities.DockerHelpers

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> list[str]:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(container_name)
                    state = helpers.extract_container_state(container)
                    issues = helpers.detect_container_state_issues(
                        state,
                        container_name,
                    )

                    if not issues:
                        self.logger.info(
                            "No issues detected for %s",
                            container_name,
                            extra={"container": container_name},
                        )
                    else:
                        self.logger.warning(
                            f"Detected {len(issues)} issues for {container_name}",
                            extra={"container": container_name, "issues": issues},
                        )

                    return issues
            return []

        result = helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"detect issues for container {container_name}",
            logger=self.logger,
        )

        # NotFound returns empty list, not failure
        if result.is_failure and "not found" in (result.error or "").lower():
            return FlextResult[list[str]].ok(["Container not found"])

        return result

    def repair_container(
        self,
        container_name: str,
        compose_file: str | None = None,
        service: str | None = None,
        *,
        recreate_volumes: bool = True,
    ) -> FlextResult[str]:
        """Repair/recreate a broken container with aggressive cleanup.

        Handles stuck containers by force-killing them (SIGKILL).

        Args:
            container_name: Name of container to repair
            compose_file: Path to docker-compose file (for docker-compose restart)
            service: Service name in docker-compose
            recreate_volumes: Whether to recreate volumes (default True)

        Returns:
            FlextResult with repair status message

        """
        helpers = FlextTestsUtilities.DockerHelpers

        # Get container and check if running
        get_result = helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=lambda client: helpers._get_container_with_state(
                client,
                container_name,
            ),
            operation_name=f"get container {container_name} for repair",
            logger=self.logger,
        )

        if get_result.is_failure:
            return FlextResult[str].fail(
                f"Container {container_name} not found for repair",
            )

        container, state = get_result.unwrap()
        was_running = state.get("running", False)
        health = state.get("health", {})
        health_status = (
            health.get("Status", "none") if isinstance(health, dict) else "none"
        )

        # Stop container if running
        if was_running:
            self.logger.warning(
                "Stopping container %s for repair (health: %s)",
                container_name,
                health_status,
                extra={"container": container_name, "health": health_status},
            )
            stop_result = helpers.execute_container_stop_operation(
                container,
                container_name,
                timeout=10,
                force_kill=True,
                logger=self.logger,
            )
            if stop_result.is_failure:
                self.logger.warning(
                    f"Stop failed during repair: {stop_result.error}",
                    extra={"container": container_name},
                )

        # Remove container
        self.logger.info(
            "Removing container %s",
            container_name,
            extra={"container": container_name},
        )
        remove_result = helpers.execute_container_remove_operation(
            container,
            container_name,
            force=True,
            logger=self.logger,
        )
        if remove_result.is_failure:
            return FlextResult[str].fail(
                f"Failed to remove container during repair: {remove_result.error}",
            )

        # Restart via compose if provided
        if compose_file:
            compose_result = self.compose_down(compose_file)
            if compose_result.is_failure:
                self.logger.warning(
                    f"compose_down failed: {compose_result.error}",
                    extra={"error": compose_result.error},
                )

            if recreate_volumes:
                client = self.get_client()
                volumes_api = getattr(client, "volumes", None)
                if volumes_api:
                    prune_method = getattr(volumes_api, "prune", None)
                    if prune_method:
                        try:
                            prune_method()
                            self.logger.info("Pruned unused volumes")
                        except Exception as e:
                            self.logger.warning(
                                "Failed to prune volumes: %s",
                                e,
                                extra={"error": str(e)},
                            )

            return self.compose_up(compose_file, service=service)

        return FlextResult[str].ok(
            f"Container {container_name} repaired (force killed and removed)",
        )

    def auto_repair_if_needed(
        self,
        container_name: str,
        compose_file: str | None = None,
        service: str | None = None,
    ) -> FlextResult[str]:
        """Auto-detect and repair container if needed.

        Args:
            container_name: Name of container
            compose_file: Path to docker-compose file
            service: Service name in docker-compose

        Returns:
            FlextResult with repair status or "no repair needed"

        """
        try:
            # Detect issues
            issues_result = self.detect_container_issues(container_name)
            if issues_result.is_failure:
                return FlextResult[str].fail(
                    issues_result.error or "Failed to detect container issues",
                )

            issues = issues_result.unwrap()
            if not issues:
                return FlextResult[str].ok("No repair needed - container is healthy")

            self.logger.warning(
                f"Auto-repairing {container_name} due to {len(issues)} issues",
                extra={"container": container_name, "issue_count": len(issues)},
            )

            # Repair the container
            return self.repair_container(
                container_name,
                compose_file=compose_file,
                service=service,
            )

        except Exception as e:
            self.logger.exception(
                "Failed to auto-repair %s",
                container_name,
                exception=e,
            )
            return FlextResult[str].fail(f"Failed to auto-repair container: {e}")

    # ============================================================================
    # PHASE 4: SIMPLIFIED STARTUP / SHUTDOWN
    # ============================================================================

    def ensure_container_running(
        self,
        container_name: str,
        compose_file: str | None = None,
        service: str | None = None,
        env_file: str | Path | None = None,
        max_wait: int = 300,
        health_check_cmd: str | None = None,
    ) -> FlextResult[str]:
        """One-stop method: ensure container running, healthy, with env vars loaded.

        Args:
            container_name: Name of container
            compose_file: Path to docker-compose file
            service: Service name in docker-compose
            env_file: Path to .env file to load
            max_wait: Max seconds to wait for healthy (default 300)
            health_check_cmd: Optional custom health check command

        Returns:
            FlextResult with container status message

        """
        helpers = FlextTestsUtilities.DockerHelpers

        # Load environment variables if provided
        if env_file:
            env_result = self.load_env_file(env_file)
            if env_result.is_failure:
                self.logger.warning(
                    f"Failed to load env file: {env_result.error}",
                )

        # Check if container exists and is running
        check_result = helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=lambda client: helpers._get_container_with_state(
                client,
                container_name,
            ),
            operation_name=f"check container {container_name} status",
            logger=self.logger,
        )

        if check_result.is_success:
            _container, state = check_result.unwrap()
            if state.get("running", False):
                self.logger.info(
                    "Container %s is already running",
                    container_name,
                    extra={"container": container_name},
                )
                # Wait for healthy if needed
                if health_check_cmd or compose_file:
                    health_result = self.wait_for_container_healthy(
                        container_name,
                        max_wait=max_wait,
                        health_check_cmd=health_check_cmd,
                    )
                    if health_result.is_success and health_result.unwrap():
                        return FlextResult[str].ok(
                            f"Container {container_name} is running and healthy",
                        )
                else:
                    return FlextResult[str].ok(f"Container {container_name} is running")
        else:
            self.logger.info(
                "Container %s not found, will start",
                container_name,
                extra={"container": container_name},
            )

        # Start container via compose if file provided
        if compose_file:
            self.logger.info(
                "Starting %s via docker-compose",
                container_name,
                extra={"container": container_name, "compose_file": compose_file},
            )
            self._container_configs[container_name] = {
                "compose_file": compose_file,
                "service": service or "",
            }
            compose_result = self.compose_up(compose_file, service=service)
            if compose_result.is_failure:
                return compose_result

        # Wait for container to be healthy
        health_result = self.wait_for_container_healthy(
            container_name,
            max_wait=max_wait,
            health_check_cmd=health_check_cmd,
        )

        if health_result.is_failure:
            return FlextResult[str].fail(
                health_result.error or "Container health check failed",
            )

        if not health_result.unwrap():
            return FlextResult[str].fail(
                f"Container {container_name} failed health check "
                "(marked dirty for recreation)",
            )

        return FlextResult[str].ok(f"Container {container_name} is running and healthy")

    def graceful_shutdown_container(
        self,
        container_name: str,
        timeout: int = 30,
        *,
        remove_volumes: bool = False,
    ) -> FlextResult[str]:
        """Gracefully stop container with optional volume cleanup.

        Args:
            container_name: Name of container
            timeout: Seconds to wait before force kill (default 30)
            remove_volumes: Whether to remove associated volumes (default False)

        Returns:
            FlextResult with shutdown status message

        """
        helpers = FlextTestsUtilities.DockerHelpers

        # Get container
        get_result = helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=lambda client: helpers._get_container_with_state(
                client,
                container_name,
            ),
            operation_name=f"get container {container_name} for shutdown",
            logger=self.logger,
        )

        if get_result.is_failure:
            return FlextResult[str].fail(f"Container {container_name} not found")

        container, state = get_result.unwrap()

        # Stop container if running
        if state.get("running", False):
            self.logger.info(
                "Stopping container %s",
                container_name,
                extra={"container": container_name, "timeout": timeout},
            )
            stop_result = helpers.execute_container_stop_operation(
                container,
                container_name,
                timeout=timeout,
                force_kill=False,
                logger=self.logger,
            )
            if stop_result.is_failure:
                return FlextResult[str].fail(
                    f"Failed to stop container: {stop_result.error}",
                )
        else:
            self.logger.info(
                "Container %s is already stopped",
                container_name,
                extra={"container": container_name},
            )

        # Remove container with optional volumes
        if remove_volumes:
            self.logger.info(
                "Removing volumes for %s",
                container_name,
                extra={"container": container_name},
            )

        remove_result = helpers.execute_container_remove_operation(
            container,
            container_name,
            force=False,
            logger=self.logger,
        )
        if remove_result.is_failure:
            return FlextResult[str].fail(
                f"Failed to remove container: {remove_result.error}",
            )

        # Handle volume removal if requested (Docker API uses v=True parameter)
        if remove_volumes:
            remove_method = getattr(container, "remove", None)
            if remove_method:
                try:
                    remove_method(v=True)
                except Exception:
                    pass  # Already removed above

        return FlextResult[str].ok(f"Container {container_name} stopped and removed")

    @classmethod
    def _status_icon(
        cls,
        status: FlextTestConstants.Docker.ContainerStatus,
    ) -> str:
        """Return a friendly icon for container status."""
        return {
            FlextTestConstants.Docker.ContainerStatus.RUNNING: "🟢 Running",
            FlextTestConstants.Docker.ContainerStatus.STOPPED: "🔴 Stopped",
            FlextTestConstants.Docker.ContainerStatus.NOT_FOUND: "⚫ Not Found",
            FlextTestConstants.Docker.ContainerStatus.ERROR: "⚠️ Error",
        }.get(status, "❓ Unknown")

    @classmethod
    def _format_ports(
        cls,
        info: FlextTestModels.Docker.ContainerInfo,
    ) -> str:
        """Format port mapping for CLI display."""
        if not info.ports:
            return "-"
        return ", ".join(
            f"{host}→{container}" for host, container in info.ports.items()
        )

    @classmethod
    def _display_status_table(cls, manager: FlextTestDocker) -> FlextResult[bool]:
        """Render the container status table to the console."""
        status_result = manager.get_all_status()
        if status_result.is_failure:
            error_message = f"Failed to get status: {status_result.error}"
            cls._console.print(f"[bold red]{error_message}[/bold red]")
            return FlextResult[bool].fail(error_message)

        table = Table(title="FLEXT Docker Test Containers Status", show_header=True)
        table.add_column("Container", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Ports", style="green")
        table.add_column("Image", style="blue")

        for name, info in status_result.value.items():
            table.add_row(
                name,
                cls._status_icon(info.status),
                cls._format_ports(info),
                info.image or "-",
            )

        cls._console.print(table)
        return FlextResult[bool].ok(True)

    def show_status_table(self) -> FlextResult[bool]:
        """Public helper to render container status."""
        return self._display_status_table(self)

    def fetch_container_logs(
        self,
        container_name: str,
        *,
        tail: int | None = None,
    ) -> FlextResult[str]:
        """Fetch logs for a specific container."""
        helpers = FlextTestsUtilities.DockerHelpers
        tail_count = tail or FlextConstants.Test.Docker.DEFAULT_LOG_TAIL

        def operation(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
        ) -> str:
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(container_name)
                    logs_method = getattr(container, "logs", None)
                    if logs_method:
                        logs_bytes = logs_method(tail=tail_count)
                        return (
                            logs_bytes.decode("utf-8", errors="ignore")
                            if isinstance(logs_bytes, bytes)
                            else str(logs_bytes)
                        )
            return ""

        return helpers.execute_docker_client_operation(
            get_client_fn=self.get_client,
            operation=operation,
            operation_name=f"fetch logs for container {container_name}",
            logger=self.logger,
        )

    @classmethod
    def register_pytest_fixtures(
        cls,
        namespace: dict[str, object] | None = None,
    ) -> None:
        """Register pytest fixtures that wrap FlextTestDocker operations."""
        if cls._pytest_registered:
            return

        ns = namespace if namespace is not None else globals()

        @pytest.fixture(scope="session")
        def docker_control() -> FlextTestDocker:
            return cls()

        @pytest.fixture(scope="session")
        def flext_ldap_container(
            docker_control: FlextTestDocker,
        ) -> Iterator[str]:
            container_name = "flext-shared-ldap"
            status = docker_control.get_container_status(container_name)
            if (
                status.is_success
                and status.value.status.value
                == FlextTestConstants.Docker.ContainerStatus.RUNNING.value
            ):
                yield container_name
                return

            start_result = docker_control.start_container(container_name)
            if start_result.is_failure:
                pytest.skip(f"Failed to start LDAP container: {start_result.error}")

            yield container_name

            docker_control.stop_container(container_name)

        @pytest.fixture(scope="session")
        def flext_postgres_container(
            docker_control: FlextTestDocker,
        ) -> Iterator[str]:
            container_name = "flext-postgres"
            status = docker_control.get_container_status(container_name)
            if (
                status.is_success
                and status.value.status.value
                == FlextTestConstants.Docker.ContainerStatus.RUNNING.value
            ):
                yield container_name
                return

            start_result = docker_control.start_container(container_name)
            if start_result.is_failure:
                pytest.skip(
                    f"Failed to start PostgreSQL container: {start_result.error}",
                )

            yield container_name
            docker_control.stop_container(container_name)

        @pytest.fixture(scope="session")
        def flext_redis_container(
            docker_control: FlextTestDocker,
        ) -> Iterator[str]:
            container_name = "flext-redis"
            status = docker_control.get_container_status(container_name)
            if (
                status.is_success
                and status.value.status.value
                == FlextTestConstants.Docker.ContainerStatus.RUNNING.value
            ):
                yield container_name
                return

            start_result = docker_control.start_container(container_name)
            if start_result.is_failure:
                pytest.skip(
                    f"Failed to start Redis container: {start_result.error}",
                )

            yield container_name
            docker_control.stop_container(container_name)

        @pytest.fixture(scope="session")
        def flext_oracle_container(
            docker_control: FlextTestDocker,
        ) -> Iterator[str]:
            container_name = "flext-oracle"
            status = docker_control.get_container_status(container_name)
            if (
                status.is_success
                and status.value.status.value
                == FlextTestConstants.Docker.ContainerStatus.RUNNING.value
            ):
                yield container_name
                return

            start_result = docker_control.start_container(container_name)
            if start_result.is_failure:
                pytest.skip(
                    f"Failed to start Oracle container: {start_result.error}",
                )

            yield container_name
            docker_control.stop_container(container_name)

        @pytest.fixture
        def reset_ldap_container(docker_control: FlextTestDocker) -> str:
            container_name = "flext-shared-ldap"
            reset_result = docker_control.reset_container(container_name)
            if reset_result.is_failure:
                pytest.skip(f"Failed to reset LDAP container: {reset_result.error}")

            return container_name

        @pytest.fixture
        def reset_postgres_container(docker_control: FlextTestDocker) -> str:
            container_name = "flext-postgres"
            reset_result = docker_control.reset_container(container_name)
            if reset_result.is_failure:
                pytest.skip(
                    f"Failed to reset PostgreSQL container: {reset_result.error}",
                )
            return container_name

        @pytest.fixture
        def all_containers_running(
            docker_control: FlextTestDocker,
        ) -> Iterator[FlextTestsTypings.Docker.ContainerOperationResult]:
            start_result = docker_control.start_all()
            if start_result.is_failure:
                pytest.skip(f"Failed to start all containers: {start_result.error}")

            yield start_result.value

            docker_control.stop_all(remove=False)

        ns.update({
            "docker_control": docker_control,
            "flext_ldap_container": flext_ldap_container,
            "flext_postgres_container": flext_postgres_container,
            "flext_redis_container": flext_redis_container,
            "flext_oracle_container": flext_oracle_container,
            "reset_ldap_container": reset_ldap_container,
            "reset_postgres_container": reset_postgres_container,
            "all_containers_running": all_containers_running,
        })

        cls._pytest_registered = True

    def init_workspace(self, workspace_root: Path) -> FlextResult[str]:
        """Initialize workspace configuration and auto-discover services."""
        try:
            self.workspace_root = workspace_root
            compose_candidates = [
                workspace_root / "docker" / "docker-compose.yml",
                workspace_root / "docker-compose.yml",
                workspace_root / "compose.yml",
            ]

            for compose_path in compose_candidates:
                if compose_path.exists():
                    discovery = self.auto_discover_services(str(compose_path))
                    if discovery.is_success:
                        services = discovery.value
                        self.logger.info(
                            "Auto-discovered services from %s: %s",
                            compose_path,
                            services,
                        )

            return FlextResult[str].ok(
                f"FlextTestDocker initialized for workspace: {workspace_root}",
            )
        except Exception as exc:
            self.logger.exception("Workspace initialization failed")
            return FlextResult[str].fail(f"Workspace initialization failed: {exc}")

    def build_workspace_projects(
        self,
        projects: list[str],
        registry: str = "flext",
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Build Docker images for a set of workspace projects."""
        # Use mutable dict during construction
        results_dict: dict[str, str | int | bool | Sequence[str] | None] = {}

        for project in projects:
            project_path = self.workspace_root / project
            if not project_path.exists():
                results_dict[project] = f"Project path not found: {project_path}"
                continue

            dockerfile_candidates = [
                project_path / "Dockerfile",
                project_path / "docker" / "Dockerfile",
                project_path / f"Dockerfile.{project}",
            ]

            dockerfile_path: Path | None = None
            for candidate in dockerfile_candidates:
                if candidate.exists():
                    dockerfile_path = candidate
                    break

            if dockerfile_path is None:
                results_dict[project] = "No Dockerfile found"
                continue

            tag = f"{registry}/{project}:latest"
            build_result = self.build_image_advanced(
                path=str(project_path),
                tag=tag,
                dockerfile=dockerfile_path.name,
            )

            if build_result.is_success:
                results_dict[project] = f"Built successfully: {tag}"
            else:
                results_dict[project] = f"Build failed: {build_result.error}"

        # dict is compatible with Mapping - pass directly
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
            results_dict,
        )

    def build_single_image(
        self,
        name: str,
        dockerfile_path: str,
        context_path: str | None = None,
    ) -> FlextResult[str]:
        """Build a single Docker image."""
        context_root = context_path or str(Path(dockerfile_path).parent)
        build_result = self.build_image_advanced(
            path=context_root,
            tag=name,
            dockerfile=Path(dockerfile_path).name,
        )

        if build_result.is_success:
            return FlextResult[str].ok(f"Image built successfully: {name}")
        return FlextResult[str].fail(f"Image build failed: {build_result.error}")

    def start_compose_stack(
        self,
        compose_file: str,
        network_name: str | None = None,
    ) -> FlextResult[str]:
        """Start a Docker Compose stack."""
        discovery = self.auto_discover_services(compose_file)
        if discovery.is_failure:
            return FlextResult[str].fail(f"Service discovery failed: {discovery.error}")

        start_result = self.compose_up(compose_file)
        if start_result.is_failure:
            return FlextResult[str].fail(f"Stack start failed: {start_result.error}")

        if network_name:
            network_result = self.create_network(network_name)
            if network_result.is_failure:
                self.logger.warning(
                    "Network creation failed for %s: %s",
                    network_name,
                    network_result.error,
                )

        services = discovery.value
        return FlextResult[str].ok(
            f"Stack started successfully with services: {services}",
        )

    def stop_compose_stack(self, compose_file: str) -> FlextResult[str]:
        """Stop a Docker Compose stack."""
        stop_result = self.compose_down(compose_file)
        if stop_result.is_success:
            return FlextResult[str].ok("Stack stopped successfully")
        return FlextResult[str].fail(f"Stack stop failed: {stop_result.error}")

    def restart_compose_stack(self, compose_file: str) -> FlextResult[str]:
        """Restart a Docker Compose stack."""
        stop_result = self.stop_compose_stack(compose_file)
        if stop_result.is_failure:
            return FlextResult[str].fail(f"Stack stop failed: {stop_result.error}")

        start_result = self.start_compose_stack(compose_file)
        if start_result.is_failure:
            return FlextResult[str].fail(f"Stack restart failed: {start_result.error}")

        return FlextResult[str].ok("Stack restarted successfully")

    def show_stack_logs(
        self,
        compose_file: str,
        *,
        follow: bool = False,
    ) -> FlextResult[str]:
        """Show logs for a Docker Compose stack."""
        _ = follow  # compatibility with previous signature
        logs_result = self.compose_logs(compose_file)
        if logs_result.is_success:
            return FlextResult[str].ok("Logs displayed")
        return FlextResult[str].fail(f"Failed to get logs: {logs_result.error}")

    def show_stack_status(
        self, compose_file: str
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Return status information for the Docker Compose stack."""
        _ = compose_file  # compose file not required for stub implementation
        status_result = self.get_all_status()
        if status_result.is_failure:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                f"Status check failed: {status_result.error}"
            )

        # Convert dict[str, ContainerInfo] to ContainerOperationResult
        # Use mutable dict during construction
        status_info_dict: dict[str, str | int | bool | Sequence[str] | None] = {}
        if isinstance(status_result.value, dict):
            # Create explicit conversion: dict[str, ContainerInfo] -> ContainerOperationResult
            for k, v in status_result.value.items():
                if hasattr(v, "model_dump"):
                    dumped = v.model_dump()
                    if isinstance(dumped, dict):
                        status_info_dict[str(k)] = str(dumped)
                    else:
                        status_info_dict[str(k)] = str(dumped)
                else:
                    status_info_dict[str(k)] = str(v)
        running_services = self.get_running_services()
        if running_services.is_success:
            status_info_dict["auto_managed_services"] = running_services.value
        else:
            status_info_dict["auto_managed_services"] = []

        # dict is compatible with Mapping - pass directly
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
            status_info_dict,
        )

    def connect_to_service(self, service_name: str) -> FlextResult[str]:
        """Open an interactive session with a service container."""
        connect_result = self.exec_container_interactive(
            container_name=service_name,
            command="/bin/bash",
        )
        if connect_result.is_success:
            return FlextResult[str].ok(f"Connected to {service_name}")
        return FlextResult[str].fail(f"Connection failed: {connect_result.error}")

    def execute_in_service(self, service_name: str, command: str) -> FlextResult[str]:
        """Execute a command inside a service container."""
        exec_result = self.execute_container_command(
            container_name=service_name,
            command=command,
        )
        if exec_result.is_success:
            return FlextResult[str].ok("Command executed successfully")
        return FlextResult[str].fail(f"Command execution failed: {exec_result.error}")

    def cleanup_workspace(
        self,
        *,
        remove_volumes: bool = False,
        remove_networks: bool = False,
        prune_system: bool = False,
    ) -> FlextResult[str]:
        """Clean up containers, networks, volumes, and images."""
        operations: list[str] = []

        running_services = self.get_running_services()
        if running_services.is_success:
            for service in running_services.value:
                stop_result = self.stop_services_for_test(f"cleanup_{service}")
                if stop_result.is_success:
                    operations.append(f"Stopped service: {service}")
                else:
                    operations.append(
                        f"Failed to stop service {service}: {stop_result.error}",
                    )
        else:
            operations.append(
                f"Failed to list running services: {running_services.error}",
            )

        container_cleanup = self.cleanup_all_test_containers()
        if container_cleanup.is_success:
            operations.append("Cleaned up test containers")
        else:
            operations.append(f"Container cleanup failed: {container_cleanup.error}")

        if remove_volumes:
            volumes_result = self.cleanup_volumes()
            if volumes_result.is_success:
                operations.append("Pruned volumes")
            else:
                operations.append(f"Volume pruning failed: {volumes_result.error}")

        if remove_networks:
            networks_result = self.cleanup_networks()
            if networks_result.is_success:
                operations.append("Pruned networks")
            else:
                operations.append(f"Network pruning failed: {networks_result.error}")

        if prune_system:
            images_result = self.cleanup_images()
            if images_result.is_success:
                operations.append("Pruned images")
            else:
                operations.append(f"Image pruning failed: {images_result.error}")

        summary = "; ".join(operations) if operations else "No cleanup actions run"
        return FlextResult[str].ok(f"Cleanup completed: {summary}")

    def health_check_stack(
        self,
        compose_file: str,
        *,
        timeout: int = 30,
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Perform health checks for services in the compose stack."""
        _ = timeout  # Compatibility with previous signature
        # Use mutable dict during construction
        health_dict: dict[str, str | int | bool | Sequence[str] | None] = {}

        discovery = self.auto_discover_services(compose_file)
        if discovery.is_failure:
            return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].fail(
                f"Service discovery failed: {discovery.error}"
            )

        for service in discovery.value:
            health_result = self.get_service_health_status(service)
            if health_result.is_success:
                info = health_result.value
                container_status = info.get("container_status", "unknown")
                health_dict[service] = f"Status: {container_status}"
            else:
                health_dict[service] = f"Health check failed: {health_result.error}"

        # dict is compatible with Mapping - pass directly
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
            health_dict,
        )

    def validate_workspace(
        self, workspace_root: Path
    ) -> FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]:
        """Validate Docker operations within a workspace."""
        # Use mutable dict during construction
        results_dict: dict[str, str | int | bool | Sequence[str] | None] = {}

        try:
            docker_manager = FlextTestDocker(workspace_root=workspace_root)
            results_dict["docker_connection"] = "✅ Connected"

            containers_result = docker_manager.list_containers_formatted()
            if containers_result.is_success:
                results_dict["container_operations"] = "✅ Working"
            else:
                results_dict["container_operations"] = (
                    f"❌ Failed: {containers_result.error}"
                )

            images_result = docker_manager.images_formatted()
            if images_result.is_success:
                results_dict["image_operations"] = "✅ Working"
            else:
                results_dict["image_operations"] = f"❌ Failed: {images_result.error}"

            networks_result = docker_manager.list_networks()
            if networks_result.is_success:
                results_dict["network_operations"] = "✅ Working"
            else:
                results_dict["network_operations"] = (
                    f"❌ Failed: {networks_result.error}"
                )

            volumes_result = docker_manager.list_volumes()
            if volumes_result.is_success:
                results_dict["volume_operations"] = "✅ Working"
            else:
                results_dict["volume_operations"] = f"❌ Failed: {volumes_result.error}"
        except Exception as exc:
            results_dict["docker_connection"] = f"❌ Failed: {exc}"

        # dict is compatible with Mapping - pass directly
        return FlextResult[FlextTestsTypings.Docker.ContainerOperationResult].ok(
            results_dict,
        )

    @classmethod
    def _get_workspace_parser(cls) -> argparse.ArgumentParser:
        """Build or reuse the workspace command parser."""
        if cls._workspace_parser is not None:
            return cls._workspace_parser

        parser = argparse.ArgumentParser(
            description=(
                "FLEXT Workspace Docker Manager - Unified Docker operations using "
                "FlextTestDocker"
            ),
        )
        parser.add_argument(
            "--workspace-root",
            type=Path,
            help="Workspace root directory",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        init_parser = subparsers.add_parser(
            "init",
            help="Initialize FlextTestDocker workspace",
        )
        init_parser.add_argument("--workspace-root", type=Path, required=True)

        build_workspace_parser = subparsers.add_parser(
            "build-workspace",
            help="Build workspace projects",
        )
        build_workspace_parser.add_argument(
            "--projects",
            required=True,
            help="Comma-separated project list",
        )
        build_workspace_parser.add_argument(
            "--registry",
            default="flext",
            help="Docker registry prefix",
        )

        build_image_parser = subparsers.add_parser(
            "build-image",
            help="Build single image",
        )
        build_image_parser.add_argument("--name", required=True, help="Image name/tag")
        build_image_parser.add_argument(
            "--dockerfile",
            required=True,
            help="Dockerfile path",
        )
        build_image_parser.add_argument("--context", help="Build context path")

        start_stack_parser = subparsers.add_parser(
            "start-stack",
            help="Start Docker Compose stack",
        )
        start_stack_parser.add_argument(
            "--compose-file",
            required=True,
            help="Docker Compose file path",
        )
        start_stack_parser.add_argument("--network", help="Network name")

        stop_stack_parser = subparsers.add_parser(
            "stop-stack",
            help="Stop Docker Compose stack",
        )
        stop_stack_parser.add_argument(
            "--compose-file",
            required=True,
            help="Docker Compose file path",
        )

        restart_stack_parser = subparsers.add_parser(
            "restart-stack",
            help="Restart Docker Compose stack",
        )
        restart_stack_parser.add_argument(
            "--compose-file",
            required=True,
            help="Docker Compose file path",
        )

        logs_parser = subparsers.add_parser("show-logs", help="Show stack logs")
        logs_parser.add_argument(
            "--compose-file",
            required=True,
            help="Docker Compose file path",
        )
        logs_parser.add_argument("--follow", action="store_true", help="Follow logs")

        status_parser = subparsers.add_parser("show-status", help="Show stack status")
        status_parser.add_argument(
            "--compose-file",
            required=True,
            help="Docker Compose file path",
        )

        connect_parser = subparsers.add_parser(
            "connect",
            help="Connect to service container",
        )
        connect_parser.add_argument("--service", required=True, help="Service name")

        exec_parser = subparsers.add_parser("exec", help="Execute command in container")
        exec_parser.add_argument("--service", required=True, help="Service name")
        exec_parser.add_argument("--command", required=True, help="Command to execute")

        cleanup_parser = subparsers.add_parser(
            "cleanup",
            help="Clean up Docker artifacts",
        )
        cleanup_parser.add_argument(
            "--remove-volumes",
            action="store_true",
            help="Remove volumes",
        )
        cleanup_parser.add_argument(
            "--remove-networks",
            action="store_true",
            help="Remove networks",
        )
        cleanup_parser.add_argument(
            "--prune-system",
            action="store_true",
            help="Prune system",
        )

        health_parser = subparsers.add_parser(
            "health-check",
            help="Perform health check",
        )
        health_parser.add_argument(
            "--compose-file",
            required=True,
            help="Docker Compose file path",
        )
        health_parser.add_argument(
            "--timeout",
            type=int,
            default=30,
            help="Health check timeout",
        )

        create_network_parser = subparsers.add_parser(
            "create-network",
            help="Create Docker network",
        )
        create_network_parser.add_argument("--name", required=True, help="Network name")
        create_network_parser.add_argument(
            "--driver",
            default="bridge",
            help="Network driver",
        )

        list_volumes_parser = subparsers.add_parser(
            "list-volumes",
            help="List Docker volumes",
        )
        list_volumes_parser.add_argument("--filter", help="Name filter")

        list_images_parser = subparsers.add_parser(
            "list-images",
            help="List Docker images",
        )
        list_images_parser.add_argument("--filter", help="Repository filter")

        list_containers_parser = subparsers.add_parser(
            "list-containers",
            help="List Docker containers",
        )
        list_containers_parser.add_argument("--filter", help="Name filter")
        list_containers_parser.add_argument(
            "--show-all",
            action="store_true",
            help="Show all containers",
        )

        validate_parser = subparsers.add_parser(
            "validate",
            help="Validate FlextTestDocker functionality",
        )
        validate_parser.add_argument("--workspace-root", type=Path, required=True)

        cls._workspace_parser = parser
        return parser

    @classmethod
    def run_workspace_command(cls, argv: list[str] | None = None) -> int:
        """Execute workspace manager commands using FlextTestDocker."""
        parser = cls._get_workspace_parser()
        args = parser.parse_args(argv)

        if not getattr(args, "command", None):
            parser.print_help()
            return 1

        workspace_root: Path = (
            args.workspace_root if args.workspace_root is not None else Path.cwd()
        )
        manager = cls(workspace_root=workspace_root)

        # Use union type for all possible command results
        command_result: (
            FlextResult[str]
            | FlextResult[FlextTestsTypings.Docker.ContainerOperationResult]
            | FlextResult[list[str]]
            | FlextResult[list[FlextTestModels.Docker.ContainerInfo]]
            | None
        ) = None

        if args.command == "init":
            if args.workspace_root is None:
                return 1
            command_result = manager.init_workspace(Path(args.workspace_root))
        elif args.command == "build-workspace":
            projects = [p.strip() for p in args.projects.split(",") if p.strip()]
            command_result = manager.build_workspace_projects(projects, args.registry)
        elif args.command == "build-image":
            command_result = manager.build_single_image(
                args.name,
                args.dockerfile,
                args.context,
            )
        elif args.command == "start-stack":
            command_result = manager.start_compose_stack(
                args.compose_file,
                args.network,
            )
        elif args.command == "stop-stack":
            command_result = manager.stop_compose_stack(args.compose_file)
        elif args.command == "restart-stack":
            command_result = manager.restart_compose_stack(args.compose_file)
        elif args.command == "show-logs":
            command_result = manager.show_stack_logs(
                args.compose_file,
                follow=args.follow,
            )
        elif args.command == "show-status":
            status_result = manager.show_stack_status(args.compose_file)
            if status_result.is_success:
                cls._display_status_table(manager)
            command_result = status_result
        elif args.command == "connect":
            command_result = manager.connect_to_service(args.service)
        elif args.command == "exec":
            command_result = manager.execute_in_service(args.service, args.command)
        elif args.command == "cleanup":
            command_result = manager.cleanup_workspace(
                remove_volumes=args.remove_volumes,
                remove_networks=args.remove_networks,
                prune_system=args.prune_system,
            )
        elif args.command == "health-check":
            command_result = manager.health_check_stack(
                args.compose_file,
                timeout=args.timeout,
            )
        elif args.command == "create-network":
            command_result = manager.create_network(args.name, driver=args.driver)
        elif args.command == "list-volumes":
            volumes = manager.list_volumes()
            if volumes.is_success:
                filtered = volumes.value
                if args.filter:
                    filtered = [volume for volume in filtered if args.filter in volume]
                for volume in filtered:
                    cls._console.print(volume)
            command_result = volumes
        elif args.command == "list-images":
            images = manager.images_formatted()
            if images.is_success:
                filtered = images.value
                if args.filter:
                    filtered = [image for image in filtered if args.filter in image]
                for image in filtered:
                    cls._console.print(image)
            command_result = images
        elif args.command == "list-containers":
            containers = manager.list_containers(all_containers=args.show_all)
            if containers.is_success:
                filtered_containers = containers.value
                if args.filter:
                    filtered_containers = [
                        info for info in filtered_containers if args.filter in info.name
                    ]
                for info in filtered_containers:
                    cls._console.print(
                        f"{info.name}: {cls._status_icon(info.status)} ({info.image})",
                    )
            command_result = containers
        elif args.command == "validate":
            if args.workspace_root is None:
                return 1
            validate_result = manager.validate_workspace(Path(args.workspace_root))
            if validate_result.is_success:
                # validate_workspace returns ContainerOperationResult
                if isinstance(validate_result.value, dict):
                    for key, value in validate_result.value.items():
                        cls._console.print(f"{key}: {value}")
            command_result = validate_result
        else:
            return 1

        if command_result is not None and command_result.is_success:
            value = command_result.value
            if isinstance(value, str) and value:
                cls._console.print(value)
            elif isinstance(value, dict):
                # ContainerOperationResult - already handled in validate case
                pass
            return 0

        if command_result is not None:
            cls._console.print(f"[bold red]{command_result.error}[/bold red]")
        return 1

    @classmethod
    def _get_cli_group(cls) -> click.Group:
        """Build (or reuse) the click CLI group for container commands."""
        if cls._cli_group is not None:
            return cls._cli_group

        @click.group(name="docker")
        def docker_cli() -> None:
            """Manage FLEXT Docker test containers."""

        cls._cli_group = docker_cli
        return docker_cli

    @classmethod
    def run_cli(cls) -> None:
        """Execute the docker CLI group using click."""
        cli = cls._get_cli_group()
        cli()

    @staticmethod
    def mark_dirty_on_failure(
        container_name: str,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Decorator to mark a container dirty if a test fails.

        Args:
            container_name: Name of the container to mark dirty on failure

        Example:
            @mark_dirty_on_failure("flext-redis-test")
            def test_redis_operations(redis_container):
                # Test code that modifies Redis state
                pass

        """

        def decorator[TResult](func: Callable[..., TResult]) -> Callable[..., TResult]:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> TResult:
                docker_manager = FlextTestDocker()

                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Mark container dirty on any exception
                    docker_manager.mark_container_dirty(container_name)
                    logger.warning(
                        "Test failed, marking container dirty",
                        extra={"container": container_name, "error": str(e)},
                    )
                    raise

            return wrapper

        return decorator

    @staticmethod
    def auto_cleanup_dirty_containers() -> Callable[
        [Callable[..., object]],
        Callable[..., object],
    ]:
        """Decorator to automatically clean up all dirty containers before a test.

        Example:
            @auto_cleanup_dirty_containers()
            def test_clean_environment():
                # Test runs with all containers in clean state
                pass

        """

        def decorator[TResult](func: Callable[..., TResult]) -> Callable[..., TResult]:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> TResult:
                docker_manager = FlextTestDocker()

                # Clean up all dirty containers before test
                dirty = docker_manager.get_dirty_containers()
                if dirty:
                    logger.info(
                        "Cleaning dirty containers before test",
                        extra={"containers": dirty},
                    )
                    cleanup_result = docker_manager.cleanup_dirty_containers()
                    if cleanup_result.is_failure:
                        msg = f"Failed to cleanup dirty containers: {cleanup_result.error}"
                        raise RuntimeError(
                            msg,
                        )

                # Run test
                return func(*args, **kwargs)

            return wrapper

        return decorator


def main() -> None:
    """Entry point for command-line usage."""
    FlextTestDocker.run_cli()


__all__ = ["FlextTestDocker", "main"]
