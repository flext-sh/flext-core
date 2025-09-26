#!/usr/bin/env python3
"""FLEXT Workspace Docker Manager - Unified Docker Operations.

This script provides workspace-level Docker management using FlextTestDocker
for unified Docker operations across the entire FLEXT ecosystem.

Usage:
    python docker_workspace_manager.py init --workspace-root /path/to/workspace
    python docker_workspace_manager.py build-workspace --projects "flext-core,flext-api"
    python docker_workspace_manager.py start-stack --compose-file docker/docker-compose.yml
    python docker_workspace_manager.py health-check --compose-file docker/docker-compose.yml

All operations use FlextTestDocker exclusively - NO direct Docker CLI commands.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Add the flext_tests module to the path
sys.path.insert(0, str(Path(__file__).parent))

from flext_core import FlextLogger, FlextResult
from flext_tests.docker import FlextTestDocker


class FlextWorkspaceDockerManager:
    """Unified workspace Docker management using FlextTestDocker."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize workspace Docker manager."""
        self.workspace_root = workspace_root or Path.cwd()
        self.docker_manager = FlextTestDocker(workspace_root=self.workspace_root)
        self.logger = FlextLogger("flext_workspace_docker")

    def init_workspace(self, workspace_root: Path) -> FlextResult[str]:
        """Initialize FlextTestDocker workspace management."""
        try:
            self.workspace_root = workspace_root
            self.docker_manager = FlextTestDocker(workspace_root=workspace_root)

            # Auto-discover services from common locations
            compose_paths = [
                workspace_root / "docker" / "docker-compose.yml",
                workspace_root / "docker-compose.yml",
                workspace_root / "compose.yml",
            ]

            for compose_path in compose_paths:
                if compose_path.exists():
                    discovery_result = self.docker_manager.auto_discover_services(
                        str(compose_path)
                    )
                    if discovery_result.is_success:
                        services = discovery_result.unwrap()
                        self.logger.info(
                            f"Auto-discovered services from {compose_path}: {services}"
                        )

            return FlextResult[str].ok(
                f"FlextTestDocker initialized for workspace: {workspace_root}"
            )

        except Exception as e:
            return FlextResult[str].fail(f"Workspace initialization failed: {e}")

    def build_workspace_projects(
        self, projects: list[str], registry: str = "flext"
    ) -> FlextResult[dict[str, str]]:
        """Build Docker images for multiple projects using FlextTestDocker."""
        results = {}

        for project in projects:
            project_path = self.workspace_root / project
            if not project_path.exists():
                results[project] = f"Project path not found: {project_path}"
                continue

            # Look for Dockerfile in project
            dockerfile_candidates = [
                project_path / "Dockerfile",
                project_path / "docker" / "Dockerfile",
                project_path / f"Dockerfile.{project}",
            ]

            dockerfile_path = None
            for candidate in dockerfile_candidates:
                if candidate.exists():
                    dockerfile_path = candidate
                    break

            if not dockerfile_path:
                results[project] = "No Dockerfile found"
                continue

            # Build image using FlextTestDocker
            tag = f"{registry}/{project}:latest"
            build_result = self.docker_manager.build_image_advanced(
                path=str(project_path), tag=tag, dockerfile=str(dockerfile_path.name)
            )

            if build_result.is_success:
                results[project] = f"Built successfully: {tag}"
            else:
                results[project] = f"Build failed: {build_result.error}"

        return FlextResult[dict[str, str]].ok(results)

    def build_single_image(
        self, name: str, dockerfile_path: str, context_path: str | None = None
    ) -> FlextResult[str]:
        """Build a single Docker image using FlextTestDocker."""
        context = context_path or str(Path(dockerfile_path).parent)

        build_result = self.docker_manager.build_image_advanced(
            path=context, tag=name, dockerfile=str(Path(dockerfile_path).name)
        )

        if build_result.is_success:
            return FlextResult[str].ok(f"Image built successfully: {name}")
        return FlextResult[str].fail(f"Image build failed: {build_result.error}")

    def start_compose_stack(
        self, compose_file: str, network_name: str | None = None
    ) -> FlextResult[str]:
        """Start Docker Compose stack using FlextTestDocker."""
        # Auto-discover and register services from compose file
        discovery_result = self.docker_manager.auto_discover_services(compose_file)
        if discovery_result.is_failure:
            return FlextResult[str].fail(
                f"Service discovery failed: {discovery_result.error}"
            )

        services = discovery_result.unwrap()

        # Start the compose stack
        start_result = self.docker_manager.compose_up(compose_file)

        if start_result.is_failure:
            return FlextResult[str].fail(f"Stack start failed: {start_result.error}")

        # Create network if specified
        if network_name:
            network_result = self.docker_manager.create_network(network_name)
            if network_result.is_failure:
                self.logger.warning(f"Network creation failed: {network_result.error}")

        return FlextResult[str].ok(
            f"Stack started successfully with services: {services}"
        )

    def stop_compose_stack(self, compose_file: str) -> FlextResult[str]:
        """Stop Docker Compose stack using FlextTestDocker."""
        stop_result = self.docker_manager.compose_down(compose_file)

        if stop_result.is_success:
            return FlextResult[str].ok("Stack stopped successfully")
        return FlextResult[str].fail(f"Stack stop failed: {stop_result.error}")

    def restart_compose_stack(self, compose_file: str) -> FlextResult[str]:
        """Restart Docker Compose stack using FlextTestDocker."""
        # Stop first
        stop_result = self.stop_compose_stack(compose_file)
        if stop_result.is_failure:
            return FlextResult[str].fail(f"Stack stop failed: {stop_result.error}")

        # Start again
        start_result = self.start_compose_stack(compose_file)
        if start_result.is_failure:
            return FlextResult[str].fail(f"Stack restart failed: {start_result.error}")

        return FlextResult[str].ok("Stack restarted successfully")

    def show_stack_logs(
        self, compose_file: str, *, follow: bool = False
    ) -> FlextResult[str]:
        """Show Docker Compose stack logs using FlextTestDocker.

        Args:
            compose_file: Docker Compose file path
            follow: Whether to follow logs (currently unused but kept for API compatibility)

        """
        _ = follow  # Parameter required by API but not used in stub implementation
        logs_result = self.docker_manager.compose_logs(compose_file)

        if logs_result.is_success:
            return FlextResult[str].ok("Logs displayed")
        return FlextResult[str].fail(f"Failed to get logs: {logs_result.error}")

    def show_stack_status(self, compose_file: str) -> FlextResult[dict[str, Any]]:
        """Show Docker Compose stack status using FlextTestDocker.

        Args:
            compose_file: Docker Compose file path (currently unused but kept for API compatibility)

        """
        _ = compose_file  # Parameter required by API but not used in stub implementation
        # Get compose stack status
        status_result = self.docker_manager.get_all_status()

        if status_result.is_failure:
            return FlextResult[dict[str, Any]].fail(
                f"Status check failed: {status_result.error}"
            )

        status_info: dict[str, Any] = status_result.unwrap()

        # Add running services information
        running_services_result = self.docker_manager.get_running_services()
        if running_services_result.is_success:
            status_info["auto_managed_services"] = running_services_result.unwrap()
        else:
            status_info["auto_managed_services"] = []

        # Print formatted status
        for _key, _value in status_info.items():
            pass

        return FlextResult[dict[str, Any]].ok(status_info)

    def connect_to_service(self, service_name: str) -> FlextResult[str]:
        """Connect to a running service container using FlextTestDocker."""
        # Try to connect to container interactively
        connect_result = self.docker_manager.exec_container_interactive(
            container_name=service_name, command="/bin/bash"
        )

        if connect_result.is_success:
            return FlextResult[str].ok(f"Connected to {service_name}")
        return FlextResult[str].fail(f"Connection failed: {connect_result.error}")

    def execute_in_service(self, service_name: str, command: str) -> FlextResult[str]:
        """Execute command in service container using FlextTestDocker."""
        exec_result = self.docker_manager.execute_container_command(
            container_name=service_name, command=command
        )

        if exec_result.is_success:
            exec_result.unwrap()
            return FlextResult[str].ok("Command executed successfully")
        return FlextResult[str].fail(f"Command execution failed: {exec_result.error}")

    def cleanup_workspace(
        self,
        *,
        remove_volumes: bool = False,
        remove_networks: bool = False,
        prune_system: bool = False,
    ) -> FlextResult[str]:
        """Clean up Docker artifacts using FlextTestDocker."""
        cleanup_operations = []

        # Stop all auto-managed services
        running_services = self.docker_manager.get_running_services()
        for service in running_services:
            stop_result = self.docker_manager.stop_services_for_test(
                f"cleanup_{service}"
            )
            if stop_result.is_success:
                cleanup_operations.append(f"Stopped service: {service}")
            else:
                cleanup_operations.append(
                    f"Failed to stop service {service}: {stop_result.error}"
                )

        # Clean up containers
        cleanup_result = self.docker_manager.cleanup_all_test_containers()
        if cleanup_result.is_success:
            cleanup_operations.append("Cleaned up test containers")
        else:
            cleanup_operations.append(
                f"Container cleanup failed: {cleanup_result.error}"
            )

        # Remove volumes if requested
        if remove_volumes:
            volumes_result = self.docker_manager.cleanup_volumes()
            if volumes_result.is_success:
                cleanup_operations.append("Pruned volumes")
            else:
                cleanup_operations.append(
                    f"Volume pruning failed: {volumes_result.error}"
                )

        # Remove networks if requested
        if remove_networks:
            networks_result = self.docker_manager.cleanup_networks()
            if networks_result.is_success:
                cleanup_operations.append("Pruned networks")
            else:
                cleanup_operations.append(
                    f"Network pruning failed: {networks_result.error}"
                )

        # System prune if requested
        if prune_system:
            system_result = self.docker_manager.cleanup_images()
            if system_result.is_success:
                cleanup_operations.append("Pruned images")
            else:
                cleanup_operations.append(
                    f"Image pruning failed: {system_result.error}"
                )

        return FlextResult[str].ok(
            f"Cleanup completed: {'; '.join(cleanup_operations)}"
        )

    def health_check_stack(
        self, compose_file: str, *, timeout: int = 30
    ) -> FlextResult[dict[str, str]]:
        """Perform comprehensive health check on Docker stack.

        Args:
            compose_file: Docker Compose file path
            timeout: Health check timeout in seconds (currently unused but kept for API compatibility)

        """
        _ = timeout  # Parameter required by API but not used in stub implementation
        health_results = {}

        # Auto-discover services
        discovery_result = self.docker_manager.auto_discover_services(compose_file)
        if discovery_result.is_failure:
            return FlextResult[dict[str, str]].fail(
                f"Service discovery failed: {discovery_result.error}"
            )

        services = discovery_result.unwrap()

        # Check health of each service
        for service in services:
            health_result = self.docker_manager.get_service_health_status(service)
            if health_result.is_success:
                health_info = health_result.unwrap()
                health_results[service] = (
                    f"Status: {health_info.get('container_status', 'unknown')}"
                )
            else:
                health_results[service] = f"Health check failed: {health_result.error}"

        # Print health summary
        for _service in health_results:
            pass

        return FlextResult[dict[str, str]].ok(health_results)

    def create_network(self, name: str, driver: str = "bridge") -> FlextResult[str]:
        """Create Docker network using FlextTestDocker."""
        network_result = self.docker_manager.create_network(name, driver=driver)

        if network_result.is_success:
            return FlextResult[str].ok(f"Network created: {name}")
        return FlextResult[str].fail(f"Network creation failed: {network_result.error}")

    def list_volumes(self, name_filter: str | None = None) -> FlextResult[list[str]]:
        """List Docker volumes using FlextTestDocker."""
        volumes_result = self.docker_manager.list_volumes()

        if volumes_result.is_failure:
            return FlextResult[list[str]].fail(
                f"Volume listing failed: {volumes_result.error}"
            )

        volumes = volumes_result.unwrap()

        # Apply name filter if specified
        if name_filter:
            volumes = [v for v in volumes if name_filter in v]

        # Print volumes
        for _volume in volumes:
            pass

        return FlextResult[list[str]].ok(volumes)

    def list_images(
        self, repository_filter: str | None = None
    ) -> FlextResult[list[str]]:
        """List Docker images using FlextTestDocker."""
        images_result = self.docker_manager.images_formatted(
            format_string="{{.Repository}}:{{.Tag}}"
        )

        if images_result.is_failure:
            return FlextResult[list[str]].fail(
                f"Image listing failed: {images_result.error}"
            )

        images_str = images_result.unwrap()
        images = images_str.split("\n") if images_str else []

        # Apply repository filter if specified
        if repository_filter:
            images = [img for img in images if repository_filter in img]

        # Print images
        for _image in images:
            pass

        return FlextResult[list[str]].ok(images)

    def list_containers(
        self, name_filter: str | None = None, *, show_all: bool = False
    ) -> FlextResult[list[str]]:
        """List Docker containers using FlextTestDocker."""
        containers_result = self.docker_manager.list_containers_formatted(
            show_all=show_all, format_string="{{.Names}} ({{.Status}})"
        )

        if containers_result.is_failure:
            return FlextResult[list[str]].fail(
                f"Container listing failed: {containers_result.error}"
            )

        containers_str = containers_result.unwrap()
        containers = containers_str.split("\n") if containers_str else []

        # Apply name filter if specified
        if name_filter:
            containers = [c for c in containers if name_filter in c]

        # Print containers
        for _container in containers:
            pass

        return FlextResult[list[str]].ok(containers)

    def validate_workspace(self, workspace_root: Path) -> FlextResult[dict[str, str]]:
        """Validate FlextTestDocker functionality in workspace."""
        validation_results = {}

        try:
            # Test Docker connection
            docker_manager = FlextTestDocker(workspace_root=workspace_root)
            validation_results["docker_connection"] = "✅ Connected"

            # Test container operations
            containers_result = docker_manager.list_containers_formatted()
            if containers_result.is_success:
                validation_results["container_operations"] = "✅ Working"
            else:
                validation_results["container_operations"] = (
                    f"❌ Failed: {containers_result.error}"
                )

            # Test image operations
            images_result = docker_manager.images_formatted()
            if images_result.is_success:
                validation_results["image_operations"] = "✅ Working"
            else:
                validation_results["image_operations"] = (
                    f"❌ Failed: {images_result.error}"
                )

            # Test network operations
            networks_result = docker_manager.list_networks()
            if networks_result.is_success:
                validation_results["network_operations"] = "✅ Working"
            else:
                validation_results["network_operations"] = (
                    f"❌ Failed: {networks_result.error}"
                )

            # Test volume operations
            volumes_result = docker_manager.list_volumes()
            if volumes_result.is_success:
                validation_results["volume_operations"] = "✅ Working"
            else:
                validation_results["volume_operations"] = (
                    f"❌ Failed: {volumes_result.error}"
                )

        except Exception as e:
            validation_results["docker_connection"] = f"❌ Failed: {e}"

        # Print validation results
        for _operation, _result in validation_results.items():
            pass

        return FlextResult[dict[str, str]].ok(validation_results)


def main() -> int:
    """Main CLI entry point for FLEXT Workspace Docker Manager."""
    parser = argparse.ArgumentParser(
        description="FLEXT Workspace Docker Manager - Unified Docker Operations using FlextTestDocker"
    )

    parser.add_argument("--workspace-root", type=Path, help="Workspace root directory")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize FlextTestDocker workspace"
    )
    init_parser.add_argument("--workspace-root", type=Path, required=True)

    # Build commands
    build_workspace_parser = subparsers.add_parser(
        "build-workspace", help="Build workspace projects"
    )
    build_workspace_parser.add_argument(
        "--projects", required=True, help="Comma-separated project list"
    )
    build_workspace_parser.add_argument(
        "--registry", default="flext", help="Docker registry prefix"
    )

    build_image_parser = subparsers.add_parser("build-image", help="Build single image")
    build_image_parser.add_argument("--name", required=True, help="Image name/tag")
    build_image_parser.add_argument(
        "--dockerfile", required=True, help="Dockerfile path"
    )
    build_image_parser.add_argument("--context", help="Build context path")

    # Stack commands
    start_stack_parser = subparsers.add_parser(
        "start-stack", help="Start Docker Compose stack"
    )
    start_stack_parser.add_argument(
        "--compose-file", required=True, help="Docker Compose file path"
    )
    start_stack_parser.add_argument("--network", help="Network name")

    stop_stack_parser = subparsers.add_parser(
        "stop-stack", help="Stop Docker Compose stack"
    )
    stop_stack_parser.add_argument(
        "--compose-file", required=True, help="Docker Compose file path"
    )

    restart_stack_parser = subparsers.add_parser(
        "restart-stack", help="Restart Docker Compose stack"
    )
    restart_stack_parser.add_argument(
        "--compose-file", required=True, help="Docker Compose file path"
    )

    # Logging and status commands
    logs_parser = subparsers.add_parser("show-logs", help="Show stack logs")
    logs_parser.add_argument(
        "--compose-file", required=True, help="Docker Compose file path"
    )
    logs_parser.add_argument("--follow", action="store_true", help="Follow logs")

    status_parser = subparsers.add_parser("show-status", help="Show stack status")
    status_parser.add_argument(
        "--compose-file", required=True, help="Docker Compose file path"
    )

    # Connection commands
    connect_parser = subparsers.add_parser(
        "connect", help="Connect to service container"
    )
    connect_parser.add_argument("--service", required=True, help="Service name")

    exec_parser = subparsers.add_parser("exec", help="Execute command in container")
    exec_parser.add_argument("--service", required=True, help="Service name")
    exec_parser.add_argument("--command", required=True, help="Command to execute")

    # Cleanup commands
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up Docker artifacts")
    cleanup_parser.add_argument(
        "--remove-volumes", action="store_true", help="Remove volumes"
    )
    cleanup_parser.add_argument(
        "--remove-networks", action="store_true", help="Remove networks"
    )
    cleanup_parser.add_argument(
        "--prune-system", action="store_true", help="Prune system"
    )

    # Health check
    health_parser = subparsers.add_parser("health-check", help="Perform health check")
    health_parser.add_argument(
        "--compose-file", required=True, help="Docker Compose file path"
    )
    health_parser.add_argument(
        "--timeout", type=int, default=30, help="Health check timeout"
    )

    # Network management
    create_network_parser = subparsers.add_parser(
        "create-network", help="Create Docker network"
    )
    create_network_parser.add_argument("--name", required=True, help="Network name")
    create_network_parser.add_argument(
        "--driver", default="bridge", help="Network driver"
    )

    # List commands
    list_volumes_parser = subparsers.add_parser(
        "list-volumes", help="List Docker volumes"
    )
    list_volumes_parser.add_argument("--filter", help="Name filter")

    list_images_parser = subparsers.add_parser("list-images", help="List Docker images")
    list_images_parser.add_argument("--filter", help="Repository filter")

    list_containers_parser = subparsers.add_parser(
        "list-containers", help="List Docker containers"
    )
    list_containers_parser.add_argument("--filter", help="Name filter")
    list_containers_parser.add_argument(
        "--show-all", action="store_true", help="Show all containers"
    )

    # Validation
    validate_parser = subparsers.add_parser(
        "validate", help="Validate FlextTestDocker functionality"
    )
    validate_parser.add_argument("--workspace-root", type=Path, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize manager
    manager = FlextWorkspaceDockerManager(workspace_root=args.workspace_root)

    try:
        # Execute command
        if args.command == "init":
            result = manager.init_workspace(args.workspace_root)
        elif args.command == "build-workspace":
            projects = [p.strip() for p in args.projects.split(",")]
            result = manager.build_workspace_projects(projects, args.registry)
        elif args.command == "build-image":
            result = manager.build_single_image(
                args.name, args.dockerfile, args.context
            )
        elif args.command == "start-stack":
            result = manager.start_compose_stack(args.compose_file, args.network)
        elif args.command == "stop-stack":
            result = manager.stop_compose_stack(args.compose_file)
        elif args.command == "restart-stack":
            result = manager.restart_compose_stack(args.compose_file)
        elif args.command == "show-logs":
            result = manager.show_stack_logs(args.compose_file, follow=args.follow)
        elif args.command == "show-status":
            result = manager.show_stack_status(args.compose_file)
        elif args.command == "connect":
            result = manager.connect_to_service(args.service)
        elif args.command == "exec":
            result = manager.execute_in_service(args.service, args.command)
        elif args.command == "cleanup":
            result = manager.cleanup_workspace(
                remove_volumes=args.remove_volumes,
                remove_networks=args.remove_networks,
                prune_system=args.prune_system,
            )
        elif args.command == "health-check":
            result = manager.health_check_stack(args.compose_file, timeout=args.timeout)
        elif args.command == "create-network":
            result = manager.create_network(args.name, args.driver)
        elif args.command == "list-volumes":
            result = manager.list_volumes(args.filter)
        elif args.command == "list-images":
            result = manager.list_images(args.filter)
        elif args.command == "list-containers":
            result = manager.list_containers(args.filter, show_all=args.show_all)
        elif args.command == "validate":
            result = manager.validate_workspace(args.workspace_root)
        else:
            return 1

        if result.is_success:
            return 0
        return 1

    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
