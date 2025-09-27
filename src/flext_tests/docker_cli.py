"""CLI interface for FLEXT Docker container management.

Provides command-line interface to start, stop, and reset test containers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

from flext_core import FlextLogger, FlextResult
from flext_tests.docker import ContainerStatus, FlextTestDocker


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


console = Console()


@click.group(name="docker")
def docker_cli() -> None:
    """Manage FLEXT Docker test containers."""


@docker_cli.command()
@click.option(
    "--container",
    "-c",
    type=click.Choice(
        ["flext-shared-ldap", "flext-postgres", "flext-redis", "flext-oracle", "all"],
        case_sensitive=False,
    ),
    default="all",
    help="Container to start",
)
def start(container: str) -> None:
    """Start FLEXT test container(s)."""
    control = FlextTestDocker()

    if container == "all":
        console.print("[bold blue]Starting all FLEXT test containers...[/bold blue]")
        all_result = control.start_all()
        if all_result.is_success:
            console.print(
                "[bold green]✓ All containers started successfully[/bold green]"
            )
            _display_status_table(control)
        else:
            console.print(
                f"[bold red]✗ Failed to start containers: {all_result.error}[/bold red]"
            )
            sys.exit(1)
    else:
        console.print(f"[bold blue]Starting container: {container}[/bold blue]")
        container_result = control.start_container(container)
        if container_result.is_success:
            console.print(f"[bold green]✓ {container_result.value}[/bold green]")
        else:
            console.print(f"[bold red]✗ {container_result.error}[/bold red]")
            sys.exit(1)


@docker_cli.command()
@click.option(
    "--container",
    "-c",
    type=click.Choice(
        ["flext-shared-ldap", "flext-postgres", "flext-redis", "flext-oracle", "all"],
        case_sensitive=False,
    ),
    default="all",
    help="Container to stop",
)
@click.option("--remove", "-r", is_flag=True, help="Remove container after stopping")
def stop(container: str, remove: bool) -> None:
    """Stop FLEXT test container(s)."""
    control = FlextTestDocker()

    if container == "all":
        action = "Stopping and removing" if remove else "Stopping"
        console.print(f"[bold blue]{action} all FLEXT test containers...[/bold blue]")
        result = control.stop_all(remove=remove)
        if result.is_success:
            console.print(f"[bold green]✓ All containers {action.lower()}[/bold green]")
        else:
            console.print("[bold red]✗ Failed to stop containers[/bold red]")
            sys.exit(1)
    else:
        action = "Stopping and removing" if remove else "Stopping"
        console.print(f"[bold blue]{action} container: {container}[/bold blue]")
        stop_result: FlextResult[str] = control.stop_container(container, remove=remove)
        if stop_result.is_success:
            console.print(f"[bold green]✓ {stop_result.value}[/bold green]")
        else:
            console.print(f"[bold red]✗ {stop_result.error}[/bold red]")
            sys.exit(1)


@docker_cli.command()
@click.option(
    "--container",
    "-c",
    type=click.Choice(
        ["flext-shared-ldap", "flext-postgres", "flext-redis", "flext-oracle", "all"],
        case_sensitive=False,
    ),
    default="all",
    help="Container to reset",
)
def reset(container: str) -> None:
    """Reset FLEXT test container(s) (stop, remove, start fresh)."""
    control = FlextTestDocker()

    if container == "all":
        console.print("[bold blue]Resetting all FLEXT test containers...[/bold blue]")
        result = control.reset_all()
        if result.is_success:
            console.print(
                "[bold green]✓ All containers reset successfully[/bold green]"
            )
            _display_status_table(control)
        else:
            console.print(
                f"[bold red]✗ Failed to reset containers: {result.error}[/bold red]"
            )
            sys.exit(1)
    else:
        console.print(f"[bold blue]Resetting container: {container}[/bold blue]")
        reset_result: FlextResult[str] = control.reset_container(container)
        if reset_result.is_success:
            console.print(f"[bold green]✓ {reset_result.value}[/bold green]")
        else:
            console.print(f"[bold red]✗ {reset_result.error}[/bold red]")
            sys.exit(1)


@docker_cli.command()
def status() -> None:
    """Show status of all FLEXT test containers."""
    control = FlextTestDocker()
    _display_status_table(control)


def _display_status_table(control: FlextTestDocker) -> None:
    """Display container status in a formatted table."""
    status_result = control.get_all_status()

    if status_result.is_failure:
        console.print(
            f"[bold red]Failed to get status: {status_result.error}[/bold red]"
        )
        return

    table = Table(title="FLEXT Docker Test Containers Status", show_header=True)
    table.add_column("Container", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Ports", style="green")
    table.add_column("Image", style="blue")

    for name, info in status_result.value.items():
        status_icon = {
            ContainerStatus.RUNNING: "🟢 Running",
            ContainerStatus.STOPPED: "🔴 Stopped",
            ContainerStatus.NOT_FOUND: "⚫ Not Found",
            ContainerStatus.ERROR: "⚠️ Error",
        }.get(info.status, "❓ Unknown")

        ports_str = (
            ", ".join(f"{k}→{v}" for k, v in info.ports.items()) if info.ports else "-"
        )

        table.add_row(name, status_icon, ports_str, info.image or "-")

    console.print(table)


@docker_cli.command()
@click.option(
    "--container",
    "-c",
    type=click.Choice(
        ["flext-shared-ldap", "flext-postgres", "flext-redis", "flext-oracle"],
        case_sensitive=False,
    ),
    required=True,
    help="Container to inspect",
)
def logs(container: str) -> None:
    """Show logs for a specific container."""
    control = FlextTestDocker()
    config = control.SHARED_CONTAINERS.get(container)

    if not config:
        console.print(f"[bold red]Unknown container: {container}[/bold red]")
        sys.exit(1)

    console.print(f"[bold blue]Fetching logs for {container}...[/bold blue]")

    try:
        # Get container logs using Docker client
        client = (
            control.get_client()
        )  # Use the public method to ensure client is initialized
        docker_container = client.containers.get(container)
        logs = docker_container.logs(tail=100).decode("utf-8", errors="ignore")
        console.print(logs)
    except Exception as e:
        console.print(f"[bold red]Failed to get logs: {e}[/bold red]")
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    docker_cli()


if __name__ == "__main__":
    main()
