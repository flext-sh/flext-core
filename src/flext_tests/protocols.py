"""Protocol definitions for FLEXT tests.

Provides FlextTestProtocols, extending p with test-specific protocol
definitions for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from flext_core.protocols import p


class FlextTestProtocols:
    """Protocol definitions for FLEXT tests - extends p.

    Architecture: Extends p with test-specific protocol definitions.
    All base protocols from p are available through inheritance pattern.
    Protocols cannot import models - only other protocols and types.
    """

    # Re-export base protocols from p
    ResultProtocol = p.ResultProtocol
    ResultLike = p.ResultLike
    ConfigProtocol = p.ConfigProtocol
    ModelProtocol = p.ModelProtocol
    Service = p.Service
    Repository = p.Repository
    Handler = p.Handler
    CommandBus = p.CommandBus
    Middleware = p.Middleware
    LoggerProtocol = p.LoggerProtocol
    Connection = p.Connection

    class Docker:
        """Docker-specific protocol definitions."""

        @runtime_checkable
        class ContainerProtocol(Protocol):
            """Protocol for Docker container objects."""

            @property
            def id(self) -> str:
                """Container ID."""
                ...

            @property
            def name(self) -> str:
                """Container name."""
                ...

            @property
            def status(self) -> str:
                """Container status."""
                ...

            @property
            def image(self) -> str:
                """Container image."""
                ...

            def start(self) -> None:
                """Start the container."""
                ...

            def stop(self) -> None:
                """Stop the container."""
                ...

            def remove(
                self,
                *,
                force: bool = False,
                **kwargs: str | int | bool | None,
            ) -> None:
                """Remove the container.

                Args:
                    force: Force removal of running container
                    **kwargs: Additional removal options (labels, volumes, etc.)

                """
                ...

        @runtime_checkable
        class DockerClientProtocol(Protocol):
            """Protocol for Docker client operations."""

            def containers(
                self,
            ) -> FlextTestProtocols.Docker.ContainerCollectionProtocol:
                """Get container collection."""
                ...

            def images(self) -> FlextTestProtocols.Docker.ImageCollectionProtocol:
                """Get image collection."""
                ...

            def networks(self) -> FlextTestProtocols.Docker.NetworkCollectionProtocol:
                """Get network collection."""
                ...

            def volumes(self) -> FlextTestProtocols.Docker.VolumeCollectionProtocol:
                """Get volume collection."""
                ...

        @runtime_checkable
        class ContainerCollectionProtocol(Protocol):
            """Protocol for container collection operations."""

            def get(
                self, container_id: str
            ) -> FlextTestProtocols.Docker.ContainerProtocol:
                """Get container by ID."""
                ...

            def list(
                self,
                *,
                show_all: bool = False,
                filters: Mapping[str, str | Sequence[str]] | None = None,
                **kwargs: str | int | bool | None,
            ) -> list[FlextTestProtocols.Docker.ContainerProtocol]:
                """List containers with filters.

                Args:
                    show_all: Show all containers (including stopped)
                    filters: Filter mapping (e.g., {"status": ["running"]})
                    **kwargs: Additional filter options

                """
                ...

            def run(
                self,
                image: str,
                *,
                name: str | None = None,
                detach: bool = True,
                ports: Mapping[str, str | int] | None = None,
                environment: Sequence[str] | None = None,
                **kwargs: str | int | bool | None,
            ) -> FlextTestProtocols.Docker.ContainerProtocol:
                """Run a new container.

                Args:
                    image: Container image name
                    name: Container name
                    detach: Run in detached mode
                    ports: Port mappings
                    environment: Environment variables
                    **kwargs: Additional container options

                """
                ...

        @runtime_checkable
        class ImageCollectionProtocol(Protocol):
            """Protocol for image collection operations."""

            def get(self, image_id: str) -> FlextTestProtocols.Docker.ImageProtocol:
                """Get image by ID."""
                ...

            def list(
                self,
                *,
                show_all: bool = False,
                filters: Mapping[str, str | Sequence[str]] | None = None,
                **kwargs: str | int | bool | None,
            ) -> list[FlextTestProtocols.Docker.ImageProtocol]:
                """List images with filters.

                Args:
                    show_all: Show all images (including intermediate)
                    filters: Filter mapping
                    **kwargs: Additional filter options

                """
                ...

            def build(
                self,
                path: str,
                *,
                tag: str | None = None,
                build_args: Mapping[str, str] | None = None,
                **kwargs: str | int | bool | None,
            ) -> FlextTestProtocols.Docker.ImageProtocol:
                """Build an image.

                Args:
                    path: Build context path
                    tag: Image tag
                    build_args: Build arguments
                    **kwargs: Additional build options

                """
                ...

        @runtime_checkable
        class ImageProtocol(Protocol):
            """Protocol for Docker image objects."""

            @property
            def id(self) -> str:
                """Image ID."""
                ...

            @property
            def tags(self) -> list[str]:
                """Image tags."""
                ...

        @runtime_checkable
        class NetworkCollectionProtocol(Protocol):
            """Protocol for network collection operations."""

            def get(self, network_id: str) -> FlextTestProtocols.Docker.NetworkProtocol:
                """Get network by ID."""
                ...

            def list(
                self,
                *,
                filters: Mapping[str, str | Sequence[str]] | None = None,
                **kwargs: str | int | bool | None,
            ) -> list[FlextTestProtocols.Docker.NetworkProtocol]:
                """List networks with filters.

                Args:
                    filters: Filter mapping
                    **kwargs: Additional filter options

                """
                ...

            def create(
                self,
                name: str,
                *,
                driver: str = "bridge",
                ipam: Mapping[str, str | Sequence[str]] | None = None,
                **kwargs: str | int | bool | None,
            ) -> FlextTestProtocols.Docker.NetworkProtocol:
                """Create a network.

                Args:
                    name: Network name
                    driver: Network driver
                    ipam: IPAM configuration
                    **kwargs: Additional network options

                """
                ...

        @runtime_checkable
        class NetworkProtocol(Protocol):
            """Protocol for Docker network objects."""

            @property
            def id(self) -> str:
                """Network ID."""
                ...

            @property
            def name(self) -> str:
                """Network name."""
                ...

        @runtime_checkable
        class VolumeCollectionProtocol(Protocol):
            """Protocol for volume collection operations."""

            def get(self, volume_id: str) -> FlextTestProtocols.Docker.VolumeProtocol:
                """Get volume by ID."""
                ...

            def list(
                self,
                *,
                filters: Mapping[str, str | Sequence[str]] | None = None,
                **kwargs: str | int | bool | None,
            ) -> list[FlextTestProtocols.Docker.VolumeProtocol]:
                """List volumes with filters.

                Args:
                    filters: Filter mapping
                    **kwargs: Additional filter options

                """
                ...

            def create(
                self,
                name: str,
                *,
                driver: str | None = None,
                driver_opts: Mapping[str, str] | None = None,
                **kwargs: str | int | bool | None,
            ) -> FlextTestProtocols.Docker.VolumeProtocol:
                """Create a volume.

                Args:
                    name: Volume name
                    driver: Volume driver
                    driver_opts: Driver options
                    **kwargs: Additional volume options

                """
                ...

        @runtime_checkable
        class VolumeProtocol(Protocol):
            """Protocol for Docker volume objects."""

            @property
            def id(self) -> str:
                """Volume ID."""
                ...

            @property
            def name(self) -> str:
                """Volume name."""
                ...

        @runtime_checkable
        class ComposeClientProtocol(Protocol):
            """Protocol for docker-compose operations.

            Compatible with python-on-whales DockerClient.
            Uses structural typing - any object with compose and client_config attributes is accepted.
            """

            compose: object
            """Compose API access (python-on-whales style)."""

            client_config: object
            """Client configuration (python-on-whales style)."""

            def up(
                self,
                services: Sequence[str] | None = None,
                *,
                detach: bool = True,
                build: bool = False,
                **kwargs: str | int | bool | None,
            ) -> None:
                """Start compose services.

                Args:
                    services: Specific services to start
                    detach: Run in detached mode
                    build: Build images before starting
                    **kwargs: Additional compose options

                """
                ...

            def down(
                self,
                *,
                volumes: bool = False,
                remove_orphans: bool = False,
                timeout: int | None = None,
                **kwargs: str | int | bool | None,
            ) -> None:
                """Stop compose services.

                Args:
                    volumes: Remove volumes
                    remove_orphans: Remove orphan containers
                    timeout: Stop timeout in seconds
                    **kwargs: Additional compose options

                """
                ...

            def restart(
                self,
                services: Sequence[str] | None = None,
                *,
                timeout: int | None = None,
                **kwargs: str | int | bool | None,
            ) -> None:
                """Restart compose services.

                Args:
                    services: Specific services to restart
                    timeout: Restart timeout in seconds
                    **kwargs: Additional compose options

                """
                ...


__all__ = ["FlextTestProtocols"]
