"""Models for FLEXT tests.

Provides FlextTestModels, extending FlextModels with test-specific model definitions
for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from flext_core._models.entity import FlextModelsEntity
from flext_core.constants import FlextConstants
from flext_core.models import FlextModels
from flext_tests.constants import FlextTestConstants
from flext_tests.typings import FlextTestsTypings


class FlextTestModels(FlextModels):
    """Models for FLEXT tests - extends FlextModels.

    Architecture: Extends FlextModels with test-specific model definitions.
    All base models from FlextModels are available through inheritance.
    """

    # Re-export base models - use actual classes for inheritance
    Value = FlextModelsEntity.Value
    Entity = FlextModelsEntity.Core
    AggregateRoot = FlextModelsEntity.AggregateRoot
    DomainEvent = FlextModelsEntity.DomainEvent

    class Docker:
        """Docker-specific models for test infrastructure."""

        class ContainerInfo(FlextModelsEntity.Value):
            """Container information model.

            Represents Docker container state and configuration.
            Uses FlextModels.Value for immutability and value comparison.
            """

            name: str
            status: FlextTestConstants.Docker.ContainerStatus
            ports: Mapping[str, str]
            image: str
            container_id: str = ""

            def model_post_init(self, __context: object, /) -> None:
                """Validate container info after initialization."""
                super().model_post_init(__context)
                if not self.name:
                    msg = "Container name cannot be empty"
                    raise ValueError(msg)
                if not self.image:
                    msg = "Container image cannot be empty"
                    raise ValueError(msg)

        class ContainerConfig(FlextModelsEntity.Value):
            """Container configuration model.

            Represents docker-compose container configuration.
            """

            compose_file: Path
            service: str
            port: int

            def model_post_init(self, __context: object, /) -> None:
                """Validate container config after initialization."""
                super().model_post_init(__context)
                if not self.compose_file.exists():
                    msg = f"Compose file not found: {self.compose_file}"
                    raise ValueError(msg)
                if not self.service:
                    msg = "Service name cannot be empty"
                    raise ValueError(msg)
                if not (
                    FlextConstants.Network.MIN_PORT
                    <= self.port
                    <= FlextConstants.Network.MAX_PORT
                ):
                    msg = f"Port {self.port} out of valid range"
                    raise ValueError(msg)

        class ContainerState(FlextModelsEntity.Value):
            """Container state tracking model.

            Represents persistent container state for dirty tracking.
            """

            container_name: str
            is_dirty: bool
            worker_id: str
            last_updated: str | None = None

        class ComposeConfig(FlextModelsEntity.Value):
            """Docker compose configuration model."""

            compose_file: Path
            services: Mapping[str, FlextTestsTypings.Docker.ComposeFileConfig]
            networks: Mapping[str, FlextTestsTypings.Docker.NetworkMapping] | None = (
                None
            )
            volumes: Mapping[str, FlextTestsTypings.Docker.VolumeMapping] | None = None

            def model_post_init(self, __context: object, /) -> None:
                """Validate compose config after initialization."""
                super().model_post_init(__context)
                if not self.compose_file.exists():
                    msg = f"Compose file not found: {self.compose_file}"
                    raise ValueError(msg)
                if not self.services:
                    msg = "Compose config must have at least one service"
                    raise ValueError(msg)


__all__ = ["FlextTestModels"]
