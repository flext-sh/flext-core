"""Constants for FLEXT tests.

Provides FlextTestConstants, extending FlextConstants with test-specific constants
for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Literal

from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes


class FlextTestConstants(FlextConstants):
    """Constants for FLEXT tests - extends FlextConstants.

    Architecture: Extends FlextConstants with test-specific constants.
    All base constants from FlextConstants are available through inheritance.
    Uses StrEnum and Literals for type-safe constants following Python 3.13+ patterns.
    """

    class Docker:
        """Docker test infrastructure constants - extends FlextConstants.Test.Docker."""

        # Inherit from parent
        DEFAULT_LOG_TAIL: Final[int] = FlextConstants.Test.Docker.DEFAULT_LOG_TAIL
        DEFAULT_CONTAINER_CHOICES: Final[tuple[str, ...]] = (
            FlextConstants.Test.Docker.DEFAULT_CONTAINER_CHOICES
        )
        SHARED_CONTAINERS: Final[Mapping[str, FlextTypes.Types.ContainerConfigDict]] = (
            FlextConstants.Test.Docker.SHARED_CONTAINERS
        )

        # Test-specific Docker constants
        DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
        MAX_TIMEOUT_SECONDS: Final[int] = 300
        DEFAULT_HEALTH_CHECK_INTERVAL: Final[int] = 2
        DEFAULT_HEALTH_CHECK_RETRIES: Final[int] = 10
        DEFAULT_STARTUP_WAIT_SECONDS: Final[int] = 5

        class ContainerStatus(StrEnum):
            """Container status enumeration for test infrastructure."""

            RUNNING = "running"
            STOPPED = "stopped"
            NOT_FOUND = "not_found"
            ERROR = "error"
            STARTING = "starting"
            STOPPING = "stopping"
            RESTARTING = "restarting"

        class Operation(StrEnum):
            """Docker operation types."""

            START = "start"
            STOP = "stop"
            RESTART = "restart"
            REMOVE = "remove"
            BUILD = "build"
            PULL = "pull"
            LOGS = "logs"
            EXEC = "exec"

        # Literal types for type-safe operations
        type OperationLiteral = Literal[
            "start", "stop", "restart", "remove", "build", "pull", "logs", "exec"
        ]
        """Type-safe literal for Docker operations."""

        type ContainerStatusLiteral = Literal[
            "running",
            "stopped",
            "not_found",
            "error",
            "starting",
            "stopping",
            "restarting",
        ]
        """Type-safe literal for container status."""

        # Error messages
        ERROR_CONTAINER_NOT_FOUND: Final[str] = "Container not found"
        ERROR_CONTAINER_ALREADY_RUNNING: Final[str] = "Container already running"
        ERROR_CONTAINER_NOT_RUNNING: Final[str] = "Container not running"
        ERROR_DOCKER_NOT_AVAILABLE: Final[str] = "Docker not available"
        ERROR_COMPOSE_FILE_NOT_FOUND: Final[str] = "Docker compose file not found"
        ERROR_OPERATION_TIMEOUT: Final[str] = "Docker operation timed out"

    class Execution:
        """Test execution constants for test infrastructure.

        Extends FlextConstants.Test with test-specific execution constants.
        Does not override FlextConstants.Test to avoid MRO conflicts.
        """

        # Test execution timeouts
        DEFAULT_TEST_TIMEOUT_SECONDS: Final[int] = 60
        MAX_TEST_TIMEOUT_SECONDS: Final[int] = 600

        # Test data generation
        DEFAULT_BATCH_SIZE: Final[int] = 10
        MAX_BATCH_SIZE: Final[int] = 1000

        # Test fixture constants
        DEFAULT_FIXTURE_COUNT: Final[int] = 5
        MAX_FIXTURE_COUNT: Final[int] = 100

    # Network constants are available via FlextConstants.Network
    # Access via: FlextConstants.Network.MIN_PORT, FlextConstants.Network.MAX_PORT


__all__ = ["FlextTestConstants"]
