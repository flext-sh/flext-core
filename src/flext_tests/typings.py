"""Type system foundation for FLEXT tests.

Provides FlextTestsTypings, extending t with test-specific type definitions
for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from flext_core.typings import T, T_co, T_contra

# Test-specific TypeVars (module-level only, as per FLEXT standards)
TTestResult = TypeVar("TTestResult")
TTestModel = TypeVar("TTestModel")
TTestService = TypeVar("TTestService")


class FlextTestsTypings:
    """Type system foundation for FLEXT tests - extends t.

    Architecture: Extends t with test-specific type aliases and definitions.
    All base types from t are available through direct reference.
    Uses specific, directed types instead of GeneralValueType where possible.
    """

    # Test-specific type aliases using Python 3.13+ PEP 695
    # Use specific types instead of GeneralValueType where possible
    type ContainerPortMapping = Mapping[str, str]
    """Mapping of container port names to host port bindings."""

    type ContainerConfigMapping = Mapping[
        str,
        str | int | float | bool | Sequence[str | int | float | bool] | None,
    ]
    """Mapping for container configuration data with specific value types."""

    type DockerComposeServiceMapping = Mapping[
        str,
        str
        | int
        | float
        | bool
        | Sequence[str | int | float | bool]
        | Mapping[str, str | int | float | bool]
        | None,
    ]
    """Mapping for docker-compose service configuration with specific types."""

    type ContainerStateMapping = Mapping[
        str,
        str | int | float | bool | Sequence[str] | Mapping[str, str | int] | None,
    ]
    """Mapping for container state information with specific value types."""

    type TestDataMapping = Mapping[
        str,
        str | int | float | bool | Sequence[str | int | float | bool] | None,
    ]
    """Mapping for test data with specific value types."""

    type TestConfigMapping = Mapping[
        str,
        str | int | float | bool | Sequence[str] | Mapping[str, str | int] | None,
    ]
    """Mapping for test configuration with specific value types."""

    type TestResultValue = (
        str
        | int
        | float
        | bool
        | Sequence[str | int | float | bool]
        | Mapping[str, str | int | float | bool]
        | None
    )
    """Type for test result values with specific constraints."""

    # Note: Generic callable types cannot use module-level TypeVars in class-level type aliases
    # Use Callable[..., T] or Callable[[T], bool] directly with TypeVar T when needed

    class Docker:
        """Docker-specific type definitions with specific types."""

        type ContainerPorts = Mapping[str, str]
        """Container port mappings (container_port -> host:port)."""

        type ContainerLabels = Mapping[str, str]
        """Container labels mapping."""

        type ContainerEnvironment = Sequence[str]
        """Container environment variables as sequence."""

        type ComposeFileConfig = Mapping[
            str,
            str
            | int
            | float
            | bool
            | Sequence[str | int | float | bool]
            | Mapping[str, str | int | float | bool]
            | None,
        ]
        """Docker compose file configuration structure with specific types."""

        type VolumeMapping = Mapping[str, str]
        """Volume mappings (host_path -> container_path)."""

        type NetworkMapping = Mapping[
            str,
            str | int | float | bool | Sequence[str] | Mapping[str, str | int] | None,
        ]
        """Network configuration mapping with specific types."""

        type ContainerHealthStatus = str
        """Container health status type (healthy, unhealthy, starting, none)."""

        type ContainerHealthStatusLiteral = str  # Will be Literal["healthy", "unhealthy", "starting", "none"] when needed
        """Type-safe literal for container health status."""

        type ContainerOperationResult = Mapping[
            str,
            str | int | bool | Sequence[str] | None,
        ]
        """Result type for container operations with specific fields."""

    class Test:
        """Test-specific type definitions."""

        type TestCaseData = Mapping[
            str,
            str | int | float | bool | Sequence[str | int | float | bool] | None,
        ]
        """Test case data structure with specific value types."""

        type TestFixtureData = Mapping[
            str,
            str
            | int
            | float
            | bool
            | Path
            | Sequence[str | int | float | bool]
            | Mapping[str, str | int | float | bool]
            | None,
        ]
        """Test fixture data structure with specific value types."""

        type TestAssertionResult = Mapping[str, str | bool | int | None]
        """Test assertion result structure."""

        type TestExecutionContext = Mapping[
            str,
            str | int | float | bool | Sequence[str] | Mapping[str, str] | None,
        ]
        """Test execution context with specific metadata types."""


__all__ = [
    "FlextTestsTypings",
    "T",
    "TTestModel",
    "TTestResult",
    "TTestService",
    "T_co",
    "T_contra",
]
