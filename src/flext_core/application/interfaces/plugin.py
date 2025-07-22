"""Plugin management interfaces for flext_core.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module defines abstract interfaces for plugin management that concrete
projects can implement, following Clean Architecture and DDD principles.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from flext_core.domain.plugin_types import PluginSource
from flext_core.domain.plugin_types import PluginStatus
from flext_core.domain.plugin_types import PluginType

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from flext_core.domain.shared_types import ServiceResult


@dataclass(frozen=True)
class PluginInfo:
    """Plugin information data structure."""

    id: UUID | str
    name: str
    version: str
    description: str
    plugin_type: PluginType
    source: PluginSource
    status: PluginStatus
    configuration: dict[str, Any]
    dependencies: list[str]
    tags: list[str]
    documentation_url: str | None = None
    repository_url: str | None = None
    installed_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class PluginInstallationRequest:
    """Plugin installation request data."""

    name: str
    version: str
    source: PluginSource = PluginSource.PYPI
    requirements: list[str] | None = None
    force: bool = False

    def __post_init__(self) -> None:
        """Set default requirements if None."""
        if self.requirements is None:
            object.__setattr__(self, "requirements", [])


@dataclass(frozen=True)
class PluginInstallationResult:
    """Plugin installation result data."""

    success: bool
    installed_version: str | None = None
    error_message: str | None = None
    logs: list[str] | None = None
    dependencies: list[str] | None = None

    def __post_init__(self) -> None:
        """Set default values for optional fields."""
        if self.logs is None:
            object.__setattr__(self, "logs", [])
        if self.dependencies is None:
            object.__setattr__(self, "dependencies", [])


@dataclass(frozen=True)
class PluginUpdateRequest:
    """Plugin update request data."""

    version: str
    force: bool = False
    requirements: list[str] | None = None

    def __post_init__(self) -> None:
        """Set default requirements if None."""
        if self.requirements is None:
            object.__setattr__(self, "requirements", [])


@dataclass(frozen=True)
class PluginUpdateResult:
    """Plugin update result data."""

    success: bool
    updated_version: str | None = None
    error_message: str | None = None
    logs: list[str] | None = None
    dependencies: list[str] | None = None

    def __post_init__(self) -> None:
        """Set default values for optional fields."""
        if self.logs is None:
            object.__setattr__(self, "logs", [])
        if self.dependencies is None:
            object.__setattr__(self, "dependencies", [])


@dataclass(frozen=True)
class PluginUninstallRequest:
    """Plugin uninstall request data."""

    force: bool = False
    keep_config: bool = True


@dataclass(frozen=True)
class PluginHealthResult:
    """Plugin health check result data."""

    success: bool
    message: str | None = None
    error_message: str | None = None
    data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Set default data if None."""
        if self.data is None:
            object.__setattr__(self, "data", {})


class PluginManagerProvider(ABC):
    """Abstract plugin manager interface for dependency injection.

    This interface defines the contract for plugin management operations
    that concrete implementations (like flext-plugin) can provide.
    Following Clean Architecture, this interface belongs in flext_core
    and concrete implementations are injected at runtime.
    """

    @abstractmethod
    async def install_plugin(
        self,
        request: PluginInstallationRequest,
    ) -> ServiceResult[PluginInstallationResult]:
        """Install a plugin.

        Args:
            request: Plugin installation request data

        Returns:
            ServiceResult containing installation result or error

        """
        ...

    @abstractmethod
    async def update_plugin(
        self,
        name: str,
        request: PluginUpdateRequest,
    ) -> ServiceResult[PluginUpdateResult]:
        """Update an installed plugin.

        Args:
            name: Plugin name to update
            request: Plugin update request data

        Returns:
            ServiceResult containing update result or error

        """
        ...

    @abstractmethod
    async def uninstall_plugin(
        self,
        name: str,
        request: PluginUninstallRequest,
    ) -> ServiceResult[bool]:
        """Uninstall a plugin.

        Args:
            name: Plugin name to uninstall
            request: Plugin uninstall request data

        Returns:
            ServiceResult indicating success or error

        """
        ...

    @abstractmethod
    def list_plugins(
        self,
        plugin_type: PluginType | None = None,
        *,
        enabled_only: bool = False,
    ) -> list[PluginInfo]:
        """List installed plugins.

        Args:
            plugin_type: Optional filter by plugin type
            enabled_only: If True, only return enabled plugins

        Returns:
            List of plugin information

        """
        ...

    @abstractmethod
    async def check_plugin_health(self, name: str) -> ServiceResult[PluginHealthResult]:
        """Check health of a specific plugin.

        Args:
            name: Plugin name to check

        Returns:
            ServiceResult containing health check result or error

        """
        ...

    @abstractmethod
    def get_project_name(self) -> str:
        """Get the project name that provides this plugin manager.

        Returns:
            Project name identifier

        """
        ...

    @abstractmethod
    def get_project_identifier(self) -> str:
        """Get the unique project identifier.

        Returns:
            Unique project identifier

        """
        ...
