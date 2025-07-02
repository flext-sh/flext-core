"""Plugin service for FLEXT Core."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from flext_core.domain.entities import Plugin
from flext_core.domain.value_objects import PluginId


class PluginService(ABC):
    """Abstract plugin service."""

    @abstractmethod
    async def install_plugin(
        self, name: str, version: str, config: dict[str, Any]
    ) -> Plugin:
        """Install a plugin."""

    @abstractmethod
    async def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin."""

    @abstractmethod
    async def get_plugin(self, plugin_id: str) -> Plugin | None:
        """Get plugin by ID."""

    @abstractmethod
    async def list_plugins(self, enabled_only: bool = False) -> list[Plugin]:
        """List plugins."""

    @abstractmethod
    async def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin."""

    @abstractmethod
    async def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin."""

    @abstractmethod
    async def update_plugin(self, plugin_id: str, version: str) -> Plugin:
        """Update a plugin to a new version."""


class DefaultPluginService(PluginService):
    """Default implementation of plugin service."""

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}

    async def install_plugin(
        self, name: str, version: str, config: dict[str, Any]
    ) -> Plugin:
        """Install a plugin."""
        plugin = Plugin(
            id=PluginId(str(uuid4())),
            name=name,
            version=version,
            description=f"Plugin {name} version {version}",
            configuration=config,
            is_enabled=True,
        )
        self._plugins[str(plugin.id)] = plugin
        return plugin

    async def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin."""
        if plugin_id in self._plugins:
            del self._plugins[plugin_id]
            return True
        return False

    async def get_plugin(self, plugin_id: str) -> Plugin | None:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)

    async def list_plugins(self, enabled_only: bool = False) -> list[Plugin]:
        """List plugins."""
        plugins = list(self._plugins.values())
        if enabled_only:
            plugins = [p for p in plugins if p.is_enabled]
        return plugins

    async def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin."""
        plugin = self._plugins.get(plugin_id)
        if plugin:
            plugin.is_enabled = True
            return True
        return False

    async def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin."""
        plugin = self._plugins.get(plugin_id)
        if plugin:
            plugin.is_enabled = False
            return True
        return False

    async def update_plugin(self, plugin_id: str, version: str) -> Plugin:
        """Update a plugin to a new version."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            msg = f"Plugin not found: {plugin_id}"
            raise ValueError(msg)

        plugin.version = version
        return plugin
