"""Dependency injection container for the dispatcher-first CQRS stack.

This module wraps dependency_injector behind a result-bearing API so handlers
and decorators can register/resolve dependencies without importing the
underlying infrastructure. Configuration stays isolated from dispatcher code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from abc import ABC
from collections.abc import (
    Callable,
    MutableMapping,
)
from typing import ClassVar, Self, override

from dependency_injector import containers as di_containers

from flext_core.context import FlextContext
from flext_core.models import m
from flext_core.protocols import p
from flext_core.settings import FlextSettings
from flext_core.typings import t


class FlextContainer(p.Container, ABC):
    """Singleton DI container wrapping dependency_injector with result-bearing API.

    Services and factories remain local to the container, keeping dispatcher and
    domain code free from infrastructure imports. All operations surface
    ``r`` (Result) so failures are explicit. Thread-safe initialization
    guarantees one global instance for runtime usage while allowing scoped
    containers in tests.
    """

    _global_instance: Self | None = None

    _global_lock: threading.RLock = threading.RLock()

    _settings_type: ClassVar[p.NamespacedSettingsType] = FlextSettings

    _context_type: ClassVar[p.ContextType] = FlextContext

    _context: p.Context

    _config: p.Settings

    _user_overrides: m.ConfigMap

    _di_bridge: di_containers.DeclarativeContainer

    _di_services: di_containers.DynamicContainer

    _di_resources: di_containers.DynamicContainer

    _di_container: di_containers.DynamicContainer

    _services: MutableMapping[str, m.ServiceRegistration]

    _factories: MutableMapping[str, m.FactoryRegistration]

    _resources: MutableMapping[str, m.ResourceRegistration]

    _internal_registrations: set[str]

    _global_config: m.ContainerConfig

    def __new__(cls, *, registration: m.ServiceRegistrationSpec | None = None) -> Self:
        """Create or return the global singleton instance."""
        _ = registration
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    instance = super().__new__(cls)
                    cls._global_instance = instance
        return cls._global_instance

    @property
    @override
    def settings(self) -> p.Settings:
        """Return configuration bound to this container."""
        return self._config

    @property
    @override
    def context(self) -> p.Context:
        """Return the execution context bound to this container."""
        return self._context

    @property
    @override
    def provide(self) -> Callable[[str], t.RegisterableService]:
        """Return the dependency-injector Provide helper scoped to the bridge."""
        return self._di_bridge.provide

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset singleton instance for testing purposes."""
        with cls._global_lock:
            cls._global_instance = None


__all__: list[str] = ["FlextContainer"]
