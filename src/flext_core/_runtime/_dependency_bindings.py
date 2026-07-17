"""Dependency-injector runtime bridge bindings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect

from dependency_injector import containers, providers, wiring

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.file import FlextConstantsFile as cf

from ._dependency_options import FlextRuntimeDependencyOptions


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core._typings.services import FlextTypesServices as ts
    from flext_core._typings.base import FlextTypingBase as tb
    from flext_core._protocols.container import FlextProtocolsContainer as pc
    from types import ModuleType
    from collections.abc import Callable, MutableSequence


class FlextRuntimeDependencyBindings(FlextRuntimeDependencyOptions):
    """Register dependency-injector providers for runtime containers."""

    @classmethod
    def _populate_container(
        cls,
        di_container: containers.DynamicContainer,
        opts: pc.ContainerCreationOptions,
    ) -> None:
        """Register settings, services, factories, resources, and wiring."""
        if opts.settings is not None:
            _ = cls.bind_configuration(di_container, opts.settings)
        if opts.services:
            for name, instance in opts.services.items():
                _ = cls.register_object(di_container, name, instance)
        if opts.factories:
            for name, factory in opts.factories.items():
                _ = cls.register_factory(
                    di_container, name, factory, cache=opts.factory_cache
                )
        if opts.resources:
            for name, resource_factory in opts.resources.items():
                _ = cls.register_resource(di_container, name, resource_factory)
        if opts.wire_modules or opts.wire_packages or opts.wire_classes:
            cls.wire(
                di_container,
                modules=opts.wire_modules,
                packages=opts.wire_packages,
                classes=opts.wire_classes,
            )

    @classmethod
    def create_container(
        cls,
        container_options: pc.ContainerCreationOptions
        | tb.MappingKV[str, ts.JsonPayload]
        | None = None,
        **runtime_kwargs: ts.JsonPayload,
    ) -> containers.DynamicContainer:
        """Create a DynamicContainer with optional pre-registration and wiring."""
        base = cls._parse_options(container_options)
        opts = cls._merge_options(base, runtime_kwargs) if runtime_kwargs else base
        di_container = cls.DynamicContainerWithConfig()
        cls._populate_container(di_container, opts)
        return di_container

    @classmethod
    def create_layered_bridge(
        cls, settings: pc.RootDict[ts.JsonPayload] | None = None
    ) -> tuple[
        containers.DeclarativeContainer,
        containers.DynamicContainer,
        containers.DynamicContainer,
    ]:
        """Create a DeclarativeContainer bridged to dynamic modules."""
        bridge = cls.BridgeContainer()
        service_module = containers.DynamicContainer()
        resource_module = containers.DynamicContainer()
        bridge.services = providers.Object(service_module)
        bridge.resources = providers.Object(resource_module)
        cls.bind_configuration_provider(bridge.settings, settings)
        return (bridge, service_module, resource_module)

    @classmethod
    def bind_configuration(
        cls,
        di_container: containers.DynamicContainer,
        settings: pc.RootDict[ts.JsonPayload] | None,
    ) -> providers.Configuration:
        """Bind configuration mapping to the DI container."""
        configuration_provider = providers.Configuration()
        if settings:
            configuration_provider.from_dict(dict(settings.root))
        if isinstance(di_container, cls.DynamicContainerWithConfig):
            configured_container: FlextRuntimeDependencyBindings.DynamicContainerWithConfig = di_container
            configured_container.settings = configuration_provider
        else:
            setattr(di_container, cf.Directory.CONFIG, configuration_provider)
        return configuration_provider

    @staticmethod
    def bind_configuration_provider(
        configuration_provider: providers.Configuration,
        settings: pc.RootDict[ts.JsonPayload] | None,
    ) -> providers.Configuration:
        """Bind configuration directly to an existing provider."""
        if settings:
            configuration_provider.from_dict(dict(settings.root))
        return configuration_provider

    @staticmethod
    def register_factory[T](
        di_container: containers.DynamicContainer,
        name: str,
        factory: Callable[[], T],
        *,
        cache: bool = True,
    ) -> providers.Provider[T]:
        """Register a factory using Singleton/Factory providers."""
        if hasattr(di_container, name):
            raise ValueError(
                ce.ERR_RUNTIME_PROVIDER_ALREADY_REGISTERED.format(name=name)
            )
        provider: providers.Provider[T] = (
            providers.Singleton(factory) if cache else providers.Factory(factory)
        )
        setattr(di_container, name, provider)
        return provider

    @staticmethod
    def register_object[T](
        di_container: containers.DynamicContainer, name: str, instance: T
    ) -> providers.Provider[T]:
        """Register a concrete instance using ``providers.Object``."""
        if hasattr(di_container, name):
            raise ValueError(
                ce.ERR_RUNTIME_PROVIDER_ALREADY_REGISTERED.format(name=name)
            )
        provider: providers.Provider[T] = providers.Object(instance)
        setattr(di_container, name, provider)
        return provider

    @staticmethod
    def register_resource[T](
        di_container: containers.DynamicContainer, name: str, factory: Callable[[], T]
    ) -> providers.Provider[T]:
        """Register a resource provider for lifecycle-managed dependencies."""
        if hasattr(di_container, name):
            raise ValueError(
                ce.ERR_RUNTIME_PROVIDER_ALREADY_REGISTERED.format(name=name)
            )
        provider: providers.Provider[T] = providers.Resource(factory)
        setattr(di_container, name, provider)
        return provider

    @staticmethod
    def wire(
        container: containers.Container,
        *,
        modules: tb.SequenceOf[ModuleType] | None = None,
        packages: tb.StrSequence | None = None,
        classes: tb.SequenceOf[type] | None = None,
    ) -> None:
        """Wire modules or packages to a dependency-injector container."""
        modules_to_wire: MutableSequence[ModuleType] = list(modules or [])
        if classes:
            for target_class in classes:
                module = inspect.getmodule(target_class)
                if module is not None:
                    modules_to_wire.append(module)
        _ = packages
        wire_runtime = wiring.wire
        wire_runtime(
            modules=modules_to_wire or None, packages=None, container=container
        )


__all__: list[str] = ["FlextRuntimeDependencyBindings"]
