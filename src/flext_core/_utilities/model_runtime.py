"""Runtime DI builders + ``build_service_runtime`` orchestration.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from importlib import import_module
from types import MethodType

from flext_core import FlextModels, c, m, p, r, t
from flext_core._utilities import FlextUtilitiesModel, FlextUtilitiesModelOptions


class FlextUtilitiesModelRuntime(
    FlextUtilitiesModelOptions,
    FlextModels,
):
    """Runtime DSL: dispatcher, registry, and service-runtime construction.

    Namespace enforcement composes independently of the service-runtime builders.
    """

    @classmethod
    def build_dispatcher(cls) -> p.Dispatcher:
        """Materialize the canonical dispatcher implementation behind ``p.Dispatcher``.

        Returns:
            The resulting ``p.Dispatcher``.

        Raises:
            TypeError: If Resolved dispatcher implementation does not satisfy
                p.Dispatcher.

        """
        dispatcher_module = import_module("flext_core.dispatcher")
        dispatcher_candidate = dispatcher_module.FlextDispatcher()
        if not isinstance(dispatcher_candidate, p.Dispatcher):
            msg = "Resolved dispatcher implementation does not satisfy p.Dispatcher"
            raise TypeError(msg)
        return dispatcher_candidate

    @classmethod
    def build_registry(
        cls,
        dispatcher: p.Dispatcher | None = None,
        *,
        discovery_module: t.RegistryModule | None = None,
    ) -> p.Registry:
        """Materialize the canonical registry implementation behind ``p.Registry``.

        Returns:
            The resulting ``p.Registry``.

        Raises:
            TypeError: If Resolved registry implementation does not satisfy p.Registry.

        """
        registry_module = import_module("flext_core.registry")
        elected_dispatcher = (
            dispatcher if dispatcher is not None else cls.build_dispatcher()
        )
        registry_candidate = registry_module.FlextRegistry(
            registry_runtime=cls.build_service_runtime(
                m.RuntimeBootstrapOptions(
                    dispatcher=elected_dispatcher,
                )
            ),
        )
        if not isinstance(registry_candidate, p.Registry):
            msg = "Resolved registry implementation does not satisfy p.Registry"
            raise TypeError(msg)
        if discovery_module is not None:
            handlers_module = import_module("flext_core.handlers")
            elected_dispatcher.register_handlers(
                tuple(
                    m.HandlerRegistration(
                        handler=handler, message_type=metadata.command
                    )
                    for _, handler, metadata in handlers_module.h.Discovery.scan_module(
                        discovery_module
                    )
                )
            ).unwrap()
        return registry_candidate

    @staticmethod
    def registry_same_value(
        previous: p.RegisterableService | p.DispatchableHandler,
        candidate: p.RegisterableService | p.DispatchableHandler,
    ) -> bool:
        """Compare collaborators by identity, immutable scalars by type and value.

        Returns:
            Whether both registrations identify the same collaborator, bound
            method, or equally valued immutable scalar of the same type.

        """
        if previous is candidate:
            return True
        if isinstance(previous, MethodType) and isinstance(candidate, MethodType):
            return (
                previous.__self__ is candidate.__self__
                and previous.__func__ is candidate.__func__
            )
        return (
            type(previous) is type(candidate)
            and any(type(previous) is scalar_type for scalar_type in c.SCALAR_TYPES)
            and previous == candidate
        )

    @staticmethod
    def registry_plugin_key(category: str, name: str | None = None) -> p.Result[str]:
        """Validate a plugin key, or a category prefix when no name is supplied.

        Returns:
            The validated key or category prefix, or a failure when a component
            is empty or contains the configured separator.

        """
        separator = c.REGISTRY_KEY_SEPARATOR
        if (
            not category
            or separator in category
            or (name is not None and (not name or separator in name))
        ):
            return r[str].fail(
                c.ERR_REGISTRY_PLUGIN_KEY_INVALID.format(separator=separator)
            )
        return r[str].ok(
            f"{category}{separator}{name}"
            if name is not None
            else f"{category}{separator}"
        )

    @staticmethod
    def registry_registration_name(binding: p.HandlerRegistration) -> str:
        """Read the normalized route published by the dispatcher.

        Returns:
            The declared string route, otherwise the handler's qualified type
            name used to identify the registration.

        """
        route = binding.message_type
        return route if isinstance(route, str) else type(binding.handler).__qualname__

    @staticmethod
    def handler_metadata(
        candidate: p.RegisterableService
        | p.DispatchableHandler
        | p.ResolvedHandlerCallable,
        *,
        name: str,
    ) -> p.DecoratorConfig | None:
        """Read validated decorator data without wrapping or invoking a callable.

        Returns:
            The validated decorator metadata, or None when none is attached.

        Raises:
            TypeError: If attached metadata does not satisfy its model contract.
            ValueError: If metadata requests an unsupported timeout or middleware.

        """
        metadata = getattr(candidate, c.HANDLER_ATTR, None)
        if metadata is None:
            return None
        if not isinstance(metadata, m.DecoratorConfig):
            raise TypeError(c.ERR_HANDLER_METADATA_INVALID.format(name=name))
        if metadata.timeout is not None or metadata.middleware:
            raise ValueError(c.ERR_HANDLER_POLICY_UNSUPPORTED)
        return metadata

    @classmethod
    def registry_registration_details(
        cls, binding: p.HandlerRegistration
    ) -> p.RegistrationDetails:
        """Project a normalized binding without losing the handler's declared mode.

        Returns:
            Registration details containing the normalized identifier and the
            declared or inferred handler mode.

        """
        handler = binding.handler
        return m.RegistrationDetails(
            registration_id=cls.registry_registration_name(binding),
            handler_mode=(
                handler.mode
                if isinstance(handler, p.Handler)
                else c.HandlerType.EVENT
                if binding.is_event
                else c.HandlerType.COMMAND
            ),
        )

    @classmethod
    def _resolve_runtime_settings(
        cls,
        runtime_options: p.RuntimeBootstrapOptions,
    ) -> p.Settings:
        """Return the injected provider or load the declared settings class."""
        settings = runtime_options.settings
        overrides = runtime_options.settings_overrides
        if settings is not None:
            return settings.clone(**overrides) if overrides else settings
        settings_type = (
            runtime_options.settings_type or FlextUtilitiesModel.settings_base()
        )
        return settings_type.fetch_global(overrides=overrides)

    @classmethod
    def build_service_runtime(
        cls,
        source: p.RuntimeBootstrapProvider
        | p.RuntimeBootstrapOptions
        | p.MixinsInfrastructure
        | None = None,
    ) -> p.ServiceRuntime:
        """Materialize settings, context, container and dispatcher for one component.

        The container scopes the injected container, or the shared container,
        preserving public DI registrations. Its settings are the runtime's
        actual elected provider. A dispatcher the options do not inject is the
        container's command bus; failing to resolve it raises with its cause.

        Returns:
            The resulting ``p.ServiceRuntime``.

        """
        options = cls.resolve_runtime_options(source)
        settings = cls._resolve_runtime_settings(options)
        context = (
            options.context
            if options.context is not None
            else FlextUtilitiesModel.context_type().create()
        )
        source_container = options.container
        if source_container is None:
            source_container = FlextUtilitiesModel.container_type().shared()
        container = source_container.scope(
            registration=m.ServiceRegistrationSpec(
                settings=settings,
                context=context,
                dispatcher=options.dispatcher,
            ),
        )
        return m.ServiceRuntime(
            settings=container.settings,
            context=container.context,
            container=container,
            dispatcher=container.dispatcher().unwrap(),
        )


__all__: list[str] = ["FlextUtilitiesModelRuntime"]
