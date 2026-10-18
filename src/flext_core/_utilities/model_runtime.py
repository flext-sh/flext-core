"""Runtime DI builders + ``build_service_runtime`` orchestration.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib import import_module

from flext_core import FlextRuntime, m, p
from flext_core._models import FlextModelsNamespace
from flext_core._utilities import FlextUtilitiesModelOptions


class FlextUtilitiesModelRuntime(
    FlextUtilitiesModelOptions,
    FlextRuntime,
    FlextModelsNamespace,
):
    """Runtime DSL: dispatcher, registry, and service-runtime construction.

    The facade receives the runtime metadata helpers (``u.normalize_to_metadata``)
    and namespace enforcement through this declared owner: the facade generator
    composes only classes the ``_utilities`` family declares.
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
        auto_discover_handlers: bool = False,
    ) -> p.Registry:
        """Materialize the canonical registry implementation behind ``p.Registry``.

        Returns:
            The resulting ``p.Registry``.

        Raises:
            TypeError: If Resolved registry implementation does not satisfy p.Registry.

        """
        registry_module = import_module("flext_core.registry")
        registry_candidate = registry_module.FlextRegistry.create(
            dispatcher=dispatcher,
            auto_discover_handlers=auto_discover_handlers,
        )
        if not isinstance(registry_candidate, p.Registry):
            msg = "Resolved registry implementation does not satisfy p.Registry"
            raise TypeError(msg)
        return registry_candidate

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
        settings_type = runtime_options.settings_type or cls._settings_base()
        return settings_type.fetch_global(overrides=overrides)

    @classmethod
    def build_service_runtime(
        cls,
        source: p.RuntimeBootstrapOptions | p.MixinsInfrastructure | None = None,
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
            else cls._context_type().create()
        )
        source_container = options.container
        if source_container is None:
            source_container = cls._container_type().shared()
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
