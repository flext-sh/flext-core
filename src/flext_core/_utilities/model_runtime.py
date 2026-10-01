"""Runtime DI builders + ``build_service_runtime`` orchestration."""

from __future__ import annotations

from importlib import import_module

from flext_core import m, p

from .model_options import FlextUtilitiesModelOptions


class FlextUtilitiesModelRuntime(FlextUtilitiesModelOptions):
    """Runtime DSL: dispatcher, registry, and service-runtime construction."""

    @classmethod
    def build_dispatcher(cls) -> p.Dispatcher:
        """Materialize the canonical dispatcher implementation behind ``p.Dispatcher``."""
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
        """Materialize the canonical registry implementation behind ``p.Registry``."""
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
        runtime_options: m.RuntimeBootstrapOptions,
    ) -> p.Settings:
        """Return the injected settings, or load the declared settings class."""
        settings = runtime_options.settings
        overrides = runtime_options.settings_overrides
        if settings is not None:
            return settings.clone(**overrides) if overrides else settings
        settings_type = runtime_options.settings_type or cls._settings_base()
        return settings_type.fetch_global(overrides=overrides)

    @classmethod
    def build_service_runtime(
        cls,
        source: m.RuntimeBootstrapOptions | p.MixinsInfrastructure | None = None,
    ) -> m.ServiceRuntime:
        """Materialize settings, context, container and dispatcher for one component.

        The container is a scope of the shared container bound to the resolved
        settings and context. A dispatcher the options do not inject is the
        container's command bus; failing to resolve it raises with its cause.
        """
        options = cls.resolve_runtime_options(source)
        settings = cls._resolve_runtime_settings(options)
        context = (
            options.context
            if options.context is not None
            else cls._context_type().create()
        )
        container = (
            cls
            ._container_type()
            .shared()
            .scope(
                registration=m.ServiceRegistrationSpec(
                    settings=settings,
                    context=context,
                ),
            )
        )
        dispatcher = (
            options.dispatcher
            if options.dispatcher is not None
            else container.dispatcher().unwrap()
        )
        return m.ServiceRuntime(
            settings=settings,
            context=container.context,
            container=container,
            dispatcher=dispatcher,
        )


__all__: list[str] = ["FlextUtilitiesModelRuntime"]
