"""Runtime DI builders + ``build_service_runtime`` orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

from flext_core import (
    FlextModelsService as ms,
    FlextUtilitiesDiscovery as ud,
    FlextUtilitiesGuardsTypeProtocol as ugp,
    m,
    p,
    t,
)
from flext_core._utilities.model_options import FlextUtilitiesModelOptions


class FlextUtilitiesModelRuntime(FlextUtilitiesModelOptions):
    """Runtime DSL: dispatcher, registry, and service-runtime construction."""

    @staticmethod
    def _resolve_runtime_dispatcher(
        runtime_options: ms.RuntimeBootstrapOptions,
        runtime_container: p.Container,
    ) -> p.Dispatcher | None:
        """Resolve the dispatcher from explicit options or the built container."""
        explicit_dispatcher = runtime_options.dispatcher
        if isinstance(explicit_dispatcher, p.Dispatcher):
            return explicit_dispatcher
        dispatcher_result = runtime_container.dispatcher()
        return None if dispatcher_result.failure else dispatcher_result.value

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
        runtime: ms.ServiceRuntime | None = None,
        auto_discover_handlers: bool = False,
    ) -> p.Registry:
        """Materialize the canonical registry implementation behind ``p.Registry``."""
        resolved_dispatcher = dispatcher or (runtime.dispatcher if runtime else None)
        registry_module = import_module("flext_core.registry")
        registry_candidate = registry_module.FlextRegistry.create(
            dispatcher=resolved_dispatcher,
            runtime=runtime,
            auto_discover_handlers=auto_discover_handlers,
        )
        if not isinstance(registry_candidate, p.Registry):
            msg = "Resolved registry implementation does not satisfy p.Registry"
            raise TypeError(msg)
        return registry_candidate

    @classmethod
    def _resolve_runtime_registry(
        cls,
        runtime_options: ms.RuntimeBootstrapOptions,
        runtime: ms.ServiceRuntime,
    ) -> p.Registry:
        """Resolve the registry from explicit options or the shared runtime DSL."""
        explicit_registry = runtime_options.registry
        if isinstance(explicit_registry, p.Registry):
            return explicit_registry
        return cls.build_registry(dispatcher=runtime.dispatcher, runtime=runtime)

    @classmethod
    def _resolve_runtime_settings(
        cls,
        runtime_options: ms.RuntimeBootstrapOptions,
    ) -> p.Settings:
        """Resolve the runtime settings instance + apply overrides."""
        settings_instance = (
            runtime_options.settings
            if isinstance(runtime_options.settings, p.Settings)
            else None
        )
        settings_cls = (
            type(settings_instance)
            if settings_instance is not None
            else cls.service_settings_type(runtime_options.settings_type)
        )
        settings_overrides = cls._normalize_runtime_override_mapping(
            runtime_options.settings_overrides,
        )
        if settings_instance is not None:
            return (
                settings_instance.model_copy(update=settings_overrides, deep=True)
                if settings_overrides
                else settings_instance
            )
        fetch_global = getattr(settings_cls, "fetch_global")
        candidate = fetch_global(overrides=settings_overrides)
        if not isinstance(candidate, p.Settings):
            msg = "Resolved settings class returned non-settings instance"
            raise TypeError(msg)
        return candidate

    @classmethod
    def _build_runtime_container(
        cls,
        runtime_options: ms.RuntimeBootstrapOptions,
        runtime_settings: p.Settings,
        runtime_context: p.Context,
    ) -> p.Container:
        """Construct + wire the runtime container from resolved options."""
        container_type = cls._container_type()
        bootstrap_services = {
            name: service
            for name, service in (runtime_options.services or {}).items()
            if ugp.registerable_service(service)
        }
        wire_modules, wire_packages, wire_classes = ud.resolve_wire_targets(
            runtime_options.wire_modules,
            runtime_options.wire_packages,
            runtime_options.wire_classes,
        )
        runtime_container = container_type.shared().scope(
            subproject=runtime_options.subproject,
            registration=m.ServiceRegistrationSpec.model_validate({
                "settings": runtime_settings,
                "context": runtime_context,
                "services": bootstrap_services,
                "factories": runtime_options.factories,
                "resources": runtime_options.resources,
            }),
        )
        normalized_container_overrides = cls._normalize_runtime_override_mapping(
            runtime_options.container_overrides,
        )
        if normalized_container_overrides:
            runtime_container.apply(normalized_container_overrides)
        if wire_modules or wire_packages or wire_classes:
            runtime_container.wire(
                modules=wire_modules,
                packages=wire_packages,
                classes=wire_classes,
            )
        return runtime_container

    @classmethod
    def build_service_runtime(
        cls,
        source: (
            ms.RuntimeBootstrapOptions | Mapping[str, t.JsonPayload] | p.Base | None
        ) = None,
        **overrides: t.JsonPayload,
    ) -> ms.ServiceRuntime:
        """Materialize settings, context, and container from one runtime specification."""
        runtime_options = cls.resolve_runtime_options(source, **overrides)
        context_type = cls._context_type()
        runtime_settings = cls._resolve_runtime_settings(runtime_options)
        runtime_context = (
            runtime_options.context
            if ugp.context(runtime_options.context)
            else context_type.create()
        )
        runtime_container = cls._build_runtime_container(
            runtime_options,
            runtime_settings,
            runtime_context,
        )
        runtime_container_context = getattr(runtime_container, "context", None)
        resolved_context = (
            runtime_container_context
            if ugp.context(runtime_container_context)
            else runtime_context
        )
        runtime_dispatcher = cls._resolve_runtime_dispatcher(
            runtime_options,
            runtime_container,
        )
        service_runtime = ms.ServiceRuntime(
            settings=runtime_settings,
            context=resolved_context,
            container=runtime_container,
            dispatcher=runtime_dispatcher,
        )
        runtime_registry = cls._resolve_runtime_registry(
            runtime_options, service_runtime
        )
        return service_runtime.model_copy(update={"registry": runtime_registry})


__all__: list[str] = ["FlextUtilitiesModelRuntime"]
