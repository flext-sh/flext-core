"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from importlib import import_module

from flext_core import (
    FlextModelsBase as m,
    FlextModelsPydantic as mp,
    FlextModelsService as ms,
    FlextUtilitiesArgs as ua,
    FlextUtilitiesDiscovery as ud,
    FlextUtilitiesGuardsTypeProtocol as ugp,
    FlextUtilitiesPydantic as up,
    c,
    e,
    p,
    r,
    t,
)


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    class ModelDumpOptions(m.FlexibleInternalModel):
        """Options controlling Pydantic model_dump() serialization behavior."""

        by_alias: bool | None = up.Field(
            None,
            description="Serialize using field aliases",
            validate_default=True,
        )
        exclude_none: bool | None = up.Field(
            None,
            description="Exclude None-valued fields",
            validate_default=True,
        )
        exclude_unset: bool | None = up.Field(
            None,
            description="Exclude fields not explicitly set",
            validate_default=True,
        )
        exclude_defaults: bool | None = up.Field(
            None,
            description="Exclude fields matching defaults",
            validate_default=True,
        )
        include: set[str] | None = up.Field(
            None,
            description="Whitelist of field names to include",
            validate_default=True,
        )
        exclude: set[str] | None = up.Field(
            None,
            description="Blacklist of field names to exclude",
            validate_default=True,
        )

    @staticmethod
    def dump(
        model: mp.BaseModel,
        options: FlextUtilitiesModel.ModelDumpOptions | None = None,
        **kwargs: t.JsonPayload,
    ) -> Mapping[str, t.JsonPayload]:
        """Unified Pydantic serialization with options.

        Generic replacement for: model.model_dump() with consistent return type.

        Args:
            model: Pydantic model instance to serialize.
            options: Optional Pydantic model_dump arguments within the settings model.
            **kwargs: Inline fallback serialization arguments mapped to ModelDumpOptions automatically.

        Returns:
            Dictionary representation of the model.

        """
        opts = ua.resolve_options(
            options,
            kwargs,
            FlextUtilitiesModel.ModelDumpOptions,
        ).unwrap_or(FlextUtilitiesModel.ModelDumpOptions())
        opts_dict = opts.model_dump(exclude_none=True)
        return model.model_dump(**opts_dict)

    # DEPRECATED: load method removed - depends on _normalize_model_input
    # @classmethod
    # def load(
    #     cls,
    #     model_cls: t.ModelClass[T_Model],
    #     data: mp.BaseModel | Mapping[str, t.JsonPayload] | m.ConfigMap,
    # ) -> p.Result[T_Model]:
    #     """Load a model from a mapping-like input using Pydantic validation."""
    #     return r[T_Model].create_from_callable(
    #         lambda: model_cls.model_validate(cls._normalize_model_input(data)),
    #     )

    @staticmethod
    def _settings_base() -> t.SettingsClass:
        """Resolve FlextSettings lazily to avoid runtime import cycles."""
        settings_module = import_module("flext_core")
        settings_cls: t.SettingsClass = settings_module.FlextSettings
        return settings_cls

    @staticmethod
    def _container_type() -> p.ContainerType:
        """Resolve FlextContainer lazily to avoid runtime import cycles."""
        container_module = import_module("flext_core")
        container_cls: p.ContainerType = container_module.FlextContainer
        return container_cls

    @staticmethod
    def _context_type() -> p.ContextType:
        """Resolve FlextContext lazily to avoid runtime import cycles."""
        context_module = import_module("flext_core")
        context_cls: p.ContextType = context_module.FlextContext
        return context_cls

    @staticmethod
    def _runtime_type() -> type:
        """Resolve FlextRuntime lazily to avoid runtime import cycles."""
        runtime_module = import_module("flext_core")
        runtime_cls: type = runtime_module.FlextRuntime
        return runtime_cls

    @classmethod
    def _normalize_runtime_override_mapping(
        cls,
        value: Mapping[str, t.JsonPayload | t.Scalar] | None,
    ) -> t.JsonMapping | None:
        """Normalize runtime override mappings to canonical JsonMapping."""
        if value is None:
            return None
        runtime_type = cls._runtime_type()
        validated: t.JsonMapping = t.json_mapping_adapter().validate_python(
            {
                str(key): runtime_type.normalize_to_metadata(item)
                for key, item in value.items()
            },
        )
        return validated

    @staticmethod
    def service_settings_type(
        service_or_cls: p.Base | p.SettingsType | t.SettingsClass,
    ) -> t.SettingsClass:
        """Resolve the concrete settings type used by a service-like object."""
        settings_base = FlextUtilitiesModel._settings_base()
        if isinstance(
            service_or_cls, type
        ) and FlextUtilitiesModel._matches_settings_type(
            service_or_cls,
            settings_base,
        ):
            return service_or_cls
        candidate = getattr(service_or_cls, "settings_type", None)
        if isinstance(candidate, type) and FlextUtilitiesModel._matches_settings_type(
            candidate,
            settings_base,
        ):
            return candidate
        settings_type_resolver = getattr(
            service_or_cls, "_get_service_settings_type", None
        )
        if callable(settings_type_resolver):
            resolved = settings_type_resolver()
            if isinstance(
                resolved, type
            ) and FlextUtilitiesModel._matches_settings_type(
                resolved,
                settings_base,
            ):
                return resolved
        return settings_base

    @staticmethod
    def _matches_settings_type(
        candidate: type,
        settings_base: t.SettingsClass,
    ) -> bool:
        """Check whether candidate behaves as a settings class.

        Avoid direct Protocol subclass checks when runtime-checkable protocols
        include non-method members, which raises ``TypeError`` in ``issubclass``.
        """
        _ = settings_base
        fetch_global = getattr(candidate, "fetch_global", None)
        model_copy = getattr(candidate, "model_copy", None)
        return callable(fetch_global) and callable(model_copy)

    @staticmethod
    def validate_value[TValue](
        target: t.ValueAdapter[TValue] | t.TypeHintSpecifier,
        data: t.JsonPayload,
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> p.Result[TValue]:
        """Validate one value through a model class or TypeAdapter."""
        try:
            adapter = (
                target if isinstance(target, mp.TypeAdapter) else mp.TypeAdapter(target)
            )
            if from_json:
                if not isinstance(data, (str, bytes, bytearray)):
                    return e.fail_validation(
                        "json_input",
                        error="JSON validation requires str or bytes input",
                    )
                return r[TValue].ok(adapter.validate_json(data, strict=strict))
            return r[TValue].ok(adapter.validate_python(data, strict=strict))
        except (
            c.ValidationError,
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
        ) as exc:
            return e.fail_validation(error=exc)

    @staticmethod
    def _runtime_option_updates_from_source(
        source: p.Base,
    ) -> Mapping[str, t.JsonPayload]:
        """Extract runtime bootstrap fields from a service-like instance."""
        field_map = {
            "runtime_settings": "settings",
            "initial_context": "context",
            "runtime_dispatcher": "dispatcher",
            "runtime_registry": "registry",
        }
        option_fields = (
            "runtime_settings",
            "settings_type",
            "settings_overrides",
            "initial_context",
            "dispatcher",
            "registry",
            "runtime_dispatcher",
            "runtime_registry",
            "subproject",
            "services",
            "factories",
            "resources",
            "container_overrides",
            "wire_modules",
            "wire_packages",
            "wire_classes",
        )
        updates: dict[str, t.JsonPayload] = {}
        for attr_name in option_fields:
            value = getattr(source, attr_name, None)
            if value is not None:
                updates[field_map.get(attr_name, attr_name)] = value
        return updates

    @classmethod
    def resolve_runtime_options(
        cls,
        source: (
            ms.RuntimeBootstrapOptions | Mapping[str, t.JsonPayload] | p.Base | None
        ) = None,
        **overrides: t.JsonPayload,
    ) -> ms.RuntimeBootstrapOptions:
        """Resolve runtime options from models, mappings, or service instances."""
        resolved = ms.RuntimeBootstrapOptions()
        if source is not None:
            if isinstance(source, ms.RuntimeBootstrapOptions):
                resolved = source
            elif isinstance(source, Mapping):
                source_dict = dict(source)
                try:
                    resolved = ms.RuntimeBootstrapOptions.model_validate(
                        source_dict,
                    )
                except c.ValidationError:
                    sanitized_source: dict[str, t.JsonPayload | None] = dict(
                        source_dict
                    )
                    wire_packages_raw = sanitized_source.get("wire_packages")
                    if isinstance(wire_packages_raw, (list, tuple)):
                        normalized_wire_packages: list[t.JsonValue] = [
                            wire_pkg
                            for wire_pkg in wire_packages_raw
                            if isinstance(wire_pkg, str)
                        ]
                        if len(normalized_wire_packages) == len(wire_packages_raw):
                            sanitized_source["wire_packages"] = normalized_wire_packages
                        else:
                            sanitized_source["wire_packages"] = None
                    resolved = ms.RuntimeBootstrapOptions.model_validate(
                        sanitized_source,
                    )
            else:
                options_resolver = getattr(source, "_runtime_bootstrap_options", None)
                if callable(options_resolver):
                    raw_options = options_resolver()
                    if raw_options is not None:
                        resolved = cls.resolve_runtime_options(raw_options)
                source_updates = cls._runtime_option_updates_from_source(source)
                if source_updates:
                    resolved = resolved.model_copy(update=source_updates)
        if overrides:
            override_options = ms.RuntimeBootstrapOptions.model_validate(
                overrides,
            )
            override_updates = {
                field: getattr(override_options, field)
                for field in ms.RuntimeBootstrapOptions.model_fields
                if getattr(override_options, field) is not None
            }
            if override_updates:
                resolved = resolved.model_copy(update=override_updates)
        return resolved

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
        """Materialize the canonical dispatcher implementation behind `p.Dispatcher`."""
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
        """Materialize the canonical registry implementation behind `p.Registry`."""
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
        container_type = cls._container_type()
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
        runtime_settings: p.Settings
        if settings_instance is not None:
            runtime_settings = (
                settings_instance.model_copy(update=settings_overrides, deep=True)
                if settings_overrides
                else settings_instance
            )
        else:
            fetch_global = getattr(settings_cls, "fetch_global")
            runtime_settings_candidate = fetch_global(overrides=settings_overrides)
            if not isinstance(runtime_settings_candidate, p.Settings):
                msg = "Resolved settings class returned non-settings instance"
                raise TypeError(msg)
            runtime_settings = runtime_settings_candidate
        runtime_context = (
            runtime_options.context
            if ugp.context(runtime_options.context)
            else context_type.create()
        )
        bootstrap_services = ugp.filter_registerable_services(
            runtime_options.services,
        )
        wire_modules, wire_packages, wire_classes = ud.resolve_wire_targets(
            runtime_options.wire_modules,
            runtime_options.wire_packages,
            runtime_options.wire_classes,
        )
        runtime_container = container_type.shared().scope(
            settings=runtime_settings,
            context=runtime_context,
            subproject=runtime_options.subproject,
            services=bootstrap_services,
            factories=runtime_options.factories,
            resources=runtime_options.resources,
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


__all__: list[str] = ["FlextUtilitiesModel"]
