"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Literal

from flext_core import T_Model, c, e, p, r, t
from flext_core._constants.pydantic import FlextConstantsPydantic
from flext_core._models.base import FlextModelsBase
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._models.service import FlextModelsService
from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.discovery import FlextUtilitiesDiscovery
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
from flext_core._utilities.guards_type_protocol import FlextUtilitiesGuardsTypeProtocol
from flext_core._utilities.pydantic import FlextUtilitiesPydantic


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    class ModelDumpOptions(FlextModelsBase.FlexibleInternalModel):
        """Options controlling Pydantic model_dump() serialization behavior."""

        by_alias: bool | None = FlextUtilitiesPydantic.Field(
            default=None,
            description="Serialize using field aliases",
        )
        exclude_none: bool | None = FlextUtilitiesPydantic.Field(
            default=None,
            description="Exclude None-valued fields",
        )
        exclude_unset: bool | None = FlextUtilitiesPydantic.Field(
            default=None,
            description="Exclude fields not explicitly set",
        )
        exclude_defaults: bool | None = FlextUtilitiesPydantic.Field(
            default=None,
            description="Exclude fields matching defaults",
        )
        include: set[str] | None = FlextUtilitiesPydantic.Field(
            default=None,
            description="Whitelist of field names to include",
        )
        exclude: set[str] | None = FlextUtilitiesPydantic.Field(
            default=None,
            description="Blacklist of field names to exclude",
        )

    @staticmethod
    def safe_get_attribute(
        obj: p.Base | Mapping[str, t.ValueOrModel] | type,
        attr: str,
        default: t.ValueOrModel | None = None,
    ) -> t.ValueOrModel | None:
        """Safe attribute access without raising AttributeError."""
        return getattr(obj, attr) if hasattr(obj, attr) else default

    @staticmethod
    def _normalize_model_input(
        data: t.ModelCarrier | Mapping[str, t.ValueOrModel] | t.ConfigMap,
    ) -> Mapping[str, t.ValueOrModel]:
        if FlextUtilitiesGuardsTypeModel.pydantic_model(data):
            root_value = getattr(data, "root", None)
            if isinstance(
                root_value,
                Mapping,
            ) and FlextUtilitiesGuardsTypeCore.mapping(root_value):
                return {str(key): value for key, value in root_value.items()}
            dumped = data.model_dump()
            return {str(key): value for key, value in dumped.items()}
        return {str(key): value for key, value in data.items()}

    @staticmethod
    def dump(
        model: t.ModelCarrier,
        options: FlextUtilitiesModel.ModelDumpOptions | None = None,
        **kwargs: t.ValueOrModel,
    ) -> Mapping[str, t.ValueOrModel]:
        """Unified Pydantic serialization with options.

        Generic replacement for: model.model_dump() with consistent return type.

        Args:
            model: Pydantic model instance to serialize.
            options: Optional Pydantic model_dump arguments within the settings model.
            **kwargs: Inline fallback serialization arguments mapped to ModelDumpOptions automatically.

        Returns:
            Dictionary representation of the model.

        """
        opts = FlextUtilitiesArgs.resolve_options(
            options,
            kwargs,
            FlextUtilitiesModel.ModelDumpOptions,
        ).unwrap_or(FlextUtilitiesModel.ModelDumpOptions())
        opts_dict = opts.model_dump(exclude_none=True)
        return model.model_dump(**opts_dict)

    @classmethod
    def load(
        cls,
        model_cls: t.ModelClass[T_Model],
        data: t.ModelCarrier | Mapping[str, t.ValueOrModel] | t.ConfigMap,
    ) -> r[T_Model]:
        """Load a model from a mapping-like input using Pydantic validation."""
        return r[T_Model].create_from_callable(
            lambda: model_cls.model_validate(cls._normalize_model_input(data)),
        )

    @staticmethod
    def _settings_base() -> t.SettingsClass:
        """Resolve FlextSettings lazily to avoid runtime import cycles."""
        settings_module = import_module("flext_core.settings")
        return settings_module.FlextSettings

    @staticmethod
    def _container_type() -> p.ContainerType:
        """Resolve FlextContainer lazily to avoid runtime import cycles."""
        container_module = import_module("flext_core.container")
        return container_module.FlextContainer

    @staticmethod
    def _context_type() -> p.ContextType:
        """Resolve FlextContext lazily to avoid runtime import cycles."""
        context_module = import_module("flext_core.context")
        return context_module.FlextContext

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
        try:
            return issubclass(candidate, settings_base)
        except TypeError:
            fetch_global = getattr(candidate, "fetch_global", None)
            return callable(fetch_global)

    @staticmethod
    def validate_value[TValue](
        target: t.ValueAdapter[TValue] | t.TypeHintSpecifier,
        data: t.ValueOrModel,
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> r[TValue]:
        """Validate one value through a model class or TypeAdapter."""
        try:
            adapter = (
                target
                if isinstance(target, FlextModelsPydantic.TypeAdapter)
                else FlextModelsPydantic.TypeAdapter(target)
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
            FlextConstantsPydantic.ValidationError,
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
        ) as exc:
            return e.fail_validation(error=exc)

    @staticmethod
    def _runtime_option_updates_from_source(
        source: p.Base,
    ) -> Mapping[str, t.ValueOrModel]:
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
        updates: dict[str, t.ValueOrModel] = {}
        for attr_name in option_fields:
            value = getattr(source, attr_name, None)
            if value is not None:
                updates[field_map.get(attr_name, attr_name)] = value
        return updates

    @classmethod
    def resolve_runtime_options(
        cls,
        source: (
            FlextModelsService.RuntimeBootstrapOptions
            | Mapping[str, t.ValueOrModel]
            | p.Base
            | None
        ) = None,
        **overrides: t.ValueOrModel,
    ) -> FlextModelsService.RuntimeBootstrapOptions:
        """Resolve runtime options from models, mappings, or service instances."""
        resolved = FlextModelsService.RuntimeBootstrapOptions()
        if source is not None:
            if isinstance(source, FlextModelsService.RuntimeBootstrapOptions):
                resolved = source
            elif isinstance(source, Mapping):
                source_dict = dict(source)
                try:
                    resolved = (
                        FlextModelsService.RuntimeBootstrapOptions.model_validate(
                            source_dict,
                        )
                    )
                except FlextConstantsPydantic.ValidationError:
                    sanitized_source = dict(source_dict)
                    wire_packages_raw = sanitized_source.get("wire_packages")
                    if isinstance(wire_packages_raw, (list, tuple)):
                        has_only_strings = all(
                            isinstance(wire_pkg, str) for wire_pkg in wire_packages_raw
                        )
                        if has_only_strings:
                            sanitized_source["wire_packages"] = list(wire_packages_raw)
                        else:
                            sanitized_source["wire_packages"] = None
                    resolved = (
                        FlextModelsService.RuntimeBootstrapOptions.model_validate(
                            sanitized_source,
                        )
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
            override_options = (
                FlextModelsService.RuntimeBootstrapOptions.model_validate(
                    overrides,
                )
            )
            override_updates = {
                field: getattr(override_options, field)
                for field in FlextModelsService.RuntimeBootstrapOptions.model_fields
                if getattr(override_options, field) is not None
            }
            if override_updates:
                resolved = resolved.model_copy(update=override_updates)
        return resolved

    @staticmethod
    def _resolve_runtime_dispatcher(
        runtime_options: FlextModelsService.RuntimeBootstrapOptions,
        runtime_container: p.Container,
    ) -> p.Dispatcher | None:
        """Resolve the dispatcher from explicit options or the built container."""
        explicit_dispatcher = runtime_options.dispatcher
        if isinstance(explicit_dispatcher, p.Dispatcher):
            return explicit_dispatcher
        dispatcher_result = runtime_container.get(c.ServiceName.COMMAND_BUS)
        if dispatcher_result.failure:
            return None
        dispatcher_candidate = dispatcher_result.value
        if isinstance(dispatcher_candidate, p.Dispatcher):
            return dispatcher_candidate
        return None

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
        runtime: FlextModelsService.ServiceRuntime | None = None,
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
        runtime_options: FlextModelsService.RuntimeBootstrapOptions,
        runtime: FlextModelsService.ServiceRuntime,
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
            FlextModelsService.RuntimeBootstrapOptions
            | Mapping[str, t.ValueOrModel]
            | p.Base
            | None
        ) = None,
        **overrides: t.ValueOrModel,
    ) -> FlextModelsService.ServiceRuntime:
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
        settings_overrides = (
            {
                key: value
                for key, value in runtime_options.settings_overrides.items()
                if FlextUtilitiesGuardsTypeCore.scalar(value)
            }
            if runtime_options.settings_overrides is not None
            else None
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
            if FlextUtilitiesGuardsTypeProtocol.context(runtime_options.context)
            else context_type.create()
        )
        bootstrap_services = (
            FlextUtilitiesGuardsTypeProtocol.filter_registerable_services(
                runtime_options.services,
            )
        )
        wire_modules, wire_packages, wire_classes = (
            FlextUtilitiesDiscovery.resolve_wire_targets(
                runtime_options.wire_modules,
                runtime_options.wire_packages,
                runtime_options.wire_classes,
            )
        )
        runtime_container = container_type.create().scoped(
            settings=runtime_settings,
            context=runtime_context,
            subproject=runtime_options.subproject,
            services=bootstrap_services,
            factories=runtime_options.factories,
            resources=runtime_options.resources,
        )
        if runtime_options.container_overrides:
            runtime_container.configure(runtime_options.container_overrides)
        if wire_modules or wire_packages or wire_classes:
            runtime_container.wire_modules(
                modules=wire_modules,
                packages=wire_packages,
                classes=wire_classes,
            )
        runtime_container_context = getattr(runtime_container, "context", None)
        resolved_context = (
            runtime_container_context
            if FlextUtilitiesGuardsTypeProtocol.context(runtime_container_context)
            else runtime_context
        )
        runtime_dispatcher = cls._resolve_runtime_dispatcher(
            runtime_options,
            runtime_container,
        )
        service_runtime = FlextModelsService.ServiceRuntime(
            settings=runtime_settings,
            context=resolved_context,
            container=runtime_container,
            dispatcher=runtime_dispatcher,
        )
        runtime_registry = cls._resolve_runtime_registry(
            runtime_options, service_runtime
        )
        return service_runtime.model_copy(update={"registry": runtime_registry})

    @staticmethod
    def service_context_scope(
        service_name: str,
        version: str | None = None,
    ) -> AbstractContextManager[None]:
        """Wrap the canonical service context manager as one central DSL entrypoint."""
        return FlextUtilitiesModel._context_type().Service.service_context(
            service_name,
            version,
        )

    @staticmethod
    def append_metadata_sequence_item(
        metadata: t.Dict,
        key: Literal["failed_items", "warning_items"],
        item: t.ValueOrModel,
    ) -> None:
        """Append one normalized item to a metadata sequence bucket."""
        raw_items = metadata.root.get(key)
        result_list: t.MutableContainerList = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if isinstance(
                    raw_item,
                    (str, int, float, bool, datetime, Path, list, dict, tuple),
                ):
                    result_list.append(raw_item)
                elif raw_item is not None:
                    result_list.append(str(raw_item))
        if isinstance(item, (str, int, float, bool, datetime, Path, list, dict, tuple)):
            result_list.append(item)
        elif item is not None:
            result_list.append(str(item))
        metadata.root[key] = result_list

    @staticmethod
    def upsert_skip_reason(
        metadata: t.Dict,
        item: t.ValueOrModel,
        reason: str,
    ) -> None:
        """Store one skip reason keyed by the stringified item representation."""
        raw_reasons = metadata.root.get("skip_reasons", {})
        reasons: t.MutableStrMapping = {}
        if isinstance(raw_reasons, Mapping):
            reasons = {str(key): str(value) for key, value in raw_reasons.items()}
        reasons[str(item)] = reason
        metadata.root["skip_reasons"] = reasons


__all__: list[str] = ["FlextUtilitiesModel"]
