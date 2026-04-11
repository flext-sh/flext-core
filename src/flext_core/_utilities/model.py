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
from types import ModuleType
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from flext_core._models.base import FlextModelsBase
from flext_core._models.service import FlextModelsService
from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.discovery import FlextUtilitiesDiscovery
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_protocol import FlextUtilitiesGuardsTypeProtocol
from flext_core.protocols import p
from flext_core.result import r
from flext_core.typings import t

if TYPE_CHECKING:
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.settings import FlextSettings

T_Model = TypeVar("T_Model", bound=BaseModel)


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    class ModelDumpOptions(FlextModelsBase.FlexibleInternalModel):
        """Options controlling Pydantic model_dump() serialization behavior."""

        by_alias: bool | None = Field(
            default=None,
            description="Serialize using field aliases",
        )
        exclude_none: bool | None = Field(
            default=None,
            description="Exclude None-valued fields",
        )
        exclude_unset: bool | None = Field(
            default=None,
            description="Exclude fields not explicitly set",
        )
        exclude_defaults: bool | None = Field(
            default=None,
            description="Exclude fields matching defaults",
        )
        include: set[str] | None = Field(
            default=None,
            description="Whitelist of field names to include",
        )
        exclude: set[str] | None = Field(
            default=None,
            description="Blacklist of field names to exclude",
        )

    @staticmethod
    def safe_get_attribute(
        obj: t.RuntimeData | type | ModuleType,
        attr: str,
        default: t.ValueOrModel | None = None,
    ) -> t.ValueOrModel | None:
        """Safe attribute access without raising AttributeError."""
        return getattr(obj, attr) if hasattr(obj, attr) else default

    @staticmethod
    def _normalize_model_input(
        data: BaseModel | Mapping[str, object] | t.ConfigMap,
    ) -> Mapping[str, object]:
        if isinstance(data, BaseModel):
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
        model: BaseModel,
        options: FlextUtilitiesModel.ModelDumpOptions | None = None,
        **kwargs: t.ValueOrModel,
    ) -> t.ScalarMapping:
        """Unified Pydantic serialization with options.

        Generic replacement for: model.model_dump() with consistent return type.

        Args:
            model: Pydantic model instance to serialize.
            options: Optional Pydantic model_dump arguments within the config model.
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
        model_cls: type[T_Model],
        data: BaseModel | Mapping[str, object] | t.ConfigMap,
    ) -> r[T_Model]:
        """Load a model from a mapping-like input using Pydantic validation."""
        return r[T_Model].create_from_callable(
            lambda: model_cls.model_validate(cls._normalize_model_input(data)),
        )

    @staticmethod
    def _settings_base() -> type[FlextSettings]:
        """Resolve FlextSettings lazily to avoid runtime import cycles."""
        settings_module = import_module("flext_core.settings")
        return settings_module.FlextSettings

    @staticmethod
    def _container_type() -> type[FlextContainer]:
        """Resolve FlextContainer lazily to avoid runtime import cycles."""
        container_module = import_module("flext_core.container")
        return container_module.FlextContainer

    @staticmethod
    def _context_type() -> type[FlextContext]:
        """Resolve FlextContext lazily to avoid runtime import cycles."""
        context_module = import_module("flext_core.context")
        return context_module.FlextContext

    @staticmethod
    def service_settings_type(
        service_or_cls: object,
    ) -> type[FlextSettings]:
        """Resolve the concrete settings type used by a service-like object."""
        settings_base = FlextUtilitiesModel._settings_base()
        candidate = getattr(service_or_cls, "config_type", None)
        if isinstance(candidate, type) and issubclass(candidate, settings_base):
            return candidate
        config_type_resolver = getattr(service_or_cls, "_get_service_config_type", None)
        if callable(config_type_resolver):
            resolved = config_type_resolver()
            if isinstance(resolved, type) and issubclass(resolved, settings_base):
                return resolved
        return settings_base

    @staticmethod
    def validate_value[TValue](
        target: TypeAdapter[TValue] | t.TypeHintSpecifier,
        data: object,
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> r[TValue]:
        """Validate one value through a model class or TypeAdapter."""
        try:
            adapter = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
            if from_json:
                if not isinstance(data, (str, bytes, bytearray)):
                    return r[TValue].fail(
                        "JSON validation requires str or bytes input",
                    )
                return r[TValue].ok(adapter.validate_json(data, strict=strict))
            return r[TValue].ok(adapter.validate_python(data, strict=strict))
        except (
            ValidationError,
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
        ) as exc:
            return r[TValue].fail(f"Validation failed: {exc}", exception=exc)

    @staticmethod
    def _runtime_option_updates_from_source(
        source: object,
    ) -> Mapping[str, t.ValueOrModel]:
        """Extract runtime bootstrap fields from a service-like instance."""
        field_map = {
            "runtime_config": "config",
            "initial_context": "context",
        }
        option_fields = (
            "runtime_config",
            "config_type",
            "config_overrides",
            "initial_context",
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
            | object
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
                resolved = FlextModelsService.RuntimeBootstrapOptions.model_validate(
                    dict(source),
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

    @classmethod
    def build_service_runtime(
        cls,
        source: (
            FlextModelsService.RuntimeBootstrapOptions
            | Mapping[str, t.ValueOrModel]
            | object
            | None
        ) = None,
        **overrides: t.ValueOrModel,
    ) -> FlextModelsService.ServiceRuntime:
        """Materialize config, context, and container from one runtime specification."""
        runtime_options = cls.resolve_runtime_options(source, **overrides)
        settings_base = cls._settings_base()
        context_type = cls._context_type()
        container_type = cls._container_type()
        config_instance = (
            runtime_options.config
            if isinstance(runtime_options.config, settings_base)
            else None
        )
        config_cls = (
            type(config_instance)
            if config_instance is not None
            else cls.service_settings_type(runtime_options.config_type)
        )
        config_overrides = (
            {
                key: value
                for key, value in runtime_options.config_overrides.items()
                if FlextUtilitiesGuardsTypeCore.scalar(value)
            }
            if runtime_options.config_overrides is not None
            else None
        )
        runtime_config: p.Settings
        if config_instance is not None:
            runtime_config = (
                config_instance.model_copy(update=config_overrides, deep=True)
                if config_overrides
                else config_instance
            )
        else:
            runtime_config = config_cls.fetch_global(overrides=config_overrides)
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
            config=runtime_config,
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
        return FlextModelsService.ServiceRuntime(
            config=runtime_config,
            context=runtime_container.context,
            container=runtime_container,
        )

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


__all__ = ["FlextUtilitiesModel"]
