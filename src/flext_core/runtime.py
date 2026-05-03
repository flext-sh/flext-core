"""Runtime bridge exposing external libraries with dispatcher-safe boundaries.

**ARCHITECTURE LAYER 0.5** - Integration Bridge (Minimal Dependencies)

This module provides runtime utilities that consume patterns from c and
expose external library APIs to higher-level modules, maintaining proper dependency
hierarchy while eliminating code duplication. Implements structural typing via
p (duck typing - no inheritance required).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import inspect
from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set as AbstractSet,
)
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import (
    ClassVar,
    Literal,
    no_type_check,
)

from dependency_injector import containers, providers, wiring
from pydantic import BaseModel, ConfigDict

from flext_core import (
    FlextConstants as c,
    FlextModelsContainers as mc,
    FlextModelsPydantic as mp,
    FlextProtocols as p,
    FlextTypes as t,
    FlextUtilitiesGuardsTypeCore as ugc,
)
from flext_core._utilities.guards_type_model import (
    FlextUtilitiesGuardsTypeModel as ugm,
)


class FlextRuntime:
    """Expose runtime normalization, DI, and validation helpers to higher layers."""

    Metadata: ClassVar[type[p.Metadata] | None] = None

    @classmethod
    def _require_metadata_model(cls) -> type[p.Metadata]:
        """Return the bound metadata model class or raise a runtime contract error."""
        metadata_cls = cls.Metadata
        if metadata_cls is None:
            msg = c.ERR_RUNTIME_METADATA_MODEL_NOT_BOUND
            raise RuntimeError(msg)
        return metadata_cls

    @staticmethod
    def create_instance[T](class_type: type[T]) -> T:
        """Type-safe factory for creating instances via object.__new__.

        Args:
            class_type: The class to instantiate

        Returns:
            An instance of type T

        Raises:
            TypeError: If object.__new__() does not return instance of expected type

        Example:
            >>> instance = FlextRuntime.create_instance(MyClass)
            >>> # instance is properly typed as MyClass

        """
        instance = object.__new__(class_type)
        if not isinstance(instance, class_type):
            msg = f"object.__new__ did not return instance of {class_type.__name__}"
            raise TypeError(msg)
        return instance

    @staticmethod
    def ensure_utc_datetime(value: datetime | None) -> datetime | None:
        """Attach UTC timezone to naive datetimes while preserving None."""
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        return providers

    @staticmethod
    def to_scalar(item: t.GuardInput | None) -> t.Scalar:
        """Coerce any runtime value to ``t.Scalar`` (flat Container invariant)."""
        if item is None:
            return ""
        return item if isinstance(item, t.SCALAR_TYPES) else str(item)

    @staticmethod
    def normalize_to_json_value(
        value: t.JsonPayload
        | t.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[t.Scalar]
        | p.Model
        | p.HasModelDump
        | None,
    ) -> t.JsonValue:
        """Normalize arbitrary runtime input to one validated ``JsonValue``."""
        normalized: t.JsonValue | t.MappingKV[str, t.JsonPayload] | t.Scalar
        if value is None:
            normalized = ""
        elif ugm.has_model_dump(value):
            normalized = value.model_dump()
        elif isinstance(value, p.Model):
            normalized = str(value)
        else:
            normalized = FlextRuntime.normalize_to_metadata(value)
        validated: t.JsonValue = t.json_value_adapter().validate_python(normalized)
        return validated

    @staticmethod
    def normalize_to_json_mapping(
        value: t.MappingKV[str, t.JsonPayload | t.Scalar],
    ) -> t.JsonMapping:
        """Normalize a mapping to a validated ``JsonMapping``."""
        return FlextRuntime._normalize_dict_entries(
            [(key, item) for key, item in value.items()],
        )

    @staticmethod
    def _normalize_dict_entries(
        items: t.SequenceOf[tuple[str, t.JsonPayload | t.Scalar]],
    ) -> dict[str, t.JsonValue]:
        """Normalize key-value pairs for container dict construction."""
        return dict(
            t.json_mapping_adapter().validate_python(
                {
                    key: FlextRuntime.normalize_to_json_value(item)
                    for key, item in items
                },
            ),
        )

    @staticmethod
    def normalize_model_input_mapping(
        value: p.HasModelDump | mc.Dict | t.ScalarMapping | None,
    ) -> t.MappingKV[str, t.JsonPayload] | None:
        """Normalize model-like input to a plain mapping."""
        if value is None:
            return None
        if isinstance(value, mc.Dict):
            raw_mapping: t.MappingKV[str, t.JsonPayload | t.Scalar] = value.root
        elif ugm.has_model_dump(value):
            dumped_mapping = value.model_dump()
            if not isinstance(dumped_mapping, Mapping):
                msg = c.ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE
                raise TypeError(msg)
            raw_mapping = dumped_mapping
        else:
            raw_mapping = value
        return FlextRuntime._normalize_dict_entries(
            [(key, item) for key, item in raw_mapping.items()],
        )

    @staticmethod
    def normalize_metadata_input_mapping(
        value: t.MappingKV[str, t.JsonPayload | None] | p.HasModelDump | None,
    ) -> t.MappingKV[str, t.JsonPayload | None] | None:
        """Normalize mapping-like metadata input while preserving explicit None.

        Defensively handles broken ``model_dump`` overrides that return a
        non-Mapping value by raising ``TypeError`` instead of surfacing an
        opaque ``AttributeError`` from downstream iteration.
        """
        if value is None:
            return None
        if isinstance(value, Mapping):
            raw: t.MappingKV[str, t.JsonPayload | None] = value
        else:
            raw_dump = value.model_dump()
            if not isinstance(raw_dump, Mapping):
                msg = c.ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE
                raise TypeError(msg)
            raw = raw_dump
        return {
            key: (None if item is None else FlextRuntime.normalize_to_json_value(item))
            for key, item in raw.items()
        }

    @staticmethod
    def validate_metadata_attributes(
        value: t.MetadataInput,
    ) -> t.MappingKV[str, t.JsonValue]:
        """Normalize and validate metadata attributes input.

        Defensively asserts that a BaseModel's ``model_dump`` actually yields
        a Mapping — broken models that violate the :class:`p.HasModelDump`
        protocol (e.g. returning a scalar) MUST surface as ``TypeError`` and
        not as opaque ``AttributeError`` from downstream iteration.
        """
        if value is None:
            return {}
        if not (isinstance(value, Mapping) or ugm.has_model_dump(value)):
            msg = c.ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE
            raise TypeError(msg)
        normalized_result = FlextRuntime.normalize_metadata_input_mapping(value)
        if normalized_result is None:
            return {}
        for key in normalized_result:
            if key.startswith("_"):
                raise ValueError(
                    c.ERR_RUNTIME_KEYS_WITH_UNDERSCORE_RESERVED.format(key=key),
                )
        validated_metadata: t.MappingKV[str, t.JsonValue] = (
            t.metadata_map_adapter().validate_python({
                key: item for key, item in normalized_result.items() if item is not None
            })
        )
        return validated_metadata

    @staticmethod
    def validate_metadata_model_input[TModel: BaseModel](
        value: t.MetadataInput,
        metadata_model: type[TModel],
    ) -> TModel:
        """Normalize metadata-like input into the provided metadata model."""
        if value is None:
            validated_model = metadata_model.model_validate({
                c.FIELD_ATTRIBUTES: {},
            })
            if isinstance(validated_model, metadata_model):
                return validated_model
            msg = f"{metadata_model.__name__} validation did not return model instance"
            raise TypeError(msg)
        if isinstance(value, metadata_model):
            return value
        if isinstance(value, Mapping):
            raw_mapping = value
        elif ugm.has_model_dump(value):
            raw_dump = value.model_dump()
            if not isinstance(raw_dump, Mapping):
                msg = c.ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE
                raise TypeError(msg)
            raw_mapping = raw_dump
        else:
            msg = (
                "metadata must be None, dict, or "
                f"{metadata_model.__name__}, got {value.__class__.__name__}"
            )
            raise TypeError(msg)
        validated_model = metadata_model.model_validate({
            c.FIELD_ATTRIBUTES: dict(raw_mapping),
        })
        if isinstance(validated_model, metadata_model):
            return validated_model
        msg = f"{metadata_model.__name__} validation did not return model instance"
        raise TypeError(msg)

    @staticmethod
    def _normalize_payload_item(
        item: p.AttributeProbe,
        *,
        container_kind: Literal["mapping", "sequence"],
    ) -> t.JsonPayload:
        """Normalize one container item to its canonical t.JsonPayload form.

        Single SSOT for the per-item normalization shared by the Mapping and
        Sequence branches of ``normalize_registerable_service``. Uses Python
        3.13 ``match/case`` to collapse the prior duplicate isinstance ladders.
        """
        normalized_item: t.JsonPayload
        match item:
            case datetime():
                normalized_item = (
                    item.replace(tzinfo=UTC) if item.tzinfo is None else item
                ).isoformat()
            case Path():
                normalized_item = str(item)
            case tuple():
                normalized_item = FlextRuntime.normalize_to_metadata(item)
            case dict():
                normalized_item = dict(t.json_mapping_adapter().validate_python(item))
            case list():
                normalized_item = list(t.json_list_adapter().validate_python(item))
            case bool() | int() | float() | str() | None | mp.BaseModel():
                normalized_item = item
            case _:
                err_template = (
                    c.ERR_RUNTIME_MAPPING_INVALID_TYPE
                    if container_kind == "mapping"
                    else c.ERR_RUNTIME_SEQUENCE_INVALID_TYPE
                )
                msg = err_template.format(type_name=type(item))
                raise TypeError(msg)
        return normalized_item

    @staticmethod
    def normalize_registerable_service(
        value: t.RegisterableService,
    ) -> t.RegisterableService | mc.ConfigMap | mc.ObjectList:
        """Normalize container registration payloads to canonical runtime types."""
        normalized_service: t.RegisterableService | mc.ConfigMap | mc.ObjectList
        match value:
            case Mapping():
                normalized_service = mc.ConfigMap(
                    root={
                        key_s: FlextRuntime._normalize_payload_item(
                            item, container_kind="mapping"
                        )
                        for key_s, item in value.items()
                    }
                )
            case Sequence() if not isinstance(value, (str, bytes, bytearray)):
                normalized_service = mc.ObjectList(
                    root=[
                        FlextRuntime._normalize_payload_item(
                            item, container_kind="sequence"
                        )
                        for item in value
                    ]
                )
            case (
                None
                | str()
                | int()
                | float()
                | bool()
                | bytes()
                | datetime()
                | Path()
                | BaseModel()
            ):
                normalized_service = value
            case _ if callable(value) or isinstance(
                value,
                (p.Logger, p.Settings, p.Context, p.Dispatcher),
            ):
                normalized_service = value
            case _:
                raise ValueError(
                    c.ERR_RUNTIME_SERVICE_MUST_BE_REGISTERABLE.format(
                        type_name=type(value).__name__,
                    ),
                )
        return normalized_service

    @staticmethod
    def validate_callable_input[TCallable](
        value: TCallable,
        subject: str,
    ) -> TCallable:
        """Validate that a single runtime input is callable."""
        if not callable(value):
            msg = f"{subject} must be callable, got {value.__class__.__name__}"
            raise TypeError(msg)
        return value

    @staticmethod
    def normalize_to_container(
        val: t.JsonPayload
        | t.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[t.Scalar],
    ) -> t.RuntimeData:
        """Normalize any value to RuntimeData.

        Args:
            val: Value to normalize

        Returns:
            JsonValue | BaseModel

        """
        normalized_data: t.RuntimeData
        if val is None:
            normalized_data = ""
        elif isinstance(val, (mc.ConfigMap, mc.Dict)):
            entries = [(k, v) for k, v in val.root.items()]
            normalized_data = FlextRuntime._normalize_dict_entries(entries)
        elif isinstance(val, mc.ObjectList):
            normalized_data = list(
                t.json_list_adapter().validate_python(
                    [FlextRuntime.normalize_to_json_value(v) for v in val.root],
                ),
            )
        elif isinstance(val, BaseModel):
            normalized_data = val
        elif isinstance(val, Path):
            normalized_data = str(val)
        elif ugc.scalar(val):
            normalized_data = FlextRuntime.normalize_to_json_value(val)
        elif isinstance(val, Mapping):
            entries = [(k, v) for k, v in val.items()]
            normalized_data = FlextRuntime._normalize_dict_entries(entries)
        elif isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            normalized_data = list(
                t.json_list_adapter().validate_python(
                    [
                        FlextRuntime.normalize_to_json_value(item_raw)
                        for item_raw in val
                    ],
                )
            )
        else:
            normalized_data = str(val)
        return normalized_data

    @staticmethod
    def normalize_to_metadata(
        val: t.JsonPayload
        | t.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[t.Scalar]
        | None,
    ) -> t.JsonValue:
        """Normalize input into metadata-compatible JSON-native values.

        ``None`` is normalized to an empty string so metadata payloads stay
        JSON-compatible without dropping the original key.
        """
        normalized_value: t.JsonValue
        if isinstance(val, (mc.ConfigMap, mc.Dict)):
            normalized_value = FlextRuntime._normalize_dict_entries([
                (key, item) for key, item in val.root.items()
            ])
        elif val is None:
            normalized_value = ""
        elif isinstance(val, datetime):
            normalized_value = val.isoformat()
        elif isinstance(val, Path):
            normalized_value = str(val)
        elif isinstance(val, (str, int, float, bool)):
            normalized_value = val
        elif ugm.has_model_dump(val):
            normalized_value = FlextRuntime.normalize_to_json_value(val)
        elif isinstance(val, Mapping):
            normalized_value = FlextRuntime._normalize_dict_entries([
                (key, item) for key, item in val.items()
            ])
        elif isinstance(val, AbstractSet):
            normalized_value = list(
                t.json_list_adapter().validate_python([
                    FlextRuntime.normalize_to_json_value(item) for item in val
                ])
            )
        elif isinstance(val, (bytes, bytearray)):
            normalized_value = str(val)
        else:
            normalized_value = list(
                t.json_list_adapter().validate_python([
                    FlextRuntime.normalize_to_json_value(item) for item in val
                ])
            )
        return normalized_value

    @no_type_check
    class DependencyIntegration:
        """Centralize dependency-injector wiring with provider helpers."""

        class DynamicContainerWithConfig(containers.DynamicContainer):
            """Dynamic container with declared configuration provider."""

            settings: providers.Configuration = providers.Configuration()

        class BridgeContainer(containers.DeclarativeContainer):
            """Declarative container grouping settings and resource modules."""

            settings = providers.Configuration()
            services = providers.Object(containers.DynamicContainer())
            resources = providers.Object(containers.DynamicContainer())

        class ContainerCreationOptions(BaseModel):
            """Default validated options for dependency container creation."""

            model_config: ClassVar[ConfigDict] = ConfigDict(
                arbitrary_types_allowed=True,
            )

            settings: mc.ConfigMap | None = None
            services: t.MappingKV[str, t.RegisterableService] | None = None
            factories: t.MappingKV[str, t.FactoryCallable] | None = None
            resources: t.MappingKV[str, t.ResourceCallable] | None = None
            wire_modules: t.SequenceOf[ModuleType] | None = None
            wire_packages: t.StrSequence | None = None
            wire_classes: t.SequenceOf[type] | None = None
            factory_cache: bool = True

        ContainerCreationOptionsModel: ClassVar[
            p.ContainerCreationOptionsType | None
        ] = ContainerCreationOptions

        @classmethod
        def _require_container_creation_options_model(
            cls,
        ) -> p.ContainerCreationOptionsType:
            """Return the bound container options model or raise a contract error."""
            options_model = cls.ContainerCreationOptionsModel
            if options_model is None:
                msg = (
                    "FlextRuntime.DependencyIntegration.ContainerCreationOptionsModel "
                    "is not bound to a concrete implementation"
                )
                raise RuntimeError(msg)
            return options_model

        _OPTION_FIELDS: ClassVar[t.StrSequence] = (
            "settings",
            "services",
            "factories",
            "resources",
            "wire_modules",
            "wire_packages",
            "wire_classes",
        )

        @classmethod
        def _parse_options(
            cls,
            container_options: p.ContainerCreationOptions
            | t.MappingKV[str, t.JsonPayload]
            | None,
        ) -> p.ContainerCreationOptions:
            """Parse raw container options into a validated model."""
            options_model = cls._require_container_creation_options_model()
            match container_options:
                case None:
                    return options_model.model_validate({})
                case Mapping():
                    return options_model.model_validate(container_options)
                case _:
                    return options_model.model_validate(
                        {
                            field: getattr(container_options, field)
                            for field in cls._OPTION_FIELDS
                        }
                        | {"factory_cache": container_options.factory_cache},
                    )

        @classmethod
        def _merge_options(
            cls,
            base: p.ContainerCreationOptions,
            overrides: t.MappingKV[str, t.JsonPayload],
        ) -> p.ContainerCreationOptions:
            """Merge runtime kwargs over base options (override wins if not None)."""
            options_model = cls._require_container_creation_options_model()
            override_opts = options_model.model_validate(overrides)
            merged: MutableMapping[str, t.JsonPayload] = {
                field: (
                    getattr(override_opts, field)
                    if getattr(override_opts, field) is not None
                    else getattr(base, field)
                )
                for field in cls._OPTION_FIELDS
            }
            merged["factory_cache"] = override_opts.factory_cache
            merged_options: t.MappingKV[str, t.JsonPayload] = dict(merged)
            return options_model.model_validate(merged_options)

        @classmethod
        def _populate_container(
            cls,
            di_container: containers.DynamicContainer,
            opts: p.ContainerCreationOptions,
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
                        di_container,
                        name,
                        factory,
                        cache=opts.factory_cache,
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
            container_options: p.ContainerCreationOptions
            | t.MappingKV[str, t.JsonPayload]
            | None = None,
            **runtime_kwargs: t.JsonPayload,
        ) -> containers.DynamicContainer:
            """Create a DynamicContainer with optional pre-registration and wiring.

            Args:
                container_options: Options as protocol, mapping, or None.
                **runtime_kwargs: Override individual option fields.

            Returns:
                A dynamic container ready for immediate ``@inject`` consumption
                without manual follow-up registration calls.

            """
            base = cls._parse_options(container_options)
            opts = cls._merge_options(base, runtime_kwargs) if runtime_kwargs else base
            di_container = cls.DynamicContainerWithConfig()
            cls._populate_container(di_container, opts)
            return di_container

        @classmethod
        def create_layered_bridge(
            cls,
            settings: mc.ConfigMap | None = None,
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

        @staticmethod
        def bind_configuration(
            di_container: containers.DynamicContainer,
            settings: mc.ConfigMap | None,
        ) -> providers.Configuration:
            """Bind configuration mapping to the DI container.

            Uses ``providers.Configuration`` to expose values to downstream
            providers without higher layers interacting with dependency-injector
            directly.
            """
            configuration_provider = providers.Configuration()
            if settings:
                configuration_provider.from_dict(dict(settings))
            if isinstance(
                di_container,
                FlextRuntime.DependencyIntegration.DynamicContainerWithConfig,
            ):
                configured_container: FlextRuntime.DependencyIntegration.DynamicContainerWithConfig = di_container
                configured_container.settings = configuration_provider
            else:
                setattr(di_container, c.Directory.CONFIG, configuration_provider)
            return configuration_provider

        @staticmethod
        def bind_configuration_provider(
            configuration_provider: providers.Configuration,
            settings: mc.ConfigMap | None,
        ) -> providers.Configuration:
            """Bind configuration directly to an existing provider."""
            if settings:
                configuration_provider.from_dict(dict(settings))
            return configuration_provider

        @staticmethod
        def register_factory[T](
            di_container: containers.DynamicContainer,
            name: str,
            factory: Callable[[], T],
            *,
            cache: bool = True,
        ) -> providers.Provider[T]:
            """Register a factory using Singleton/Factory providers.

            Args:
                di_container: DynamicContainer instance for provider registration.

            """
            if hasattr(di_container, name):
                raise ValueError(
                    c.ERR_RUNTIME_PROVIDER_ALREADY_REGISTERED.format(name=name),
                )
            provider: providers.Provider[T] = (
                providers.Singleton(factory) if cache else providers.Factory(factory)
            )
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def register_object[T](
            di_container: containers.DynamicContainer,
            name: str,
            instance: T,
        ) -> providers.Provider[T]:
            """Register a concrete instance using ``providers.Object``.

            Args:
                di_container: DynamicContainer instance for provider registration.

            """
            if hasattr(di_container, name):
                raise ValueError(
                    c.ERR_RUNTIME_PROVIDER_ALREADY_REGISTERED.format(name=name),
                )
            provider: providers.Provider[T] = providers.Object(instance)
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def register_resource[T](
            di_container: containers.DynamicContainer,
            name: str,
            factory: Callable[[], T],
        ) -> providers.Provider[T]:
            """Register a resource provider for lifecycle-managed dependencies.

            Args:
                di_container: DynamicContainer instance for provider registration.

            """
            if hasattr(di_container, name):
                raise ValueError(
                    c.ERR_RUNTIME_PROVIDER_ALREADY_REGISTERED.format(name=name),
                )
            provider: providers.Provider[T] = providers.Resource(factory)
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def wire(
            container: containers.Container,
            *,
            modules: t.SequenceOf[ModuleType] | None = None,
            packages: t.StrSequence | None = None,
            classes: t.SequenceOf[type] | None = None,
        ) -> None:
            """Wire modules or packages to a dependency-injector container for @inject usage."""
            modules_to_wire: MutableSequence[ModuleType] = list(modules or [])
            if classes:
                for target_class in classes:
                    module = inspect.getmodule(target_class)
                    if module is not None:
                        modules_to_wire.append(module)
            _ = packages
            wire_runtime = getattr(wiring, "wire")
            wire_runtime(
                modules=modules_to_wire or None,
                packages=None,
                container=container,
            )


__all__: list[str] = ["FlextRuntime"]
