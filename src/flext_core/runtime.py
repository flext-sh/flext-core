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
import sys
from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import (
    ClassVar,
    cast,
)

import orjson
from dependency_injector import containers, providers, wiring
from pydantic import BaseModel, ConfigDict, ValidationError

from flext_core import (
    FlextUtilitiesGenerators,
    FlextUtilitiesGuardsTypeCore,
    T,
    c,
    p,
    t,
)


class FlextRuntime:
    """Expose runtime normalization, DI, and validation helpers to higher layers."""

    Metadata: ClassVar[type[p.Metadata] | None] = None

    @classmethod
    def _require_metadata_model(cls) -> type[p.Metadata]:
        """Return the bound metadata model class or raise a runtime contract error."""
        metadata_cls = cls.Metadata
        if metadata_cls is None:
            msg = "FlextRuntime.Metadata is not bound to a concrete model"
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

    class Bootstrap:
        """Bootstrap helpers for instantiation without calling ``__init__``."""

        @staticmethod
        def create_instance[T](class_type: type[T]) -> T:
            """Create instance using the runtime low-level constructor path."""
            return FlextRuntime.create_instance(class_type)

    @staticmethod
    def ensure_utc_datetime(value: datetime | None) -> datetime | None:
        """Attach UTC timezone to naive datetimes while preserving None."""
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @staticmethod
    def dependency_containers() -> ModuleType:
        """Return the dependency-injector containers module."""
        return containers

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        return providers

    @staticmethod
    def _to_plain_container(value: t.RuntimeAtomic) -> t.RecursiveContainer:
        """Flatten a runtime atomic value to plain Python types."""
        match value:
            case t.ConfigMap() | t.Dict():
                return {
                    str(k): FlextRuntime._to_plain_container(
                        FlextRuntime.normalize_to_container(v),
                    )
                    for k, v in value.root.items()
                }
            case t.ObjectList():
                return list(value.root)
            case bool() | str() | int() | float() | datetime() | Path():
                return value
            case _:
                return str(value)

    @staticmethod
    def _normalize_dict_entries(
        items: Sequence[tuple[str, t.RuntimeData]],
    ) -> MutableMapping[str, t.ValueOrModel]:
        """Normalize key-value pairs for container dict construction."""
        result: MutableMapping[str, t.ValueOrModel] = {}
        for key, item in items:
            normalized = FlextRuntime.normalize_to_container(item)
            result[key] = FlextRuntime._to_plain_container(normalized)
        return result

    @staticmethod
    def resolve_nested_model_class[TModel: BaseModel](
        *,
        module_name: str,
        qualname: str,
        models_module_name: str,
        attribute_name: str,
        fallback: type[TModel],
    ) -> type[TModel]:
        """Resolve a nested override model class from a facade module path."""
        min_qualname_parts = 2
        if module_name != models_module_name or "." not in qualname:
            return fallback
        parts = qualname.split(".")
        models_module = sys.modules.get(models_module_name)
        if not isinstance(models_module, ModuleType) or len(parts) < min_qualname_parts:
            return fallback
        obj: type | ModuleType | None = models_module.__dict__.get(parts[0])
        for part in parts[1:-1]:
            if isinstance(obj, ModuleType):
                obj = obj.__dict__.get(part)
                continue
            if isinstance(obj, type):
                obj = obj.__dict__.get(part)
                continue
            obj = None
            break
        if not isinstance(obj, type):
            return fallback
        resolved_attr = obj.__dict__.get(attribute_name)
        if isinstance(resolved_attr, type) and issubclass(resolved_attr, fallback):
            return resolved_attr
        return fallback

    @staticmethod
    def normalize_model_input_mapping(
        value: BaseModel | t.Dict | t.ScalarMapping | None,
    ) -> Mapping[str, t.ValueOrModel] | None:
        """Normalize model-like input to a plain mapping."""
        if value is None:
            return None
        if isinstance(value, t.Dict):
            return dict(value.root)
        if isinstance(value, BaseModel):
            return value.model_dump()
        return dict(value)

    @staticmethod
    def validate_metadata_attributes(
        value: t.MetadataValue | Mapping[str, t.MetadataValue] | BaseModel | None,
    ) -> Mapping[str, t.MetadataValue]:
        """Normalize and validate metadata attributes input."""
        if value is None:
            return {}
        if isinstance(value, BaseModel):
            result = value.model_dump()
        elif isinstance(value, Mapping):
            result = dict(value)
        else:
            msg = "attributes must be dict-like"
            raise TypeError(msg)
        for key in result:
            if key.startswith("_"):
                msg = f"Keys starting with '_' are reserved: {key}"
                raise ValueError(msg)
        return t.metadata_map_adapter().validate_python(result)

    @staticmethod
    def validate_metadata_model_input[TModel: BaseModel](
        value: t.MetadataInput,
        metadata_model: type[TModel],
    ) -> TModel:
        """Normalize metadata-like input into the provided metadata model."""
        if value is None:
            return metadata_model.model_validate({
                c.FIELD_ATTRIBUTES: {},
            })
        if isinstance(value, metadata_model):
            return value
        if not isinstance(value, Mapping):
            msg = (
                "metadata must be None, dict, or "
                f"{metadata_model.__name__}, got {value.__class__.__name__}"
            )
            raise TypeError(msg)
        return metadata_model.model_validate({
            c.FIELD_ATTRIBUTES: dict(value),
        })

    @staticmethod
    def normalize_registerable_service(
        value: t.RegisterableService,
    ) -> t.RegisterableService | t.ConfigMap | t.ObjectList:
        """Normalize container registration payloads to canonical runtime types."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (BaseModel, Path)):
            return value
        if callable(value):
            return value
        if isinstance(value, Mapping):
            normalized_mapping: MutableMapping[str, t.ValueOrModel] = {}
            for key_s, item in value.items():
                if isinstance(item, datetime):
                    normalized_mapping[key_s] = (
                        item.replace(tzinfo=UTC) if item.tzinfo is None else item
                    )
                elif isinstance(item, Path):
                    normalized_mapping[key_s] = str(item)
                elif isinstance(
                    item,
                    (
                        str,
                        int,
                        float,
                        list,
                        dict,
                        tuple,
                        type(None),
                        BaseModel,
                    ),
                ):
                    normalized_mapping[key_s] = item
                else:
                    msg = f"Invalid type in Mapping: {type(item)}"
                    raise TypeError(msg)
            return t.ConfigMap(root=normalized_mapping)
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            normalized_sequence: MutableSequence[t.Container] = []
            for item in value:
                if isinstance(item, datetime):
                    item = item.replace(tzinfo=UTC) if item.tzinfo is None else item
                elif isinstance(item, Path):
                    item = str(item)
                elif not isinstance(
                    item,
                    (
                        str,
                        int,
                        float,
                        bool,
                        list,
                        dict,
                        tuple,
                        type(None),
                        BaseModel,
                    ),
                ):
                    msg = f"Invalid type in Sequence: {type(item)}"
                    raise TypeError(msg)
                normalized_sequence.append(str(item))
            return t.ObjectList(root=normalized_sequence)
        if hasattr(value, "__dict__"):
            return value
        if hasattr(value, "bind") and hasattr(value, "info"):
            return value
        msg = f"Service must be a RegisterableService type, got {type(value).__name__}"
        raise ValueError(msg)

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
    def validate_callable_sequence[TCallable](
        values: Sequence[TCallable],
        subject: str,
    ) -> Sequence[TCallable]:
        """Validate that every item in a sequence is callable."""
        for value in values:
            FlextRuntime.validate_callable_input(value, subject)
        return values

    @staticmethod
    def validate_model_sequence[TModel: BaseModel](
        values: Sequence[t.ValueOrModel],
        model_cls: type[TModel],
    ) -> Sequence[TModel]:
        """Validate a heterogeneous input sequence into one Pydantic model type."""
        validated_items: list[TModel] = []
        item_errors: list[str] = []
        for index, value in enumerate(values):
            try:
                if isinstance(value, model_cls):
                    validated_items.append(value)
                elif isinstance(value, BaseModel):
                    validated_items.append(model_cls.model_validate(value.model_dump()))
                elif isinstance(value, Mapping):
                    validated_items.append(model_cls.model_validate(dict(value)))
                else:
                    validated_items.append(model_cls.model_validate(value))
            except ValidationError as exc:
                item_errors.extend(
                    f"{index}.{'.'.join(str(part) for part in err.get('loc', ()))}: {err.get('msg', 'validation error')}"
                    for err in exc.errors()
                )
        if item_errors:
            msg = f"Batch validation failed: {'; '.join(item_errors)}"
            raise TypeError(msg)
        return validated_items

    @classmethod
    def normalize_metadata_input(cls, value: t.MetadataInput) -> p.Metadata:
        """Normalize metadata input into the bound metadata model."""
        metadata_cls = cast("type[BaseModel]", cls._require_metadata_model())
        if value is None:
            return cast(
                "p.Metadata", metadata_cls.model_validate({c.FIELD_ATTRIBUTES: {}})
            )
        if isinstance(value, metadata_cls):
            return cast("p.Metadata", value)
        if not isinstance(value, Mapping):
            msg = (
                "metadata must be None, dict, or bound metadata model, got "
                f"{value.__class__.__name__}"
            )
            raise TypeError(msg)
        return cast(
            "p.Metadata", metadata_cls.model_validate({c.FIELD_ATTRIBUTES: dict(value)})
        )

    @staticmethod
    def normalize_to_container(
        val: t.RuntimeData,
    ) -> t.RuntimeAtomic:
        """Normalize any value to t.Container | BaseModel.

        Args:
            val: Value to normalize

        Returns:
            Scalar | Path | BaseModel

        """
        match val:
            case None:
                return ""
            case BaseModel():
                return val
            case Path():
                return val
            case _ if FlextUtilitiesGuardsTypeCore.is_scalar(val):
                return val
            case _ if FlextUtilitiesGuardsTypeCore.is_dict_like(val):
                if isinstance(val, t.ConfigMap):
                    entries = [(k, v) for k, v in val.root.items()]
                else:
                    entries = [(str(k), v) for k, v in val.items()]
                return t.Dict(root=FlextRuntime._normalize_dict_entries(entries))
            case _ if FlextUtilitiesGuardsTypeCore.is_list_like(val):
                normalized_list: t.FlatContainerList = [
                    item
                    for v in val
                    if isinstance(
                        item := FlextRuntime.normalize_to_container(v),
                        (str, int, float, bool, datetime, Path),
                    )
                ]
                return t.ObjectList(root=normalized_list)
            case _:
                return str(val)

    @staticmethod
    def _normalize_to_metadata_scalar(val: t.RuntimeData) -> t.Primitives:
        if val is None:
            return ""
        if FlextUtilitiesGuardsTypeCore.is_primitive(val):
            return val
        if isinstance(val, datetime):
            return val.isoformat()
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, BaseModel):
            return val.model_dump_json()
        return str(val)

    @staticmethod
    def _normalize_metadata_dict_value(
        v: t.RuntimeData,
    ) -> t.Scalar | t.ScalarList:
        """Normalize a single dict value for metadata context."""
        match v:
            case None:
                return ""
            case Path():
                return str(v)
            case BaseModel():
                return v.model_dump_json()
            case _ if FlextUtilitiesGuardsTypeCore.is_scalar(v):
                return v
            case _ if FlextUtilitiesGuardsTypeCore.is_list_like(v):
                return [FlextRuntime._normalize_to_metadata_scalar(item) for item in v]
            case _ if FlextUtilitiesGuardsTypeCore.is_dict_like(v):
                inner: MutableMapping[str, t.Primitives] = {}
                for ik, iv in v.items():
                    inner[str(ik)] = FlextRuntime._normalize_to_metadata_scalar(iv)
                return orjson.dumps(inner).decode()
            case _:
                return str(v)

    @staticmethod
    def normalize_to_metadata(
        val: t.RuntimeData,
    ) -> t.MetadataValue:
        """Normalize input into metadata-compatible scalar, list, or mapping values."""
        match val:
            case None:
                return ""
            case Path():
                return str(val)
            case BaseModel():
                return val.model_dump_json()
            case datetime():
                return val
            case _ if FlextUtilitiesGuardsTypeCore.is_primitive(val):
                return val
            case _ if FlextUtilitiesGuardsTypeCore.is_dict_like(val):
                normalized: MutableMapping[str, t.Scalar | t.ScalarList] = {}
                for k, v in val.items():
                    normalized[str(k)] = FlextRuntime._normalize_metadata_dict_value(v)
                return normalized
            case _ if FlextUtilitiesGuardsTypeCore.is_list_like(val):
                return [
                    FlextRuntime._normalize_to_metadata_scalar(item) for item in val
                ]
            case _:
                return str(val)

    class DependencyIntegration:
        """Centralize dependency-injector wiring with provider helpers."""

        class DynamicContainerWithConfig(containers.DynamicContainer):
            """Dynamic container with declared configuration provider."""

            config: providers.Configuration = providers.Configuration()

        class BridgeContainer(containers.DeclarativeContainer):
            """Declarative container grouping config and resource modules."""

            config = providers.Configuration()
            services = providers.Object(containers.DynamicContainer())
            resources = providers.Object(containers.DynamicContainer())

        class ContainerCreationOptions(BaseModel):
            """Default validated options for dependency container creation."""

            model_config: ClassVar[ConfigDict] = ConfigDict(
                arbitrary_types_allowed=True,
            )

            config: t.ConfigMap | None = None
            services: Mapping[str, t.RegisterableService] | None = None
            factories: Mapping[str, t.FactoryCallable] | None = None
            resources: Mapping[str, t.ResourceCallable] | None = None
            wire_modules: Sequence[ModuleType] | None = None
            wire_packages: t.StrSequence | None = None
            wire_classes: Sequence[type] | None = None
            factory_cache: bool = True

        ContainerCreationOptionsModel: ClassVar[
            p.ContainerCreationOptionsType | None
        ] = ContainerCreationOptions

        Provide = wiring.Provide
        inject = staticmethod(wiring.inject)

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
            "config",
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
            | Mapping[str, t.RuntimeData]
            | None,
        ) -> p.ContainerCreationOptions:
            """Parse raw container options into a validated model."""
            options_model = cls._require_container_creation_options_model()
            match container_options:
                case None:
                    return options_model.model_validate({})
                case Mapping():
                    return options_model.model_validate(dict(container_options))
                case _:
                    return options_model.model_validate(
                        {
                            field: getattr(container_options, field)
                            for field in cls._OPTION_FIELDS
                        }
                        | {"factory_cache": container_options.factory_cache}
                    )

        @classmethod
        def _merge_options(
            cls,
            base: p.ContainerCreationOptions,
            overrides: Mapping[str, t.RuntimeData],
        ) -> p.ContainerCreationOptions:
            """Merge runtime kwargs over base options (override wins if not None)."""
            options_model = cls._require_container_creation_options_model()
            override_opts = options_model.model_validate(overrides)
            merged: MutableMapping[str, t.RuntimeData] = {
                field: (
                    getattr(override_opts, field)
                    if getattr(override_opts, field) is not None
                    else getattr(base, field)
                )
                for field in cls._OPTION_FIELDS
            }
            merged["factory_cache"] = override_opts.factory_cache
            return options_model.model_validate(merged)

        @classmethod
        def _populate_container(
            cls,
            di_container: containers.DynamicContainer,
            opts: p.ContainerCreationOptions,
        ) -> None:
            """Register config, services, factories, resources, and wiring."""
            if opts.config is not None:
                _ = cls.bind_configuration(di_container, opts.config)
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
            | Mapping[str, t.RuntimeData]
            | None = None,
            **runtime_kwargs: t.RuntimeData,
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
            config: t.ConfigMap | None = None,
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
            cls.bind_configuration_provider(bridge.config, config)
            return (bridge, service_module, resource_module)

        @staticmethod
        def bind_configuration(
            di_container: containers.DynamicContainer,
            config: t.ConfigMap | None,
        ) -> providers.Configuration:
            """Bind configuration mapping to the DI container.

            Uses ``providers.Configuration`` to expose values to downstream
            providers without higher layers interacting with dependency-injector
            directly.
            """
            configuration_provider = providers.Configuration()
            if config:
                configuration_provider.from_dict(dict(config))
            if isinstance(
                di_container,
                FlextRuntime.DependencyIntegration.DynamicContainerWithConfig,
            ):
                configured_container: FlextRuntime.DependencyIntegration.DynamicContainerWithConfig = di_container
                configured_container.config = configuration_provider
            else:
                setattr(di_container, c.DIR_CONFIG, configuration_provider)
            return configuration_provider

        @staticmethod
        def bind_configuration_provider(
            configuration_provider: providers.Configuration,
            config: t.ConfigMap | None,
        ) -> providers.Configuration:
            """Bind configuration directly to an existing provider."""
            if config:
                configuration_provider.from_dict(dict(config))
            return configuration_provider

        @staticmethod
        def register_factory(
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
                msg = f"Provider '{name}' is already registered"
                raise ValueError(msg)
            provider: providers.Provider[T] = (
                providers.Singleton(factory) if cache else providers.Factory(factory)
            )
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def register_object(
            di_container: containers.DynamicContainer,
            name: str,
            instance: T,
        ) -> providers.Provider[T]:
            """Register a concrete instance using ``providers.Object``.

            Args:
                di_container: DynamicContainer instance for provider registration.

            """
            if hasattr(di_container, name):
                msg = f"Provider '{name}' is already registered"
                raise ValueError(msg)
            provider: providers.Provider[T] = providers.Object(instance)
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def register_resource(
            di_container: containers.DynamicContainer,
            name: str,
            factory: Callable[[], T],
        ) -> providers.Provider[T]:
            """Register a resource provider for lifecycle-managed dependencies.

            Args:
                di_container: DynamicContainer instance for provider registration.

            """
            if hasattr(di_container, name):
                msg = f"Provider '{name}' is already registered"
                raise ValueError(msg)
            provider: providers.Provider[T] = providers.Resource(factory)
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def wire(
            container: containers.DeclarativeContainer | containers.DynamicContainer,
            *,
            modules: Sequence[ModuleType] | None = None,
            packages: t.StrSequence | None = None,
            classes: Sequence[type] | None = None,
        ) -> None:
            """Wire modules or packages to a DeclarativeContainer or DynamicContainer for @inject usage."""
            modules_to_wire: MutableSequence[ModuleType] = list(modules or [])
            if classes:
                for target_class in classes:
                    module = inspect.getmodule(target_class)
                    if module is not None:
                        modules_to_wire.append(module)
            _ = packages
            wiring.wire(
                modules=modules_to_wire or None,
                packages=None,
                container=container,
            )

    @classmethod
    def ensure_trace_context(
        cls,
        context: t.ScalarMapping | t.ScalarOrModel,
        *,
        include_correlation_id: bool = False,
        include_timestamp: bool = False,
    ) -> t.StrMapping:
        """Ensure context dict has distributed tracing fields (bridge for _models).

        Args:
            context: Context dictionary or recursive payload to enrich
            include_correlation_id: If True, ensure correlation_id exists
            include_timestamp: If True, ensure timestamp exists

        Returns:
            t.StrMapping: Enriched context with trace fields

        """
        context_dict = t.ConfigMap(root={})
        if isinstance(context, Mapping):
            parsed_context: MutableMapping[str, t.ValueOrModel] = {
                str(k): str(v) for k, v in context.items()
            }
            context_dict = t.ConfigMap(root=parsed_context)
        elif not isinstance(
            context, Mapping
        ) and FlextUtilitiesGuardsTypeCore.is_scalar(context):
            context_dict = t.ConfigMap(root={})
        elif isinstance(context, BaseModel):
            context_dict.update(context.model_dump())
        else:
            context_dict = t.ConfigMap(root={})
        result: t.MutableStrMapping = {}
        for key, value in context_dict.items():
            result[key] = str(value)
        if "trace_id" not in result:
            result["trace_id"] = cls.generate_id()
        if "span_id" not in result:
            result["span_id"] = cls.generate_id()
        if include_correlation_id and c.KEY_CORRELATION_ID not in result:
            result[c.KEY_CORRELATION_ID] = cls.generate_id()
        if include_timestamp and "timestamp" not in result:
            result["timestamp"] = cls.generate_datetime_utc().isoformat()
        return result

    @staticmethod
    def generate_datetime_utc() -> datetime:
        """Generate current UTC datetime for runtime-scoped timestamps."""
        return FlextUtilitiesGenerators.generate_datetime_utc()

    @staticmethod
    def generate_id() -> str:
        """Generate unique ID for runtime-scoped correlation and tracing."""
        return FlextUtilitiesGenerators.generate_id()


__all__ = ["FlextRuntime"]
