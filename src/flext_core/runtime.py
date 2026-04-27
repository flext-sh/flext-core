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
    FlextModelsContainers as mc,
    FlextModelsPydantic as mp,
    FlextUtilitiesGuardsTypeCore as ugc,
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
    def _normalize_to_json_value(
        value: t.JsonPayload
        | t.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[t.Scalar],
    ) -> t.JsonValue:
        """Normalize arbitrary runtime input to one validated ``JsonValue``."""
        normalized = FlextRuntime.normalize_to_metadata(value)
        validated: t.JsonValue = t.json_value_adapter().validate_python(normalized)
        return validated

    @staticmethod
    def _normalize_dict_entries(
        items: Sequence[tuple[str, t.JsonPayload | t.Scalar]],
    ) -> dict[str, t.JsonValue]:
        """Normalize key-value pairs for container dict construction."""
        return dict(
            t.json_mapping_adapter().validate_python(
                {
                    str(key): FlextRuntime._normalize_to_json_value(item)
                    for key, item in items
                },
            ),
        )

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
        value: BaseModel | mc.Dict | t.ScalarMapping | None,
    ) -> Mapping[str, t.JsonPayload] | None:
        """Normalize model-like input to a plain mapping."""
        if value is None:
            return None
        if isinstance(value, mc.Dict):
            raw_mapping: Mapping[str, t.JsonPayload | t.Scalar] = value.root
        elif isinstance(value, BaseModel):
            raw_mapping = value.model_dump()
        else:
            raw_mapping = value
        return FlextRuntime._normalize_dict_entries(
            [(str(key), item) for key, item in raw_mapping.items()],
        )

    @staticmethod
    def normalize_metadata_input_mapping(
        value: Mapping[str, t.JsonPayload | None] | p.HasModelDump | None,
    ) -> Mapping[str, t.JsonPayload | None] | None:
        """Normalize mapping-like metadata input while preserving explicit None.

        Defensively handles broken ``model_dump`` overrides that return a
        non-Mapping value by raising ``TypeError`` instead of surfacing an
        opaque ``AttributeError`` from downstream iteration.
        """
        if value is None:
            return None
        raw: Mapping[str, t.JsonPayload | None]
        if isinstance(value, Mapping):
            raw = value
        else:
            dumped: object = value.model_dump()
            if not isinstance(dumped, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = c.ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE
                raise TypeError(msg)
            raw = dumped
        return {
            str(key): (
                None if item is None else FlextRuntime._normalize_to_json_value(item)
            )
            for key, item in raw.items()
        }

    @staticmethod
    def validate_metadata_attributes(
        value: t.JsonValue | Mapping[str, t.JsonValue] | BaseModel | None,
    ) -> Mapping[str, t.JsonValue]:
        """Normalize and validate metadata attributes input.

        Defensively asserts that a BaseModel's ``model_dump`` actually yields
        a Mapping — broken models that violate the :class:`p.HasModelDump`
        protocol (e.g. returning a scalar) MUST surface as ``TypeError`` and
        not as opaque ``AttributeError`` from downstream iteration.
        """
        if value is None:
            return {}
        if not isinstance(value, (BaseModel, Mapping)):
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
        validated_metadata: Mapping[str, t.JsonValue] = (
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
    def _normalize_payload_item(
        item: object,
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
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (mp.BaseModel, Path)):
            return value
        if callable(value):
            return value
        if isinstance(value, Mapping):
            return mc.ConfigMap(
                root={
                    str(key_s): FlextRuntime._normalize_payload_item(
                        item, container_kind="mapping"
                    )
                    for key_s, item in value.items()
                }
            )
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return mc.ObjectList(
                root=[
                    FlextRuntime._normalize_payload_item(
                        item, container_kind="sequence"
                    )
                    for item in value
                ]
            )
        if hasattr(value, "__dict__"):
            return value
        if hasattr(value, "bind") and hasattr(value, "info"):
            return value
        raise ValueError(
            c.ERR_RUNTIME_SERVICE_MUST_BE_REGISTERABLE.format(
                type_name=type(value).__name__,
            ),
        )

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
        if val is None:
            return ""
        if isinstance(val, (mc.ConfigMap, mc.Dict)):
            entries = [(k, v) for k, v in val.root.items()]
            return FlextRuntime._normalize_dict_entries(entries)
        if isinstance(val, mc.ObjectList):
            return list(
                t.json_list_adapter().validate_python(
                    [FlextRuntime._normalize_to_json_value(v) for v in val.root],
                ),
            )
        if isinstance(val, BaseModel):
            return val
        if isinstance(val, Path):
            return str(val)
        if ugc.scalar(val):
            return FlextRuntime._normalize_to_json_value(val)
        if ugc.dict_like(val):
            entries = [(str(k), v) for k, v in val.items()]
            return FlextRuntime._normalize_dict_entries(entries)
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            return list(
                t.flat_container_list_adapter().validate_python(
                    [
                        FlextRuntime._normalize_to_json_value(item_raw)
                        for item_raw in val
                    ],
                )
            )
        return str(val)

    @staticmethod
    def normalize_to_metadata(
        val: t.JsonPayload
        | t.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[t.Scalar],
    ) -> t.JsonValue:
        """Normalize input into metadata-compatible JSON-native values."""
        if val is None:
            return ""
        if isinstance(val, datetime):
            return val.isoformat()
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, (str, int, float, bool)):
            return val
        if isinstance(val, BaseModel):
            return FlextRuntime.normalize_to_metadata(val.model_dump())
        normalized: (
            t.JsonPayload
            | AbstractSet[t.Scalar]
            | dict[str, t.JsonPayload]
            | list[t.JsonPayload]
            | list[t.JsonValue]
        )
        if isinstance(val, (mc.ConfigMap, mc.Dict, mc.ObjectList)):
            normalized = val.root
        elif isinstance(val, AbstractSet):
            normalized = [FlextRuntime._normalize_to_json_value(item) for item in val]
        else:
            normalized = val
        if isinstance(normalized, Mapping):
            return {
                str(key): FlextRuntime.normalize_to_metadata(item)
                for key, item in normalized.items()
            }
        if isinstance(normalized, (str, bytes, bytearray)):
            return str(normalized)
        try:
            iter(normalized)
        except TypeError:
            return str(normalized)
        return [FlextRuntime.normalize_to_metadata(item) for item in normalized]

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
            | Mapping[str, t.JsonPayload]
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
            overrides: Mapping[str, t.JsonPayload],
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
            merged_options: Mapping[str, t.JsonPayload] = dict(merged)
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
            | Mapping[str, t.JsonPayload]
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


__all__: list[str] = ["FlextRuntime"]
