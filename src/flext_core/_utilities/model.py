"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

import orjson
from pydantic import BaseModel

from flext_core import c, r, t
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core.models import m
from flext_core.runtime import FlextRuntime


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization.

    PHILOSOPHY:
    ──────────
    - model_validate() to create from dicts
    - Automatic StrEnum coercion
    - Merge defaults with overrides
    - No initialization code bloat

    References:
    ────────────
    - model_validate: https://docs.pydantic.dev/latest/api/base_model/
    - ConfigDict: https://docs.pydantic.dev/latest/api/config/

    """

    _V = m.Validators

    @staticmethod
    def _normalize_str_object_mapping(
        value: t.ValueOrModel | Mapping[str, t.ValueOrModel],
    ) -> dict[str, t.ValueOrModel]:
        normalized_result = r[dict[str, t.NormalizedValue]].create_from_callable(
            lambda: FlextUtilitiesModel._V.dict_str_metadata_adapter().validate_python(
                value,
            ),
        )
        if normalized_result.is_failure:
            return {}
        validated = normalized_result.value
        return {
            str(k): FlextRuntime.normalize_to_container(v)
            for k, v in dict(validated).items()
        }

    @staticmethod
    def _normalize_to_pydantic_value(
        value: t.ValueOrModel,
    ) -> t.Scalar | list[t.Primitives]:
        """Normalize object to Pydantic-safe PydanticConfigValue.

        Converts complex types to strings, preserves primitives.

        Args:
            value: input value to normalize

        Returns:
            t.PydanticConfigValue: Pydantic-safe value

        """
        if value is None:
            return ""
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            sequence_items: list[t.ValueOrModel] = [
                FlextRuntime.normalize_to_container(item_value) for item_value in value
            ]
            normalized_items: list[t.Primitives] = []
            for item in sequence_items:
                if item is None:
                    normalized_items.append("")
                elif isinstance(item, (bool, int, float, str)):
                    normalized_items.append(item)
                else:
                    normalized_items.append(str(item))
            return normalized_items
        return str(value)

    @staticmethod
    def dump(
        model: BaseModel,
        *,
        by_alias: bool = False,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> Mapping[str, t.Scalar]:
        """Unified Pydantic serialization with options.

        Generic replacement for: model.model_dump() with consistent return type.

        Common usage patterns from codebase:
        - dump(model) - no arguments
        - dump(model, exclude_none=True) - bool flag
        - dump(model, exclude={"key"}) - set[str] for exclude/include
        - dump(model, exclude_unset=True) - bool flag

        Args:
            model: Pydantic model instance to serialize.
            by_alias: Whether to use field aliases.
            exclude_none: Whether to exclude None values.
            exclude_unset: Whether to exclude unset values.
            exclude_defaults: Whether to exclude default values.
            include: Set of field names to include.
            exclude: Set of field names to exclude.

        Returns:
            Dictionary representation of the model.

        Example:
            >>> user = UserModel(status=Status.ACTIVE, name="John")
            >>> data = u.dump(user, exclude_none=True)
            >>> # {"status": "active", "name": "John"}

        """
        return model.model_dump(
            by_alias=by_alias,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            include=include,
            exclude=exclude,
        )

    @staticmethod
    def from_kwargs[M: BaseModel](model_cls: type[M], **kwargs: t.Scalar) -> r[M]:
        """Create Pydantic model from kwargs with r.

        Accepts any type in kwargs - Pydantic 2 field_validators will handle
        type conversions automatically (e.g., str → Path, dict → BaseModel, etc.).
        All parameter validation and conversion happens via Pydantic 2 Field constraints
        and field_validators defined in the model.

        Example:
             result = u.from_kwargs(
                 CreateParams,
                 content={"key": "value"},
                 name="file.json",
                 directory=Path("/tmp"),  # Pydantic field_validator converts str → Path
                 indent=2,                # Pydantic Field(ge=0) validates
             )
             if result.is_success:
                 params: CreateParams = result.value

        """
        instance_result = r[M].create_from_callable(lambda: model_cls(**kwargs))
        if instance_result.is_failure:
            return r[M].fail(f"Model validation failed: {instance_result.error}")
        instance = instance_result.value
        if not isinstance(instance, model_cls):
            return r[M].fail(
                f"Expected {model_cls.__name__}, got {type(instance).__name__}",
            )
        return r[M].ok(instance)

    @staticmethod
    def load[T_Model: BaseModel](
        model_cls: type[T_Model],
        data: t.ConfigMap,
        *,
        strict: bool = False,
    ) -> r[T_Model]:
        """Load Pydantic model from mapping with r.

        Generic replacement for: Model(data) with error handling.

        Args:
            model_cls: Pydantic model class to instantiate.
            data: Dictionary or mapping to validate.
            strict: If True, enforce strict type checking during validation.

        Returns:
            r containing model instance or error message.

        Example:
            >>> result = u.load(UserModel, {"status": "active", "name": "John"})
            >>> if result.is_success:
            ...     user: UserModel = result.value

        """
        instance_result = r[T_Model].create_from_callable(
            lambda: model_cls.model_validate(data, strict=strict),
        )
        if instance_result.is_failure:
            return r[T_Model].fail(f"Model validation failed: {instance_result.error}")
        instance = instance_result.value
        if not isinstance(instance, model_cls):
            return r[T_Model].fail(
                f"Expected {model_cls.__name__}, got {type(instance).__name__}",
            )
        return r[T_Model].ok(instance)

    @staticmethod
    def merge_defaults[M: BaseModel](
        model_cls: type[M],
        defaults: Mapping[str, t.NormalizedValue],
        overrides: Mapping[str, t.NormalizedValue],
    ) -> r[M]:
        """Merge defaults with overrides and create model.

        Example:
             DEFAULTS = {"status": Status.PENDING, "retries": 3}

             result = u.merge_defaults(
                 ConfigModel,
                 defaults=DEFAULTS,
                 overrides={"status": "active"},  # Overrides
             )
             # result.value.status = Status.ACTIVE
             # result.value.retries = 3

        """
        merged = {**defaults, **overrides}
        instance_result = r[M].create_from_callable(
            lambda: model_cls.model_validate(merged),
        )
        if instance_result.is_failure:
            return r[M].fail(f"Model validation failed: {instance_result.error}")
        instance = instance_result.value
        if not isinstance(instance, model_cls):
            return r[M].fail(
                f"Expected {model_cls.__name__}, got {type(instance).__name__}",
            )
        return r[M].ok(instance)

    @staticmethod
    def ensure_metadata(
        value: t.Scalar | t.ConfigMap | m.Metadata | None,
    ) -> m.Metadata:
        """Normalize any value to m.Metadata.

        Business Rule: Always returns Metadata, never None.
        Uses FlextRuntime guards and normalization methods for automatic
        type checking and value normalization. Eliminates need for defensive
        fallbacks by centralizing all metadata normalization logic.

        Args:
            value: None, dict, Mapping, Metadata, or any object

        Returns:
            m.Metadata: Normalized metadata (empty attributes
                if input was None or empty dict)

        Raises:
            TypeError: If value is not None, dict-like, or Metadata instance

        Example:
            >>> u.ensure_metadata(None)
            Metadata(attributes={})
            >>> u.ensure_metadata({"key": "value"})
            Metadata(attributes={"key": "value"})
            >>> u.ensure_metadata(Metadata(attributes={"a": 1}))
            Metadata(attributes={"a": 1})

        """
        if value is None:
            return m.Metadata.model_validate({c.Mixins.FIELD_ATTRIBUTES: {}})
        if isinstance(value, m.Metadata):
            return value
        if FlextRuntime.is_dict_like(value):
            safe_attrs: dict[str, t.MetadataValue] = {}
            for k, v in value.items():
                str_k = str(k)
                if v is None:
                    safe_attrs[str_k] = ""
                elif FlextUtilitiesGuardsTypeCore.is_primitive(v):
                    safe_attrs[str_k] = v
                elif FlextRuntime.is_dict_like(v):
                    nested_mapping = FlextUtilitiesModel._normalize_str_object_mapping(
                        v,
                    )
                    plain_mapping: dict[str, t.NormalizedValue] = {}
                    for nested_key, nested_value in nested_mapping.items():
                        if isinstance(nested_value, BaseModel):
                            dumped_nested = nested_value.model_dump()
                            plain_mapping[str(nested_key)] = dumped_nested
                        else:
                            plain_mapping[str(nested_key)] = nested_value
                    # Bridge-level: orjson used for non-model dict serialization at infrastructure boundary
                    safe_attrs[str_k] = orjson.dumps(plain_mapping).decode()
                else:
                    safe_attrs[str_k] = str(v)
            return m.Metadata.model_validate({c.Mixins.FIELD_ATTRIBUTES: safe_attrs})
        msg = f"metadata must be None, dict, or m.Metadata, got {value.__class__.__name__}"
        raise TypeError(msg)

    @staticmethod
    def normalize_to_pydantic_dict(
        data: t.ConfigMap | None,
    ) -> Mapping[str, t.Scalar | list[t.Primitives]]:
        """Convert EventDataMapping to Pydantic-safe PydanticConfigDict.

        Normalizes object values to the restricted PydanticConfigValue type
        that Pydantic can generate schemas for without recursion issues.

        Args:
            data: EventDataMapping (Mapping[str, Any]) or None

        Returns:
            Mapping[str, Any]: Mapping with Pydantic-safe values

        Example:
            >>> u.normalize_to_pydantic_dict(None)
            {}
            >>> u.normalize_to_pydantic_dict({"key": "value"})
            {"key": "value"}
            >>> u.normalize_to_pydantic_dict({"obj": SomeModel()})
            {"obj": "SomeModel(...)"}  # Complex types converted to string

        """
        if not data:
            empty_result: dict[str, t.Scalar | list[t.Primitives]] = {}
            return empty_result
        result: dict[str, t.Scalar | list[t.Primitives]] = {}
        for key, value in data.root.items():
            result[key] = FlextUtilitiesModel._normalize_to_pydantic_value(value)
        return result

    @staticmethod
    def update[M: BaseModel](instance: M, **updates: t.Scalar) -> r[M]:
        """Update existing model with new values.

        Example:
             user = UserModel(status=Status.ACTIVE, name="John")
             result = u.update(user, status="inactive")
             # result.value = UserModel with status=Status.INACTIVE

        """
        updated_result = r[M].create_from_callable(
            lambda: instance.model_copy(update=updates),
        )
        if updated_result.is_failure:
            return r[M].fail(f"Model update failed: {updated_result.error}")
        updated = updated_result.value
        if not isinstance(updated, instance.__class__):
            return r[M].fail(
                f"Expected {instance.__class__.__name__}, got {type(updated).__name__}",
            )
        return r[M].ok(updated)

    @staticmethod
    def to_config_map(
        obj: BaseModel | Mapping[str, t.NormalizedValue] | t.NormalizedValue | None,
    ) -> t.ConfigMap:
        """Convert BaseModel/dict to ConfigMap (None → empty ConfigMap)."""
        if obj is None:
            return t.ConfigMap(root={})
        if isinstance(obj, t.ConfigMap):
            return obj
        if isinstance(obj, BaseModel):
            model_dump_result = obj.model_dump()
            normalized_model_dump: dict[str, t.ValueOrModel] = {}
            for key, value in model_dump_result.items():
                normalized_value: t.ValueOrModel
                if value is None:
                    normalized_value = ""
                elif isinstance(value, (str, int, float, bool, type(None), BaseModel)):
                    normalized_value = FlextRuntime.normalize_to_container(value)
                else:
                    normalized_value = str(value)
                normalized_model_dump[str(key)] = normalized_value
            config_map_result = r[t.ConfigMap].create_from_callable(
                lambda: t.ConfigMap(normalized_model_dump),
            )
            if config_map_result.is_failure:
                return t.ConfigMap(root={"value": str(model_dump_result)})
            return config_map_result.value
        if isinstance(obj, Mapping):
            obj_mapping_result = r[dict[str, t.NormalizedValue]].create_from_callable(
                lambda: (
                    FlextUtilitiesModel._V.dict_str_metadata_adapter().validate_python(
                        obj,
                    )
                ),
            )
            if obj_mapping_result.is_failure:
                return t.ConfigMap(root={})
            normalized_mapping: dict[str, t.ValueOrModel] = {}
            for key, value in obj_mapping_result.value.items():
                normalized_mapping_value: t.ValueOrModel = (
                    FlextRuntime.normalize_to_container(value)
                    if isinstance(value, (str, int, float, bool, type(None), BaseModel))
                    else str(value)
                )
                normalized_mapping[str(key)] = normalized_mapping_value
            config_map_result = r[t.ConfigMap].create_from_callable(
                lambda: t.ConfigMap(normalized_mapping),
            )
            if config_map_result.is_failure:
                return t.ConfigMap(root={})
            return config_map_result.value

        # Fallback to general value normalization
        normalized = FlextRuntime.normalize_to_container(obj)
        if isinstance(normalized, Mapping):
            normalized_obj: t.ValueOrModel = normalized
            normalized_map = FlextUtilitiesModel._normalize_str_object_mapping(
                normalized_obj,
            )
            return t.ConfigMap(root=normalized_map)
        return t.ConfigMap(root={})


__all__ = ["FlextUtilitiesModel"]
