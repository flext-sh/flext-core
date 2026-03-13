"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import ClassVar, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core import FlextRuntime, m, r, t

T_Model = TypeVar("T_Model", bound=BaseModel)
_MODEL_DICT_STR_OBJECT_ADAPTER = TypeAdapter(dict[str, object])
_MODEL_LIST_OBJECT_ADAPTER = TypeAdapter(list[object])
_MODEL_TUPLE_OBJECT_ADAPTER = TypeAdapter(tuple[object, ...])


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

    _pydantic_scalar_adapter: ClassVar[TypeAdapter[t.Primitives]] = TypeAdapter(
        t.Primitives
    )

    @staticmethod
    def _normalize_str_object_mapping(value: object) -> dict[str, object]:
        try:
            return _MODEL_DICT_STR_OBJECT_ADAPTER.validate_python(value)
        except ValidationError:
            return {}

    @staticmethod
    def _normalize_to_pydantic_value(
        value: object,
    ) -> t.Scalar | list[t.Primitives]:
        """Normalize object to Pydantic-safe PydanticConfigValue.

        Converts complex types to strings, preserves primitives.

        Args:
            value: object value to normalize

        Returns:
            t.PydanticConfigValue: Pydantic-safe value

        """
        if value is None:
            return ""
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, list):
            list_items = _MODEL_LIST_OBJECT_ADAPTER.validate_python(value)
            normalized_items: list[t.Primitives] = []
            for item in list_items:
                if item is None:
                    normalized_items.append("")
                    continue
                try:
                    normalized_items.append(
                        FlextUtilitiesModel._pydantic_scalar_adapter.validate_python(
                            item
                        )
                    )
                except ValidationError:
                    normalized_items.append(str(item))
            return normalized_items
        if isinstance(value, tuple):
            tuple_items = _MODEL_TUPLE_OBJECT_ADAPTER.validate_python(value)
            normalized_tuple_items: list[t.Primitives] = []
            for item in tuple_items:
                if item is None:
                    normalized_tuple_items.append("")
                    continue
                try:
                    normalized_tuple_items.append(
                        FlextUtilitiesModel._pydantic_scalar_adapter.validate_python(
                            item
                        )
                    )
                except ValidationError:
                    normalized_tuple_items.append(str(item))
            return normalized_tuple_items
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
        try:
            instance = model_cls.model_validate(kwargs)
            return r[M].ok(instance)
        except (ValidationError, TypeError, ValueError) as e:
            return r[M].fail(f"Model validation failed: {e}")

    @staticmethod
    def load[T_Model: BaseModel](
        model_cls: type[T_Model], data: m.ConfigMap, *, strict: bool = False
    ) -> r[T_Model]:
        """Load Pydantic model from mapping with r.

        Generic replacement for: Model.model_validate(data) with error handling.

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
        try:
            instance = model_cls.model_validate(data, strict=strict)
            return r[T_Model].ok(instance)
        except ValidationError as e:
            return r[T_Model].fail(f"Model validation failed: {e}")

    @staticmethod
    def merge_defaults[M: BaseModel](
        model_cls: type[M],
        defaults: Mapping[str, object],
        overrides: Mapping[str, object],
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
        try:
            instance = model_cls.model_validate(merged)
            return r[M].ok(instance)
        except (ValidationError, TypeError, ValueError) as e:
            return r[M].fail(f"Model validation failed: {e}")

    @staticmethod
    def ensure_metadata(
        value: t.Scalar | m.ConfigMap | m.Metadata | None,
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
            return m.Metadata(attributes={})
        if isinstance(value, m.Metadata):
            return value
        if isinstance(value, m.Metadata):
            return m.Metadata.model_validate(value.model_dump())
        if FlextRuntime.is_dict_like(value):
            safe_attrs: dict[str, t.MetadataValue] = {}
            for k, v in value.items():
                str_k = str(k)
                if v is None:
                    safe_attrs[str_k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    safe_attrs[str_k] = v
                elif FlextRuntime.is_dict_like(v):
                    nested_mapping = FlextUtilitiesModel._normalize_str_object_mapping(
                        v
                    )
                    safe_attrs[str_k] = json.dumps(nested_mapping)
                else:
                    safe_attrs[str_k] = str(v)
            return m.Metadata(attributes=safe_attrs)
        msg = f"metadata must be None, dict, or m.Metadata, got {value.__class__.__name__}"
        raise TypeError(msg)

    @staticmethod
    def normalize_to_pydantic_dict(
        data: m.ConfigMap | None,
    ) -> Mapping[str, t.Scalar | list[t.Primitives]]:
        """Convert EventDataMapping to Pydantic-safe PydanticConfigDict.

        Normalizes object values to the restricted PydanticConfigValue type
        that Pydantic can generate schemas for without recursion issues.

        Args:
            data: EventDataMapping (Mapping[str, object]) or None

        Returns:
            Mapping[str, object]: Mapping with Pydantic-safe values

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
        try:
            updated_instance = instance.model_copy(update=updates)
            return r[M].ok(updated_instance)
        except (AttributeError, TypeError, ValueError) as e:
            return r[M].fail(f"Model update failed: {e}")

    @staticmethod
    def to_config_map(obj: BaseModel | object | None) -> m.ConfigMap:
        """Convert BaseModel/dict to ConfigMap (None → empty ConfigMap)."""
        if obj is None:
            return m.ConfigMap(root={})
        if isinstance(obj, m.ConfigMap):
            return obj
        if isinstance(obj, BaseModel):
            model_dump_result = obj.model_dump()
            try:
                normalized_model_dump: dict[str, object] = {}
                for key, value in model_dump_result.items():
                    normalized_value: object
                    if value is None:
                        normalized_value = ""
                    elif isinstance(
                        value, (str, int, float, bool, type(None), BaseModel)
                    ):
                        normalized_value = FlextRuntime.normalize_to_container(value)
                    else:
                        normalized_value = str(value)
                    normalized_model_dump[str(key)] = normalized_value
                return m.ConfigMap.model_validate(normalized_model_dump)
            except (TypeError, ValueError, AttributeError):
                return m.ConfigMap(root={"value": str(model_dump_result)})
        if isinstance(obj, Mapping):
            try:
                normalized_mapping: dict[str, object] = {}
                obj_mapping = _MODEL_DICT_STR_OBJECT_ADAPTER.validate_python(obj)
                for key, value in obj_mapping.items():
                    normalized_mapping_value: object = (
                        FlextRuntime.normalize_to_container(value)
                        if isinstance(
                            value, (str, int, float, bool, type(None), BaseModel)
                        )
                        else str(value)
                    )
                    normalized_mapping[str(key)] = normalized_mapping_value
                return m.ConfigMap.model_validate(normalized_mapping)
            except (TypeError, ValueError, AttributeError):
                return m.ConfigMap(root={})

        # Fallback to general value normalization
        normalized = FlextRuntime.normalize_to_container(obj)
        if isinstance(normalized, Mapping):
            normalized_obj: object = normalized
            normalized_map = FlextUtilitiesModel._normalize_str_object_mapping(
                normalized_obj
            )
            return m.ConfigMap(root=normalized_map)
        return m.ConfigMap(root={})


__all__ = ["FlextUtilitiesModel"]
