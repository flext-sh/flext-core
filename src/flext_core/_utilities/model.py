"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core._models.base import FlextModelFoundation
from flext_core.models import m
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

T_Model = TypeVar("T_Model", bound=BaseModel)


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

    _pydantic_scalar_adapter: ClassVar[TypeAdapter[str | int | float | bool | None]] = (
        TypeAdapter(str | int | float | bool | None)
    )

    @staticmethod
    def from_dict[M: BaseModel](
        model_cls: type[M],
        data: Mapping[str, t.JsonValue],
        *,
        strict: bool = False,
    ) -> r[M]:
        """Create Pydantic model from dict with r.

        Example:
             result = u.Model.from_dict(
                 UserModel,
                 {"status": "active", "name": "John"},
             )
             if result.is_success:
                 user: UserModel = result.value

        """
        try:
            # model_validate returns M (the model type)
            instance = model_cls.model_validate(data, strict=strict)
            # Type narrowing: instance is M from model_validate return type
            return r[M].ok(instance)
        except (ValidationError, TypeError, ValueError) as e:
            return r[M].fail(f"Model validation failed: {e}")

    @staticmethod
    def from_kwargs[M: BaseModel](
        model_cls: type[M],
        **kwargs: t.ScalarValue,
    ) -> r[M]:
        """Create Pydantic model from kwargs with r.

        Accepts any type in kwargs - Pydantic 2 field_validators will handle
        type conversions automatically (e.g., str → Path, dict → BaseModel, etc.).
        All parameter validation and conversion happens via Pydantic 2 Field constraints
        and field_validators defined in the model.

        Example:
             result = u.Model.from_kwargs(
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
            # Pydantic 2 model_validate() accepts any dict-like structure
            # field_validators in the model will handle type conversions automatically
            # model_validate returns M (the model type)
            instance = model_cls.model_validate(kwargs)
            # Type narrowing: instance is M from model_validate return type
            return r[M].ok(instance)
        except (ValidationError, TypeError, ValueError) as e:
            return r[M].fail(f"Model validation failed: {e}")

    @staticmethod
    def merge_defaults[M: BaseModel](
        model_cls: type[M],
        defaults: Mapping[str, t.JsonValue],
        overrides: Mapping[str, t.JsonValue],
    ) -> r[M]:
        """Merge defaults with overrides and create model.

        Example:
             DEFAULTS = {"status": Status.PENDING, "retries": 3}

             result = u.Model.merge_defaults(
                 ConfigModel,
                 defaults=DEFAULTS,
                 overrides={"status": "active"},  # Overrides
             )
             # result.value.status = Status.ACTIVE
             # result.value.retries = 3

        """
        merged = {**defaults, **overrides}
        return FlextUtilitiesModel.from_dict(model_cls, merged)

    @staticmethod
    def update[M: BaseModel](
        instance: M,
        **updates: t.JsonValue,
    ) -> r[M]:
        """Update existing model with new values.

        Example:
             user = UserModel(status=Status.ACTIVE, name="John")
             result = u.Model.update(user, status="inactive")
             # result.value = UserModel with status=Status.INACTIVE

        """
        try:
            # Use model_copy with update - modern Pydantic approach
            # This preserves the type M without needing casts or recreating
            # model_copy returns M (same type as instance)
            updated_instance = instance.model_copy(update=updates)
            # Type narrowing: updated_instance is M from model_copy return type
            return r[M].ok(updated_instance)
        except (AttributeError, TypeError, ValueError) as e:
            return r[M].fail(f"Model update failed: {e}")

    @staticmethod
    def to_dict(
        instance: BaseModel,
        *,
        by_alias: bool = False,
        exclude_none: bool = False,
    ) -> Mapping[str, t.ScalarValue]:
        """Convert model to dict (simple wrapper).

        Example:
             user = UserModel(status=Status.ACTIVE, name="John")
             data = u.Model.to_dict(user)
             # data = {"status": "active", "name": "John"}

        """
        return instance.model_dump(
            by_alias=by_alias,
            exclude_none=exclude_none,
        )

    @staticmethod
    def normalize_to_metadata(
        value: t.ScalarValue | m.ConfigMap | FlextModelFoundation.Metadata | None,
    ) -> FlextModelFoundation.Metadata:  # Returns m.Metadata at runtime
        """Normalize any value to FlextModelFoundation.Metadata.

        Business Rule: Always returns Metadata, never None.
        Uses FlextRuntime guards and normalization methods for automatic
        type checking and value normalization. Eliminates need for defensive
        fallbacks by centralizing all metadata normalization logic.

        Args:
            value: None, dict, Mapping, Metadata, or any t.ConfigMapValue

        Returns:
            FlextModelFoundation.Metadata: Normalized metadata (empty attributes
                if input was None or empty dict)

        Raises:
            TypeError: If value is not None, dict-like, or Metadata instance

        Example:
            >>> u.Model.normalize_to_metadata(None)
            Metadata(attributes={})
            >>> u.Model.normalize_to_metadata({"key": "value"})
            Metadata(attributes={"key": "value"})
            >>> u.Model.normalize_to_metadata(Metadata(attributes={"a": 1}))
            Metadata(attributes={"a": 1})

        """
        # Handle None - return empty Metadata
        if value is None:
            return FlextModelFoundation.Metadata(attributes={})

        # Handle existing Metadata instance - return as-is
        if isinstance(value, FlextModelFoundation.Metadata):
            return value

        # Handle dict-like values (dict or m.ConfigMap)
        if FlextRuntime.is_dict_like(value):
            attributes = {}
            for key, val in value.items():
                attributes[str(key)] = FlextRuntime.normalize_to_metadata_value(val)

            return FlextModelFoundation.Metadata.model_validate({
                "attributes": attributes
            })

        # Invalid type - raise TypeError
        msg = (
            f"metadata must be None, dict, or FlextModelFoundation.Metadata, "
            f"got {value.__class__.__name__}"
        )
        raise TypeError(msg)

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
    ) -> Mapping[str, t.ScalarValue]:
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
            >>> data = u.Model.dump(user, exclude_none=True)
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
    def load[T_Model: BaseModel](
        model_cls: type[T_Model],
        data: t.ConfigMap,
    ) -> r[T_Model]:
        """Load Pydantic model from mapping with FlextResult.

        Generic replacement for: Model.model_validate(data) with error handling.

        Args:
            model_cls: Pydantic model class to instantiate.
            data: Dictionary or mapping to validate.

        Returns:
            FlextResult containing model instance or error message.

        Example:
            >>> result = u.Model.load(UserModel, {"status": "active", "name": "John"})
            >>> if result.is_success:
            ...     user: UserModel = result.value

        """
        try:
            instance = model_cls.model_validate(data)
            return r[T_Model].ok(instance)
        except ValidationError as e:
            return r[T_Model].fail(f"Model validation failed: {e}")

    @staticmethod
    def normalize_to_pydantic_dict(
        data: t.ConfigMap | None,
    ) -> Mapping[str, t.PydanticConfigValue]:
        """Convert EventDataMapping to Pydantic-safe PydanticConfigDict.

        Normalizes PayloadValue values to the restricted PydanticConfigValue type
        that Pydantic can generate schemas for without recursion issues.

        Args:
            data: EventDataMapping (Mapping[str, PayloadValue]) or None

        Returns:
            Mapping[str, t.ConfigMapValue]: Mapping with Pydantic-safe values

        Example:
            >>> u.Model.normalize_to_pydantic_dict(None)
            {}
            >>> u.Model.normalize_to_pydantic_dict({"key": "value"})
            {"key": "value"}
            >>> u.Model.normalize_to_pydantic_dict({"obj": SomeModel()})
            {"obj": "SomeModel(...)"}  # Complex types converted to string

        """
        if not data:
            return {}
        result = {}
        for key, value in data.items():
            result[key] = FlextUtilitiesModel._normalize_to_pydantic_value(value)
        return result

    @staticmethod
    def _normalize_to_pydantic_value(
        value: t.ConfigMapValue,
    ) -> t.PydanticConfigValue:
        """Normalize PayloadValue to Pydantic-safe PydanticConfigValue.

        Converts complex types to strings, preserves primitives.

        Args:
            value: PayloadValue value to normalize

        Returns:
            t.PydanticConfigValue: Pydantic-safe value

        """
        match value:
            case None:
                return None
            case bool() | int() | float() | str():
                return value
            case list() as items:
                normalized_items: list[str | int | float | bool | None] = []
                for item in items:
                    try:
                        normalized_items.append(
                            FlextUtilitiesModel._pydantic_scalar_adapter.validate_python(
                                item
                            )
                        )
                    except ValidationError:
                        normalized_items.append(str(item))
                return normalized_items
            case tuple() as items:
                normalized_tuple_items: list[str | int | float | bool | None] = []
                for item in items:
                    try:
                        normalized_tuple_items.append(
                            FlextUtilitiesModel._pydantic_scalar_adapter.validate_python(
                                item
                            )
                        )
                    except ValidationError:
                        normalized_tuple_items.append(str(item))
                return normalized_tuple_items
            case _:
                # Convert any other type to string representation
                return str(value)


__all__ = [
    "FlextUtilitiesModel",
]
