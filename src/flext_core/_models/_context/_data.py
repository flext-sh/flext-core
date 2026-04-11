"""Context data models with serialization validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    field_validator,
)

from flext_core import (
    FlextModelsBase,
    FlextRuntime,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    c,
    t,
)


class FlextModelsContextData:
    """Namespace for context data models."""

    @staticmethod
    def normalize_to_mapping(
        v: t.ValueOrModel,
    ) -> t.ContainerMapping:
        if v is None:
            out: t.ContainerMapping = {}
            return out
        if FlextUtilitiesGuardsTypeCore.mapping(v):
            validated = t.dict_str_metadata_adapter().validate_python(
                v,
            )
            return dict(validated)
        if FlextUtilitiesGuardsTypeModel.pydantic_model(v):
            return v.model_dump()
        msg = c.ERR_CONTEXT_CANNOT_NORMALIZE_TYPE_TO_MAPPING.format(
            type_name=type(v).__name__,
        )
        raise ValueError(msg)

    @staticmethod
    def normalize_metadata_before(
        v: t.ValueOrModel | None,
    ) -> t.ValueOrModel | None:
        if v is None:
            return None
        if isinstance(v, FlextModelsBase.Metadata):
            return v
        if FlextUtilitiesGuardsTypeCore.mapping(v):
            return FlextModelsBase.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: t.dict_str_metadata_adapter().validate_python(
                    v,
                ),
            })
        return v

    class SerializableDataValidatorMixin:
        """Mixin validating that data is JSON-serializable for context models."""

        @field_validator("data", mode="before")
        @classmethod
        def validate_dict_serializable(
            cls,
            v: t.Dict | t.ScalarMapping | BaseModel | None,
        ) -> t.ContainerMapping:
            """Validate that data values are JSON-serializable."""
            working_value: t.ContainerMapping
            normalized_mapping: Mapping[str, t.ValueOrModel]
            if v is None:
                empty: t.ContainerMapping = {}
                return empty
            if isinstance(v, FlextModelsBase.Metadata):
                normalized_metadata: Mapping[str, t.ValueOrModel] = {
                    key: FlextRuntime.normalize_to_container(value)
                    for key, value in v.attributes.items()
                }
                normalized_mapping = normalized_metadata
            elif FlextUtilitiesGuardsTypeModel.pydantic_model(v):
                dump_result = v.model_dump()
                normalized_mapping = {
                    str(k): FlextRuntime.normalize_to_container(dv)
                    for k, dv in dump_result.items()
                }
            else:
                normalized_mapping = dict(v)
            working_value = {
                str(
                    k,
                ): FlextModelsContextData.ContextData.normalize_to_serializable_value(
                    val,
                )
                for k, val in normalized_mapping.items()
            }
            FlextModelsContextData.ContextData.check_json_serializable(working_value)
            return dict(working_value)

    class ContextData(
        SerializableDataValidatorMixin,
        FlextModelsBase.FlexibleInternalModel,
    ):
        """Lightweight container for initializing context state."""

        data: Annotated[
            t.Dict,
            Field(
                description="Initial context data as key-value pairs",
            ),
        ] = Field(default_factory=t.Dict)
        metadata: Annotated[
            FlextModelsBase.Metadata | t.Dict | None,
            BeforeValidator(
                lambda v: FlextModelsContextData.normalize_metadata_before(v),
            ),
            Field(
                default=None,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = None

        @classmethod
        def check_json_serializable(cls, obj: t.ValueOrModel, path: str = "") -> None:
            """Recursively check if a canonical container value is JSON-serializable."""
            if obj is None or FlextUtilitiesGuardsTypeCore.primitive(obj):
                return
            if FlextUtilitiesGuardsTypeCore.dict_like(obj):
                for key, val in obj.items():
                    cls.check_json_serializable(val, f"{path}.{key}")
                return
            if FlextUtilitiesGuardsTypeCore.list_like(obj) and (
                not isinstance(obj, (str, bytes))
            ):
                seq_obj: t.ContainerList = obj
                for i, item in enumerate(seq_obj):
                    cls.check_json_serializable(item, f"{path}[{i}]")
                return
            msg = f"Non-JSON-serializable type {obj.__class__.__name__} at {path}"
            raise TypeError(msg)

        @classmethod
        def normalize_to_serializable_value(
            cls,
            val: t.ValueOrModel,
        ) -> t.RecursiveContainer:
            normalized = cls.normalize_to_container(val)
            if normalized is None or FlextUtilitiesGuardsTypeCore.primitive(normalized):
                return normalized
            if isinstance(
                normalized,
                (t.ConfigMap, t.Dict),
            ):
                return {
                    str(key): cls.normalize_to_serializable_value(item_value)
                    for key, item_value in normalized.root.items()
                }
            if FlextUtilitiesGuardsTypeModel.pydantic_model(normalized):
                dumped_model = normalized.model_dump()
                return {
                    str(key): cls.normalize_to_serializable_value(item_value)
                    for key, item_value in dumped_model.items()
                }
            if FlextUtilitiesGuardsTypeCore.mapping(normalized):
                normalized_map = t.dict_str_metadata_adapter().validate_python(
                    normalized,
                )
                return {
                    str(key): cls.normalize_to_serializable_value(val)
                    for key, val in normalized_map.items()
                }
            if FlextUtilitiesGuardsTypeCore.list_like(normalized):
                return [
                    cls.normalize_to_serializable_value(item) for item in normalized
                ]
            return str(normalized)

        @staticmethod
        def normalize_to_container(
            val: t.ValueOrModel,
        ) -> t.ValueOrModel:
            """Normalize to container; raises TypeError for non-normalizable types."""
            if val is None:
                return ""
            if isinstance(val, t.SCALAR_TYPES):
                return val
            if FlextUtilitiesGuardsTypeModel.pydantic_model(val):
                return val
            if FlextUtilitiesGuardsTypeCore.dict_like(
                val,
            ) or FlextUtilitiesGuardsTypeCore.list_like(val):
                return FlextRuntime.normalize_to_container(val)
            if hasattr(val, "__iter__"):
                return str(val)
            msg = f"Non-normalizable type {type(val).__name__}"
            raise TypeError(msg)


__all__ = ["FlextModelsContextData"]
