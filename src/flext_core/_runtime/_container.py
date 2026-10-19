"""Runtime container and service normalization helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeGuard

from pydantic import BaseModel
from typing_extensions import TypeForm

from flext_core import c, p, t
from flext_core._models import FlextModelsContainers
from flext_core._runtime import FlextRuntimeMetadataValidation
from flext_core._utilities import FlextUtilitiesBase, FlextUtilitiesGuardsTypeCore


class FlextRuntimeContainer(FlextRuntimeMetadataValidation):
    """Normalize runtime container and service registration payloads."""

    @staticmethod
    def _is_registerable_runtime_service(
        value: p.GuardInput,
    ) -> TypeGuard[p.RegisterableService]:
        """Narrow runtime service values accepted by the dependency container.

        Returns:
            The resulting ``TypeGuard[p.RegisterableService]``.

        """
        return callable(value) or isinstance(
            value,
            (
                p.Model,
                p.Logger,
                p.Settings,
                p.Context,
                p.Dispatcher,
            ),
        )

    @staticmethod
    def _normalize_payload_item[T](
        item: T,
        *,
        container_kind: Literal["mapping", "sequence"],
    ) -> t.JsonPayload:
        """Normalize one container item to its canonical payload form.

        Returns:
            The resulting ``ts.JsonPayload``.

        Raises:
            TypeError: Always.

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
                normalized_item = FlextRuntimeContainer.normalize_to_metadata(
                    item,
                )
            case dict():
                normalized_item = FlextUtilitiesBase.type_adapter(
                    TypeForm(t.JsonDict)
                ).validate_python(item)
            case list():
                normalized_item = list(
                    FlextUtilitiesBase.type_adapter(
                        TypeForm(t.JsonList)
                    ).validate_python(item),
                )
            case bool() | int() | float() | str() | None | BaseModel():
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
        value: p.GuardInput,
    ) -> p.RegisterableService:
        """Normalize container registration payloads to canonical runtime types.

        Returns:
            The resulting ``p.RegisterableService``.

        Raises:
            ValueError: Always.

        """
        if isinstance(value, Mapping):
            mapping_value = FlextModelsContainers.ConfigMap.model_validate(value)
            for key, item in mapping_value.root.items():
                mapping_value.root[key] = FlextRuntimeContainer._normalize_payload_item(
                    item, container_kind="mapping"
                )
            return mapping_value
        if isinstance(value, Sequence) and not isinstance(value, c.STR_BINARY_TYPES):
            sequence_value = FlextModelsContainers.ObjectList.model_validate(value)
            for index, item in enumerate(sequence_value.root):
                sequence_value.root[index] = (
                    FlextRuntimeContainer._normalize_payload_item(
                        item, container_kind="sequence"
                    )
                )
            return sequence_value
        if (
            isinstance(value, (str, int, float, bool, bytes, datetime, Path, BaseModel))
            or value is None
        ):
            return value
        if FlextRuntimeContainer._is_registerable_runtime_service(value):
            return value
        raise ValueError(
            c.ERR_RUNTIME_SERVICE_MUST_BE_REGISTERABLE.format(
                type_name=type(value).__name__,
            ),
        )

    @staticmethod
    def validate_callable_input[TCallable](value: TCallable, subject: str) -> TCallable:
        """Validate that a single runtime input is callable.

        Returns:
            The resulting ``TCallable``.

        Raises:
            TypeError: If ``not callable(value)``.

        """
        if not callable(value):
            msg = f"{subject} must be callable, got {value.__class__.__name__}"
            raise TypeError(msg)
        return value

    @staticmethod
    def normalize_to_container(
        val: t.JsonPayload | AbstractSet[t.Scalar],
    ) -> t.RuntimeData:
        """Normalize any value to RuntimeData.

        Returns:
            The resulting ``ts.RuntimeData``.

        Raises:
            TypeError: If the value has no supported container representation.

        """
        normalized_data: t.RuntimeData
        if val is None:
            normalized_data = ""
        elif isinstance(
            val,
            (FlextModelsContainers.ConfigMap, FlextModelsContainers.Dict),
        ):
            normalized_data = FlextRuntimeContainer._normalize_dict_entries(
                list(val.root.items()),
            )
        elif isinstance(val, FlextModelsContainers.ObjectList):
            normalized_data = list(
                FlextUtilitiesBase.type_adapter(TypeForm(t.JsonList)).validate_python([
                    FlextRuntimeContainer.normalize_to_json_value(v) for v in val.root
                ]),
            )
        elif isinstance(val, BaseModel):
            normalized_data = val
        elif isinstance(val, Path):
            normalized_data = str(val)
        elif FlextUtilitiesGuardsTypeCore.scalar(val):
            normalized_data = FlextRuntimeContainer.normalize_to_json_value(val)
        elif isinstance(val, Mapping):
            normalized_data = FlextRuntimeContainer._normalize_dict_entries(
                list(val.items()),
            )
        elif isinstance(val, Sequence) and not isinstance(val, c.STR_BYTES_TYPES):
            normalized_data = list(
                FlextUtilitiesBase.type_adapter(TypeForm(t.JsonList)).validate_python([
                    FlextRuntimeContainer.normalize_to_json_value(item_raw)
                    for item_raw in val
                ]),
            )
        elif isinstance(val, AbstractSet):
            normalized_data = FlextRuntimeContainer.normalize_to_json_value(val)
        else:
            msg = c.ERR_RUNTIME_CONTAINER_INVALID_TYPE.format(
                type_name=type(val).__name__,
            )
            raise TypeError(msg)
        return normalized_data


__all__: list[str] = ["FlextRuntimeContainer"]
