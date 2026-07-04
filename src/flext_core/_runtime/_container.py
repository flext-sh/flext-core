"""Runtime container and service normalization helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._models.containers import FlextModelsContainers as mc
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._protocols.context import FlextProtocolsContext as pcx
from flext_core._protocols.handler import FlextProtocolsHandler as ph
from flext_core._protocols.logging import FlextProtocolsLogging as pl
from flext_core._protocols.settings import FlextProtocolsSettings as ps
from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.typeadapters import FlextTypesTypeAdapters as tta
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore as ugc

from ._metadata_validation import FlextRuntimeMetadataValidation

if TYPE_CHECKING:
    from flext_core._protocols.base import FlextProtocolsBase as pb
    from flext_core._typings.services import FlextTypesServices as ts


class FlextRuntimeContainer(FlextRuntimeMetadataValidation):
    """Normalize runtime container and service registration payloads."""

    @staticmethod
    def _normalize_payload_item(
        item: pb.AttributeProbe,
        *,
        container_kind: Literal["mapping", "sequence"],
    ) -> ts.JsonPayload:
        """Normalize one container item to its canonical payload form."""
        normalized_item: ts.JsonPayload
        match item:
            case datetime():
                normalized_item = (
                    item.replace(tzinfo=UTC) if item.tzinfo is None else item
                ).isoformat()
            case Path():
                normalized_item = str(item)
            case tuple():
                normalized_item = FlextRuntimeContainer.normalize_to_metadata(item)
            case dict():
                normalized_item = tta.json_dict_adapter().validate_python(item)
            case list():
                normalized_item = list(tta.json_list_adapter().validate_python(item))
            case bool() | int() | float() | str() | None | mp.BaseModel():
                normalized_item = item
            case _:
                err_template = (
                    ce.ERR_RUNTIME_MAPPING_INVALID_TYPE
                    if container_kind == "mapping"
                    else ce.ERR_RUNTIME_SEQUENCE_INVALID_TYPE
                )
                msg = err_template.format(type_name=type(item))
                raise TypeError(msg)
        return normalized_item

    @staticmethod
    def normalize_registerable_service(
        value: ts.RegisterableService,
    ) -> ts.RegisterableService | mc.ConfigMap | mc.ObjectList:
        """Normalize container registration payloads to canonical runtime types."""
        normalized_service: ts.RegisterableService | mc.ConfigMap | mc.ObjectList
        match value:
            case Mapping():
                normalized_service = mc.ConfigMap(
                    root={
                        key_s: FlextRuntimeContainer._normalize_payload_item(
                            item,
                            container_kind="mapping",
                        )
                        for key_s, item in value.items()
                    },
                )
            case Sequence() if not isinstance(value, tb.STR_BINARY_TYPES):
                normalized_service = mc.ObjectList(
                    root=[
                        FlextRuntimeContainer._normalize_payload_item(
                            item,
                            container_kind="sequence",
                        )
                        for item in value
                    ],
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
                (pl.Logger, ps.Settings, pcx.Context, ph.Dispatcher),
            ):
                normalized_service = value
            case _:
                raise ValueError(
                    ce.ERR_RUNTIME_SERVICE_MUST_BE_REGISTERABLE.format(
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
        val: ts.JsonPayload
        | tb.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[tb.Scalar],
    ) -> ts.RuntimeData:
        """Normalize any value to RuntimeData."""
        normalized_data: ts.RuntimeData
        if val is None:
            normalized_data = ""
        elif isinstance(val, (mc.ConfigMap, mc.Dict)):
            normalized_data = FlextRuntimeContainer._normalize_dict_entries(
                list(val.root.items()),
            )
        elif isinstance(val, mc.ObjectList):
            normalized_data = list(
                tta.json_list_adapter().validate_python(
                    [
                        FlextRuntimeContainer.normalize_to_json_value(v)
                        for v in val.root
                    ],
                ),
            )
        elif isinstance(val, BaseModel):
            normalized_data = val
        elif isinstance(val, Path):
            normalized_data = str(val)
        elif ugc.scalar(val):
            normalized_data = FlextRuntimeContainer.normalize_to_json_value(val)
        elif isinstance(val, Mapping):
            normalized_data = FlextRuntimeContainer._normalize_dict_entries(
                list(val.items()),
            )
        elif isinstance(val, Sequence) and not isinstance(val, tb.STR_BYTES_TYPES):
            normalized_data = list(
                tta.json_list_adapter().validate_python(
                    [
                        FlextRuntimeContainer.normalize_to_json_value(item_raw)
                        for item_raw in val
                    ],
                ),
            )
        else:
            normalized_data = str(val)
        return normalized_data


__all__: list[str] = ["FlextRuntimeContainer"]
