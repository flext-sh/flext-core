"""Exception metadata normalization.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c, m
from flext_core._exceptions import FlextExceptionsHelpers
from flext_core._protocols import FlextProtocolsResult

if TYPE_CHECKING:
    from flext_core._typings import FlextTypesServices, FlextTypingBase


class FlextBaseErrorMetadataMixin:
    @staticmethod
    def normalize_metadata(
        metadata: FlextProtocolsResult.HasModelDump | FlextTypingBase.JsonValue | None,
        merged_kwargs: FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload],
    ) -> m.Metadata:
        """Normalize metadata from various input types to m.Metadata model.

        Returns:
            The resulting ``m.Metadata``.

        """
        metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
        source = (
            metadata_model.model_dump(mode="json") if metadata_model is not None else {}
        )
        merged_attrs: FlextTypingBase.MutableMappingKV[
            str, FlextTypesServices.JsonPayload | None
        ] = dict(metadata_model.attributes) if metadata_model is not None else {}
        merged_attrs.update(merged_kwargs)
        return m.Metadata.model_validate({
            **source,
            c.FIELD_ATTRIBUTES: merged_attrs,
        })


__all__: list[str] = ["FlextBaseErrorMetadataMixin"]
