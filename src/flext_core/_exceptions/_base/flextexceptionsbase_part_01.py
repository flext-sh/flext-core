"""Exception metadata normalization.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from collections.abc import Mapping, MutableMapping

from flext_core import c, m, p, t
from flext_core._exceptions import FlextExceptionsHelpers


class FlextBaseErrorMetadata:
    @staticmethod
    def normalize_metadata(
        metadata: p.HasModelDump | t.JsonValue | None,
        merged_kwargs: Mapping[str, t.JsonPayload],
    ) -> m.Metadata:
        """Normalize metadata from various input types to m.Metadata model.

        Returns:
            The resulting ``m.Metadata``.

        """
        metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
        source = (
            metadata_model.model_dump(mode="json") if metadata_model is not None else {}
        )
        merged_attrs: MutableMapping[str, t.JsonPayload | None] = (
            dict(metadata_model.attributes) if metadata_model is not None else {}
        )
        merged_attrs.update(merged_kwargs)
        return m.Metadata.model_validate({
            **source,
            c.FIELD_ATTRIBUTES: merged_attrs,
        })


__all__: list[str] = ["FlextBaseErrorMetadata"]
