"""Logging context binding and value normalization.

Extracted from FlextUtilitiesLogging as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic_core import to_jsonable_python
from typing_extensions import TypeForm

from flext_core import FlextRuntime, p, t
from flext_core._utilities import FlextUtilitiesBase, FlextUtilitiesGuardsTypeModel
from flext_core._utilities._logging_context_parts import logging_context_part_01


class FlextUtilitiesLoggingContext(
    logging_context_part_01.FlextUtilitiesLoggingContext
):
    @staticmethod
    def _to_container_value(value: p.LogValue | t.JsonValue) -> t.JsonValue:
        """Adapt owned values only at the native logging SDK boundary.

        Returns:
            The JSON-compatible logging value.

        """
        if isinstance(value, BaseException):
            return str(value)
        if isinstance(value, tuple):
            return FlextUtilitiesBase.type_adapter(
                TypeForm(t.JsonValue)
            ).validate_python(to_jsonable_python(value), strict=True)
        if FlextUtilitiesGuardsTypeModel.has_model_dump(value):
            return FlextUtilitiesBase.type_adapter(
                TypeForm(t.JsonValue)
            ).validate_python(
                value.model_dump(mode="json"),
            )
        return FlextRuntime.normalize_to_json_value(value)

    @staticmethod
    def to_container_context(
        context: t.MappingKV[str, p.LogValue | t.JsonValue],
    ) -> t.JsonMapping:
        """Serialize owned fields once before entering the native processor chain.

        Returns:
            The JSON-compatible context mapping.

        """
        return {
            key: FlextUtilitiesLoggingContext._to_container_value(value)
            for key, value in context.items()
        }


__all__: list[str] = ["FlextUtilitiesLoggingContext"]
