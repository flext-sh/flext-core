"""Logging context binding and value normalization.

Extracted from FlextUtilitiesLogging as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypeForm

from flext_core import FlextRuntime, p, t
from flext_core._models import FlextModelsBase
from flext_core._typings import FlextTypesServices
from flext_core._utilities import FlextUtilitiesBase
from flext_core._utilities._logging_context_parts import logging_context_part_01


class FlextUtilitiesLoggingContext(
    logging_context_part_01.FlextUtilitiesLoggingContext
):
    @staticmethod
    def _to_container_value(value: p.LogValue | t.JsonValue) -> t.JsonValue:
        """Admit ordinary fields before canonical JSON normalization.

        Native exception triples belong to reserved ``exc_info`` and bypass
        this boundary in the bound logger's native emission path.

        Returns:
            The JSON-compatible logging value.

        """
        if isinstance(value, BaseException):
            return str(value)
        admitted = FlextUtilitiesBase.type_adapter(
            TypeForm(FlextTypesServices.JsonPayload | Any),
            config=FlextModelsBase.ArbitraryTypesModel.model_config,
        ).validate_python(value)
        return FlextRuntime.normalize_to_json_value(admitted)

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
