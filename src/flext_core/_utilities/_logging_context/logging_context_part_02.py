"""Logging context binding and value normalization.

Extracted from FlextUtilitiesLogging as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from flext_core import m, p, t
from flext_core._utilities import FlextUtilitiesBase, FlextUtilitiesPydantic
from flext_core._utilities._logging_context import FlextUtilitiesLoggingContextPart01


class FlextUtilitiesLoggingContextPart02(FlextUtilitiesLoggingContextPart01):
    @staticmethod
    def to_container_value(value: p.LogValue | t.JsonValue) -> t.JsonValue:
        """Admit ordinary fields before canonical JSON normalization.

        Native exception triples belong to reserved ``exc_info`` and bypass
        this boundary in the bound logger's native emission path.

        Returns:
            The JSON-compatible logging value.

        """
        if isinstance(value, BaseException):
            return str(value)
        if isinstance(value, tuple):
            admitted = FlextUtilitiesBase.type_adapter(
                t.TypeForm(t.JsonPayload),
                config=m.ArbitraryTypesModel.model_config,
            ).validate_python(list(value))
            return FlextUtilitiesPydantic.normalize_to_json_value(admitted)
        return FlextUtilitiesPydantic.normalize_to_json_value(value)

    @staticmethod
    def to_container_context(
        context: Mapping[str, p.LogValue | t.JsonValue],
    ) -> t.JsonMapping:
        """Serialize owned fields once before entering the native processor chain.

        Returns:
            The JSON-compatible context mapping.

        """
        return {
            key: FlextUtilitiesLoggingContextPart02.to_container_value(value)
            for key, value in context.items()
        }


__all__: list[str] = ["FlextUtilitiesLoggingContextPart02"]
