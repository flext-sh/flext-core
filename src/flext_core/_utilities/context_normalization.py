"""Context normalization helpers for converting values to container types.

Extracted from FlextContext as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from typing import ClassVar

from flext_core import FlextRuntime, m, p, t


class FlextUtilitiesContextNormalization:
    """Static normalization helpers used by FlextContext."""

    _logger: ClassVar[p.Logger]

    @staticmethod
    def _narrow_contextvar_to_configuration_dict(
        ctx_value: m.ConfigMap | Mapping[str, t.JsonPayload] | p.Model | None,
    ) -> t.JsonMapping:
        """Return contextvar payload as a flat container mapping with safe default."""
        if ctx_value is None:
            return {}

        payload: Mapping[str, t.JsonPayload]
        if isinstance(ctx_value, m.ConfigMap):
            payload = ctx_value.root
        elif isinstance(ctx_value, p.Model):
            dumped = ctx_value.model_dump(mode="python")
            if not isinstance(dumped, Mapping):
                return {}
            payload = t.flat_container_mapping_adapter().validate_python(dumped)
        elif isinstance(ctx_value, Mapping):
            payload = ctx_value
        else:
            return {}

        try:
            normalized: dict[str, t.JsonPayload] = {}
            for key, value in payload.items():
                if str(key) != key:
                    return {}
                normalized[key] = FlextRuntime.normalize_to_container(value)
            return t.flat_container_mapping_adapter().validate_python(normalized)
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            FlextUtilitiesContextNormalization._logger.debug(
                "Failed to normalize contextvar payload to configuration dict",
                exc_info=exc,
            )
            return {}


__all__: list[str] = ["FlextUtilitiesContextNormalization"]
