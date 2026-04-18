"""Context normalization helpers for converting values to container types.

Extracted from FlextContext as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from flext_core import FlextRuntime, p, t, u


class FlextUtilitiesContextNormalization:
    """Static normalization helpers used by FlextContext."""

    _logger: ClassVar[p.Logger]

    @staticmethod
    def _to_normalized(value: t.ValueOrModel | t.ConfigMap) -> t.RecursiveContainer:
        """Narrow ``Container | t.ModelCarrier`` to ``RecursiveContainer``.

        Pydantic model instances are converted via ``model_dump()`` so the result
        is always a plain ``RecursiveContainer`` with no model carrier left inside.
        """
        if u.pydantic_model(value):
            raw = value.model_dump()
            result: t.MutableRecursiveContainerMapping = {}
            for k, v in raw.items():
                container_val = u.normalize_to_container(v)
                if u.pydantic_model(container_val):
                    if isinstance(container_val, (t.ConfigMap, t.Dict)):
                        nested: t.MutableRecursiveContainerMapping = {}
                        for nk, nv in container_val.root.items():
                            nested[str(nk)] = (
                                FlextUtilitiesContextNormalization._to_normalized(
                                    u.normalize_to_container(nv),
                                )
                            )
                        result[str(k)] = nested
                        continue
                    result[str(k)] = str(container_val)
                else:
                    result[str(k)] = FlextRuntime.to_plain_container(container_val)
            return result
        if isinstance(value, (t.ConfigMap, t.Dict)):
            normalized_root: t.MutableRecursiveContainerMapping = {}
            for k, v in value.root.items():
                normalized_root[str(k)] = (
                    FlextUtilitiesContextNormalization._to_normalized(
                        u.normalize_to_container(v),
                    )
                )
            return normalized_root
        return value

    @staticmethod
    def _narrow_contextvar_to_configuration_dict(
        ctx_value: t.ConfigMap | t.RecursiveContainerMapping | t.ModelCarrier | None,
    ) -> t.RecursiveContainerMapping:
        """Return contextvar payload as ConfigMap with safe default."""
        if ctx_value is None:
            empty: t.RecursiveContainerMapping = {}
            return empty

        payload: Mapping[str, t.ValueOrModel] | t.RecursiveContainerMapping
        if isinstance(ctx_value, (t.ConfigMap, t.Dict)):
            payload = ctx_value.root
        elif u.pydantic_model(ctx_value):
            payload = ctx_value.model_dump()
        elif u.mapping(ctx_value):
            payload = ctx_value
        else:
            empty_fallback: t.RecursiveContainerMapping = {}
            return empty_fallback

        try:
            normalized: t.MutableRecursiveContainerMapping = {}
            mapping_value: Mapping[str, t.ValueOrModel] = dict(
                payload.items(),
            )
            for key, value in mapping_value.items():
                if str(key) != key:
                    empty_key: t.RecursiveContainerMapping = {}
                    return empty_key
                if value is None:
                    normalized[key] = None
                    continue
                normalized_value = u.normalize_to_container(value)
                metadata_target: t.ValueOrModel | t.Dict | t.ObjectList = (
                    normalized_value
                )
                if isinstance(normalized_value, dict):
                    metadata_target = t.Dict(root=normalized_value)
                elif isinstance(normalized_value, list):
                    metadata_target = t.ObjectList(
                        root=[
                            v
                            if isinstance(v, (str, int, float, bool, datetime, Path))
                            else str(v)
                            for v in normalized_value
                        ],
                    )
                if isinstance(
                    metadata_target,
                    (t.ConfigMap, t.Dict, t.ObjectList),
                ) or u.pydantic_model(metadata_target):
                    metadata_normalized = u.normalize_to_container(
                        u.normalize_to_metadata(metadata_target),
                    )
                    if isinstance(metadata_normalized, (*t.CONTAINER_TYPES,)):
                        normalized[key] = metadata_normalized
                    else:
                        normalized[key] = str(metadata_normalized)
                else:
                    normalized[key] = FlextRuntime.to_plain_container(
                        normalized_value,
                    )
            return normalized
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            FlextUtilitiesContextNormalization._logger.debug(
                "Failed to normalize contextvar payload to configuration dict",
                exc_info=exc,
            )
            empty_err: t.RecursiveContainerMapping = {}
            return empty_err


__all__: list[str] = ["FlextUtilitiesContextNormalization"]
