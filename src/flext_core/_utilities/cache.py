"""Cache utilities for deterministic normalization and key management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import BaseModel

from flext_core import FlextUtilitiesGuardsTypeCore, FlextUtilitiesGuardsTypeModel, t


class FlextUtilitiesCache:
    """Cache utilities for deterministic normalization and key management."""

    @staticmethod
    def normalize_component(
        component: t.ValueOrModel | set[t.RecursiveContainer],
    ) -> t.RecursiveContainer:
        """Normalize a component recursively for consistent representation."""
        if isinstance(
            component,
            BaseModel,
        ) and FlextUtilitiesGuardsTypeModel.is_pydantic_model(component):
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in component.model_dump().items()
            }
        if FlextUtilitiesGuardsTypeCore.is_mapping(component):
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in component.items()
            }
        if isinstance(component, set):
            normalized_set_items: t.ContainerList = [
                FlextUtilitiesCache.normalize_component(item) for item in component
            ]
            return tuple(normalized_set_items)
        if FlextUtilitiesGuardsTypeCore.is_primitive(component) or component is None:
            return component
        if isinstance(component, (list, tuple)):
            return [FlextUtilitiesCache.normalize_component(item) for item in component]
        return str(component)


__all__ = ["FlextUtilitiesCache"]
