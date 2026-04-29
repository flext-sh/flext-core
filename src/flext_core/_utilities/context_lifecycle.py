"""Context lifecycle operations (merge, export).

Extracted from FlextContext as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from typing import ClassVar, Self

from flext_core import (
    FlextConstants as c,
    FlextModels as m,
    FlextProtocols as p,
    FlextRuntime,
    FlextTypes as t,
    FlextUtilitiesContextCrud,
    FlextUtilitiesGuardsTypeCore,
)


class FlextUtilitiesContextLifecycle(FlextUtilitiesContextCrud):
    """Lifecycle operations (merge/export) for FlextContext."""

    logger: ClassVar[p.Logger]
    state: m.ContextRuntimeState
    initial_data: m.ContextData | t.JsonValue | None

    def export(
        self,
        *,
        include_statistics: bool = False,
        include_metadata: bool = False,
        as_dict: bool = True,
    ) -> m.ContextExport | Mapping[str, t.JsonPayload]:
        """Export context state using canonical Pydantic models."""
        all_data: dict[str, t.JsonPayload] = {}
        all_scopes = self._scope_payloads()
        for scope_name, scope_payload in all_scopes.items():
            all_data[scope_name] = self._normalize_mapping_payload(scope_payload)
        stats_dict_export: dict[str, t.JsonValue] = {}
        if include_statistics and self.state.statistics:
            stats_dict_export = dict(
                self._normalize_mapping_payload(
                    self.state.statistics.model_dump(mode="python"),
                ),
            )
        metadata_attributes: dict[str, t.JsonValue] | None = None
        if include_metadata:
            metadata_attributes = {
                k: FlextRuntime.normalize_to_metadata(v)
                for k, v in self._metadata_map().items()
            }
        export_model = m.ContextExport.model_validate({
            "data": dict(all_data),
            "metadata": (
                m.Metadata.model_validate({"attributes": metadata_attributes})
                if metadata_attributes
                else None
            ),
            "statistics": dict(stats_dict_export),
        })
        if as_dict:
            return export_model.model_dump(mode="python")
        return export_model

    @staticmethod
    def _normalize_mapping_payload(
        source: (Mapping[str, t.JsonPayload] | t.JsonMapping),
    ) -> t.JsonMapping:
        """Normalize and validate mapping payloads through canonical adapters."""
        normalized = {
            k: FlextRuntime.normalize_to_container(v) for k, v in source.items()
        }
        validated: t.JsonMapping = t.json_mapping_adapter().validate_python(normalized)
        return validated

    @staticmethod
    def _as_config_map(
        source: (Mapping[str, t.JsonPayload] | t.JsonMapping),
        label: str,
    ) -> m.ConfigMap | None:
        """Normalize an arbitrary mapping into a scope-compatible map."""
        try:
            normalized_payload = (
                FlextUtilitiesContextLifecycle._normalize_mapping_payload(
                    source,
                )
            )
            return m.ConfigMap(root=dict(normalized_payload))
        except (TypeError, ValueError, AttributeError) as exc:
            FlextUtilitiesContextLifecycle.logger.debug(
                f"Context {label} validation failed",
                exc_info=exc,
            )
            return None

    def _extract_config_map(
        self,
        other: p.Context | Mapping[str, t.JsonPayload] | t.JsonMapping,
    ) -> m.ConfigMap | None:
        """Extract a ConfigMap from any supported merge source."""
        if isinstance(other, p.Context):
            exported_result = other.export(as_dict=True)
            if isinstance(exported_result, m.ContextExport):
                exported_payload = exported_result.model_dump(mode="python")
            elif isinstance(exported_result, Mapping):
                exported_payload = exported_result
            else:
                return None
            return self._as_config_map(exported_payload, "export payload")
        return self._as_config_map(other, "export payload")

    def _apply_scoped_merge(self, exported_map: m.ConfigMap) -> None:
        """Merge exported scopes from another FlextContext."""
        for scope_name, scope_payload in exported_map.items():
            if scope_name not in c.CONTEXT_MERGEABLE_SCOPES:
                continue
            if not FlextUtilitiesGuardsTypeCore.mapping(scope_payload):
                continue
            scope_data = self._as_config_map(scope_payload, "scope payload")
            if scope_data is not None:
                self._update_contextvar(scope_name, scope_data)

    def merge(
        self,
        other: p.Context | Mapping[str, t.JsonPayload] | t.JsonMapping,
    ) -> Self:
        """Merge another context or dictionary into this context."""
        if not self.state.active:
            return self
        exported_map = self._extract_config_map(other)
        if exported_map is None:
            return self
        if isinstance(other, p.Context):
            self._apply_scoped_merge(exported_map)
        else:
            self._update_contextvar(c.ContextScope.GLOBAL, exported_map)
        return self


__all__: list[str] = ["FlextUtilitiesContextLifecycle"]
