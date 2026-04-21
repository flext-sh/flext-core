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

from flext_core import FlextRuntime, FlextUtilitiesContextCrud, c, m, p, t, u


class FlextUtilitiesContextLifecycle(FlextUtilitiesContextCrud):
    """Lifecycle operations (merge/export) for FlextContext."""

    _logger: ClassVar[p.Logger]
    _state: m.ContextRuntimeState
    initial_data: m.ContextData | m.ConfigMap | None

    _MERGEABLE_SCOPES: ClassVar[frozenset[str]] = frozenset({
        c.ContextScope.GLOBAL,
        c.ContextScope.USER,
        c.ContextScope.SESSION,
    })

    def export(
        self,
        *,
        include_statistics: bool = False,
        include_metadata: bool = False,
        as_dict: bool = True,
    ) -> m.ContextExport | Mapping[str, t.RuntimeData]:
        """Export context state using canonical Pydantic models."""
        all_data: dict[str, t.RuntimeData] = {}
        all_scopes = self._scope_payloads()
        for scope_name, scope_payload in all_scopes.items():
            all_data[scope_name] = self._normalize_mapping_payload(scope_payload)
        stats_dict_export: dict[str, t.Container] = {}
        if include_statistics and self._state.statistics:
            stats_dict_export = dict(
                self._normalize_mapping_payload(
                    self._state.statistics.model_dump(mode="python"),
                ),
            )
        metadata_attributes: dict[str, t.MetadataValue] | None = None
        if include_metadata:
            metadata_attributes = {
                str(k): u.normalize_to_metadata(
                    FlextRuntime.to_plain_container(
                        FlextRuntime.normalize_to_container(v)
                    ),
                )
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
        source: (
            Mapping[str, t.RuntimeData]
            | Mapping[str, t.ValueOrModel]
            | Mapping[str, t.Container]
        ),
    ) -> t.FlatContainerMapping:
        """Normalize and validate mapping payloads through canonical adapters."""
        normalized = {
            str(k): FlextRuntime.to_plain_container(
                FlextRuntime.normalize_to_container(v),
            )
            for k, v in source.items()
        }
        return t.flat_container_mapping_adapter().validate_python(normalized)

    @staticmethod
    def _as_config_map(
        source: (
            Mapping[str, t.RuntimeData]
            | Mapping[str, t.ValueOrModel]
            | Mapping[str, t.Container]
        ),
        label: str,
    ) -> m.ConfigMap | None:
        """Normalize an arbitrary mapping into a scope-compatible map."""
        try:
            normalized_payload = (
                FlextUtilitiesContextLifecycle._normalize_mapping_payload(source)
            )
            return m.ConfigMap(root=dict(normalized_payload))
        except (TypeError, ValueError, AttributeError) as exc:
            FlextUtilitiesContextLifecycle._logger.debug(
                f"Context {label} validation failed",
                exc_info=exc,
            )
            return None

    def _extract_config_map(
        self,
        other: p.Context | Mapping[str, t.RuntimeData] | Mapping[str, t.Container],
    ) -> m.ConfigMap | None:
        """Extract a ConfigMap from any supported merge source."""
        match other:
            case _ if isinstance(other, p.Context):
                exported_result = other.export(as_dict=True)
                exported_payload: Mapping[str, t.RuntimeData] | None
                if isinstance(exported_result, m.ContextExport):
                    exported_payload = exported_result.model_dump(mode="python")
                elif isinstance(exported_result, Mapping):
                    exported_payload = exported_result
                else:
                    exported_payload = None
                if exported_payload is None:
                    return None
                return self._as_config_map(exported_payload, "export payload")
            case _ if isinstance(other, Mapping):
                return self._as_config_map(other, "export payload")
            case _:
                return None

    def _apply_scoped_merge(self, exported_map: m.ConfigMap) -> None:
        """Merge exported scopes from another FlextContext."""
        for scope_name, scope_payload in exported_map.items():
            if scope_name not in self._MERGEABLE_SCOPES:
                continue
            if not u.mapping(scope_payload):
                continue
            scope_data = self._as_config_map(scope_payload, "scope payload")
            if scope_data is not None:
                self._update_contextvar(scope_name, scope_data)

    def merge(
        self,
        other: p.Context | Mapping[str, t.RuntimeData] | Mapping[str, t.Container],
    ) -> Self:
        """Merge another context or dictionary into this context."""
        if not self._state.active:
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
