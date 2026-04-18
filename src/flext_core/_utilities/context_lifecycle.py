"""Context lifecycle operations (merge, export).

Extracted from FlextContext as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import ClassVar, Self

from flext_core import FlextRuntime, c, m, p, t, u
from flext_core._utilities.context_crud import FlextUtilitiesContextCrud


class FlextUtilitiesContextLifecycle(FlextUtilitiesContextCrud):
    """Lifecycle operations (merge/export) for FlextContext."""

    _logger: ClassVar[p.Logger]
    _state: m.ContextRuntimeState
    initial_data: m.ContextData | t.ConfigMap | None

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
    ) -> m.ContextExport | t.RecursiveContainerMapping:
        """Export context data for serialization or debugging."""
        all_data: t.ConfigMap = t.ConfigMap(root={})
        all_scopes = self._scope_payloads()
        all_data.update(dict(all_scopes))
        stats_dict_export: t.ConfigMap | None = None
        if include_statistics and self._state.statistics:
            stats_dict_export = t.ConfigMap(root=self._state.statistics.model_dump())
        metadata_dict_export: t.RecursiveContainerMapping | None = None
        if include_metadata:
            metadata_dict_export = self._metadata_map()
        metadata_for_model: t.ConfigMap | None = None
        if metadata_dict_export:
            normalized_metadata_map: MutableMapping[str, t.ValueOrModel] = {}
            for k, v in metadata_dict_export.items():
                metadata_value: t.ValueOrModel | t.ConfigMap = v
                if u.mapping(v):
                    metadata_value = t.ConfigMap(
                        root=dict(v),
                    )
                normalized_metadata_map[k] = FlextRuntime.to_plain_container(
                    u.normalize_to_container(
                        u.normalize_to_metadata(metadata_value),
                    ),
                )
            metadata_for_model = t.ConfigMap(root=normalized_metadata_map)
        statistics_mapping: t.Dict = t.Dict(
            root=dict(stats_dict_export or t.ConfigMap(root={})),
        )
        if as_dict:
            result_dict: t.MutableRecursiveContainerMapping = dict(all_scopes)
            if include_statistics and stats_dict_export:
                stats_items: t.RecursiveContainerMapping = {
                    sk: self._to_normalized(sv) for sk, sv in stats_dict_export.items()
                }
                result_dict["statistics"] = stats_items
            if include_metadata and metadata_dict_export:
                metadata_container: t.ConfigMap = t.ConfigMap(
                    root=dict(metadata_dict_export),
                )
                meta_items: t.RecursiveContainerMapping = {
                    mk: self._to_normalized(mv) for mk, mv in metadata_container.items()
                }
                result_dict[c.FIELD_METADATA] = meta_items
            return result_dict
        metadata_root: t.ConfigMap | None = (
            t.ConfigMap(
                root={
                    k: u.normalize_to_container(v)
                    for k, v in metadata_for_model.items()
                },
            )
            if metadata_for_model
            else None
        )
        return m.ContextExport(
            data=dict(all_data),
            metadata=m.Metadata(
                attributes={
                    key: u.normalize_to_metadata(value)
                    for key, value in metadata_root.items()
                },
            )
            if metadata_root
            else None,
            statistics={
                key: self._to_normalized(
                    u.normalize_to_container(
                        u.normalize_to_metadata(value),
                    ),
                )
                for key, value in statistics_mapping.items()
            },
        )

    @staticmethod
    def _as_config_map(
        source: Mapping[str, t.ValueOrModel] | t.RecursiveContainerMapping,
        label: str,
    ) -> t.ConfigMap | None:
        """Try to wrap a mapping as ConfigMap, logging on failure."""
        try:
            return t.ConfigMap(root=dict(source))
        except (TypeError, ValueError, AttributeError) as exc:
            FlextUtilitiesContextLifecycle._logger.debug(
                f"Context {label} validation failed",
                exc_info=exc,
            )
            return None

    def _extract_config_map(
        self,
        other: p.Context | t.ConfigMap | t.RecursiveContainerMapping,
    ) -> t.ConfigMap | None:
        """Extract a ConfigMap from any supported merge source."""
        match other:
            case _ if u.context(other):
                exported_result = other.export(as_dict=True)
                if u.pydantic_model(exported_result):
                    return None
                return self._as_config_map(exported_result, "export payload")
            case t.ConfigMap():
                return other
            case _ if isinstance(other, Mapping):
                return self._as_config_map(other, "export payload")
            case _:
                return None

    def _apply_scoped_merge(self, exported_map: t.ConfigMap) -> None:
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
        other: p.Context | t.ConfigMap | t.RecursiveContainerMapping,
    ) -> Self:
        """Merge another context or dictionary into this context."""
        if not self._state.active:
            return self
        exported_map = self._extract_config_map(other)
        if exported_map is None:
            return self
        if u.context(other):
            self._apply_scoped_merge(exported_map)
        else:
            self._update_contextvar(c.ContextScope.GLOBAL, exported_map)
        return self


__all__: list[str] = ["FlextUtilitiesContextLifecycle"]
