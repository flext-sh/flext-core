"""Runtime bootstrap option resolution.

Houses :meth:`resolve_runtime_options` and its helpers. Split from
``model_runtime.py`` so each layer stays under the 200-LOC cap while
the runtime DSL keeps composing via MRO inheritance.
"""

from __future__ import annotations

from collections.abc import Mapping

from flext_core import (
    FlextModelsService as ms,
    FlextUtilitiesModel,
    c,
    p,
    t,
)


class FlextUtilitiesModelOptions(FlextUtilitiesModel):
    """Bootstrap-options resolver — RuntimeBootstrapOptions normalization."""

    @staticmethod
    def _runtime_option_updates_from_source(
        source: p.Base,
    ) -> Mapping[str, t.JsonPayload]:
        """Extract runtime bootstrap fields from a service-like instance."""
        field_map = {
            "runtime_settings": "settings",
            "initial_context": "context",
            "runtime_dispatcher": "dispatcher",
            "runtime_registry": "registry",
        }
        option_fields = (
            "runtime_settings",
            "settings_type",
            "settings_overrides",
            "initial_context",
            "dispatcher",
            "registry",
            "runtime_dispatcher",
            "runtime_registry",
            "subproject",
            "services",
            "factories",
            "resources",
            "container_overrides",
            "wire_modules",
            "wire_packages",
            "wire_classes",
        )
        updates: dict[str, t.JsonPayload] = {}
        for attr_name in option_fields:
            value = getattr(source, attr_name, None)
            if value is not None:
                updates[field_map.get(attr_name, attr_name)] = value
        return updates

    @classmethod
    def _resolve_from_mapping(
        cls,
        source: Mapping[str, t.JsonPayload],
    ) -> ms.RuntimeBootstrapOptions:
        """Validate a mapping into RuntimeBootstrapOptions, sanitizing wire_packages."""
        source_dict = dict(source)
        try:
            return ms.RuntimeBootstrapOptions.model_validate(source_dict)
        except c.ValidationError:
            sanitized: dict[str, t.JsonPayload | None] = dict(source_dict)
            wire_packages_raw = sanitized.get("wire_packages")
            if isinstance(wire_packages_raw, (list, tuple)):
                normalized = [w for w in wire_packages_raw if isinstance(w, str)]
                sanitized["wire_packages"] = (
                    normalized if len(normalized) == len(wire_packages_raw) else None
                )
            return ms.RuntimeBootstrapOptions.model_validate(sanitized)

    @classmethod
    def resolve_runtime_options(
        cls,
        source: (
            ms.RuntimeBootstrapOptions | Mapping[str, t.JsonPayload] | p.Base | None
        ) = None,
        **overrides: t.JsonPayload,
    ) -> ms.RuntimeBootstrapOptions:
        """Resolve runtime options from models, mappings, or service instances."""
        resolved = ms.RuntimeBootstrapOptions()
        if source is not None:
            if isinstance(source, ms.RuntimeBootstrapOptions):
                resolved = source
            elif isinstance(source, Mapping):
                resolved = cls._resolve_from_mapping(source)
            else:
                options_resolver = getattr(source, "_runtime_bootstrap_options", None)
                if callable(options_resolver):
                    raw_options = options_resolver()
                    if raw_options is not None:
                        resolved = cls.resolve_runtime_options(raw_options)
                source_updates = cls._runtime_option_updates_from_source(source)
                if source_updates:
                    resolved = resolved.model_copy(update=source_updates)
        if overrides:
            override_options = ms.RuntimeBootstrapOptions.model_validate(overrides)
            override_updates = dict(
                cls.dump(override_options, exclude_none=True),
            )
            if override_updates:
                resolved = resolved.model_copy(update=override_updates)
        return resolved


__all__: list[str] = ["FlextUtilitiesModelOptions"]
