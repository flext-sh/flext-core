"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from importlib import import_module

from flext_core import (
    FlextModelsBase as m,
    FlextModelsPydantic as mp,
    FlextUtilitiesArgs as ua,
    c,
    e,
    p,
    r,
    t,
)


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    class ModelDumpOptions(m.FlexibleInternalModel):
        """Options controlling Pydantic model_dump() serialization behavior."""

        by_alias: bool | None = mp.Field(
            None,
            description="Serialize using field aliases",
            validate_default=True,
        )
        exclude_none: bool | None = mp.Field(
            None,
            description="Exclude None-valued fields",
            validate_default=True,
        )
        exclude_unset: bool | None = mp.Field(
            None,
            description="Exclude fields not explicitly set",
            validate_default=True,
        )
        exclude_defaults: bool | None = mp.Field(
            None,
            description="Exclude fields matching defaults",
            validate_default=True,
        )
        include: set[str] | None = mp.Field(
            None,
            description="Whitelist of field names to include",
            validate_default=True,
        )
        exclude: set[str] | None = mp.Field(
            None,
            description="Blacklist of field names to exclude",
            validate_default=True,
        )

    @staticmethod
    def dump(
        model: mp.BaseModel,
        options: FlextUtilitiesModel.ModelDumpOptions | None = None,
        **kwargs: t.JsonPayload,
    ) -> Mapping[str, t.JsonPayload]:
        """Unified Pydantic serialization with options.

        Generic replacement for: model.model_dump() with consistent return type.

        Args:
            model: Pydantic model instance to serialize.
            options: Optional Pydantic model_dump arguments within the settings model.
            **kwargs: Inline fallback serialization arguments mapped to ModelDumpOptions automatically.

        Returns:
            Dictionary representation of the model.

        """
        opts = ua.resolve_options(
            options,
            kwargs,
            FlextUtilitiesModel.ModelDumpOptions,
        ).unwrap_or(FlextUtilitiesModel.ModelDumpOptions())
        opts_dict = opts.model_dump(exclude_none=True)
        return model.model_dump(**opts_dict)

    @staticmethod
    def _settings_base() -> t.SettingsClass:
        """Resolve FlextSettings lazily to avoid runtime import cycles."""
        settings_module = import_module("flext_core")
        settings_cls: t.SettingsClass = settings_module.FlextSettings
        return settings_cls

    @staticmethod
    def _container_type() -> p.ContainerType:
        """Resolve FlextContainer lazily to avoid runtime import cycles."""
        container_module = import_module("flext_core")
        container_cls: p.ContainerType = container_module.FlextContainer
        return container_cls

    @staticmethod
    def _context_type() -> p.ContextType:
        """Resolve FlextContext lazily to avoid runtime import cycles."""
        context_module = import_module("flext_core")
        context_cls: p.ContextType = context_module.FlextContext
        return context_cls

    @staticmethod
    def _runtime_type() -> type:
        """Resolve FlextRuntime lazily to avoid runtime import cycles."""
        runtime_module = import_module("flext_core")
        runtime_cls: type = runtime_module.FlextRuntime
        return runtime_cls

    @classmethod
    def _normalize_runtime_override_mapping(
        cls,
        value: Mapping[str, t.JsonPayload | t.Scalar] | None,
    ) -> t.JsonMapping | None:
        """Normalize runtime override mappings to canonical JsonMapping."""
        if value is None:
            return None
        runtime_type = cls._runtime_type()
        validated: t.JsonMapping = t.json_mapping_adapter().validate_python(
            {
                str(key): runtime_type.normalize_to_metadata(item)
                for key, item in value.items()
            },
        )
        return validated

    @staticmethod
    def service_settings_type(
        service_or_cls: p.Base | p.SettingsType | t.SettingsClass,
    ) -> t.SettingsClass:
        """Resolve the concrete settings type used by a service-like object."""
        settings_base = FlextUtilitiesModel._settings_base()
        fetch_global = getattr(service_or_cls, "fetch_global", None)
        model_copy = getattr(service_or_cls, "model_copy", None)
        if (
            isinstance(service_or_cls, type)
            and callable(fetch_global)
            and callable(model_copy)
        ):
            return service_or_cls
        candidate = getattr(service_or_cls, "settings_type", None)
        if (
            isinstance(candidate, type)
            and callable(getattr(candidate, "fetch_global", None))
            and callable(getattr(candidate, "model_copy", None))
        ):
            return candidate
        resolver = getattr(service_or_cls, "_get_service_settings_type", None)
        if callable(resolver):
            resolved = resolver()
            if (
                isinstance(resolved, type)
                and callable(getattr(resolved, "fetch_global", None))
                and callable(getattr(resolved, "model_copy", None))
            ):
                return resolved
        return settings_base

    @staticmethod
    def validate_value[TValue](
        target: t.ValueAdapter[TValue] | t.TypeHintSpecifier,
        data: t.JsonPayload,
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> p.Result[TValue]:
        """Validate one value through a model class or TypeAdapter."""
        try:
            adapter = (
                target if isinstance(target, mp.TypeAdapter) else mp.TypeAdapter(target)
            )
            if from_json:
                if not isinstance(data, (str, bytes, bytearray)):
                    return e.fail_validation(
                        "json_input",
                        error="JSON validation requires str or bytes input",
                    )
                return r[TValue].ok(adapter.validate_json(data, strict=strict))
            return r[TValue].ok(adapter.validate_python(data, strict=strict))
        except (
            c.ValidationError,
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
        ) as exc:
            return e.fail_validation(error=exc)


__all__: list[str] = ["FlextUtilitiesModel"]
