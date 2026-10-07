"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib import import_module
from typing import overload

from pydantic import TypeAdapter

from flext_core import c, e, p, r, t
from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._utilities import FlextUtilitiesArgs


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    class ModelDumpOptions(FlextModelsBase.FlexibleInternalModel):
        """Options controlling Pydantic model_dump() serialization behavior."""

        by_alias: bool | None = FlextModelsPydantic.Field(
            None,
            description="Serialize using field aliases",
            validate_default=True,
        )
        exclude_none: bool | None = FlextModelsPydantic.Field(
            None,
            description="Exclude None-valued fields",
            validate_default=True,
        )
        exclude_unset: bool | None = FlextModelsPydantic.Field(
            None,
            description="Exclude fields not explicitly set",
            validate_default=True,
        )
        exclude_defaults: bool | None = FlextModelsPydantic.Field(
            None,
            description="Exclude fields matching defaults",
            validate_default=True,
        )
        include: set[str] | None = FlextModelsPydantic.Field(
            None,
            description="Whitelist of field names to include",
            validate_default=True,
        )
        exclude: set[str] | None = FlextModelsPydantic.Field(
            None,
            description="Blacklist of field names to exclude",
            validate_default=True,
        )

    @staticmethod
    def dump(
        model: FlextModelsPydantic.BaseModel,
        options: FlextUtilitiesModel.ModelDumpOptions | None = None,
        **kwargs: t.JsonPayload,
    ) -> t.MappingKV[str, t.JsonPayload]:
        """Unified Pydantic serialization with options.

        Generic replacement for: model.model_dump() with consistent return type.

        Args:
            model: Pydantic model instance to serialize.
            options: Optional Pydantic model_dump arguments within the settings model.
            **kwargs: Inline serialization options mapped to ModelDumpOptions;
                invalid options fail loudly instead of silently falling
                back to defaults.

        Returns:
            Dictionary representation of the model.

        """
        opts = FlextUtilitiesArgs.resolve_options(
            options,
            kwargs,
            FlextUtilitiesModel.ModelDumpOptions,
        ).unwrap()
        opts_dict = opts.model_dump(exclude_none=True)
        dumped: t.JsonMapping = t.json_mapping_adapter().validate_python(
            model.model_dump(mode="json", **opts_dict),
        )
        return dumped

    @staticmethod
    def _settings_base() -> t.SettingsClass:
        """Resolve FlextSettings lazily to avoid runtime import cycles.

        Returns:
            The resulting ``t.SettingsClass``.
        """
        settings_module = import_module("flext_core")
        settings_cls: t.SettingsClass = settings_module.FlextSettings
        return settings_cls

    @staticmethod
    def _container_type() -> p.ContainerType:
        """Resolve FlextContainer lazily to avoid runtime import cycles.

        Returns:
            The resulting ``p.ContainerType``.
        """
        container_module = import_module("flext_core")
        container_cls: p.ContainerType = container_module.FlextContainer
        return container_cls

    @staticmethod
    def _context_type() -> p.ContextType:
        """Resolve FlextContext lazily to avoid runtime import cycles.

        Returns:
            The resulting ``p.ContextType``.
        """
        context_module = import_module("flext_core")
        context_cls: p.ContextType = context_module.FlextContext
        return context_cls

    @overload
    @staticmethod
    def validate_value[TValue](
        target: type[TValue],
        data: t.JsonPayload,
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> p.Result[TValue]: ...

    @overload
    @staticmethod
    def validate_value[TValue](
        target: t.ValueAdapter[TValue],
        data: t.JsonPayload,
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> p.Result[TValue]: ...

    @overload
    @staticmethod
    def validate_value(
        target: t.DynamicTypeHint,
        data: t.JsonPayload,
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> p.Result[t.JsonValue]: ...

    @staticmethod
    def validate_value[TValue](
        target: t.ValueAdapter[TValue] | t.TypeHintSpecifier,
        data: t.JsonPayload,
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> p.Result[TValue]:
        """Validate one value through a model class or TypeAdapter.

        Returns:
            The resulting ``p.Result[TValue]``.
        """
        try:
            adapter = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
            if from_json:
                if not isinstance(data, c.STR_BINARY_TYPES):
                    return e.fail_validation(
                        "json_input",
                        error="JSON validation requires str or bytes input",
                    )
                return r[TValue].ok(adapter.validate_json(data, strict=strict))
            return r[TValue].ok(adapter.validate_python(data, strict=strict))
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return e.fail_validation(error=exc)


__all__: list[str] = ["FlextUtilitiesModel"]
