"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, overload

from pydantic import TypeAdapter

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._utilities import (
    FlextUtilitiesArgs,
    FlextUtilitiesBase,
    FlextUtilitiesTypingAdapterJson,
)


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    @staticmethod
    def dump(
        model: flext_core.p.BaseModel,
        options: flext_core.p.ModelDumpOptions | None = None,
        **kwargs: flext_core.t.JsonPayload,
    ) -> Mapping[str, flext_core.t.JsonPayload]:
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
        opts = FlextUtilitiesArgs.parse_model(
            options.model_dump(exclude_none=True) if options is not None else kwargs,
            flext_core.m.ModelDumpOptions,
        ).unwrap()
        opts_dict = opts.model_dump(exclude_none=True)
        adapter = FlextUtilitiesTypingAdapterJson.json_mapping_adapter()
        dumped: flext_core.t.JsonMapping = adapter.validate_python(
            model.model_dump(mode="json", **opts_dict),
        )
        return dumped

    @staticmethod
    def settings_base() -> flext_core.p.SettingsType:
        """Return the canonical ``FlextSettings`` class (layer zero, static).

        Returns:
            The resulting ``p.SettingsType``.

        """
        settings_cls: flext_core.p.SettingsType = flext_core.FlextSettings
        return settings_cls

    @overload
    @staticmethod
    def validate_value[TValue](
        target: type[TValue],
        data: flext_core.t.ValidationInput[flext_core.p.BaseModel],
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> flext_core.p.Result[TValue]: ...

    @overload
    @staticmethod
    def validate_value[TValue](
        target: flext_core.t.ValueAdapter[TValue],
        data: flext_core.t.ValidationInput[flext_core.p.BaseModel],
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> flext_core.p.Result[TValue]: ...

    @staticmethod
    def validate_value[TValue](
        target: flext_core.t.ValueAdapter[TValue] | type[TValue],
        data: flext_core.t.ValidationInput[flext_core.p.BaseModel],
        *,
        from_json: bool = False,
        strict: bool | None = None,
    ) -> flext_core.p.Result[TValue]:
        """Validate one value through a model class or TypeAdapter.

        Python inputs retain their native types until the target validates them.
        JSON inputs still require encoded text or bytes at the runtime boundary.

        Returns:
            The resulting ``p.Result[TValue]``.

        """
        try:
            adapter: flext_core.t.ValueAdapter[TValue] = (
                target
                if isinstance(target, TypeAdapter)
                else FlextUtilitiesBase.type_adapter(target)
            )
            if from_json:
                if not isinstance(data, flext_core.c.STR_BINARY_TYPES):
                    return flext_core.FlextExceptions.fail_validation(
                        "json_input",
                        error="JSON validation requires str or bytes input",
                    )
                return flext_core.FlextResult[TValue].ok(
                    adapter.validate_json(data, strict=strict)
                )
            return flext_core.FlextResult[TValue].ok(
                adapter.validate_python(data, strict=strict)
            )
        except flext_core.c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return flext_core.FlextExceptions.fail_validation(error=exc)


__all__: list[str] = ["FlextUtilitiesModel"]
