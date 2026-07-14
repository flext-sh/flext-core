"""Dependency-injector runtime bridge option parsing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from ._dependency_types import FlextRuntimeDependencyTypes

from flext_core._protocols.container import FlextProtocolsContainer as pc
from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.services import FlextTypesServices as ts


class FlextRuntimeDependencyOptions(FlextRuntimeDependencyTypes):
    """Parse and merge dependency container creation options."""

    @classmethod
    def _parse_options(
        cls,
        container_options: pc.ContainerCreationOptions
        | tb.MappingKV[str, ts.JsonPayload]
        | None,
    ) -> pc.ContainerCreationOptions:
        """Parse raw container options into a validated model."""
        match container_options:
            case None:
                return cls.ContainerCreationOptions.model_validate({})
            case Mapping():
                return cls.ContainerCreationOptions.model_validate(container_options)
            case _:
                return cls.ContainerCreationOptions.model_validate(
                    {
                        field: getattr(container_options, field)
                        for field in cls._OPTION_FIELDS
                    }
                    | {"factory_cache": container_options.factory_cache}
                )

    @classmethod
    def _merge_options(
        cls,
        base: pc.ContainerCreationOptions,
        overrides: tb.MappingKV[str, ts.JsonPayload],
    ) -> pc.ContainerCreationOptions:
        """Merge runtime kwargs over base options."""
        override_opts = cls.ContainerCreationOptions.model_validate(overrides)
        merged: MutableMapping[str, ts.JsonPayload] = {
            field: (
                getattr(override_opts, field)
                if getattr(override_opts, field) is not None
                else getattr(base, field)
            )
            for field in cls._OPTION_FIELDS
        }
        merged["factory_cache"] = override_opts.factory_cache
        return cls.ContainerCreationOptions.model_validate(merged)


__all__: list[str] = ["FlextRuntimeDependencyOptions"]
