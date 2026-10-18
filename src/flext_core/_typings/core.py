"""FlextTypesCore - foundational, flat type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import ModuleType

from flext_core._typings import FlextTypesPydantic, FlextTypingBase


class FlextTypesCore:
    """Type aliases for core scalar/container foundations."""

    type TextOrBinaryContent = (
        FlextTypesPydantic.StrictStr | FlextTypesPydantic.StrictBytes
    )
    type RegistryBindingKey = str | type
    type RegistryModule = ModuleType

    type FileContent = (
        FlextTypesPydantic.StrictStr
        | FlextTypesPydantic.StrictBytes
        | FlextTypingBase.SequenceOf[FlextTypingBase.StrSequence]
    )
    type GeneralValueTypeMapping = FlextTypingBase.MappingKV[
        str,
        FlextTypingBase.Scalar,
    ]


__all__: list[str] = ["FlextTypesCore"]
