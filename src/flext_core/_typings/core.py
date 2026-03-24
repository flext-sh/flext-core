"""FlextTypesCore - foundational and recursive type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from re import Pattern

from pydantic import ConfigDict
from pydantic_settings import SettingsConfigDict

from flext_core import FlextTypingBase, FlextTypingContainers


class FlextTypesCore(FlextTypingBase, FlextTypingContainers):
    """Type aliases for core scalar/container foundations."""

    type RegistryBindingKey = str | type

    type Serializable = (
        FlextTypingBase.Container
        | Sequence[FlextTypesCore.Serializable]
        | Mapping[str, FlextTypesCore.Serializable]
        | None
    )
    type ContainerValue = (
        FlextTypingBase.Scalar
        | Sequence[FlextTypesCore.ContainerValue]
        | Mapping[str, FlextTypesCore.ContainerValue]
    )
    type GeneralValueType = (
        FlextTypingBase.Scalar
        | Path
        | Sequence[FlextTypesCore.GeneralValueType]
        | Mapping[str, FlextTypesCore.GeneralValueType]
    )

    type ConstantValue = (
        FlextTypingBase.Primitives
        | ConfigDict
        | SettingsConfigDict
        | frozenset[str]
        | tuple[str, ...]
        | Mapping[str, str | int]
        | StrEnum
        | type[StrEnum]
        | Pattern[str]
        | type
    )
    type JsonValue = (
        FlextTypingBase.Primitives
        | Sequence[FlextTypesCore.JsonValue]
        | Mapping[str, FlextTypesCore.JsonValue]
        | None
    )
    type FileContent = str | bytes | Sequence[Sequence[str]]
    type GeneralValueTypeMapping = Mapping[str, FlextTypingBase.Scalar]
