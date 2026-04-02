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

from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers


class FlextTypesCore(FlextTypingBase, FlextTypingContainers):
    """Type aliases for core scalar/container foundations."""

    type TextValue = str
    type IntegerValue = int
    type FloatValue = float
    type BinaryContent = bytes
    type TextOrBinaryContent = FlextTypesCore.TextValue | FlextTypesCore.BinaryContent
    type OptionalPrimitive = FlextTypingBase.Primitives | None
    type OptionalScalar = FlextTypingBase.Scalar | None
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
    type OptionalContainerValue = FlextTypesCore.ContainerValue | None

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
    type FileContent = (
        str | FlextTypesCore.BinaryContent | Sequence[FlextTypingBase.StrSequence]
    )
    type GeneralValueTypeMapping = Mapping[str, FlextTypingBase.Scalar]

    # Short aliases for high-frequency inline patterns (annotation-only, not base classes)
    type ContainerValueMapping = Mapping[str, FlextTypesCore.ContainerValue]
    type ContainerValueList = Sequence[FlextTypesCore.ContainerValue]
    type OptionalContainerValueMapping = Mapping[
        str,
        FlextTypesCore.OptionalContainerValue,
    ]
    type JsonMapping = Mapping[str, FlextTypesCore.JsonValue]
    type JsonList = Sequence[FlextTypesCore.JsonValue]
    type GeneralValueMapping = Mapping[str, FlextTypesCore.GeneralValueType]

    type JsonObject = ContainerValueMapping
    type ApiJsonValue = ContainerValue | None

    # Runtime tuples for isinstance checks (mirrors base.py CONTAINER_TYPES pattern)
    CONTAINER_VALUE_SCALAR_TYPES: tuple[type, ...] = FlextTypingBase.SCALAR_TYPES
    """Runtime tuple for ContainerValue leaf isinstance checks."""
