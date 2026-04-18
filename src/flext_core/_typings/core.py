"""FlextTypesCore - foundational and recursive type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from enum import StrEnum
from pathlib import Path
from re import Pattern

from flext_core import (
    FlextConstantsPydantic,
    FlextTypingBase as t,
    FlextTypingContainers,
)


class FlextTypesCore(t, FlextTypingContainers):
    """Type aliases for core scalar/container foundations."""

    type TextValue = str
    type IntegerValue = int
    type FloatValue = float
    type BinaryContent = bytes
    type TextOrBinaryContent = FlextTypesCore.TextValue | FlextTypesCore.BinaryContent
    type OptionalPrimitive = t.Primitives | None
    type OptionalScalar = t.Scalar | None
    type RegistryBindingKey = str | type

    type Serializable = (
        t.Container
        | Sequence[FlextTypesCore.Serializable]
        | Mapping[str, FlextTypesCore.Serializable]
        | None
    )
    type ContainerValue = (
        t.Scalar
        | Sequence[FlextTypesCore.ContainerValue]
        | Mapping[str, FlextTypesCore.ContainerValue]
    )
    type GeneralValueType = (
        t.Scalar
        | Path
        | Sequence[FlextTypesCore.GeneralValueType]
        | Mapping[str, FlextTypesCore.GeneralValueType]
    )
    type OptionalContainerValue = FlextTypesCore.ContainerValue | None

    type ConstantValue = (
        t.Primitives
        | FlextConstantsPydantic.ConfigDict
        | frozenset[str]
        | tuple[str, ...]
        | t.HeaderMapping
        | StrEnum
        | type[StrEnum]
        | Pattern[str]
        | type
    )
    type RecursiveValue = (
        t.Primitives
        | Sequence[FlextTypesCore.RecursiveValue]
        | Mapping[str, FlextTypesCore.RecursiveValue]
        | None
    )
    type FileContent = str | FlextTypesCore.BinaryContent | Sequence[t.StrSequence]
    type GeneralValueTypeMapping = Mapping[str, t.Scalar]

    # Short aliases for high-frequency inline patterns (annotation-only, not base classes)
    type ContainerValueMapping = Mapping[str, FlextTypesCore.ContainerValue]
    type MutableContainerValueMapping = MutableMapping[
        str,
        FlextTypesCore.ContainerValue,
    ]
    type ContainerValueList = Sequence[FlextTypesCore.ContainerValue]
    type OptionalContainerValueMapping = Mapping[
        str,
        FlextTypesCore.OptionalContainerValue,
    ]
    type JsonMapping = Mapping[str, FlextTypesCore.RecursiveValue]
    type JsonList = Sequence[FlextTypesCore.RecursiveValue]
    type GeneralValueMapping = Mapping[str, FlextTypesCore.GeneralValueType]

    type JsonObject = ContainerValueMapping
    type ApiJsonValue = ContainerValue | None

    # Runtime tuples for isinstance checks (mirrors base.py CONTAINER_TYPES pattern)
    CONTAINER_VALUE_SCALAR_TYPES: tuple[type, ...] = t.SCALAR_TYPES
    """Runtime tuple for ContainerValue leaf isinstance checks."""

    class Enforcement:
        """Type aliases for the runtime enforcement engine."""

        type RuleTag = str
        type RuleSpec = tuple[str, str, str, str]
        type RuleTagSequence = Sequence[str]
