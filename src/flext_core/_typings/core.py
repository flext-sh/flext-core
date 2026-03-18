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

    # --- NON-RECURSIVE TYPES (TypeAlias - isinstance-safe) ---

    type RegistryBindingKey = str | type

    # --- RECURSIVE TYPES (PEP 695 - Annotation-only, NEVER with isinstance) ---

    type Serializable = (
        FlextTypingBase.Container
        | list[FlextTypesCore.Serializable]
        | Mapping[str, FlextTypesCore.Serializable]
        | None
    )
    type ContainerValue = (
        FlextTypingBase.Scalar
        | list[FlextTypesCore.ContainerValue]
        | dict[str, FlextTypesCore.ContainerValue]
    )
    type GeneralValueType = (
        FlextTypingBase.Scalar
        | Path
        | list[FlextTypesCore.GeneralValueType]
        | dict[str, FlextTypesCore.GeneralValueType]
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
    type FileContent = str | bytes | Sequence[Sequence[str]]
    type GeneralValueTypeMapping = Mapping[str, FlextTypingBase.Scalar]
