"""FlextTypesCore - foundational, flat type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)

from flext_core import (
    FlextTypesPydantic as tp,
    FlextTypingBase as t,
    FlextTypingContainers,
)


class FlextTypesCore(t, FlextTypingContainers):
    """Type aliases for core scalar/container foundations."""

    type TextValue = tp.StrictStr
    type IntegerValue = tp.StrictInt
    type FloatValue = tp.StrictFloat
    type BinaryContent = tp.StrictBytes
    type TextOrBinaryContent = FlextTypesCore.TextValue | FlextTypesCore.BinaryContent
    type RegistryBindingKey = str | type

    type Serializable = t.Container
    type FileContent = (
        FlextTypesCore.TextValue
        | FlextTypesCore.BinaryContent
        | Sequence[t.StrSequence]
    )
    type GeneralValueTypeMapping = Mapping[str, t.Scalar]
