"""FlextTypesCore - foundational, flat type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from collections.abc import Mapping, Sequence
from types import ModuleType

from flext_core._typings import (
    FlextTypesAnnotateds,
    FlextTypesPydantic,
    FlextTypingBase,
)


class FlextTypesCore(FlextTypesAnnotateds):
    """Type aliases for core scalar/container foundations."""

    type TextOrBinaryContent = (
        FlextTypesPydantic.StrictStr | FlextTypesPydantic.StrictBytes
    )
    type RegistryBindingKey = str | type
    type RegistryModule = ModuleType

    type FileContent = (
        FlextTypesPydantic.StrictStr
        | FlextTypesPydantic.StrictBytes
        | Sequence[FlextTypingBase.Sequence[str]]
    )
    type GeneralValueTypeMapping = Mapping[
        str,
        FlextTypingBase.Scalar,
    ]


__all__: list[str] = ["FlextTypesCore"]
