"""FlextTypesCore - foundational, flat type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._typings.base import FlextTypingBase
from flext_core._typings.pydantic import FlextTypesPydantic


class FlextTypesCore:
    """Type aliases for core scalar/container foundations."""

    type TextOrBinaryContent = (
        FlextTypesPydantic.StrictStr | FlextTypesPydantic.StrictBytes
    )
    type RegistryBindingKey = str | type

    type FileContent = (
        FlextTypesPydantic.StrictStr
        | FlextTypesPydantic.StrictBytes
        | FlextTypingBase.SequenceOf[FlextTypingBase.StrSequence]
    )
    type GeneralValueTypeMapping = FlextTypingBase.MappingKV[
        str,
        FlextTypingBase.Scalar,
    ]
