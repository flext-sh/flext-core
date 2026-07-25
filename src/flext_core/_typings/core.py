"""FlextTypesCore - foundational, flat type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .base import FlextTypingBase as t
from .pydantic import FlextTypesPydantic as tp


class FlextTypesCore:
    """Type aliases for core scalar/container foundations."""

    type TextOrBinaryContent = tp.StrictStr | tp.StrictBytes
    type RegistryBindingKey = str | type

    type FileContent = tp.StrictStr | tp.StrictBytes | t.SequenceOf[t.StrSequence]
    type GeneralValueTypeMapping = t.MappingKV[str, t.Scalar]
