"""FlextTypesCore - foundational, flat type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)

from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp


class FlextTypesCore:
    """Type aliases for core scalar/container foundations."""

    type TextOrBinaryContent = tp.StrictStr | tp.StrictBytes
    type RegistryBindingKey = str | type

    type FileContent = tp.StrictStr | tp.StrictBytes | Sequence[t.StrSequence]
    type GeneralValueTypeMapping = Mapping[str, t.Scalar]
