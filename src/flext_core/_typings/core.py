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
)


class FlextTypesCore:
    """Type aliases for core scalar/container foundations."""

    type TextOrBinaryContent = tp.StrictStr | tp.StrictBytes
    type RegistryBindingKey = str | type

    type FileContent = tp.StrictStr | tp.StrictBytes | Sequence[t.StrSequence]
    type GeneralValueTypeMapping = Mapping[str, t.Scalar]
