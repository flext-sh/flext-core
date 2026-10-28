"""FlextProtocolsContext - context and bootstrap protocols.

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

from flext_core._protocols._context import FlextProtocolsContextPart03


class FlextProtocolsContext(FlextProtocolsContextPart03):
    """Public facade for FlextProtocolsContext."""


__all__: list[str] = ["FlextProtocolsContext"]
