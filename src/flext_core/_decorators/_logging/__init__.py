# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for logging.

Copyright (c) 2026 FLEXT Team. All rights reserved.
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

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._decorators._logging.payloads import FlextDecoratorsLoggingPayloads

__all__: tuple[str, ...] = ("FlextDecoratorsLoggingPayloads",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextDecoratorsLoggingPayloads": ".payloads",
    }),
    public_exports=__all__,
)
