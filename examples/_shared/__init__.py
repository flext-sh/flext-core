# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Shared package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from examples._shared.base import ExamplesFlextSharedBase
    from examples._shared.verification import ExamplesFlextShared


__all__: tuple[str, ...] = ("ExamplesFlextShared", "ExamplesFlextSharedBase")

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "ExamplesFlextShared": ".verification",
        "ExamplesFlextSharedBase": ".base",
    }),
    public_exports=__all__,
)
