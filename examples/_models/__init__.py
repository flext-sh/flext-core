# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Models package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from examples._models.handle import ExamplesFlextSharedHandle
    from examples._models.person import ExamplesFlextSharedPerson


__all__: tuple[str, ...] = ("ExamplesFlextSharedHandle", "ExamplesFlextSharedPerson")

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "ExamplesFlextSharedHandle": ".handle",
        "ExamplesFlextSharedPerson": ".person",
    }),
    public_exports=__all__,
)
