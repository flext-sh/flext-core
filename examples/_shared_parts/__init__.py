# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Shared Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from examples._shared_parts.shared_part_01 import ExamplesFlextSharedBase
    from examples._shared_parts.shared_part_02 import ExamplesFlextShared


__all__: tuple[str, ...] = ("ExamplesFlextShared", "ExamplesFlextSharedBase")

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "ExamplesFlextShared": ".shared_part_02",
        "ExamplesFlextSharedBase": ".shared_part_01",
    }),
    public_exports=__all__,
)
