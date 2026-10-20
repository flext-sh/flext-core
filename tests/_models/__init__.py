# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Models package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from tests._models.mixins import (
        TestsFlextModelsMixins,
        TestsFlextModelsNamespace,
        m,
    )


__all__: tuple[str, ...] = ("TestsFlextModelsMixins", "TestsFlextModelsNamespace", "m")

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "TestsFlextModelsMixins": ".mixins",
        "TestsFlextModelsNamespace": ".mixins",
        "m": ".mixins",
    }),
    public_exports=__all__,
)[0]
