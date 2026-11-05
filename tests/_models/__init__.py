# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Models package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests._models.mixins import TestsFlextModelsMixins, TestsFlextModelsNamespace


__all__: tuple[str, ...] = ("TestsFlextModelsMixins", "TestsFlextModelsNamespace")

install_lazy_exports(
    __name__,
    globals(),
    {"TestsFlextModelsMixins": ".mixins", "TestsFlextModelsNamespace": ".mixins"},
    public_exports=__all__,
)
