# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Cqrs package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flext_core
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._cqrs.flextmodelscqrs_part_01 import FlextModelsCqrsPart01


__all__: tuple[str, ...] = ("FlextModelsCqrsPart01", "flext_core")

install_lazy_exports(
    __name__,
    globals(),
    {"FlextModelsCqrsPart01": ".flextmodelscqrs_part_01"},
    public_exports=__all__,
)
