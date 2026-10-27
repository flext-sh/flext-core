# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Handlers package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._handlers.flexthandlers_part_01 import FlextHandlersPart01
    from flext_core._handlers.flexthandlers_part_02 import FlextHandlersPart02
    from flext_core._handlers.flexthandlers_part_03 import FlextHandlersPart03
    from flext_core._handlers.flexthandlers_part_04 import FlextHandlersPart04
    from flext_core._handlers.flexthandlers_part_05 import FlextHandlersPart05
    from flext_core._handlers.flexthandlers_part_06 import FlextHandlersPart06
    from flext_core._handlers.flexthandlers_part_07 import FlextHandlersPart07


__all__: tuple[str, ...] = (
    "FlextHandlersPart01",
    "FlextHandlersPart02",
    "FlextHandlersPart03",
    "FlextHandlersPart04",
    "FlextHandlersPart05",
    "FlextHandlersPart06",
    "FlextHandlersPart07",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextHandlersPart01": ".flexthandlers_part_01",
        "FlextHandlersPart02": ".flexthandlers_part_02",
        "FlextHandlersPart03": ".flexthandlers_part_03",
        "FlextHandlersPart04": ".flexthandlers_part_04",
        "FlextHandlersPart05": ".flexthandlers_part_05",
        "FlextHandlersPart06": ".flexthandlers_part_06",
        "FlextHandlersPart07": ".flexthandlers_part_07",
    }),
    public_exports=__all__,
)
