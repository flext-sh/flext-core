# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Parser Targets Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._parser_targets.parser_targets_part_01 import (
        FlextUtilitiesParserTargetsPart01,
    )
    from flext_core._utilities._parser_targets.parser_targets_part_02 import (
        FlextUtilitiesParserTargetsPart02,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesParserTargetsPart01",
    "FlextUtilitiesParserTargetsPart02",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesParserTargetsPart01": ".parser_targets_part_01",
        "FlextUtilitiesParserTargetsPart02": ".parser_targets_part_02",
    }),
    public_exports=__all__,
)
