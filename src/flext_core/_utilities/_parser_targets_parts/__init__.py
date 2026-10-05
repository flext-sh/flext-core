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
    from flext_core._utilities._parser_targets_parts.parser_targets_part_02 import (
        FlextUtilitiesParserTargets,
    )


__all__: tuple[str, ...] = ("FlextUtilitiesParserTargets",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextUtilitiesParserTargets": ".parser_targets_part_02"}),
    public_exports=__all__,
)
