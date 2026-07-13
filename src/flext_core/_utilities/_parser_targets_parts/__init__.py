# AUTO-GENERATED FILE — Regenerate with: make gen
"""Parser Targets Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._parser_targets_parts.parser_targets_part_02 import (
        FlextUtilitiesParserTargets as FlextUtilitiesParserTargets,
    )
_LAZY_IMPORTS = build_lazy_import_map({
    ".parser_targets_part_02": ("FlextUtilitiesParserTargets",)
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
