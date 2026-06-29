"""FlextSettingsBase facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base_parts.flextsettingsbase_part_02 import FlextSettingsBase
else:
    from ._base_parts import FlextSettingsBase

__all__: list[str] = ["FlextSettingsBase"]
