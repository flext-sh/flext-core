"""FlextContext facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._context_parts.flextcontext_part_02 import FlextContext

if TYPE_CHECKING:
    from flext_core import t

__all__: t.StrSequence = ("FlextContext",)
