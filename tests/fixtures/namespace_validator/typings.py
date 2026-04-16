from __future__ import annotations

from typing import TypeAlias

from flext_core import t

LooseTypeAlias: TypeAlias = t.Primitives | None

__all__: list[str] = ["LooseTypeAlias"]
