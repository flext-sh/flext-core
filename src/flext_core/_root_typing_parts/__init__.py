# AUTO-GENERATED FILE — Regenerate with: make gen
"""Root Typing Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._root_typing_parts._exports import (
    FLEXT_CORE__ROOT_TYPING_PARTS_LAZY_IMPORTS,
)
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._root_typing_parts.facades import (
        c as c,
        d as d,
        e as e,
        h as h,
        m as m,
        p as p,
        r as r,
        s as s,
        t as t,
        u as u,
        x as x,
    )

_LAZY_IMPORTS = FLEXT_CORE__ROOT_TYPING_PARTS_LAZY_IMPORTS


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
