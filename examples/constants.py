"""Public examples constants facade for flext-core."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c as _c

from examples import p, t


c = _c

__all__: t.MutableSequenceOf[str] = ["c"]
