"""Tests package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports
from tests._exports import TESTS_FLEXT_CORE_LAZY_IMPORTS

if _t.TYPE_CHECKING:
    from flext_tests import (
        d as d,
        e as e,
        h as h,
        r as r,
        td as td,
        tf as tf,
        tk as tk,
        tm as tm,
        tv as tv,
        x as x,
    )

install_lazy_exports(__name__, globals(), TESTS_FLEXT_CORE_LAZY_IMPORTS, publish_all=False)
