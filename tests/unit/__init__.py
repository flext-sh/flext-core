"""Unit package."""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports
from tests.unit._exports import TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS

install_lazy_exports(__name__, globals(), TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS, publish_all=False)
