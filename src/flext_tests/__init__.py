"""FLEXT Core Test Support - Comprehensive testing utilities and fixtures.

This module provides the test support foundation for the FLEXT ecosystem with test utilities,
fixtures, builders, matchers, performance testing, and domain-specific test helpers following
modern testing patterns and SOLID principles.
"""

from __future__ import annotations


from .asyncs import *
from .builders import *
from .domains import *
from .factories import *
from .http_support import *
from .hypothesis import *
from .matchers import *
from .performance import *
from .utilities import *

# =============================================================================
# CONSOLIDATED EXPORTS - Combine all __all__ from modules
# =============================================================================

# Import modules for __all__ collection
from . import asyncs as _asyncs
from . import builders as _builders
from . import domains as _domains
from . import factories as _factories
from . import http_support as _http
from . import hypothesis as _hypothesis
from . import matchers as _matchers
from . import performance as _performance
from . import utilities as _utilities


# Collect all __all__ exports from imported modules
_temp_exports: list[str] = []

_modules_to_check = [
    _asyncs,
    _builders,
    _domains,
    _factories,
    _http,
    _hypothesis,
    _matchers,
    _performance,
    _utilities,
]


for module in _modules_to_check:
    if hasattr(module, "__all__"):
        _temp_exports.extend(module.__all__)

# Remove duplicates and sort for consistent exports - build complete list first
_seen: set[str] = set()
_final_exports: list[str] = []
for item in _temp_exports:
    if item not in _seen:
        _seen.add(item)
        _final_exports.append(item)
_final_exports.sort()

# Define __all__ as literal list for linter compatibility
# This dynamic assignment is necessary for aggregating module exports
__all__: list[str] = _final_exports  # pyright: ignore[reportUnsupportedDunderAll] # noqa: PLE0605
