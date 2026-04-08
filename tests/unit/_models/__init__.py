# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Models package."""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports

_LAZY_IMPORTS = {
    "test_base": "tests.unit._models.test_base",
    "test_cqrs": "tests.unit._models.test_cqrs",
    "test_entity": "tests.unit._models.test_entity",
    "test_exception_params": "tests.unit._models.test_exception_params",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
