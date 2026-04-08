# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Benchmark package."""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports

_LAZY_IMPORTS = {
    "test_container_memory": "tests.benchmark.test_container_memory",
    "test_container_performance": "tests.benchmark.test_container_performance",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
