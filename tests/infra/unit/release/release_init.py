"""Tests for release module __init__.py lazy imports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib

import pytest

from flext_infra.release import FlextInfraReleaseOrchestrator
from flext_tests import tm


class TestReleaseInit:
    """Test release module __init__.py lazy imports."""

    def test_lazy_import_orchestrator(self) -> None:
        release_module = importlib.import_module("flext_infra.release")
        orchestrator = release_module.FlextInfraReleaseOrchestrator()
        tm.that(isinstance(orchestrator, FlextInfraReleaseOrchestrator), eq=True)

    def test_getattr_invalid_attribute(self) -> None:
        release_module = importlib.import_module("flext_infra.release")
        with pytest.raises(AttributeError, match=r"module.*has no attribute"):
            _ = release_module.NonexistentAttribute

    def test_dir_returns_all_exports(self) -> None:
        release_module = importlib.import_module("flext_infra.release")
        exports = dir(release_module)
        tm.that("FlextInfraReleaseOrchestrator" in exports, eq=True)
