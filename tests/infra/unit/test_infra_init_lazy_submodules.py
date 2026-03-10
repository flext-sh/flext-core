"""Tests for flext_infra submodule __init__ lazy loading error paths.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

import flext_infra.basemk
import flext_infra.check
import flext_infra.codegen
import flext_infra.core
import flext_infra.deps
import flext_infra.docs
import flext_infra.github
import flext_infra.maintenance
import flext_infra.release
import flext_infra.workspace


class TestFlextInfraSubmoduleInitLazyLoading:
    """Test __getattr__ error paths in flext_infra submodule __init__ files."""

    def test_basemk_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in basemk raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.basemk.NonexistentAttribute

    def test_check_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in check raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.check.NonexistentAttribute

    def test_codegen_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in codegen raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.codegen.NonexistentAttribute

    def test_core_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in core raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.core.NonexistentAttribute

    def test_deps_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in deps raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.deps.NonexistentAttribute

    def test_docs_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in docs raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.docs.NonexistentAttribute

    def test_github_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in github raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.github.NonexistentAttribute

    def test_maintenance_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in maintenance raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.maintenance.NonexistentAttribute

    def test_release_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in release raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.release.NonexistentAttribute

    def test_workspace_getattr_nonexistent_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute in workspace raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = flext_infra.workspace.NonexistentAttribute
