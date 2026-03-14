"""Tests for FlextInfraStubSupplyChain.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.core.stub_chain import FlextInfraStubSupplyChain
from flext_tests import tm


class TestStubChainCore:
    """Core tests for FlextInfraStubSupplyChain."""

    run_mypy_hints = getattr(FlextInfraStubSupplyChain, "_run_mypy_hints")
    run_pyrefly_missing = getattr(FlextInfraStubSupplyChain, "_run_pyrefly_missing")
    is_internal = getattr(FlextInfraStubSupplyChain, "_is_internal")
    stub_exists = getattr(FlextInfraStubSupplyChain, "_stub_exists")
    discover_stub_projects = getattr(
        FlextInfraStubSupplyChain,
        "_discover_stub_projects",
    )

    def test_init_creates_service(self) -> None:
        """Service initializes with runner attribute."""
        chain = FlextInfraStubSupplyChain()
        assert chain is not None
        tm.that(hasattr(chain, "_runner"), eq=True)


class TestStubChainAnalyze:
    """Analyze method tests for FlextInfraStubSupplyChain."""

    def test_analyze_valid_project(self, tmp_path: Path) -> None:
        """Valid project returns success."""
        chain = FlextInfraStubSupplyChain()
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname = 'test'")
        result = chain.analyze(proj, tmp_path)
        tm.ok(result)

    def test_analyze_returns_flextresult(self, tmp_path: Path) -> None:
        """Analyze returns r type."""
        chain = FlextInfraStubSupplyChain()
        proj = tmp_path / "project"
        proj.mkdir()
        result = chain.analyze(proj, tmp_path)
        tm.ok(result)

    def test_analyze_nonexistent_project(self, tmp_path: Path) -> None:
        """Nonexistent project still returns success."""
        chain = FlextInfraStubSupplyChain()
        result = chain.analyze(tmp_path / "nonexistent", tmp_path)
        tm.ok(result)


class TestStubChainValidate:
    """Validate method tests for FlextInfraStubSupplyChain."""

    def test_validate_workspace(self, tmp_path: Path) -> None:
        """Workspace validation returns r."""
        chain = FlextInfraStubSupplyChain()
        result = chain.validate(tmp_path)
        tm.ok(result)

    def test_validate_nonexistent_workspace(self, tmp_path: Path) -> None:
        """Nonexistent workspace returns failure."""
        chain = FlextInfraStubSupplyChain()
        result = chain.validate(tmp_path / "nonexistent")
        tm.fail(result)

    def test_validate_with_project_dirs(self, tmp_path: Path) -> None:
        """Validate respects project_dirs filter."""
        chain = FlextInfraStubSupplyChain()
        proj = tmp_path / "project1"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("")
        (proj / "src").mkdir()
        result = chain.validate(tmp_path, project_dirs=[proj])
        tm.ok(result)


class TestStubChainIsInternal:
    """Tests for _is_internal static method."""

    is_internal = getattr(FlextInfraStubSupplyChain, "_is_internal")

    def test_flext_underscore_prefix(self) -> None:
        """flext_ prefix is internal."""
        tm.that(self.is_internal("flext_core", "project"), eq=True)
        tm.that(self.is_internal("flext_api", "project"), eq=True)

    def test_flext_dash_prefix(self) -> None:
        """flext- prefix is internal."""
        tm.that(self.is_internal("flext-core", "project"), eq=True)

    def test_project_name(self) -> None:
        """Project name is internal."""
        tm.that(self.is_internal("my_project", "my_project"), eq=True)
        tm.that(self.is_internal("my_project.sub", "my_project"), eq=True)

    def test_external_module(self) -> None:
        """External module is not internal."""
        tm.that(self.is_internal("requests", "my_project"), eq=False)


class TestStubChainStubExists:
    """Tests for _stub_exists static method."""

    stub_exists = getattr(FlextInfraStubSupplyChain, "_stub_exists")

    def test_pyi_file(self, tmp_path: Path) -> None:
        """Finds .pyi files."""
        typings = tmp_path / "typings"
        typings.mkdir()
        (typings / "requests.pyi").write_text("")
        tm.that(self.stub_exists("requests", tmp_path), eq=True)

    def test_package_init(self, tmp_path: Path) -> None:
        """Finds package __init__.pyi."""
        pkg = tmp_path / "typings" / "requests"
        pkg.mkdir(parents=True)
        (pkg / "__init__.pyi").write_text("")
        tm.that(self.stub_exists("requests", tmp_path), eq=True)

    def test_generated_stubs(self, tmp_path: Path) -> None:
        """Finds generated stubs."""
        gen = tmp_path / "typings" / "generated"
        gen.mkdir(parents=True)
        (gen / "requests.pyi").write_text("")
        tm.that(self.stub_exists("requests", tmp_path), eq=True)

    def test_missing_returns_false(self, tmp_path: Path) -> None:
        """Missing stubs return False."""
        tm.that(self.stub_exists("requests", tmp_path), eq=False)


class TestStubChainDiscoverProjects:
    """Tests for _discover_stub_projects static method."""

    discover = getattr(FlextInfraStubSupplyChain, "_discover_stub_projects")

    def test_finds_projects(self, tmp_path: Path) -> None:
        """Discovers projects with pyproject.toml and src/."""
        for name in ("project1", "project2"):
            proj = tmp_path / name
            proj.mkdir()
            (proj / "pyproject.toml").write_text("")
            (proj / "src").mkdir()
        projects = self.discover(tmp_path)
        tm.that(len(projects), eq=2)

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        """Hidden directories are skipped."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "pyproject.toml").write_text("")
        (hidden / "src").mkdir()
        tm.that(len(self.discover(tmp_path)), eq=0)

    def test_requires_src_dir(self, tmp_path: Path) -> None:
        """Projects without src/ are skipped."""
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("")
        tm.that(len(self.discover(tmp_path)), eq=0)

    def test_requires_pyproject(self, tmp_path: Path) -> None:
        """Projects without pyproject.toml are skipped."""
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "src").mkdir()
        tm.that(len(self.discover(tmp_path)), eq=0)


__all__: list[str] = []
