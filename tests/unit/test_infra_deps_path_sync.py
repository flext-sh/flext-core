"""Tests for FlextInfraDependencyPathSync."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from flext_core import r
from flext_infra import m
from flext_infra.deps.path_sync import (
    FlextInfraDependencyPathSync,
    _extract_requirement_name,
    _rewrite_pep621,
    _rewrite_poetry,
    _target_path,
    detect_mode,
    extract_dep_name,
    main,
    rewrite_dep_paths,
)
from tomlkit.toml_document import TOMLDocument


class TestFlextInfraDependencyPathSync:
    """Test FlextInfraDependencyPathSync."""

    def test_path_sync_initialization(self) -> None:
        """Test path sync initializes without errors."""
        path_sync = FlextInfraDependencyPathSync()
        assert path_sync is not None
        assert path_sync._toml is not None


class TestDetectMode:
    """Test detect_mode function."""

    def test_detect_mode_workspace(self, tmp_path: Path) -> None:
        """Test detect_mode with workspace structure."""
        gitmodules = tmp_path / ".gitmodules"
        gitmodules.touch()
        mode = detect_mode(tmp_path)
        assert mode == "workspace"

    def test_detect_mode_workspace_parent(self, tmp_path: Path) -> None:
        """Test detect_mode finds .gitmodules in parent."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".gitmodules").touch()
        project = workspace / "project"
        project.mkdir()
        mode = detect_mode(project)
        assert mode == "workspace"

    def test_detect_mode_standalone(self, tmp_path: Path) -> None:
        """Test detect_mode with standalone structure."""
        mode = detect_mode(tmp_path)
        assert mode == "standalone"


class TestExtractDepName:
    """Test extract_dep_name function."""

    def test_extract_dep_name_simple(self) -> None:
        """Test extract_dep_name with simple path."""
        name = extract_dep_name("flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_with_prefix(self) -> None:
        """Test extract_dep_name with .flext-deps prefix."""
        name = extract_dep_name(".flext-deps/flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_with_parent_ref(self) -> None:
        """Test extract_dep_name with parent directory reference."""
        name = extract_dep_name("../flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_with_slash(self) -> None:
        """Test extract_dep_name with leading slash."""
        name = extract_dep_name("/flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_with_whitespace(self) -> None:
        """Test extract_dep_name with whitespace."""
        name = extract_dep_name("  flext-core  ")
        assert name == "flext-core"

    def test_extract_dep_name_with_dot_prefix(self) -> None:
        """Test extract_dep_name with ./ prefix."""
        name = extract_dep_name("./flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_complex(self) -> None:
        """Test extract_dep_name with complex path."""
        name = extract_dep_name(".flext-deps/flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_parent_and_slash(self) -> None:
        """Test extract_dep_name with parent and slash."""
        name = extract_dep_name("/../flext-core")
        assert name == "flext-core"


class TestTargetPath:
    """Test _target_path function."""

    def test_target_path_workspace_root(self) -> None:
        """Test _target_path in workspace mode at root."""
        path = _target_path("flext-core", is_root=True, mode="workspace")
        assert path == "flext-core"

    def test_target_path_workspace_subproject(self) -> None:
        """Test _target_path in workspace mode for subproject."""
        path = _target_path("flext-core", is_root=False, mode="workspace")
        assert path == "../flext-core"

    def test_target_path_standalone_root(self) -> None:
        """Test _target_path in standalone mode at root."""
        path = _target_path("flext-core", is_root=True, mode="standalone")
        assert path == ".flext-deps/flext-core"

    def test_target_path_standalone_subproject(self) -> None:
        """Test _target_path in standalone mode for subproject."""
        path = _target_path("flext-core", is_root=False, mode="standalone")
        assert path == ".flext-deps/flext-core"


class TestExtractRequirementName:
    """Test _extract_requirement_name function."""

    def test_extract_requirement_name_pep621_path(self) -> None:
        """Test _extract_requirement_name with PEP 621 path dependency."""
        name = _extract_requirement_name("flext-core @ file://.flext-deps/flext-core")
        assert name == "flext-core"

    def test_extract_requirement_name_simple(self) -> None:
        """Test _extract_requirement_name with simple requirement."""
        name = _extract_requirement_name("flext-core")
        assert name == "flext-core"

    def test_extract_requirement_name_with_version(self) -> None:
        """Test _extract_requirement_name with version specifier."""
        name = _extract_requirement_name("flext-core>=1.0.0")
        assert name == "flext-core"

    def test_extract_requirement_name_invalid(self) -> None:
        """Test _extract_requirement_name with invalid entry."""
        name = _extract_requirement_name("@invalid")
        assert name is None

    def test_extract_requirement_name_empty(self) -> None:
        """Test _extract_requirement_name with empty string."""
        name = _extract_requirement_name("")
        assert name is None

    def test_extract_requirement_name_with_marker(self) -> None:
        """Test _extract_requirement_name with environment marker."""
        name = _extract_requirement_name(
            'flext-core @ file://.flext-deps/flext-core ; python_version >= "3.8"'
        )
        assert name == "flext-core"


class TestRewritePep621:
    """Test _rewrite_pep621 function."""

    def test_rewrite_pep621_no_project(self) -> None:
        """Test _rewrite_pep621 with no project section."""
        doc = TOMLDocument()
        changes = _rewrite_pep621(
            doc, is_root=True, mode="workspace", internal_names=set()
        )
        assert changes == []

    def test_rewrite_pep621_no_dependencies(self) -> None:
        """Test _rewrite_pep621 with no dependencies."""
        doc = TOMLDocument()
        doc["project"] = {}
        changes = _rewrite_pep621(
            doc, is_root=True, mode="workspace", internal_names=set()
        )
        assert changes == []

    def test_rewrite_pep621_non_list_dependencies(self) -> None:
        """Test _rewrite_pep621 with non-list dependencies."""
        doc = TOMLDocument()
        doc["project"] = {"dependencies": "not-a-list"}
        changes = _rewrite_pep621(
            doc, is_root=True, mode="workspace", internal_names=set()
        )
        assert changes == []

    def test_rewrite_pep621_rewrite_path_dep(self) -> None:
        """Test _rewrite_pep621 rewrites path dependency."""
        doc = TOMLDocument()
        doc["project"] = {
            "dependencies": ["flext-core @ file://.flext-deps/flext-core"]
        }
        changes = _rewrite_pep621(
            doc,
            is_root=True,
            mode="workspace",
            internal_names={"flext-core"},
        )
        assert len(changes) > 0
        assert len(changes) > 0
        assert "flext-core @ file:./flext-core" in doc["project"]["dependencies"][0]

    def test_rewrite_pep621_skip_external_dep(self) -> None:
        """Test _rewrite_pep621 skips external dependencies."""
        doc = TOMLDocument()
        doc["project"] = {"dependencies": ["requests>=2.0.0"]}
        changes = _rewrite_pep621(
            doc,
            is_root=True,
            mode="workspace",
            internal_names={"flext-core"},
        )
        assert changes == []

    def test_rewrite_pep621_with_marker(self) -> None:
        """Test _rewrite_pep621 preserves environment markers."""
        doc = TOMLDocument()
        doc["project"] = {
            "dependencies": [
                'flext-core @ file://.flext-deps/flext-core ; python_version >= "3.8"'
            ]
        }
        changes = _rewrite_pep621(
            doc,
            is_root=True,
            mode="workspace",
            internal_names={"flext-core"},
        )
        assert len(changes) > 0
        assert 'python_version >= "3.8"' in doc["project"]["dependencies"][0]

    def test_rewrite_pep621_non_string_item(self) -> None:
        """Test _rewrite_pep621 skips non-string items."""
        doc = TOMLDocument()
        doc["project"] = {
            "dependencies": [123, "flext-core @ file://.flext-deps/flext-core"]
        }
        changes = _rewrite_pep621(
            doc,
            is_root=True,
            mode="workspace",
            internal_names={"flext-core"},
        )
        assert len(changes) == 1

    def test_rewrite_pep621_subproject_mode(self) -> None:
        """Test _rewrite_pep621 in subproject mode."""
        doc = TOMLDocument()
        doc["project"] = {
            "dependencies": ["flext-core @ file://.flext-deps/flext-core"]
        }
        changes = _rewrite_pep621(
            doc,
            is_root=False,
            mode="workspace",
            internal_names={"flext-core"},
        )
        assert len(changes) > 0
        assert "../flext-core" in doc["project"]["dependencies"][0]


class TestRewritePoetry:
    """Test _rewrite_poetry function."""

    def test_rewrite_poetry_no_tool(self) -> None:
        """Test _rewrite_poetry with no tool section."""
        doc = TOMLDocument()
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert changes == []

    def test_rewrite_poetry_no_poetry(self) -> None:
        """Test _rewrite_poetry with no poetry section."""
        doc = TOMLDocument()
        doc["tool"] = {}
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert changes == []

    def test_rewrite_poetry_no_dependencies(self) -> None:
        """Test _rewrite_poetry with no dependencies."""
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {}}
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert changes == []

    def test_rewrite_poetry_non_dict_dependencies(self) -> None:
        """Test _rewrite_poetry with non-dict dependencies."""
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": "not-a-dict"}}
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert changes == []

    def test_rewrite_poetry_rewrite_path_dep(self) -> None:
        """Test _rewrite_poetry rewrites path dependency."""
        doc = TOMLDocument()
        doc["tool"] = {
            "poetry": {
                "dependencies": {"flext-core": {"path": ".flext-deps/flext-core"}}
            }
        }
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert len(changes) > 0
        assert (
            doc["tool"]["poetry"]["dependencies"]["flext-core"]["path"] == "flext-core"
        )

    def test_rewrite_poetry_skip_non_path_dep(self) -> None:
        """Test _rewrite_poetry skips non-path dependencies."""
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"requests": {"version": "^2.0.0"}}}}
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert changes == []

    def test_rewrite_poetry_non_dict_value(self) -> None:
        """Test _rewrite_poetry skips non-dict values."""
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"requests": "^2.0.0"}}}
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert changes == []

    def test_rewrite_poetry_empty_path(self) -> None:
        """Test _rewrite_poetry skips empty path."""
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"flext-core": {"path": ""}}}}
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert changes == []

    def test_rewrite_poetry_non_string_path(self) -> None:
        """Test _rewrite_poetry skips non-string path."""
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"flext-core": {"path": 123}}}}
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        assert changes == []

    def test_rewrite_poetry_subproject_mode(self) -> None:
        """Test _rewrite_poetry in subproject mode."""
        doc = TOMLDocument()
        doc["tool"] = {
            "poetry": {
                "dependencies": {"flext-core": {"path": ".flext-deps/flext-core"}}
            }
        }
        changes = _rewrite_poetry(doc, is_root=False, mode="workspace")
        assert len(changes) > 0
        assert (
            doc["tool"]["poetry"]["dependencies"]["flext-core"]["path"]
            == "../flext-core"
        )


class TestRewriteDepPaths:
    """Test rewrite_dep_paths function."""

    def test_rewrite_dep_paths_success(self, tmp_path: Path) -> None:
        """Test rewrite_dep_paths successfully rewrites paths."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\ndependencies = ["flext-core @ file://.flext-deps/flext-core"]\n'
        )
        result = rewrite_dep_paths(
            pyproject,
            mode="workspace",
            internal_names={"flext-core"},
            is_root=True,
        )
        assert result.is_success
        assert len(result.value) > 0

    def test_rewrite_dep_paths_dry_run(self, tmp_path: Path) -> None:
        """Test rewrite_dep_paths with dry_run=True."""
        pyproject = tmp_path / "pyproject.toml"
        original_content = (
            '[project]\ndependencies = ["flext-core @ file://.flext-deps/flext-core"]\n'
        )
        pyproject.write_text(original_content)
        result = rewrite_dep_paths(
            pyproject,
            mode="workspace",
            internal_names={"flext-core"},
            is_root=True,
            dry_run=True,
        )
        assert result.is_success
        # Verify file was not modified
        assert pyproject.read_text() == original_content

    def test_rewrite_dep_paths_no_changes(self, tmp_path: Path) -> None:
        """Test rewrite_dep_paths with no changes needed."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\ndependencies = ["requests>=2.0.0"]\n')
        result = rewrite_dep_paths(
            pyproject,
            mode="workspace",
            internal_names={"flext-core"},
            is_root=True,
        )
        assert result.is_success
        assert result.value == []

    def test_rewrite_dep_paths_read_failure(self, tmp_path: Path) -> None:
        """Test rewrite_dep_paths when read fails."""
        pyproject = tmp_path / "pyproject.toml"
        result = rewrite_dep_paths(
            pyproject,
            mode="workspace",
            internal_names={"flext-core"},
            is_root=True,
        )
        assert result.is_failure

    def test_rewrite_dep_paths_write_failure(self, tmp_path: Path) -> None:
        """Test rewrite_dep_paths when write fails."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\ndependencies = ["flext-core @ file://.flext-deps/flext-core"]\n'
        )
        with patch("flext_infra.FlextInfraTomlService.write_document") as mock_write:
            mock_write.return_value = r[bool].fail("write failed")
            result = rewrite_dep_paths(
                pyproject,
                mode="workspace",
                internal_names={"flext-core"},
                is_root=True,
            )
            assert result.is_failure


class TestMain:
    """Test main function."""

    def test_main_auto_detect_workspace(self, tmp_path: Path) -> None:
        """Test main with auto-detect workspace mode."""
        (tmp_path / ".gitmodules").touch()
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "sys.argv",
                ["prog", "--mode", "auto"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_explicit_workspace_mode(self, tmp_path: Path) -> None:
        """Test main with explicit workspace mode."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "sys.argv",
                ["prog", "--mode", "workspace"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_explicit_standalone_mode(self, tmp_path: Path) -> None:
        """Test main with explicit standalone mode."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "sys.argv",
                ["prog", "--mode", "standalone"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_dry_run(self, tmp_path: Path) -> None:
        """Test main with dry-run flag."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "sys.argv",
                ["prog", "--dry-run"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_specific_projects(self, tmp_path: Path) -> None:
        """Test main with specific projects."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        project_pyproject = project_dir / "pyproject.toml"
        project_pyproject.write_text('[project]\nname = "flext-core"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "sys.argv",
                ["prog", "--project", "flext-core"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_discovery_failure(self, tmp_path: Path) -> None:
        """Test main when discovery fails."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].fail("discovery failed"),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 1

    def test_main_root_rewrite_failure(self, tmp_path: Path) -> None:
        """Test main when root rewrite fails."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.deps.path_sync.rewrite_dep_paths",
                return_value=r[list].fail("rewrite failed"),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 1

    def test_main_project_rewrite_failure(self, tmp_path: Path) -> None:
        """Test main when project rewrite fails."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        project_pyproject = project_dir / "pyproject.toml"
        project_pyproject.write_text('[project]\nname = "flext-core"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([
                    m.ProjectInfo(
                        path=project_dir,
                        name="flext-core",
                        stack="python",
                        has_tests=False,
                        has_src=False,
                    )
                ]),
            ),
            patch(
                "flext_infra.deps.path_sync.rewrite_dep_paths",
                side_effect=[
                    r[list].ok([]),  # root rewrite succeeds
                    r[list].fail("project rewrite failed"),  # project rewrite fails
                ],
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 1

    def test_main_no_changes(self, tmp_path: Path) -> None:
        """Test main with no changes needed."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([]),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_with_changes(self, tmp_path: Path) -> None:
        """Test main with changes made."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        project_pyproject = project_dir / "pyproject.toml"
        project_pyproject.write_text('[project]\nname = "flext-core"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([
                    m.ProjectInfo(
                        path=project_dir,
                        name="flext-core",
                        stack="python",
                        has_tests=False,
                        has_src=False,
                    )
                ]),
            ),
            patch(
                "flext_infra.deps.path_sync.rewrite_dep_paths",
                side_effect=[
                    r[list].ok([]),  # root rewrite
                    r[list].ok(["change1"]),  # project rewrite with changes
                ],
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_root_project_name_extraction(self, tmp_path: Path) -> None:
        """Test main extracts root project name."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([]),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_project_name_extraction(self, tmp_path: Path) -> None:
        """Test main extracts project names."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        project_pyproject = project_dir / "pyproject.toml"
        project_pyproject.write_text('[project]\nname = "flext-core"\n')

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([
                    m.ProjectInfo(
                        path=project_dir,
                        name="flext-core",
                        stack="python",
                        has_tests=False,
                        has_src=False,
                    )
                ]),
            ),
            patch(
                "flext_infra.deps.path_sync.rewrite_dep_paths",
                return_value=r[list].ok([]),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_invalid_project_toml(self, tmp_path: Path) -> None:
        """Test main handles invalid project TOML."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text("invalid toml [[[")

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 1

    def test_main_missing_root_pyproject(self, tmp_path: Path) -> None:
        """Test main handles missing root pyproject."""
        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([]),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_project_without_pyproject(self, tmp_path: Path) -> None:
        """Test main handles project without pyproject."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([
                    m.ProjectInfo(
                        path=project_dir,
                        name="flext-core",
                        stack="python",
                        has_tests=False,
                        has_src=False,
                    )
                ]),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_project_invalid_toml(self, tmp_path: Path) -> None:
        """Test main handles project with invalid TOML."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        project_pyproject = project_dir / "pyproject.toml"
        project_pyproject.write_text("invalid toml [[[")

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([
                    m.ProjectInfo(
                        path=project_dir,
                        name="flext-core",
                        stack="python",
                        has_tests=False,
                        has_src=False,
                    )
                ]),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 1

    def test_main_project_no_name(self, tmp_path: Path) -> None:
        """Test main handles project without name."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        project_pyproject = project_dir / "pyproject.toml"
        project_pyproject.write_text("[project]\n")

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([
                    m.ProjectInfo(
                        path=project_dir,
                        name="flext-core",
                        stack="python",
                        has_tests=False,
                        has_src=False,
                    )
                ]),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 0

    def test_main_project_non_string_name(self, tmp_path: Path) -> None:
        """Test main handles project with non-string name."""
        root_pyproject = tmp_path / "pyproject.toml"
        root_pyproject.write_text('[project]\nname = "flext-workspace"\n')

        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        project_pyproject = project_dir / "pyproject.toml"
        project_pyproject.write_text("[project]\nname = 123\n")

        with (
            patch("flext_infra.deps.path_sync.ROOT", tmp_path),
            patch(
                "flext_infra.FlextInfraDiscoveryService.discover_projects",
                return_value=r[list].ok([
                    m.ProjectInfo(
                        path=project_dir,
                        name="flext-core",
                        stack="python",
                        has_tests=False,
                        has_src=False,
                    )
                ]),
            ),
            patch(
                "sys.argv",
                ["prog"],
            ),
        ):
            result = main()
            assert result == 0


class TestPathSyncEdgeCases:
    """Test edge cases in path sync."""

    def test_detect_mode_with_nonexistent_path(self, tmp_path: Path) -> None:
        """Test detect_mode with a path that has no poetry.lock or pdm.lock."""
        result = detect_mode(tmp_path)
        assert result is not None

    def test_extract_dep_name_with_empty_string(self) -> None:
        """Test extract_dep_name with empty string."""
        result = extract_dep_name("")
        assert result is not None

    def test_rewrite_dep_paths_with_no_deps(self, tmp_path: Path) -> None:
        """Test rewrite_dep_paths with pyproject that has no path deps."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.poetry.dependencies]\npython = "^3.13"')
        result = rewrite_dep_paths(
            pyproject, mode="poetry", internal_names=set(), dry_run=True
        )
        assert result.is_success


def test_detect_mode_with_path_object() -> None:
    """Test detect_mode accepts Path object."""
    result = detect_mode(Path("/tmp"))
    assert result in {"workspace", "standalone"}


def test_rewrite_dep_paths_with_internal_names(tmp_path: Path) -> None:
    """Test rewrite_dep_paths with internal dependency names."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\ndependencies = ["flext-core @ file:.flext-deps/flext-core"]\n'
    )
    result = rewrite_dep_paths(
        pyproject,
        mode="workspace",
        internal_names={"flext-core"},
        is_root=False,
        dry_run=False,
    )
    assert result.is_success
    # Should have changes since we're converting from .flext-deps to ../
    assert len(result.value) > 0


def test_rewrite_dep_paths_dry_run(tmp_path: Path) -> None:
    """Test rewrite_dep_paths with dry_run=True doesn't write."""
    pyproject = tmp_path / "pyproject.toml"
    original = '[project]\ndependencies = ["flext-core @ file:../flext-core"]\n'
    pyproject.write_text(original)

    result = rewrite_dep_paths(
        pyproject,
        mode="workspace",
        internal_names={"flext-core"},
        is_root=False,
        dry_run=True,
    )
    assert result.is_success
    # File should not be modified in dry-run
    assert pyproject.read_text() == original


def test_rewrite_dep_paths_read_failure(tmp_path: Path) -> None:
    """Test rewrite_dep_paths handles read failures."""
    pyproject = tmp_path / "pyproject.toml"
    # Don't create the file

    result = rewrite_dep_paths(
        pyproject,
        mode="workspace",
        internal_names={"flext-core"},
        is_root=False,
        dry_run=False,
    )
    assert result.is_failure


def test_extract_requirement_name_with_path_dep() -> None:
    """Test _extract_requirement_name with path dependency."""
    result = _extract_requirement_name("flext-core @ file:../flext-core")
    assert result == "flext-core"


def test_extract_requirement_name_simple() -> None:
    """Test _extract_requirement_name with simple requirement."""
    result = _extract_requirement_name("requests>=2.0")
    assert result == "requests"


def test_extract_requirement_name_invalid() -> None:
    """Test _extract_requirement_name with invalid requirement."""
    result = _extract_requirement_name("")
    assert result is None


def test_target_path_workspace_root() -> None:
    """Test _target_path for workspace root."""
    result = _target_path("flext-core", is_root=True, mode="workspace")
    assert result == "flext-core"


def test_target_path_workspace_subproject() -> None:
    """Test _target_path for workspace subproject."""
    result = _target_path("flext-core", is_root=False, mode="workspace")
    assert result == "../flext-core"


def test_target_path_standalone() -> None:
    """Test _target_path for standalone mode."""
    result = _target_path("flext-core", is_root=False, mode="standalone")
    assert result == ".flext-deps/flext-core"
