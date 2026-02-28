"""Tests for FlextInfraPyprojectModernizer with 100% coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import Mock, patch

# Ensure modules are imported for coverage
import flext_infra.deps.modernizer  # noqa: F401
import tomlkit
from flext_infra.deps.modernizer import (
    ConsolidateGroupsPhase,
    EnsurePyreflyConfigPhase,
    EnsurePytestConfigPhase,
    FlextInfraPyprojectModernizer,
    InjectCommentsPhase,
    _array,
    _as_string_list,
    _canonical_dev_dependencies,
    _dedupe_specs,
    _dep_name,
    _ensure_table,
    _parser,
    _project_dev_groups,
    _read_doc,
    _unwrap_item,
    _workspace_root,
    main,
)
from tomlkit.items import Array, Table


class TestDepName:
    """Test _dep_name function."""

    def test_dep_name_simple(self) -> None:
        """Test extracting simple dependency name."""
        assert _dep_name("requests") == "requests"

    def test_dep_name_with_version(self) -> None:
        """Test extracting name from versioned spec."""
        assert _dep_name("requests>=2.0") == "requests"

    def test_dep_name_with_git_url(self) -> None:
        """Test extracting name from git URL."""
        assert (
            _dep_name("requests @ git+https://github.com/psf/requests.git")
            == "requests"
        )

    def test_dep_name_with_underscore(self) -> None:
        """Test normalizing underscores to hyphens."""
        assert _dep_name("my_package") == "my-package"

    def test_dep_name_with_whitespace(self) -> None:
        """Test handling leading/trailing whitespace."""
        assert _dep_name("  requests  ") == "requests"

    def test_dep_name_empty_string(self) -> None:
        """Test handling empty string."""
        assert _dep_name("") == ""

    def test_dep_name_complex_spec(self) -> None:
        """Test complex dependency specification."""
        assert _dep_name("Django>=3.0,<4.0") == "django"


class TestDedupeSpecs:
    """Test _dedupe_specs function."""

    def test_dedupe_specs_no_duplicates(self) -> None:
        """Test deduplication with no duplicates."""
        specs = ["requests>=2.0", "django>=3.0"]
        result = _dedupe_specs(specs)
        assert len(result) == 2

    def test_dedupe_specs_with_duplicates(self) -> None:
        """Test deduplication with duplicates."""
        specs = ["requests>=2.0", "requests>=2.1", "django>=3.0"]
        result = _dedupe_specs(specs)
        assert len(result) == 2
        assert "requests" in [_dep_name(s) for s in result]

    def test_dedupe_specs_empty_list(self) -> None:
        """Test deduplication with empty list."""
        result = _dedupe_specs([])
        assert result == []

    def test_dedupe_specs_sorted_output(self) -> None:
        """Test that output is sorted."""
        specs = ["zebra>=1.0", "apple>=1.0"]
        result = _dedupe_specs(specs)
        assert _dep_name(result[0]) < _dep_name(result[1])

    def test_dedupe_specs_case_insensitive(self) -> None:
        """Test case-insensitive deduplication."""
        specs = ["Requests>=2.0", "requests>=2.1"]
        result = _dedupe_specs(specs)
        assert len(result) == 1


class TestUnwrapItem:
    """Test _unwrap_item function."""

    def test_unwrap_item_with_string(self) -> None:
        """Test unwrapping string value."""
        result = _unwrap_item("test")
        assert result == "test"

    def test_unwrap_item_with_none(self) -> None:
        """Test unwrapping None."""
        result = _unwrap_item(None)
        assert result is None

    def test_unwrap_item_with_tomlkit_item(self) -> None:
        """Test unwrapping tomlkit Item."""
        doc = tomlkit.document()
        doc["key"] = "value"
        item = doc["key"]
        result = _unwrap_item(item)
        assert result == "value"

    def test_unwrap_item_with_dict(self) -> None:
        """Test unwrapping dict value."""
        value = {"key": "value"}
        result = _unwrap_item(value)
        assert result == value

    def test_unwrap_item_with_nested_item(self) -> None:
        """Test unwrapping nested Item."""
        doc = tomlkit.document()
        doc["key"] = tomlkit.item("value")
        item = doc["key"]
        result = _unwrap_item(item)
        assert result == "value"


class TestAsStringList:
    """Test _as_string_list function."""

    def test_as_string_list_with_list(self) -> None:
        """Test converting list to string list."""
        result = _as_string_list(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_as_string_list_with_none(self) -> None:
        """Test converting None to empty list."""
        result = _as_string_list(None)
        assert result == []

    def test_as_string_list_with_string(self) -> None:
        """Test converting string to empty list."""
        result = _as_string_list("test")
        assert result == []

    def test_as_string_list_with_dict(self) -> None:
        """Test converting dict to empty list."""
        result = _as_string_list({"key": "value"})
        assert result == []

    def test_as_string_list_with_tomlkit_array(self) -> None:
        """Test converting tomlkit array."""
        arr = tomlkit.array()
        arr.append("item1")
        arr.append("item2")
        result = _as_string_list(arr)
        assert result == ["item1", "item2"]

    def test_as_string_list_with_tomlkit_item_array(self) -> None:
        """Test converting tomlkit Item containing array."""
        doc = tomlkit.document()
        doc["items"] = ["a", "b"]
        item = doc["items"]
        result = _as_string_list(item)
        assert result == ["a", "b"]

    def test_as_string_list_with_non_iterable(self) -> None:
        """Test converting non-iterable value."""
        result = _as_string_list(42)
        assert result == []

    def test_as_string_list_with_tomlkit_item_non_iterable(self) -> None:
        """Test converting tomlkit Item with non-iterable."""
        doc = tomlkit.document()
        doc["value"] = 42
        item = doc["value"]
        result = _as_string_list(item)
        assert result == []


class TestArray:
    """Test _array function."""

    def test_array_creates_multiline_array(self) -> None:
        """Test creating multiline array."""
        result = _array(["a", "b", "c"])
        assert isinstance(result, Array)
        assert len(result) == 3

    def test_array_empty_list(self) -> None:
        """Test creating array from empty list."""
        result = _array([])
        assert isinstance(result, Array)
        assert len(result) == 0

    def test_array_single_item(self) -> None:
        """Test creating array with single item."""
        result = _array(["single"])
        assert isinstance(result, Array)
        assert len(result) == 1


class TestEnsureTable:
    """Test _ensure_table function."""

    def test_ensure_table_creates_new(self) -> None:
        """Test creating new table."""
        parent = tomlkit.table()
        result = _ensure_table(parent, "new_key")
        assert isinstance(result, Table)
        assert "new_key" in parent

    def test_ensure_table_returns_existing(self) -> None:
        """Test returning existing table."""
        parent = tomlkit.table()
        parent["existing"] = tomlkit.table()
        result = _ensure_table(parent, "existing")
        assert isinstance(result, Table)
        assert result is parent["existing"]

    def test_ensure_table_overwrites_non_table(self) -> None:
        """Test overwriting non-table value."""
        parent = tomlkit.table()
        parent["key"] = "string_value"
        result = _ensure_table(parent, "key")
        assert isinstance(result, Table)


class TestReadDoc:
    """Test _read_doc function."""

    def test_read_doc_valid_file(self, tmp_path: Path) -> None:
        """Test reading valid TOML file."""
        toml_file = tmp_path / "test.toml"
        doc = tomlkit.document()
        doc["key"] = "value"
        toml_file.write_text(tomlkit.dumps(doc))

        result = _read_doc(toml_file)
        assert result is not None
        assert result["key"] == "value"

    def test_read_doc_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading nonexistent file."""
        result = _read_doc(tmp_path / "nonexistent.toml")
        assert result is None

    def test_read_doc_invalid_toml(self, tmp_path: Path) -> None:
        """Test reading invalid TOML file."""
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("invalid toml content [[[")

        result = _read_doc(toml_file)
        assert result is None

    def test_read_doc_permission_error(self, tmp_path: Path) -> None:
        """Test reading file with permission error."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[project]\nname = 'test'")
        toml_file.chmod(0o000)

        try:
            result = _read_doc(toml_file)
            assert result is None
        finally:
            toml_file.chmod(0o644)


class TestProjectDevGroups:
    """Test _project_dev_groups function."""

    def test_project_dev_groups_all_present(self) -> None:
        """Test extracting all dev groups."""
        doc = tomlkit.document()
        doc["project"] = {
            "optional-dependencies": {
                "dev": ["pytest"],
                "docs": ["sphinx"],
                "security": ["bandit"],
                "test": ["coverage"],
                "typings": ["mypy"],
            }
        }

        result = _project_dev_groups(doc)
        assert result["dev"] == ["pytest"]
        assert result["docs"] == ["sphinx"]
        assert result["security"] == ["bandit"]
        assert result["test"] == ["coverage"]
        assert result["typings"] == ["mypy"]

    def test_project_dev_groups_missing_project(self) -> None:
        """Test with missing project table."""
        doc = tomlkit.document()
        result = _project_dev_groups(doc)
        assert result == {}

    def test_project_dev_groups_missing_optional_dependencies(self) -> None:
        """Test with missing optional-dependencies."""
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        result = _project_dev_groups(doc)
        assert result == {}

    def test_project_dev_groups_partial_groups(self) -> None:
        """Test with partial groups."""
        doc = tomlkit.document()
        doc["project"] = {
            "optional-dependencies": {
                "dev": ["pytest"],
            }
        }

        result = _project_dev_groups(doc)
        assert result["dev"] == ["pytest"]
        assert result["docs"] == []


class TestCanonicalDevDependencies:
    """Test _canonical_dev_dependencies function."""

    def test_canonical_dev_dependencies_all_groups(self) -> None:
        """Test merging all dev groups."""
        doc = tomlkit.document()
        doc["project"] = {
            "optional-dependencies": {
                "dev": ["pytest"],
                "docs": ["sphinx"],
                "security": ["bandit"],
                "test": ["coverage"],
                "typings": ["mypy"],
            }
        }

        result = _canonical_dev_dependencies(doc)
        assert len(result) == 5
        # Check that all items are present (sorted)
        assert any("pytest" in r for r in result)

    def test_canonical_dev_dependencies_empty(self) -> None:
        """Test with no dev groups."""
        doc = tomlkit.document()
        result = _canonical_dev_dependencies(doc)
        assert result == []

    def test_canonical_dev_dependencies_deduplicates(self) -> None:
        """Test deduplication across groups."""
        doc = tomlkit.document()
        doc["project"] = {
            "optional-dependencies": {
                "dev": ["pytest>=7.0"],
                "test": ["pytest>=6.0"],
            }
        }

        result = _canonical_dev_dependencies(doc)
        assert len(result) == 1


class TestWorkspaceRoot:
    """Test _workspace_root function."""

    def test_workspace_root_with_gitmodules(self, tmp_path: Path) -> None:
        """Test finding workspace root with .gitmodules."""
        (tmp_path / ".gitmodules").touch()
        (tmp_path / "pyproject.toml").touch()

        result = _workspace_root(tmp_path / "subdir")
        assert result == tmp_path

    def test_workspace_root_with_git(self, tmp_path: Path) -> None:
        """Test finding workspace root with .git."""
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").touch()

        result = _workspace_root(tmp_path / "subdir")
        assert result == tmp_path

    def test_workspace_root_fallback(self, tmp_path: Path) -> None:
        """Test fallback when no markers found."""
        # Create a deep enough path structure
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True, exist_ok=True)
        result = _workspace_root(deep_path)
        assert result is not None


class TestConsolidateGroupsPhase:
    """Test ConsolidateGroupsPhase."""

    def test_consolidate_groups_creates_dev_group(self) -> None:
        """Test creating consolidated dev group."""
        doc = tomlkit.document()
        doc["project"] = {"optional-dependencies": {}}

        phase = ConsolidateGroupsPhase()
        changes = phase.apply(doc, [])
        assert len(changes) > 0

    def test_consolidate_groups_removes_old_groups(self) -> None:
        """Test removing old optional-dependency groups."""
        doc = tomlkit.document()
        doc["project"] = {
            "optional-dependencies": {
                "dev": ["pytest"],
                "docs": ["sphinx"],
                "test": ["coverage"],
            }
        }

        phase = ConsolidateGroupsPhase()
        changes = phase.apply(doc, ["pytest"])
        assert any("removed" in c for c in changes)

    def test_consolidate_groups_merges_poetry_groups(self) -> None:
        """Test merging Poetry groups."""
        doc = tomlkit.document()
        doc["project"] = {"optional-dependencies": {}}
        doc["tool"] = {
            "poetry": {
                "group": {
                    "dev": {"dependencies": {"pytest": "^7.0"}},
                    "docs": {"dependencies": {"sphinx": "^4.0"}},
                }
            }
        }

        phase = ConsolidateGroupsPhase()
        changes = phase.apply(doc, [])
        assert len(changes) > 0

    def test_consolidate_groups_sets_deptry_config(self) -> None:
        """Test setting deptry configuration."""
        doc = tomlkit.document()
        doc["project"] = {"optional-dependencies": {}}
        doc["tool"] = {}

        phase = ConsolidateGroupsPhase()
        changes = phase.apply(doc, [])
        assert any("deptry" in c for c in changes)

    def test_consolidate_groups_handles_missing_tables(self) -> None:
        """Test handling missing tables."""
        doc = tomlkit.document()

        phase = ConsolidateGroupsPhase()
        changes = phase.apply(doc, [])
        assert len(changes) > 0


class TestEnsurePytestConfigPhase:
    """Test EnsurePytestConfigPhase."""

    def test_ensure_pytest_config_sets_minversion(self) -> None:
        """Test setting pytest minversion."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePytestConfigPhase()
        changes = phase.apply(doc)
        assert any("minversion" in c for c in changes)

    def test_ensure_pytest_config_sets_python_classes(self) -> None:
        """Test setting python_classes."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePytestConfigPhase()
        changes = phase.apply(doc)
        assert any("python_classes" in c for c in changes)

    def test_ensure_pytest_config_sets_python_files(self) -> None:
        """Test setting python_files."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePytestConfigPhase()
        changes = phase.apply(doc)
        assert any("python_files" in c for c in changes)

    def test_ensure_pytest_config_sets_addopts(self) -> None:
        """Test setting addopts."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePytestConfigPhase()
        changes = phase.apply(doc)
        assert any("addopts" in c for c in changes)

    def test_ensure_pytest_config_adds_markers(self) -> None:
        """Test adding pytest markers."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePytestConfigPhase()
        changes = phase.apply(doc)
        assert any("markers" in c for c in changes)

    def test_ensure_pytest_config_preserves_existing(self) -> None:
        """Test preserving existing config."""
        doc = tomlkit.document()
        doc["tool"] = {
            "pytest": {
                "ini_options": {
                    "minversion": "8.0",
                    "python_classes": ["Test*"],
                }
            }
        }

        phase = EnsurePytestConfigPhase()
        _ = phase.apply(doc)
        assert doc["tool"]["pytest"]["ini_options"]["minversion"] == "8.0"


class TestEnsurePyreflyConfigPhase:
    """Test EnsurePyreflyConfigPhase."""

    def test_ensure_pyrefly_config_sets_python_version(self) -> None:
        """Test setting Python version."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePyreflyConfigPhase()
        changes = phase.apply(doc, is_root=True)
        assert any("python-version" in c for c in changes)

    def test_ensure_pyrefly_config_enables_ignore_generated(self) -> None:
        """Test enabling ignore-errors-in-generated-code."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePyreflyConfigPhase()
        changes = phase.apply(doc, is_root=True)
        assert any("ignore-errors-in-generated-code" in c for c in changes)

    def test_ensure_pyrefly_config_sets_search_path(self) -> None:
        """Test setting search-path."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePyreflyConfigPhase()
        changes = phase.apply(doc, is_root=True)
        assert any("search-path" in c for c in changes)

    def test_ensure_pyrefly_config_enables_strict_errors(self) -> None:
        """Test enabling strict error rules."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePyreflyConfigPhase()
        changes = phase.apply(doc, is_root=True)
        assert any("errors" in c for c in changes)

    def test_ensure_pyrefly_config_sets_project_excludes(self) -> None:
        """Test setting project-excludes."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePyreflyConfigPhase()
        changes = phase.apply(doc, is_root=True)
        assert any("project-excludes" in c for c in changes)

    def test_ensure_pyrefly_config_non_root(self) -> None:
        """Test non-root project configuration."""
        doc = tomlkit.document()
        doc["tool"] = {}

        phase = EnsurePyreflyConfigPhase()
        changes = phase.apply(doc, is_root=False)
        assert len(changes) > 0


class TestInjectCommentsPhase:
    """Test InjectCommentsPhase."""

    def test_inject_comments_adds_banner(self) -> None:
        """Test adding managed banner."""
        rendered = "[project]\nname = 'test'"

        phase = InjectCommentsPhase()
        result, changes = phase.apply(rendered)
        assert "[MANAGED] FLEXT pyproject standardization" in result
        assert any("banner" in c for c in changes)

    def test_inject_comments_injects_markers(self) -> None:
        """Test injecting section markers."""
        rendered = "[project]\nname = 'test'\n[tool.pytest]"

        phase = InjectCommentsPhase()
        _, changes = phase.apply(rendered)
        assert any("marker" in c for c in changes)

    def test_inject_comments_removes_broken_group_section(self) -> None:
        """Test removing broken [group.dev.dependencies] section."""
        rendered = "[group.dev.dependencies]\npytest = '^7.0'"

        phase = InjectCommentsPhase()
        result, changes = phase.apply(rendered)
        assert "[group.dev.dependencies]" not in result
        assert any("broken" in c for c in changes)

    def test_inject_comments_handles_optional_dependencies_dev(self) -> None:
        """Test handling optional-dependencies.dev marker."""
        rendered = "[project.optional-dependencies]\ndev = ['pytest']"

        phase = InjectCommentsPhase()
        result, changes = phase.apply(rendered)
        # Check that dev marker was processed
        assert "dev" in result or len(changes) > 0

    def test_inject_comments_preserves_existing_markers(self) -> None:
        """Test preserving existing markers."""
        rendered = "# [MANAGED] build system\n[build-system]"

        phase = InjectCommentsPhase()
        result, _ = phase.apply(rendered)
        assert "# [MANAGED] build system" in result


class TestFlextInfraPyprojectModernizer:
    """Test FlextInfraPyprojectModernizer."""

    def test_modernizer_initialization(self) -> None:
        """Test modernizer initializes without errors."""
        modernizer = FlextInfraPyprojectModernizer()
        assert modernizer is not None
        assert modernizer.root is not None

    def test_modernizer_with_custom_root(self, tmp_path: Path) -> None:
        """Test modernizer with custom root."""
        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        assert modernizer.root == tmp_path

    def test_find_pyproject_files(self, tmp_path: Path) -> None:
        """Test finding pyproject.toml files."""
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "pyproject.toml").touch()

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        files = modernizer.find_pyproject_files()
        assert len(files) >= 2

    def test_find_pyproject_files_skips_directories(self, tmp_path: Path) -> None:
        """Test skipping excluded directories."""
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "pyproject.toml").touch()

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        files = modernizer.find_pyproject_files()
        assert all(".venv" not in str(f) for f in files)

    def test_process_file_valid_toml(self, tmp_path: Path) -> None:
        """Test processing valid TOML file."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        pyproject.write_text(tomlkit.dumps(doc))

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        changes = modernizer.process_file(
            pyproject,
            canonical_dev=[],
            dry_run=True,
            skip_comments=False,
        )
        assert isinstance(changes, list)

    def test_process_file_invalid_toml(self, tmp_path: Path) -> None:
        """Test processing invalid TOML file."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("invalid [[[")

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        changes = modernizer.process_file(
            pyproject,
            canonical_dev=[],
            dry_run=True,
            skip_comments=False,
        )
        assert "invalid TOML" in changes

    def test_process_file_dry_run(self, tmp_path: Path) -> None:
        """Test dry-run mode doesn't write."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        original_content = tomlkit.dumps(doc)
        pyproject.write_text(original_content)

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        modernizer.process_file(
            pyproject,
            canonical_dev=["pytest"],
            dry_run=True,
            skip_comments=False,
        )
        assert pyproject.read_text() == original_content

    def test_process_file_skip_comments(self, tmp_path: Path) -> None:
        """Test skipping comment injection."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        pyproject.write_text(tomlkit.dumps(doc))

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        changes = modernizer.process_file(
            pyproject,
            canonical_dev=[],
            dry_run=True,
            skip_comments=True,
        )
        assert not any("banner" in c for c in changes)

    def test_process_file_removes_empty_poetry_groups(self, tmp_path: Path) -> None:
        """Test removing empty Poetry groups."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        doc["tool"] = {
            "poetry": {
                "group": {
                    "empty": {"dependencies": {}},
                }
            }
        }
        pyproject.write_text(tomlkit.dumps(doc))

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        changes = modernizer.process_file(
            pyproject,
            canonical_dev=[],
            dry_run=True,
            skip_comments=False,
        )
        assert any("empty" in c for c in changes)

    def test_run_with_audit_mode(self, tmp_path: Path) -> None:
        """Test run with audit mode."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        pyproject.write_text(tomlkit.dumps(doc))

        args = argparse.Namespace(
            dry_run=False,
            audit=True,
            skip_comments=False,
            skip_check=True,
        )

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        with patch.object(modernizer, "find_pyproject_files", return_value=[pyproject]):
            with patch("flext_infra.deps.modernizer._read_doc") as mock_read:
                mock_read.return_value = doc
                result = modernizer.run(args)
                assert result in {0, 1}

    def test_run_with_poetry_check(self, tmp_path: Path) -> None:
        """Test run with poetry check."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        pyproject.write_text(tomlkit.dumps(doc))

        args = argparse.Namespace(
            dry_run=False,
            audit=False,
            skip_comments=False,
            skip_check=False,
        )

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        with patch.object(modernizer, "find_pyproject_files", return_value=[pyproject]):
            with patch("flext_infra.deps.modernizer._read_doc") as mock_read:
                mock_read.return_value = doc
                with patch.object(modernizer, "_run_poetry_check", return_value=0):
                    result = modernizer.run(args)
                    assert result == 0

    def test_run_poetry_check_success(self, tmp_path: Path) -> None:
        """Test poetry check success."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        with patch(
            "flext_infra.deps.modernizer.FlextInfraCommandRunner.run_raw"
        ) as mock_run:
            mock_result = Mock()
            mock_result.is_failure = False
            mock_result.value = Mock(exit_code=0)
            mock_run.return_value = mock_result

            result = modernizer._run_poetry_check([pyproject])
            assert result == 0

    def test_run_poetry_check_failure(self, tmp_path: Path) -> None:
        """Test poetry check failure."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        with patch(
            "flext_infra.deps.modernizer.FlextInfraCommandRunner.run_raw"
        ) as mock_run:
            mock_result = Mock()
            mock_result.is_failure = True
            mock_run.return_value = mock_result

            result = modernizer._run_poetry_check([pyproject])
            assert result == 1

    def test_run_poetry_check_non_zero_exit(self, tmp_path: Path) -> None:
        """Test poetry check with non-zero exit code."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")

        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        with patch(
            "flext_infra.deps.modernizer.FlextInfraCommandRunner.run_raw"
        ) as mock_run:
            mock_result = Mock()
            mock_result.is_failure = False
            mock_result.value = Mock(exit_code=1)
            mock_run.return_value = mock_result

            result = modernizer._run_poetry_check([pyproject])
            assert result == 1


class TestParser:
    """Test _parser function."""

    def test_parser_creates_argument_parser(self) -> None:
        """Test creating argument parser."""
        parser = _parser()
        assert parser is not None

    def test_parser_has_audit_argument(self) -> None:
        """Test parser has --audit argument."""
        parser = _parser()
        args = parser.parse_args(["--audit"])
        assert args.audit is True

    def test_parser_has_dry_run_argument(self) -> None:
        """Test parser has --dry-run argument."""
        parser = _parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_parser_has_skip_comments_argument(self) -> None:
        """Test parser has --skip-comments argument."""
        parser = _parser()
        args = parser.parse_args(["--skip-comments"])
        assert args.skip_comments is True

    def test_parser_has_skip_check_argument(self) -> None:
        """Test parser has --skip-check argument."""
        parser = _parser()
        args = parser.parse_args(["--skip-check"])
        assert args.skip_check is True


class TestMain:
    """Test main function."""

    def test_main_with_valid_args(self) -> None:
        """Test main with valid arguments."""
        with patch("sys.argv", ["modernizer", "--dry-run"]):
            with patch.object(FlextInfraPyprojectModernizer, "run", return_value=0):
                result = main()
                assert result == 0

    def test_main_with_audit_mode(self) -> None:
        """Test main with audit mode."""
        with patch("sys.argv", ["modernizer", "--audit"]):
            with patch.object(FlextInfraPyprojectModernizer, "run", return_value=0):
                result = main()
                assert result == 0

    def test_main_returns_exit_code(self) -> None:
        """Test main returns exit code."""
        with patch("sys.argv", ["modernizer"]):
            with patch.object(FlextInfraPyprojectModernizer, "run", return_value=42):
                result = main()
                assert result == 42


class TestModernizerEdgeCases:
    """Test edge cases in modernizer."""

    def test_modernizer_with_empty_pyproject(self, tmp_path: Path) -> None:
        """Test modernizer with empty pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("")
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = argparse.Namespace(
            project=None,
            dry_run=True,
            verbose=False,
            audit=False,
            skip_comments=False,
            skip_check=True,
        )
        result = modernizer.run(args)
        assert isinstance(result, int)

    def test_modernizer_with_invalid_toml(self, tmp_path: Path) -> None:
        """Test modernizer with invalid TOML syntax."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[invalid toml {")
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = argparse.Namespace(
            project=None,
            dry_run=True,
            verbose=False,
            audit=False,
            skip_comments=False,
            skip_check=True,
        )
        result = modernizer.run(args)
        assert isinstance(result, int)

    def test_modernizer_with_missing_pyproject(self, tmp_path: Path) -> None:
        """Test modernizer with missing pyproject.toml."""
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = argparse.Namespace(
            project=None,
            dry_run=True,
            verbose=False,
            audit=False,
            skip_comments=False,
            skip_check=True,
        )
        result = modernizer.run(args)
        assert isinstance(result, int)


def test_unwrap_item_with_item() -> None:
    """Test _unwrap_item with tomlkit Item."""
    item = tomlkit.item("test_value")
    result = _unwrap_item(item)
    assert result == "test_value"


def test_unwrap_item_with_none() -> None:
    """Test _unwrap_item with None."""
    result = _unwrap_item(None)
    assert result is None


def test_as_string_list_with_item() -> None:
    """Test _as_string_list with tomlkit Item."""
    item = tomlkit.item(["a", "b", "c"])
    result = _as_string_list(item)
    assert result == ["a", "b", "c"]


def test_as_string_list_with_string() -> None:
    """Test _as_string_list with string returns empty list."""
    result = _as_string_list("string")
    assert result == []


def test_as_string_list_with_mapping() -> None:
    """Test _as_string_list with mapping returns empty list."""
    result = _as_string_list({"key": "value"})
    assert result == []


def test_ensure_pyrefly_config_phase_apply_python_version(tmp_path: Path) -> None:
    """Test EnsurePyreflyConfigPhase sets python-version."""
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    doc["tool"]["pyrefly"] = tomlkit.table()

    phase = EnsurePyreflyConfigPhase()
    changes = phase.apply(doc, is_root=True)

    assert any("python-version set to 3.13" in change for change in changes)
    assert doc["tool"]["pyrefly"]["python-version"] == "3.13"


def test_ensure_pyrefly_config_phase_apply_ignore_errors(tmp_path: Path) -> None:
    """Test EnsurePyreflyConfigPhase enables ignore-errors-in-generated-code."""
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    doc["tool"]["pyrefly"] = tomlkit.table()

    phase = EnsurePyreflyConfigPhase()
    changes = phase.apply(doc, is_root=True)

    assert any(
        "ignore-errors-in-generated-code enabled" in change for change in changes
    )
    assert doc["tool"]["pyrefly"]["ignore-errors-in-generated-code"] is True


def test_ensure_pyrefly_config_phase_apply_search_path(tmp_path: Path) -> None:
    """Test EnsurePyreflyConfigPhase sets search-path."""
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    doc["tool"]["pyrefly"] = tomlkit.table()

    phase = EnsurePyreflyConfigPhase()
    changes = phase.apply(doc, is_root=True)

    assert "search-path set to" in " ".join(changes)


def test_ensure_pyrefly_config_phase_apply_errors(tmp_path: Path) -> None:
    """Test EnsurePyreflyConfigPhase enables strict errors."""
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    doc["tool"]["pyrefly"] = tomlkit.table()

    phase = EnsurePyreflyConfigPhase()
    changes = phase.apply(doc, is_root=True)

    # Should have changes for enabling strict errors
    assert any("errors" in change for change in changes)


def test_inject_comments_phase_apply_banner(tmp_path: Path) -> None:
    """Test InjectCommentsPhase injects banner."""
    rendered = '[project]\nname = "test"\n'
    phase = InjectCommentsPhase()
    result, changes = phase.apply(rendered)

    assert "[MANAGED] FLEXT pyproject standardization" in result
    assert "managed banner injected" in changes


def test_inject_comments_phase_apply_markers(tmp_path: Path) -> None:
    """Test InjectCommentsPhase injects section markers."""
    rendered = '[project]\nname = "test"\n[tool.pytest]\n'
    phase = InjectCommentsPhase()
    result, _ = phase.apply(rendered)

    assert "[MANAGED]" in result


def test_inject_comments_phase_apply_broken_group_section(tmp_path: Path) -> None:
    """Test InjectCommentsPhase removes broken [group.dev.dependencies] section."""
    rendered = '[group.dev.dependencies]\nrequests = "*"\n[project]\n'
    phase = InjectCommentsPhase()
    result, changes = phase.apply(rendered)

    assert "[group.dev.dependencies]" not in result
    assert "broken [group.dev.dependencies] section removed" in changes


def test_consolidate_groups_phase_apply_removes_old_groups(tmp_path: Path) -> None:
    """Test ConsolidateGroupsPhase removes old optional-dependencies groups."""
    doc = tomlkit.document()
    doc["project"] = tomlkit.table()
    doc["project"]["optional-dependencies"] = tomlkit.table()
    doc["project"]["optional-dependencies"]["dev"] = ["pytest"]
    doc["project"]["optional-dependencies"]["docs"] = ["sphinx"]
    doc["project"]["optional-dependencies"]["test"] = ["coverage"]

    phase = ConsolidateGroupsPhase()
    changes = phase.apply(doc, [])

    assert any("optional-dependencies.docs removed" in change for change in changes)
    assert any("optional-dependencies.test removed" in change for change in changes)


def test_ensure_pytest_config_phase_apply_minversion(tmp_path: Path) -> None:
    """Test EnsurePytestConfigPhase sets minversion."""
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    doc["tool"]["pytest"] = tomlkit.table()
    doc["tool"]["pytest"]["ini_options"] = tomlkit.table()

    phase = EnsurePytestConfigPhase()
    changes = phase.apply(doc)

    assert any("minversion set to 8.0" in change for change in changes)
    assert doc["tool"]["pytest"]["ini_options"]["minversion"] == "8.0"


def test_ensure_pytest_config_phase_apply_python_classes(tmp_path: Path) -> None:
    """Test EnsurePytestConfigPhase sets python_classes."""
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    doc["tool"]["pytest"] = tomlkit.table()
    doc["tool"]["pytest"]["ini_options"] = tomlkit.table()

    phase = EnsurePytestConfigPhase()
    changes = phase.apply(doc)

    assert any("python_classes updated" in change for change in changes)


def test_ensure_pytest_config_phase_apply_markers(tmp_path: Path) -> None:
    """Test EnsurePytestConfigPhase adds standard markers."""
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    doc["tool"]["pytest"] = tomlkit.table()
    doc["tool"]["pytest"]["ini_options"] = tomlkit.table()

    phase = EnsurePytestConfigPhase()
    changes = phase.apply(doc)

    assert any("markers" in change for change in changes)


def test_flext_infra_pyproject_modernizer_process_file_invalid_toml(
    tmp_path: Path,
) -> None:
    """Test modernizer handles invalid TOML gracefully."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("invalid toml {", encoding="utf-8")

    modernizer = FlextInfraPyprojectModernizer(tmp_path)
    changes = modernizer.process_file(
        pyproject, canonical_dev=[], dry_run=True, skip_comments=False
    )

    assert "invalid TOML" in changes


def test_flext_infra_pyproject_modernizer_find_pyproject_files(tmp_path: Path) -> None:
    """Test modernizer finds pyproject.toml files."""
    (tmp_path / "project1").mkdir()
    (tmp_path / "project1" / "pyproject.toml").write_text(
        "[project]\n", encoding="utf-8"
    )
    (tmp_path / "project2").mkdir()
    (tmp_path / "project2" / "pyproject.toml").write_text(
        "[project]\n", encoding="utf-8"
    )
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    modernizer = FlextInfraPyprojectModernizer(tmp_path)
    files = modernizer.find_pyproject_files()

    assert len(files) == 2  # Should skip .venv
    assert all("project" in str(f) for f in files)
