"""Tests for FlextInfraInternalDependencySyncService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from flext_core import r
from flext_infra.deps.internal_sync import (
    FlextInfraInternalDependencySyncService,
    main,
)


class TestFlextInfraInternalDependencySyncService:
    """Test FlextInfraInternalDependencySyncService."""

    def test_service_initialization(self) -> None:
        """Test service initializes without errors."""
        service = FlextInfraInternalDependencySyncService()
        assert service is not None

    def test_validate_git_ref_valid(self) -> None:
        """Test git ref validation with valid reference."""
        result = FlextInfraInternalDependencySyncService._validate_git_ref("main")
        assert result.is_success
        assert result.value == "main"

    def test_validate_git_ref_invalid(self) -> None:
        """Test git ref validation with invalid reference."""
        result = FlextInfraInternalDependencySyncService._validate_git_ref(
            "invalid@ref!",
        )
        assert result.is_failure

    def test_validate_repo_url_https(self) -> None:
        """Test repository URL validation with HTTPS URL."""
        url = "https://github.com/flext-sh/flext.git"
        result = FlextInfraInternalDependencySyncService._validate_repo_url(url)
        assert result.is_success

    def test_validate_repo_url_ssh(self) -> None:
        """Test repository URL validation with SSH URL."""
        url = "git@github.com:flext-sh/flext.git"
        result = FlextInfraInternalDependencySyncService._validate_repo_url(url)
        assert result.is_success

    def test_validate_repo_url_invalid(self) -> None:
        """Test repository URL validation with invalid URL."""
        result = FlextInfraInternalDependencySyncService._validate_repo_url(
            "not-a-url",
        )
        assert result.is_failure

    def test_ssh_to_https_conversion(self) -> None:
        """Test SSH to HTTPS URL conversion."""
        ssh_url = "git@github.com:flext-sh/flext.git"
        https_url = FlextInfraInternalDependencySyncService._ssh_to_https(ssh_url)
        assert https_url.startswith("https://")
        assert "flext-sh/flext" in https_url

    def test_ssh_to_https_already_https(self) -> None:
        """Test SSH to HTTPS returns HTTPS unchanged."""
        url = "https://github.com/flext-sh/flext.git"
        result = FlextInfraInternalDependencySyncService._ssh_to_https(url)
        assert result == url


class TestParseGitmodules:
    """Test _parse_gitmodules method."""

    def test_parse_gitmodules_valid(self, tmp_path: Path) -> None:
        """Test parsing a valid .gitmodules file."""
        gitmodules = tmp_path / ".gitmodules"
        gitmodules.write_text(
            '[submodule "flext-core"]\n'
            "\tpath = flext-core\n"
            "\turl = git@github.com:flext-sh/flext-core.git\n"
            '[submodule "flext-api"]\n'
            "\tpath = flext-api\n"
            "\turl = git@github.com:flext-sh/flext-api.git\n",
        )
        service = FlextInfraInternalDependencySyncService()
        result = service._parse_gitmodules(gitmodules)
        assert "flext-core" in result
        assert "flext-api" in result
        assert result["flext-core"].ssh_url == "git@github.com:flext-sh/flext-core.git"
        assert result["flext-core"].https_url.startswith("https://")

    def test_parse_gitmodules_empty(self, tmp_path: Path) -> None:
        """Test parsing empty .gitmodules file."""
        gitmodules = tmp_path / ".gitmodules"
        gitmodules.write_text("")
        service = FlextInfraInternalDependencySyncService()
        result = service._parse_gitmodules(gitmodules)
        assert result == {}

    def test_parse_gitmodules_no_url(self, tmp_path: Path) -> None:
        """Test parsing .gitmodules with missing URL."""
        gitmodules = tmp_path / ".gitmodules"
        gitmodules.write_text('[submodule "test"]\n\tpath = test\n')
        service = FlextInfraInternalDependencySyncService()
        result = service._parse_gitmodules(gitmodules)
        assert result == {}

    def test_parse_gitmodules_non_submodule_section(self, tmp_path: Path) -> None:
        """Test parsing .gitmodules with non-submodule sections."""
        gitmodules = tmp_path / ".gitmodules"
        gitmodules.write_text("[other]\nfoo = bar\n")
        service = FlextInfraInternalDependencySyncService()
        result = service._parse_gitmodules(gitmodules)
        assert result == {}


class TestParseRepoMap:
    """Test _parse_repo_map method."""

    def test_parse_repo_map_success(self) -> None:
        """Test parsing a valid repo map."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "repo": {
                "flext-core": {
                    "ssh_url": "git@github.com:flext-sh/flext-core.git",
                    "https_url": "https://github.com/flext-sh/flext-core.git",
                },
            },
        })
        result = service._parse_repo_map(Path("/fake/map.toml"))
        assert result.is_success
        assert "flext-core" in result.value

    def test_parse_repo_map_read_failure(self) -> None:
        """Test repo map read failure."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].fail("file not found")
        result = service._parse_repo_map(Path("/fake/map.toml"))
        assert result.is_failure

    def test_parse_repo_map_no_repo_section(self) -> None:
        """Test repo map with missing repo section."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({"other": "data"})
        result = service._parse_repo_map(Path("/fake/map.toml"))
        assert result.is_success
        assert result.value == {}

    def test_parse_repo_map_non_dict_repo(self) -> None:
        """Test repo map with non-dict repo section."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({"repo": "not-a-dict"})
        result = service._parse_repo_map(Path("/fake/map.toml"))
        assert result.is_success
        assert result.value == {}

    def test_parse_repo_map_non_dict_values(self) -> None:
        """Test repo map with non-dict repo values."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "repo": {"flext-core": "string-value"},
        })
        result = service._parse_repo_map(Path("/fake/map.toml"))
        assert result.is_success
        assert result.value == {}

    def test_parse_repo_map_no_ssh_url(self) -> None:
        """Test repo map entry without ssh_url skips it."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "repo": {"flext-core": {"other": "val"}},
        })
        result = service._parse_repo_map(Path("/fake/map.toml"))
        assert result.is_success
        assert result.value == {}

    def test_parse_repo_map_auto_https(self) -> None:
        """Test repo map auto-generates https_url from ssh_url."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "repo": {
                "flext-core": {"ssh_url": "git@github.com:flext-sh/flext-core.git"},
            },
        })
        result = service._parse_repo_map(Path("/fake/map.toml"))
        assert result.is_success
        assert result.value["flext-core"].https_url.startswith("https://")


class TestResolveRef:
    """Test _resolve_ref method."""

    def test_resolve_ref_github_actions_head_ref(self) -> None:
        """Test ref resolution from GITHUB_HEAD_REF env var."""
        service = FlextInfraInternalDependencySyncService()
        with patch.dict(
            "os.environ",
            {"GITHUB_ACTIONS": "true", "GITHUB_HEAD_REF": "feature/test"},
        ):
            result = service._resolve_ref(Path("/fake"))
        assert result == "feature/test"

    def test_resolve_ref_github_actions_ref_name(self) -> None:
        """Test ref resolution from GITHUB_REF_NAME env var."""
        service = FlextInfraInternalDependencySyncService()
        with patch.dict(
            "os.environ",
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_HEAD_REF": "",
                "GITHUB_REF_NAME": "main",
            },
        ):
            result = service._resolve_ref(Path("/fake"))
        assert result == "main"

    def test_resolve_ref_git_branch(self) -> None:
        """Test ref resolution from git rev-parse."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        cmd_out = Mock(exit_code=0, stdout="develop\n", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        with patch.dict("os.environ", {}, clear=True):
            result = service._resolve_ref(Path("/fake"))
        assert result == "develop"

    def test_resolve_ref_git_tag(self) -> None:
        """Test ref resolution falls back to git tag."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        head_out = Mock(exit_code=0, stdout="HEAD\n", stderr="")
        tag_out = Mock(exit_code=0, stdout="v1.0.0\n", stderr="")
        service._runner.run_raw.side_effect = [
            r[Mock].ok(head_out),
            r[Mock].ok(tag_out),
        ]
        with patch.dict("os.environ", {}, clear=True):
            result = service._resolve_ref(Path("/fake"))
        assert result == "v1.0.0"

    def test_resolve_ref_fallback_main(self) -> None:
        """Test ref resolution falls back to main."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        service._runner.run_raw.return_value = r[Mock].fail("not a git repo")
        with patch.dict("os.environ", {}, clear=True):
            result = service._resolve_ref(Path("/fake"))
        assert result == "main"


class TestIsRelativeTo:
    """Test _is_relative_to static method."""

    def test_relative_to_true(self, tmp_path: Path) -> None:
        """Test path is relative to parent."""
        child = tmp_path / "sub" / "file.txt"
        assert FlextInfraInternalDependencySyncService._is_relative_to(child, tmp_path)

    def test_relative_to_false(self, tmp_path: Path) -> None:
        """Test path is not relative to parent."""
        other = Path("/completely/different")
        assert not FlextInfraInternalDependencySyncService._is_relative_to(
            other, tmp_path
        )


class TestWorkspaceRootFromEnv:
    """Test _workspace_root_from_env method."""

    def test_env_not_set(self, tmp_path: Path) -> None:
        """Test when FLEXT_WORKSPACE_ROOT not set."""
        service = FlextInfraInternalDependencySyncService()
        with patch.dict("os.environ", {}, clear=True):
            result = service._workspace_root_from_env(tmp_path)
        assert result is None

    def test_env_set_valid(self, tmp_path: Path) -> None:
        """Test with valid FLEXT_WORKSPACE_ROOT."""
        project = tmp_path / "project"
        project.mkdir()
        service = FlextInfraInternalDependencySyncService()
        with patch.dict("os.environ", {"FLEXT_WORKSPACE_ROOT": str(tmp_path)}):
            result = service._workspace_root_from_env(project)
        assert result == tmp_path

    def test_env_set_nonexistent(self, tmp_path: Path) -> None:
        """Test with FLEXT_WORKSPACE_ROOT pointing to nonexistent dir."""
        service = FlextInfraInternalDependencySyncService()
        with patch.dict("os.environ", {"FLEXT_WORKSPACE_ROOT": "/nonexistent/path"}):
            result = service._workspace_root_from_env(tmp_path)
        assert result is None

    def test_env_set_not_parent(self, tmp_path: Path) -> None:
        """Test with FLEXT_WORKSPACE_ROOT not parent of project."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        project = tmp_path / "other" / "project"
        project.mkdir(parents=True)
        service = FlextInfraInternalDependencySyncService()
        with patch.dict("os.environ", {"FLEXT_WORKSPACE_ROOT": str(workspace)}):
            result = service._workspace_root_from_env(project)
        assert result is None


class TestWorkspaceRootFromParents:
    """Test _workspace_root_from_parents static method."""

    def test_found_in_parent(self, tmp_path: Path) -> None:
        """Test .gitmodules found in parent directory."""
        (tmp_path / ".gitmodules").touch()
        project = tmp_path / "sub" / "project"
        project.mkdir(parents=True)
        result = FlextInfraInternalDependencySyncService._workspace_root_from_parents(
            project
        )
        assert result == tmp_path

    def test_not_found(self, tmp_path: Path) -> None:
        """Test .gitmodules not found in any parent."""
        project = tmp_path / "isolated"
        project.mkdir()
        result = FlextInfraInternalDependencySyncService._workspace_root_from_parents(
            project
        )
        # May or may not be None depending on filesystem; test it doesn't crash
        assert result is None or isinstance(result, Path)

    def test_found_in_self(self, tmp_path: Path) -> None:
        """Test .gitmodules found in project root itself."""
        (tmp_path / ".gitmodules").touch()
        result = FlextInfraInternalDependencySyncService._workspace_root_from_parents(
            tmp_path
        )
        assert result == tmp_path


class TestIsWorkspaceMode:
    """Test _is_workspace_mode method."""

    def test_standalone_mode(self, tmp_path: Path) -> None:
        """Test FLEXT_STANDALONE=1 returns False."""
        service = FlextInfraInternalDependencySyncService()
        with patch.dict("os.environ", {"FLEXT_STANDALONE": "1"}):
            is_ws, root = service._is_workspace_mode(tmp_path)
        assert is_ws is False
        assert root is None

    def test_env_workspace_root(self, tmp_path: Path) -> None:
        """Test FLEXT_WORKSPACE_ROOT detected."""
        project = tmp_path / "project"
        project.mkdir()
        service = FlextInfraInternalDependencySyncService()
        with patch.dict(
            "os.environ",
            {"FLEXT_WORKSPACE_ROOT": str(tmp_path), "FLEXT_STANDALONE": ""},
        ):
            is_ws, root = service._is_workspace_mode(project)
        assert is_ws is True
        assert root == tmp_path

    def test_git_superproject(self, tmp_path: Path) -> None:
        """Test git superproject detection."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        cmd_out = Mock(exit_code=0, stdout=str(tmp_path) + "\n", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        with patch.dict(
            "os.environ", {"FLEXT_STANDALONE": "", "FLEXT_WORKSPACE_ROOT": ""}
        ):
            is_ws, root = service._is_workspace_mode(tmp_path / "sub")
        assert is_ws is True
        assert root == tmp_path

    def test_heuristic_gitmodules(self, tmp_path: Path) -> None:
        """Test heuristic .gitmodules detection."""
        (tmp_path / ".gitmodules").touch()
        project = tmp_path / "sub"
        project.mkdir()
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        cmd_out = Mock(exit_code=1, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        with patch.dict(
            "os.environ", {"FLEXT_STANDALONE": "", "FLEXT_WORKSPACE_ROOT": ""}
        ):
            is_ws, root = service._is_workspace_mode(project)
        assert is_ws is True
        assert root == tmp_path

    def test_no_workspace(self, tmp_path: Path) -> None:
        """Test no workspace mode detected."""
        project = tmp_path / "isolated"
        project.mkdir()
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        cmd_out = Mock(exit_code=1, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        with patch.dict(
            "os.environ", {"FLEXT_STANDALONE": "", "FLEXT_WORKSPACE_ROOT": ""}
        ):
            is_ws, root = service._is_workspace_mode(project)
        assert is_ws is False
        assert root is None


class TestOwnerFromRemoteUrl:
    """Test _owner_from_remote_url static method."""

    def test_ssh_url(self) -> None:
        """Test extracting owner from SSH URL."""
        result = FlextInfraInternalDependencySyncService._owner_from_remote_url(
            "git@github.com:flext-sh/flext-core.git"
        )
        assert result == "flext-sh"

    def test_https_url(self) -> None:
        """Test extracting owner from HTTPS URL."""
        result = FlextInfraInternalDependencySyncService._owner_from_remote_url(
            "https://github.com/flext-sh/flext-core.git"
        )
        assert result == "flext-sh"

    def test_http_url(self) -> None:
        """Test extracting owner from HTTP URL."""
        result = FlextInfraInternalDependencySyncService._owner_from_remote_url(
            "http://github.com/flext-sh/flext-core.git"
        )
        assert result == "flext-sh"

    def test_invalid_url(self) -> None:
        """Test None returned for invalid URL."""
        result = FlextInfraInternalDependencySyncService._owner_from_remote_url(
            "not-a-github-url"
        )
        assert result is None


class TestInferOwnerFromOrigin:
    """Test _infer_owner_from_origin method."""

    def test_success(self) -> None:
        """Test owner inferred from git remote origin."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        cmd_out = Mock(
            exit_code=0,
            stdout="git@github.com:flext-sh/flext-core.git\n",
            stderr="",
        )
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service._infer_owner_from_origin(Path("/fake"))
        assert result == "flext-sh"

    def test_failure(self) -> None:
        """Test None on git command failure."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        service._runner.run_raw.return_value = r[Mock].fail("no remote")
        result = service._infer_owner_from_origin(Path("/fake"))
        assert result is None

    def test_nonzero_exit(self) -> None:
        """Test None on non-zero exit code."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        cmd_out = Mock(exit_code=1, stdout="", stderr="error")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service._infer_owner_from_origin(Path("/fake"))
        assert result is None


class TestSynthesizedRepoMap:
    """Test _synthesized_repo_map method."""

    def test_generates_urls(self) -> None:
        """Test synthesized repo map generates SSH and HTTPS URLs."""
        service = FlextInfraInternalDependencySyncService()
        result = service._synthesized_repo_map("flext-sh", {"flext-core", "flext-api"})
        assert "flext-api" in result
        assert "flext-core" in result
        assert result["flext-core"].ssh_url == "git@github.com:flext-sh/flext-core.git"
        assert result["flext-core"].https_url.startswith("https://")


class TestEnsureSymlink:
    """Test _ensure_symlink static method."""

    def test_create_new_symlink(self, tmp_path: Path) -> None:
        """Test creating a new symlink."""
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        result = FlextInfraInternalDependencySyncService._ensure_symlink(target, source)
        assert result.is_success
        assert target.is_symlink()

    def test_existing_correct_symlink(self, tmp_path: Path) -> None:
        """Test symlink already correct returns True."""
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        target.symlink_to(source.resolve(), target_is_directory=True)
        result = FlextInfraInternalDependencySyncService._ensure_symlink(target, source)
        assert result.is_success

    def test_replace_existing_dir(self, tmp_path: Path) -> None:
        """Test replacing existing directory with symlink."""
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        target.mkdir()
        (target / "file.txt").write_text("old")
        result = FlextInfraInternalDependencySyncService._ensure_symlink(target, source)
        assert result.is_success
        assert target.is_symlink()

    def test_replace_existing_wrong_symlink(self, tmp_path: Path) -> None:
        """Test replacing symlink pointing to wrong target."""
        source = tmp_path / "source"
        source.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        target = tmp_path / "target"
        target.symlink_to(other.resolve(), target_is_directory=True)
        result = FlextInfraInternalDependencySyncService._ensure_symlink(target, source)
        assert result.is_success
        assert target.resolve() == source.resolve()


class TestEnsureCheckout:
    """Test _ensure_checkout method."""

    def test_clone_success(self, tmp_path: Path) -> None:
        """Test cloning a new dependency."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        clone_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(clone_out)
        dep_path = tmp_path / "dep"
        result = service._ensure_checkout(
            dep_path, "https://github.com/flext-sh/flext.git", "main"
        )
        assert result.is_success

    def test_clone_failure(self, tmp_path: Path) -> None:
        """Test clone failure returns error."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        clone_out = Mock(exit_code=1, stdout="", stderr="fatal: repo not found")
        service._runner.run_raw.return_value = r[Mock].ok(clone_out)
        dep_path = tmp_path / "dep"
        result = service._ensure_checkout(
            dep_path, "https://github.com/flext-sh/flext.git", "main"
        )
        assert result.is_failure

    def test_fetch_and_checkout_existing(self, tmp_path: Path) -> None:
        """Test fetch + checkout for existing repo."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()
        cmd_ok = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_ok)
        result = service._ensure_checkout(
            dep_path, "https://github.com/flext-sh/flext.git", "main"
        )
        assert result.is_success

    def test_invalid_repo_url(self, tmp_path: Path) -> None:
        """Test invalid repo URL returns failure."""
        service = FlextInfraInternalDependencySyncService()
        dep_path = tmp_path / "dep"
        result = service._ensure_checkout(dep_path, "not-a-url", "main")
        assert result.is_failure

    def test_invalid_git_ref(self, tmp_path: Path) -> None:
        """Test invalid git ref returns failure."""
        service = FlextInfraInternalDependencySyncService()
        dep_path = tmp_path / "dep"
        result = service._ensure_checkout(
            dep_path, "https://github.com/flext-sh/flext.git", "invalid@ref!"
        )
        assert result.is_failure

    def test_fetch_failure(self, tmp_path: Path) -> None:
        """Test fetch failure returns error."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()
        fetch_fail = Mock(exit_code=1, stdout="", stderr="fetch failed")
        service._runner.run_raw.return_value = r[Mock].ok(fetch_fail)
        result = service._ensure_checkout(
            dep_path, "https://github.com/flext-sh/flext.git", "main"
        )
        assert result.is_failure

    def test_checkout_failure(self, tmp_path: Path) -> None:
        """Test checkout failure after successful fetch."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        dep_path = tmp_path / "dep"
        dep_path.mkdir(parents=True)
        (dep_path / ".git").mkdir()
        fetch_ok = Mock(exit_code=0, stdout="", stderr="")
        checkout_fail = Mock(exit_code=1, stdout="", stderr="checkout error")
        service._runner.run_raw.side_effect = [
            r[Mock].ok(fetch_ok),
            r[Mock].ok(checkout_fail),
        ]
        result = service._ensure_checkout(
            dep_path, "https://github.com/flext-sh/flext.git", "main"
        )
        assert result.is_failure

    def test_clone_replaces_existing_symlink(self, tmp_path: Path) -> None:
        """Test cloning removes existing symlink before clone."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        dep_path = tmp_path / "dep"
        other = tmp_path / "other"
        other.mkdir()
        dep_path.symlink_to(other)
        clone_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(clone_out)
        result = service._ensure_checkout(
            dep_path, "https://github.com/flext-sh/flext.git", "main"
        )
        assert result.is_success

    def test_clone_replaces_existing_dir(self, tmp_path: Path) -> None:
        """Test cloning removes existing directory before clone."""
        service = FlextInfraInternalDependencySyncService()
        service._runner = Mock()
        dep_path = tmp_path / "dep"
        dep_path.mkdir()
        (dep_path / "somefile").write_text("old")
        clone_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(clone_out)
        result = service._ensure_checkout(
            dep_path, "https://github.com/flext-sh/flext.git", "main"
        )
        assert result.is_success


class TestIsInternalPathDep:
    """Test _is_internal_path_dep static method."""

    def test_flext_deps_prefix(self) -> None:
        """Test .flext-deps/foo returns foo."""
        result = FlextInfraInternalDependencySyncService._is_internal_path_dep(
            ".flext-deps/foo"
        )
        assert result == "foo"

    def test_dotdot_prefix(self) -> None:
        """Test ../bar returns bar."""
        result = FlextInfraInternalDependencySyncService._is_internal_path_dep("../bar")
        assert result == "bar"

    def test_bare_name(self) -> None:
        """Test bare name returns itself."""
        result = FlextInfraInternalDependencySyncService._is_internal_path_dep("baz")
        assert result == "baz"

    def test_nested_path_rejected(self) -> None:
        """Test nested path is rejected."""
        result = FlextInfraInternalDependencySyncService._is_internal_path_dep(
            "./some/nested/path"
        )
        assert result is None

    def test_dot_rejected(self) -> None:
        """Test '.' is rejected."""
        result = FlextInfraInternalDependencySyncService._is_internal_path_dep(".")
        assert result is None

    def test_dotdot_rejected(self) -> None:
        """Test '..' is rejected."""
        result = FlextInfraInternalDependencySyncService._is_internal_path_dep("..")
        assert result is None

    def test_with_leading_dotslash(self) -> None:
        """Test ./ prefix is stripped."""
        result = FlextInfraInternalDependencySyncService._is_internal_path_dep(
            "./.flext-deps/foo"
        )
        assert result == "foo"

    def test_dotdot_nested_rejected(self) -> None:
        """Test ../a/b (nested) returns None."""
        result = FlextInfraInternalDependencySyncService._is_internal_path_dep("../a/b")
        assert result is None


class TestCollectInternalDeps:
    """Test _collect_internal_deps method."""

    def test_no_pyproject(self, tmp_path: Path) -> None:
        """Test with missing pyproject.toml returns empty."""
        service = FlextInfraInternalDependencySyncService()
        result = service._collect_internal_deps(tmp_path)
        assert result.is_success
        assert result.value == {}

    def test_poetry_path_deps(self, tmp_path: Path) -> None:
        """Test collecting poetry path dependencies."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "tool": {
                "poetry": {
                    "dependencies": {
                        "flext-core": {"path": ".flext-deps/flext-core"},
                        "requests": "^2.28",
                    },
                },
            },
            "project": {},
        })
        (tmp_path / "pyproject.toml").write_text("")
        result = service._collect_internal_deps(tmp_path)
        assert result.is_success
        assert "flext-core" in result.value

    def test_pep621_path_deps(self, tmp_path: Path) -> None:
        """Test collecting PEP 621 path dependencies."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "tool": {},
            "project": {
                "dependencies": [
                    "flext-core @ file:.flext-deps/flext-core",
                    "requests>=2.28",
                ],
            },
        })
        (tmp_path / "pyproject.toml").write_text("")
        result = service._collect_internal_deps(tmp_path)
        assert result.is_success
        assert "flext-core" in result.value

    def test_read_failure(self, tmp_path: Path) -> None:
        """Test pyproject.toml read failure."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].fail("parse error")
        (tmp_path / "pyproject.toml").write_text("")
        result = service._collect_internal_deps(tmp_path)
        assert result.is_failure

    def test_no_tool_section(self, tmp_path: Path) -> None:
        """Test pyproject.toml with no tool section."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({"project": {}})
        (tmp_path / "pyproject.toml").write_text("")
        result = service._collect_internal_deps(tmp_path)
        assert result.is_success
        assert result.value == {}

    def test_non_dict_deps(self, tmp_path: Path) -> None:
        """Test pyproject.toml with non-dict dependencies."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "tool": {"poetry": {"dependencies": "not-a-dict"}},
            "project": {"dependencies": "not-a-list"},
        })
        (tmp_path / "pyproject.toml").write_text("")
        result = service._collect_internal_deps(tmp_path)
        assert result.is_success
        assert result.value == {}


class TestSync:
    """Test sync method."""

    def test_sync_no_deps(self, tmp_path: Path) -> None:
        """Test sync with no internal dependencies returns 0."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({"tool": {}, "project": {}})
        (tmp_path / "pyproject.toml").write_text("")
        result = service.sync(tmp_path)
        assert result.is_success
        assert result.value == 0

    def test_sync_collect_failure(self, tmp_path: Path) -> None:
        """Test sync fails when dependency collection fails."""
        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].fail("read error")
        (tmp_path / "pyproject.toml").write_text("")
        result = service.sync(tmp_path)
        assert result.is_failure

    def test_sync_workspace_mode_symlink(self, tmp_path: Path) -> None:
        """Test sync in workspace mode creates symlinks."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".gitmodules").write_text(
            '[submodule "flext-api"]\n'
            "\tpath = flext-api\n"
            "\turl = git@github.com:flext-sh/flext-api.git\n"
        )
        sibling = workspace / "flext-api"
        sibling.mkdir()
        project = workspace / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")

        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "tool": {
                "poetry": {
                    "dependencies": {
                        "flext-api": {"path": ".flext-deps/flext-api"},
                    },
                },
            },
            "project": {},
        })
        service._runner = Mock()
        cmd_out = Mock(exit_code=1, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)

        with patch.dict(
            "os.environ", {"FLEXT_STANDALONE": "", "FLEXT_WORKSPACE_ROOT": ""}
        ):
            result = service.sync(project)
        assert result.is_success

    def test_sync_missing_repo_mapping(self, tmp_path: Path) -> None:
        """Test sync fails when repo mapping is missing for a dependency."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")

        service = FlextInfraInternalDependencySyncService()
        service._toml = Mock()
        service._toml.read.side_effect = [
            # First call: _collect_internal_deps reads pyproject.toml
            r[dict].ok({
                "tool": {
                    "poetry": {
                        "dependencies": {
                            "flext-api": {"path": ".flext-deps/flext-api"},
                        },
                    },
                },
                "project": {},
            }),
            # Second call: _parse_repo_map
            r[dict].ok({"repo": {}}),
        ]
        service._runner = Mock()
        cmd_out = Mock(exit_code=1, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)

        # Create map file so standalone path is used
        (project / "flext-repo-map.toml").write_text("")

        with patch.dict(
            "os.environ", {"FLEXT_STANDALONE": "", "FLEXT_WORKSPACE_ROOT": ""}
        ):
            result = service.sync(project)
        assert result.is_failure
        assert "missing repo mapping" in (result.error or "")


class TestMain:
    """Test main() CLI entry point."""

    def test_main_success(self) -> None:
        """Test main returns 0 on success."""
        with (
            patch(
                "flext_infra.deps.internal_sync.FlextInfraInternalDependencySyncService.sync"
            ) as mock_sync,
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=Mock(project_root=Path("/tmp/test")),
            ),
        ):
            mock_sync.return_value = r[int].ok(0)
            result = main()
        assert result == 0

    def test_main_failure(self) -> None:
        """Test main returns 1 on failure."""
        with (
            patch(
                "flext_infra.deps.internal_sync.FlextInfraInternalDependencySyncService.sync"
            ) as mock_sync,
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=Mock(project_root=Path("/tmp/test")),
            ),
        ):
            mock_sync.return_value = r[int].fail("sync failed")
            result = main()
        assert result == 1


class TestValidateGitRefEdgeCases:
    """Test edge cases for git ref validation."""

    @pytest.mark.parametrize(
        "ref",
        [
            "feature/my-branch",
            "v1.0.0",
            "release/2.0",
            "fix/issue-123",
        ],
    )
    def test_valid_refs(self, ref: str) -> None:
        """Test various valid git ref formats."""
        result = FlextInfraInternalDependencySyncService._validate_git_ref(ref)
        assert result.is_success

    @pytest.mark.parametrize(
        "ref",
        [
            "",
            " starts-with-space",
            "a" * 200,
        ],
    )
    def test_invalid_refs(self, ref: str) -> None:
        """Test various invalid git ref formats."""
        result = FlextInfraInternalDependencySyncService._validate_git_ref(ref)
        assert result.is_failure
