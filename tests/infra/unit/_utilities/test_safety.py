from __future__ import annotations

import subprocess
from pathlib import Path

from flext_infra import c, u


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        [c.Infra.Cli.GIT, *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_git_repo(repo: Path) -> None:
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "tests@flext.local")
    _run_git(repo, "config", "user.name", "Flext Tests")
    (repo / "README.md").write_text("# test repo\n", encoding="utf-8")
    _run_git(repo, "add", "README.md")
    _run_git(repo, "commit", "-m", "initial commit")


class TestSafetyCheckpoint:
    def test_create_checkpoint_returns_empty_for_clean_repo(
        self, tmp_path: Path
    ) -> None:
        _init_git_repo(tmp_path)

        result = u.Infra.create_checkpoint(tmp_path)

        assert result.is_success
        assert result.value == ""

    def test_create_checkpoint_creates_stash_for_dirty_repo(
        self, tmp_path: Path
    ) -> None:
        _init_git_repo(tmp_path)
        (tmp_path / "notes.txt").write_text("dirty\n", encoding="utf-8")

        result = u.Infra.create_checkpoint(tmp_path, label="test-checkpoint")

        assert result.is_success
        assert "stash@{0}" in result.value
        assert "test-checkpoint:" in result.value

    def test_create_checkpoint_returns_empty_for_non_git_folder(
        self, tmp_path: Path
    ) -> None:
        result = u.Infra.create_checkpoint(tmp_path)

        assert result.is_success
        assert result.value == ""


class TestSafetyRollback:
    def test_rollback_to_checkpoint_invalid_stash_ref_fails(
        self, tmp_path: Path
    ) -> None:
        _init_git_repo(tmp_path)

        result = u.Infra.rollback_to_checkpoint(tmp_path, "stash@{999}")

        assert result.is_failure

    def test_rollback_to_checkpoint_succeeds_for_non_repo(self, tmp_path: Path) -> None:
        result = u.Infra.rollback_to_checkpoint(tmp_path)

        assert result.is_success
        assert result.value is True


class TestSafetyWorkspaceValidation:
    def test_validate_workspace_clean_true_for_clean_repo(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)

        result = u.Infra.validate_workspace_clean(tmp_path)

        assert result.is_success
        assert result.value is True

    def test_validate_workspace_clean_false_for_dirty_repo(
        self, tmp_path: Path
    ) -> None:
        _init_git_repo(tmp_path)
        (tmp_path / "dirty.txt").write_text("dirty\n", encoding="utf-8")

        result = u.Infra.validate_workspace_clean(tmp_path)

        assert result.is_success
        assert result.value is False
