"""Tests for FlextInfraGitService using real git repositories.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesGit
from flext_tests import tm


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a real git repository for testing."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


class TestFlextInfraGitService:
    """Test suite for FlextInfraGitService with real repos."""

    def test_current_branch_success(self, git_repo: Path) -> None:
        result = FlextInfraUtilitiesGit.git_current_branch(git_repo)
        tm.ok(result, is_=str)

    def test_current_branch_failure(self, tmp_path: Path) -> None:
        result = FlextInfraUtilitiesGit.git_current_branch(tmp_path)
        tm.fail(result)

    def test_tag_exists_true(self, git_repo: Path) -> None:
        subprocess.run(
            ["git", "tag", "-a", "v1.0.0", "-m", "release"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )
        result = FlextInfraUtilitiesGit.git_tag_exists(git_repo, "v1.0.0")
        tm.ok(result, eq=True)

    def test_tag_exists_false(self, git_repo: Path) -> None:
        result = FlextInfraUtilitiesGit.git_tag_exists(git_repo, "v9.9.9")
        tm.ok(result, eq=False)

    def test_tag_exists_failure(self, tmp_path: Path) -> None:
        result = FlextInfraUtilitiesGit.git_tag_exists(tmp_path, "v1.0.0")
        tm.fail(result)

    def test_run_arbitrary_command(self, git_repo: Path) -> None:
        result = FlextInfraUtilitiesGit.git_run(["log", "--oneline"], cwd=git_repo)
        tm.ok(result, is_=str)

    def test_run_command_failure(self, tmp_path: Path) -> None:
        result = FlextInfraUtilitiesGit.git_run(["log"], cwd=tmp_path)
        tm.fail(result)


class TestRemovedCompatibilityMethods:
    """Removed compatibility methods are not callable anymore."""

    def test_removed_methods_raise_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            _ = getattr(FlextInfraUtilitiesGit, "smart_checkout")
        with pytest.raises(AttributeError):
            _ = getattr(FlextInfraUtilitiesGit, "checkpoint")
        with pytest.raises(AttributeError):
            _ = getattr(FlextInfraUtilitiesGit, "create_tag_if_missing")
        with pytest.raises(AttributeError):
            _ = getattr(FlextInfraUtilitiesGit, "collect_changes")


class TestGitTagOperations:
    """Tests for tag listing and creation."""

    def test_list_tags(self, git_repo: Path) -> None:
        subprocess.run(
            ["git", "tag", "-a", "v1.0.0", "-m", "r1"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )
        result = FlextInfraUtilitiesGit.git_list_tags(git_repo)
        tm.ok(result, has="v1.0.0")

    def test_list_tags_failure(self, tmp_path: Path) -> None:
        result = FlextInfraUtilitiesGit.git_list_tags(tmp_path)
        tm.fail(result)

    def test_create_tag(self, git_repo: Path) -> None:
        result = FlextInfraUtilitiesGit.git_create_tag(git_repo, "v3.0.0")
        tm.ok(result, eq=True)
        verify = FlextInfraUtilitiesGit.git_tag_exists(git_repo, "v3.0.0")
        tm.ok(verify, eq=True)


class TestGitPush:
    """Tests for git push operations."""

    def test_push_to_nonexistent_remote_fails(self, git_repo: Path) -> None:
        result = FlextInfraUtilitiesGit.git_push(
            git_repo, remote="origin", branch="main", set_upstream=True
        )
        tm.fail(result)
