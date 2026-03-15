from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import h


class TestFlextInfraInternalDependencySyncService:
    def test_service_initialization(self) -> None:
        service = FlextInfraInternalDependencySyncService()
        tm.that(
            service.__class__.__name__, eq="FlextInfraInternalDependencySyncService"
        )
        tm.that(hasattr(h, "assert_ok"), eq=True)

    def test_validate_git_ref_valid(self) -> None:
        tm.ok(
            FlextInfraInternalDependencySyncService.validate_git_ref("main"), eq="main"
        )

    def test_validate_git_ref_invalid(self) -> None:
        tm.fail(
            FlextInfraInternalDependencySyncService.validate_git_ref("invalid@ref!")
        )

    def test_validate_repo_url_https(self) -> None:
        tm.ok(
            FlextInfraInternalDependencySyncService.validate_repo_url(
                "https://github.com/flext-sh/flext.git",
            ),
        )

    def test_validate_repo_url_ssh(self) -> None:
        tm.ok(
            FlextInfraInternalDependencySyncService.validate_repo_url(
                "git@github.com:flext-sh/flext.git",
            ),
        )

    def test_validate_repo_url_invalid(self) -> None:
        tm.fail(FlextInfraInternalDependencySyncService.validate_repo_url("not-a-url"))

    def test_ssh_to_https_conversion(self) -> None:
        result = FlextInfraInternalDependencySyncService.ssh_to_https(
            "git@github.com:flext-sh/flext.git",
        )
        tm.that(result.startswith("https://"), eq=True)
        tm.that(result, contains="flext-sh/flext")

    def test_ssh_to_https_already_https(self) -> None:
        url = "https://github.com/flext-sh/flext.git"
        tm.that(FlextInfraInternalDependencySyncService.ssh_to_https(url), eq=url)


class TestValidateGitRefEdgeCases:
    @pytest.mark.parametrize(
        "ref",
        ["feature/my-branch", "v1.0.0", "release/2.0", "fix/issue-123"],
    )
    def test_valid_refs(self, ref: str) -> None:
        tm.ok(FlextInfraInternalDependencySyncService.validate_git_ref(ref))

    @pytest.mark.parametrize("ref", ["", " starts-with-space", "a" * 200])
    def test_invalid_refs(self, ref: str) -> None:
        tm.fail(FlextInfraInternalDependencySyncService.validate_git_ref(ref))


class TestOwnerFromRemoteUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("git@github.com:flext-sh/flext-core.git", "flext-sh"),
            ("https://github.com/flext-sh/flext-core.git", "flext-sh"),
            ("http://github.com/flext-sh/flext-core.git", "flext-sh"),
            ("not-a-github-url", None),
        ],
    )
    def test_owner_from_remote_url(self, url: str, expected: str | None) -> None:
        tm.that(
            FlextInfraInternalDependencySyncService.owner_from_remote_url(url),
            eq=expected,
        )


class TestIsRelativeTo:
    def test_relative_to_true(self, tmp_path: Path) -> None:
        child = tmp_path / "sub" / "file.txt"
        tm.that(
            FlextInfraInternalDependencySyncService.is_relative_to(child, tmp_path),
            eq=True,
        )

    def test_relative_to_false(self, tmp_path: Path) -> None:
        tm.that(
            FlextInfraInternalDependencySyncService.is_relative_to(
                Path("/completely/different"),
                tmp_path,
            ),
            eq=False,
        )


class TestIsInternalPathDep:
    @pytest.mark.parametrize(
        ("raw_path", "expected"),
        [
            (".flext-deps/foo", "foo"),
            ("../bar", "bar"),
            ("baz", "baz"),
            ("./some/nested/path", None),
            (".", None),
            ("..", None),
            ("./.flext-deps/foo", "foo"),
            ("../a/b", None),
        ],
    )
    def test_is_internal_path_dep(self, raw_path: str, expected: str | None) -> None:
        tm.that(
            FlextInfraInternalDependencySyncService.is_internal_path_dep(raw_path),
            eq=expected,
        )
