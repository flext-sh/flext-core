from __future__ import annotations

from pathlib import Path

from pydantic import Field

from flext_core import r
from flext_infra._utilities.git import FlextInfraUtilitiesGit
from flext_tests.base import FlextTestsUtilityBase


class RealGitService(FlextTestsUtilityBase[bool]):
    git_utility: type[FlextInfraUtilitiesGit] = Field(
        default=FlextInfraUtilitiesGit,
        description="Injected git utility implementation.",
    )

    def _as_success(self, result: r[bool], operation: str) -> r[bool]:
        if result.is_failure:
            return r[bool].fail(result.error or f"git {operation} failed")
        if not result.value:
            return r[bool].fail(f"git {operation} returned false")
        return r[bool].ok(True)

    def init_repo(self, path: Path) -> r[bool]:
        path.mkdir(parents=True, exist_ok=True)
        return self._as_success(
            self.git_utility.git_run_checked(["init"], cwd=path), "init"
        )

    def add_all(self, path: Path) -> r[bool]:
        return self._as_success(self.git_utility.git_add(path), "add")

    def commit(self, path: Path, msg: str) -> r[bool]:
        if not msg.strip():
            return r[bool].fail("commit message must not be empty")

        email_result = self.git_utility.git_run_checked(
            ["config", "user.email", "flext-tests@example.com"],
            cwd=path,
        )
        if email_result.is_failure:
            return r[bool].fail(email_result.error or "git config user.email failed")

        name_result = self.git_utility.git_run_checked(
            ["config", "user.name", "Flext Tests"],
            cwd=path,
        )
        if name_result.is_failure:
            return r[bool].fail(name_result.error or "git config user.name failed")

        return self._as_success(self.git_utility.git_commit(path, msg), "commit")

    def create_branch(self, path: Path, name: str) -> r[bool]:
        if not name.strip():
            return r[bool].fail("branch name must not be empty")
        return self._as_success(
            self.git_utility.git_run_checked(["branch", name], cwd=path),
            "branch",
        )


__all__ = ["RealGitService"]
