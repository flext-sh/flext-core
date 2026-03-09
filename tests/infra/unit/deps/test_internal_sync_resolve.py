from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra.deps import internal_sync
from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import h


class TestResolveRef:
    def test_resolve_ref_github_actions_head_ref(self, monkeypatch) -> None:
        service = FlextInfraInternalDependencySyncService()
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_HEAD_REF", "feature/test")
        tm.that(service.resolve_ref(Path("/fake")), eq="feature/test")

    def test_resolve_ref_github_actions_ref_name(self, monkeypatch) -> None:
        service = FlextInfraInternalDependencySyncService()
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_HEAD_REF", "")
        monkeypatch.setenv("GITHUB_REF_NAME", "main")
        tm.that(service.resolve_ref(Path("/fake")), eq="main")

    def test_resolve_ref_git_branch(self, monkeypatch) -> None:
        service = FlextInfraInternalDependencySyncService()
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_current_branch",
            lambda _cwd: r[str].ok("develop"),
        )
        tm.that(service.resolve_ref(Path("/fake")), eq="develop")

    def test_resolve_ref_git_tag(self, monkeypatch) -> None:
        service = FlextInfraInternalDependencySyncService()
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_current_branch", lambda _cwd: r[str].ok("HEAD")
        )
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_run", lambda _cmd, cwd: r[str].ok("v1.0.0")
        )
        tm.that(service.resolve_ref(Path("/fake")), eq="v1.0.0")

    def test_resolve_ref_fallback_main(self, monkeypatch) -> None:
        service = FlextInfraInternalDependencySyncService()
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_current_branch",
            lambda _cwd: r[str].fail("not a git repo"),
        )
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_run",
            lambda _cmd, cwd: r[str].fail("not a git repo"),
        )
        tm.that(service.resolve_ref(Path("/fake")), eq="main")


class TestInferOwnerFromOrigin:
    def test_success(self, monkeypatch) -> None:
        service = FlextInfraInternalDependencySyncService()
        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_run",
            lambda _cmd, cwd: r[str].ok("git@github.com:flext-sh/flext-core.git"),
        )
        tm.that(service.infer_owner_from_origin(Path("/fake")), eq="flext-sh")

    def test_failure(self, monkeypatch) -> None:
        service = FlextInfraInternalDependencySyncService()
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_run", lambda _cmd, cwd: r[str].fail("no remote")
        )
        tm.that(service.infer_owner_from_origin(Path("/fake")), eq=None)

    def test_nonzero_exit(self, monkeypatch) -> None:
        service = FlextInfraInternalDependencySyncService()
        monkeypatch.setattr(
            internal_sync.u.Infra, "git_run", lambda _cmd, cwd: r[str].ok("")
        )
        tm.that(service.infer_owner_from_origin(Path("/fake")), eq=None)


class TestSynthesizedRepoMap:
    def test_generates_urls(self) -> None:
        result = FlextInfraInternalDependencySyncService().synthesized_repo_map(
            "flext-sh",
            {"flext-core", "flext-api"},
        )
        tm.that("flext-api" in result, eq=True)
        tm.that("flext-core" in result, eq=True)
        tm.that(
            result["flext-core"].ssh_url, eq="git@github.com:flext-sh/flext-core.git"
        )
        tm.that(result["flext-core"].https_url.startswith("https://"), eq=True)
        tm.that(h is not None, eq=True)
