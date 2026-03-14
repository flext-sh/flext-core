"""Tests for FlextInfraPrManager operations (view, checks, merge, close, release)."""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra.github.pr import FlextInfraPrManager
from flext_tests import tm
from tests.infra.models import m
from tests.infra.unit.github._stubs import StubRunner, StubVersioning


def _mgr(
    runner: StubRunner | None = None,
    versioning: StubVersioning | None = None,
) -> FlextInfraPrManager:
    return FlextInfraPrManager(
        runner=runner or StubRunner(),
        versioning=versioning or StubVersioning(),
    )


def _ok_cmd() -> r[m.Infra.Core.CommandOutput]:
    return r[m.Infra.Core.CommandOutput].ok(
        m.Infra.Core.CommandOutput(exit_code=0, stdout="", stderr=""),
    )


class TestView:
    def test_view_success(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].ok("PR details")])
        tm.ok(_mgr(runner=runner).view(tmp_path, "42"))

    def test_view_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].fail("not found")])
        tm.fail(_mgr(runner=runner).view(tmp_path, "999"))


class TestChecks:
    def test_checks_pass(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[_ok_cmd()])
        result = _mgr(runner=runner).checks(tmp_path, "42")
        tm.ok(result)
        tm.that(result.value["status"], eq="checks-passed")

    def test_checks_fail_non_strict(self, tmp_path: Path) -> None:
        runner = StubRunner(
            run_returns=[r[m.Infra.Core.CommandOutput].fail("checks failed")]
        )
        result = _mgr(runner=runner).checks(tmp_path, "42")
        tm.ok(result)
        tm.that(result.value["status"], eq="checks-nonblocking")

    def test_checks_fail_strict(self, tmp_path: Path) -> None:
        runner = StubRunner(
            run_returns=[r[m.Infra.Core.CommandOutput].fail("checks failed")]
        )
        tm.fail(_mgr(runner=runner).checks(tmp_path, "42", strict=True))


class TestMerge:
    def test_merge_success(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[_ok_cmd()])
        result = _mgr(runner=runner).merge(
            tmp_path, "42", "feature", release_on_merge=False
        )
        tm.ok(result)
        tm.that(result.value["status"], eq="merged")

    def test_merge_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(
            run_returns=[r[m.Infra.Core.CommandOutput].fail("merge conflict")]
        )
        tm.fail(_mgr(runner=runner).merge(tmp_path, "42", "feature"))

    def test_merge_not_mergeable_retry(self, tmp_path: Path) -> None:
        runner = StubRunner(
            run_returns=[
                r[m.Infra.Core.CommandOutput].fail("not mergeable"),
                _ok_cmd(),
                _ok_cmd(),
            ]
        )
        tm.ok(
            _mgr(runner=runner).merge(tmp_path, "42", "feature", release_on_merge=False)
        )

    def test_merge_selector_same_as_head_no_pr(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].ok("[]")])
        result = _mgr(runner=runner).merge(tmp_path, "feature", "feature")
        tm.ok(result)
        tm.that(result.value["status"], eq="no-open-pr")

    def test_merge_with_release(self, tmp_path: Path) -> None:
        versioning = StubVersioning(release_tag_returns=r[str].ok("v1.0.0"))
        runner = StubRunner(run_returns=[_ok_cmd(), _ok_cmd()])
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: Release")
        tm.ok(
            _mgr(runner=runner, versioning=versioning).merge(
                tmp_path,
                "42",
                "release/1.0",
                release_on_merge=True,
            )
        )

    def test_merge_auto_and_delete_branch(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[_ok_cmd()])
        result = _mgr(runner=runner).merge(
            tmp_path,
            "42",
            "feature",
            auto=True,
            delete_branch=True,
            release_on_merge=False,
        )
        tm.ok(result)
        tm.that(runner.run_calls[0], contains="--auto")
        tm.that(runner.run_calls[0], contains="--delete-branch")

    def test_merge_rebase_method(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[_ok_cmd()])
        result = _mgr(runner=runner).merge(
            tmp_path,
            "42",
            "feature",
            method="rebase",
            release_on_merge=False,
        )
        tm.ok(result)
        tm.that(runner.run_calls[0], contains="--rebase")


class TestClose:
    def test_close_success(self, tmp_path: Path) -> None:
        runner = StubRunner(run_checked_returns=[r[bool].ok(True)])
        tm.ok(_mgr(runner=runner).close(tmp_path, "42"))

    def test_close_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(run_checked_returns=[r[bool].fail("close failed")])
        tm.fail(_mgr(runner=runner).close(tmp_path, "42"))


class TestTriggerRelease:
    def _release_setup(self, tmp_path: Path) -> None:
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: R")

    def test_no_release_workflow(self, tmp_path: Path) -> None:
        result = _mgr()._trigger_release_if_needed(tmp_path, "feature")
        tm.ok(result)
        tm.that(result.value["status"], eq="no-release-workflow")

    def test_no_release_tag(self, tmp_path: Path) -> None:
        self._release_setup(tmp_path)
        versioning = StubVersioning(release_tag_returns=r[str].fail("no tag"))
        result = _mgr(versioning=versioning)._trigger_release_if_needed(
            tmp_path, "feature"
        )
        tm.ok(result)
        tm.that(result.value["status"], eq="no-release-tag")

    def test_release_exists(self, tmp_path: Path) -> None:
        self._release_setup(tmp_path)
        versioning = StubVersioning(release_tag_returns=r[str].ok("v1.0.0"))
        runner = StubRunner(run_returns=[_ok_cmd()])
        result = _mgr(runner=runner, versioning=versioning)._trigger_release_if_needed(
            tmp_path,
            "release/1.0",
        )
        tm.ok(result)
        tm.that(result.value["status"], eq="release-exists")

    def test_release_dispatched(self, tmp_path: Path) -> None:
        self._release_setup(tmp_path)
        versioning = StubVersioning(release_tag_returns=r[str].ok("v1.0.0"))
        runner = StubRunner(
            run_returns=[r[m.Infra.Core.CommandOutput].fail("not found"), _ok_cmd()]
        )
        result = _mgr(runner=runner, versioning=versioning)._trigger_release_if_needed(
            tmp_path,
            "release/1.0",
        )
        tm.ok(result)
        tm.that(result.value["status"], eq="release-dispatched")

    def test_release_dispatch_failed(self, tmp_path: Path) -> None:
        self._release_setup(tmp_path)
        versioning = StubVersioning(release_tag_returns=r[str].ok("v1.0.0"))
        runner = StubRunner(
            run_returns=[
                r[m.Infra.Core.CommandOutput].fail("not found"),
                r[m.Infra.Core.CommandOutput].fail("dispatch failed"),
            ],
        )
        result = _mgr(runner=runner, versioning=versioning)._trigger_release_if_needed(
            tmp_path,
            "release/1.0",
        )
        tm.ok(result)
        tm.that(result.value["status"], eq="release-dispatch-failed")
