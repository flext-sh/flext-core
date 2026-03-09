"""Tests for FlextInfraPrManager operations (view, checks, merge, close, release)."""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra.github.pr import FlextInfraPrManager

from tests.infra.unit.github._stubs import StubRunner, StubVersioning


def _mgr(
    runner: StubRunner | None = None,
    versioning: StubVersioning | None = None,
) -> FlextInfraPrManager:
    return FlextInfraPrManager(
        runner=runner or StubRunner(),
        versioning=versioning or StubVersioning(),
    )


class TestView:
    def test_view_success(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].ok("PR details")])
        result = _mgr(runner=runner).view(tmp_path, "42")
        assert result.is_success

    def test_view_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].fail("not found")])
        result = _mgr(runner=runner).view(tmp_path, "999")
        assert result.is_failure


class TestChecks:
    def test_checks_pass(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[r[bool].ok(True)])
        result = _mgr(runner=runner).checks(tmp_path, "42")
        assert result.is_success
        assert result.value["status"] == "checks-passed"

    def test_checks_fail_non_strict(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[r[bool].fail("checks failed")])
        result = _mgr(runner=runner).checks(tmp_path, "42")
        assert result.is_success
        assert result.value["status"] == "checks-nonblocking"

    def test_checks_fail_strict(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[r[bool].fail("checks failed")])
        result = _mgr(runner=runner).checks(tmp_path, "42", strict=True)
        assert result.is_failure


class TestMerge:
    def test_merge_success(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[r[bool].ok(True)])
        result = _mgr(runner=runner).merge(
            tmp_path,
            "42",
            "feature",
            release_on_merge=False,
        )
        assert result.is_success
        assert result.value["status"] == "merged"

    def test_merge_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[r[bool].fail("merge conflict")])
        result = _mgr(runner=runner).merge(tmp_path, "42", "feature")
        assert result.is_failure

    def test_merge_not_mergeable_retry(self, tmp_path: Path) -> None:
        runner = StubRunner(
            run_returns=[
                r[bool].fail("not mergeable"),
                r[bool].ok(True),
                r[bool].ok(True),
            ]
        )
        result = _mgr(runner=runner).merge(
            tmp_path,
            "42",
            "feature",
            release_on_merge=False,
        )
        assert result.is_success

    def test_merge_selector_same_as_head_no_pr(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].ok("[]")])
        result = _mgr(runner=runner).merge(tmp_path, "feature", "feature")
        assert result.is_success
        assert result.value["status"] == "no-open-pr"

    def test_merge_with_release(self, tmp_path: Path) -> None:
        versioning = StubVersioning(release_tag_returns=r[str].ok("v1.0.0"))
        runner = StubRunner(run_returns=[r[bool].ok(True), r[bool].ok(True)])
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: Release")
        result = _mgr(runner=runner, versioning=versioning).merge(
            tmp_path,
            "42",
            "release/1.0",
            release_on_merge=True,
        )
        assert result.is_success

    def test_merge_auto_and_delete_branch(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[r[bool].ok(True)])
        result = _mgr(runner=runner).merge(
            tmp_path,
            "42",
            "feature",
            auto=True,
            delete_branch=True,
            release_on_merge=False,
        )
        assert result.is_success
        assert "--auto" in runner.run_calls[0]
        assert "--delete-branch" in runner.run_calls[0]

    def test_merge_rebase_method(self, tmp_path: Path) -> None:
        runner = StubRunner(run_returns=[r[bool].ok(True)])
        result = _mgr(runner=runner).merge(
            tmp_path,
            "42",
            "feature",
            method="rebase",
            release_on_merge=False,
        )
        assert result.is_success
        assert "--rebase" in runner.run_calls[0]


class TestClose:
    """Test close method."""

    def test_close_success(self, tmp_path: Path) -> None:
        runner = StubRunner(run_checked_returns=[r[bool].ok(True)])
        result = _mgr(runner=runner).close(tmp_path, "42")
        assert result.is_success

    def test_close_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(run_checked_returns=[r[bool].fail("close failed")])
        result = _mgr(runner=runner).close(tmp_path, "42")
        assert result.is_failure


class TestTriggerRelease:
    """Test _trigger_release_if_needed method."""

    def _release_setup(self, tmp_path: Path) -> None:
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text("name: R")

    def test_no_release_workflow(self, tmp_path: Path) -> None:
        result = _mgr()._trigger_release_if_needed(tmp_path, "feature")
        assert result.is_success
        assert result.value["status"] == "no-release-workflow"

    def test_no_release_tag(self, tmp_path: Path) -> None:
        self._release_setup(tmp_path)
        versioning = StubVersioning(release_tag_returns=r[str].fail("no tag"))
        result = _mgr(versioning=versioning)._trigger_release_if_needed(
            tmp_path,
            "feature",
        )
        assert result.is_success
        assert result.value["status"] == "no-release-tag"

    def test_release_exists(self, tmp_path: Path) -> None:
        self._release_setup(tmp_path)
        versioning = StubVersioning(release_tag_returns=r[str].ok("v1.0.0"))
        runner = StubRunner(run_returns=[r[bool].ok(True)])
        result = _mgr(runner=runner, versioning=versioning)._trigger_release_if_needed(
            tmp_path,
            "release/1.0",
        )
        assert result.is_success
        assert result.value["status"] == "release-exists"

    def test_release_dispatched(self, tmp_path: Path) -> None:
        self._release_setup(tmp_path)
        versioning = StubVersioning(release_tag_returns=r[str].ok("v1.0.0"))
        runner = StubRunner(
            run_returns=[
                r[bool].fail("not found"),
                r[bool].ok(True),
            ]
        )
        result = _mgr(runner=runner, versioning=versioning)._trigger_release_if_needed(
            tmp_path,
            "release/1.0",
        )
        assert result.is_success
        assert result.value["status"] == "release-dispatched"

    def test_release_dispatch_failed(self, tmp_path: Path) -> None:
        self._release_setup(tmp_path)
        versioning = StubVersioning(release_tag_returns=r[str].ok("v1.0.0"))
        runner = StubRunner(
            run_returns=[
                r[bool].fail("not found"),
                r[bool].fail("dispatch failed"),
            ]
        )
        result = _mgr(runner=runner, versioning=versioning)._trigger_release_if_needed(
            tmp_path,
            "release/1.0",
        )
        assert result.is_success
        assert result.value["status"] == "release-dispatch-failed"
