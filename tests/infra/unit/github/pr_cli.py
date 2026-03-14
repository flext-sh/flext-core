"""Tests for pr.py CLI entry point (main, _parse_args, _selector)."""

from __future__ import annotations

import argparse

import pytest

from flext_core import r
from flext_infra.github import pr as pr_module
from flext_infra.github.pr import FlextInfraPrManager, main
from flext_tests import tm
from tests.infra.helpers import h
from tests.infra.typings import t
from tests.infra.unit.github._stubs import StubPrManager, StubUtilities

_parse_args = FlextInfraPrManager.parse_args
_selector = FlextInfraPrManager.selector

_DEFAULTS: dict[str, t.Scalar | None] = {
    "action": "status",
    "repo_root": "/tmp/test",
    "base": "main",
    "head": "feature",
    "number": "",
    "title": "",
    "body": "",
    "draft": 0,
    "merge_method": "squash",
    "auto": 0,
    "delete_branch": 0,
    "checks_strict": 0,
    "release_on_merge": 1,
}


def _args(**overrides: t.Scalar | None) -> argparse.Namespace:
    return h.ns(**{**_DEFAULTS, **overrides})


class TestMainFunction:
    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        args: argparse.Namespace,
        manager: StubPrManager,
    ) -> None:
        def _parse() -> argparse.Namespace:
            return args

        def _manager_factory() -> StubPrManager:
            return manager

        monkeypatch.setattr(pr_module, "_parse_args", _parse)
        monkeypatch.setattr(pr_module, "FlextInfraPrManager", _manager_factory)
        monkeypatch.setattr(
            StubUtilities.Infra, "_git_branch_returns", r[str].ok("feature")
        )
        monkeypatch.setattr(pr_module, "u", StubUtilities)

    def test_status_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            status_returns=[r[dict[str, t.Scalar]].ok({"status": "open"})]
        )
        self._setup(monkeypatch, _args(action="status"), mgr)
        tm.that(main(), eq=0)

    def test_status_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(status_returns=[r[dict[str, t.Scalar]].fail("error")])
        self._setup(monkeypatch, _args(action="status"), mgr)
        tm.that(main(), eq=1)

    def test_create_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            create_returns=[r[dict[str, t.Scalar]].ok({"status": "created"})]
        )
        self._setup(monkeypatch, _args(action="create"), mgr)
        tm.that(main(), eq=0)

    def test_create_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            create_returns=[r[dict[str, t.Scalar]].fail("create failed")]
        )
        self._setup(monkeypatch, _args(action="create"), mgr)
        tm.that(main(), eq=1)

    def test_view_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(view_returns=[r[str].ok("PR view output")])
        self._setup(monkeypatch, _args(action="view", number="42"), mgr)
        tm.that(main(), eq=0)

    def test_view_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(view_returns=[r[str].fail("not found")])
        self._setup(monkeypatch, _args(action="view", number="42"), mgr)
        tm.that(main(), eq=1)

    def test_checks_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            checks_returns=[r[dict[str, t.Scalar]].ok({"status": "checks-passed"})]
        )
        self._setup(monkeypatch, _args(action="checks", number="42"), mgr)
        tm.that(main(), eq=0)

    def test_checks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            checks_returns=[r[dict[str, t.Scalar]].fail("checks failed")]
        )
        self._setup(
            monkeypatch, _args(action="checks", number="42", checks_strict=1), mgr
        )
        tm.that(main(), eq=1)

    def test_merge_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(merge_returns=[r[t.Container].ok("merged")])
        self._setup(monkeypatch, _args(action="merge", number="42"), mgr)
        tm.that(main(), eq=0)

    def test_merge_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(merge_returns=[r[t.Container].fail("merge failed")])
        self._setup(monkeypatch, _args(action="merge", number="42"), mgr)
        tm.that(main(), eq=1)

    def test_close_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(close_returns=[r[bool].ok(True)])
        self._setup(monkeypatch, _args(action="close", number="42"), mgr)
        tm.that(main(), eq=0)

    def test_close_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(close_returns=[r[bool].fail("close failed")])
        self._setup(monkeypatch, _args(action="close", number="42"), mgr)
        tm.that(main(), eq=1)

    def test_unknown_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._setup(monkeypatch, _args(action="invalid_action"), StubPrManager())
        with pytest.raises(RuntimeError, match="unknown action"):
            main()


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["prog"])
        args = _parse_args()
        tm.that(args.action, eq="status")
        tm.that(args.base, eq="main")
        tm.that(args.head, eq="")
        tm.that(args.number, eq="")
        tm.that(args.title, eq="")
        tm.that(args.body, eq="")
        tm.that(args.draft, eq=0)
        tm.that(args.merge_method, eq="squash")
        tm.that(args.auto, eq=0)
        tm.that(args.delete_branch, eq=0)
        tm.that(args.checks_strict, eq=0)
        tm.that(args.release_on_merge, eq=1)

    def test_custom_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                *["--action", "create", "--base", "develop", "--head", "feature/test"],
                *["--number", "42", "--title", "Test PR", "--body", "Test body"],
                *["--draft", "1", "--merge-method", "rebase", "--auto", "1"],
                *[
                    "--delete-branch",
                    "1",
                    "--checks-strict",
                    "1",
                    "--release-on-merge",
                    "0",
                ],
            ],
        )
        args = _parse_args()
        tm.that(args.action, eq="create")
        tm.that(args.base, eq="develop")
        tm.that(args.head, eq="feature/test")
        tm.that(args.number, eq="42")
        tm.that(args.title, eq="Test PR")
        tm.that(args.body, eq="Test body")
        tm.that(args.draft, eq=1)
        tm.that(args.merge_method, eq="rebase")
        tm.that(args.auto, eq=1)
        tm.that(args.delete_branch, eq=1)
        tm.that(args.checks_strict, eq=1)
        tm.that(args.release_on_merge, eq=0)


class TestSelectorFunction:
    def test_with_pr_number(self) -> None:
        tm.that(_selector("42", "feature"), eq="42")

    def test_with_head_only(self) -> None:
        tm.that(_selector("", "feature"), eq="feature")
