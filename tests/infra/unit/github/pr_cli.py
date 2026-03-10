"""Tests for pr.py CLI entry point (main, _parse_args, _selector)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from flext_core import r
from flext_infra.github import pr as pr_module
from flext_infra.github.pr import _parse_args, _selector, main
from flext_tests import tm
from tests.infra.helpers import h
from tests.infra.typings import t
from tests.infra.unit.github._stubs import StubPrManager, StubUtilities

_DEFAULTS: dict[str, t.ContainerValue] = {
    "action": "status",
    "repo_root": Path("/tmp/test"),
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


def _args(**overrides: t.ContainerValue) -> argparse.Namespace:
    return h.ns(**{**_DEFAULTS, **overrides})


class TestMainFunction:
    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        args: argparse.Namespace,
        manager: StubPrManager,
    ) -> None:
        monkeypatch.setattr(pr_module, "_parse_args", lambda: args)
        monkeypatch.setattr(pr_module, "FlextInfraPrManager", lambda **kw: manager)
        monkeypatch.setattr(
            StubUtilities.Infra, "_git_branch_returns", r[str].ok("feature")
        )
        monkeypatch.setattr(pr_module, "u", StubUtilities)

    def test_status_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            status_returns=[r[dict[str, t.ContainerValue]].ok({"status": "open"})]
        )
        self._setup(monkeypatch, _args(action="status"), mgr)
        tm.that(main(), eq=0)

    def test_status_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            status_returns=[r[dict[str, t.ContainerValue]].fail("error")]
        )
        self._setup(monkeypatch, _args(action="status"), mgr)
        tm.that(main(), eq=1)

    def test_create_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            create_returns=[r[dict[str, t.ContainerValue]].ok({"status": "created"})]
        )
        self._setup(monkeypatch, _args(action="create"), mgr)
        tm.that(main(), eq=0)

    def test_create_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            create_returns=[r[dict[str, t.ContainerValue]].fail("create failed")]
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
            checks_returns=[
                r[dict[str, t.ContainerValue]].ok({"status": "checks-passed"})
            ]
        )
        self._setup(monkeypatch, _args(action="checks", number="42"), mgr)
        tm.that(main(), eq=0)

    def test_checks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            checks_returns=[r[dict[str, t.ContainerValue]].fail("checks failed")]
        )
        self._setup(
            monkeypatch, _args(action="checks", number="42", checks_strict=1), mgr
        )
        tm.that(main(), eq=1)

    def test_merge_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            merge_returns=[r[dict[str, t.ContainerValue]].ok({"status": "merged"})]
        )
        self._setup(monkeypatch, _args(action="merge", number="42"), mgr)
        tm.that(main(), eq=0)

    def test_merge_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            merge_returns=[r[dict[str, t.ContainerValue]].fail("merge failed")]
        )
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
        expected_defaults = {
            "action": "status",
            "base": "main",
            "head": "",
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
        for attr, expected in expected_defaults.items():
            tm.that(getattr(args, attr), eq=expected)

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
        expected_values = {
            "action": "create",
            "base": "develop",
            "head": "feature/test",
            "number": "42",
            "title": "Test PR",
            "body": "Test body",
            "draft": 1,
            "merge_method": "rebase",
            "auto": 1,
            "delete_branch": 1,
            "checks_strict": 1,
            "release_on_merge": 0,
        }
        for attr, expected in expected_values.items():
            tm.that(getattr(args, attr), eq=expected)


class TestSelectorFunction:
    def test_with_pr_number(self) -> None:
        tm.that(_selector("42", "feature"), eq="42")

    def test_with_head_only(self) -> None:
        tm.that(_selector("", "feature"), eq="feature")
