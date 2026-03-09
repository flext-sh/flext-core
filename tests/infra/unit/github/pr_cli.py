"""Tests for pr.py CLI entry point (main, _parse_args, _selector).

Uses monkeypatch instead of unittest.mock.patch for module-level substitution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from flext_core import r
from flext_infra.github import pr as pr_module
from flext_infra.github.pr import _parse_args, _selector, main

from tests.infra.unit.github._stubs import StubPrManager, StubUtilities


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
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
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestMainFunction:
    """Test main() CLI entry point."""

    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        args: argparse.Namespace,
        manager: StubPrManager,
    ) -> None:
        monkeypatch.setattr(pr_module, "_parse_args", lambda: args)
        monkeypatch.setattr(pr_module, "FlextInfraPrManager", lambda **kw: manager)
        monkeypatch.setattr(
            StubUtilities.Infra,
            "_git_branch_returns",
            r[str].ok("feature"),
        )
        monkeypatch.setattr(pr_module, "u", StubUtilities)

    def test_status_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            status_returns=[r[dict[str, object]].ok({"status": "open"})]
        )
        self._setup(monkeypatch, _args(action="status"), mgr)
        assert main() == 0

    def test_status_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(status_returns=[r[dict[str, object]].fail("error")])
        self._setup(monkeypatch, _args(action="status"), mgr)
        assert main() == 1

    def test_create_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            create_returns=[r[dict[str, object]].ok({"status": "created"})],
        )
        self._setup(monkeypatch, _args(action="create"), mgr)
        assert main() == 0

    def test_create_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            create_returns=[r[dict[str, object]].fail("create failed")],
        )
        self._setup(monkeypatch, _args(action="create"), mgr)
        assert main() == 1

    def test_view_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(view_returns=[r[str].ok("PR view output")])
        self._setup(monkeypatch, _args(action="view", number="42"), mgr)
        assert main() == 0

    def test_view_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(view_returns=[r[str].fail("not found")])
        self._setup(monkeypatch, _args(action="view", number="42"), mgr)
        assert main() == 1

    def test_checks_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            checks_returns=[r[dict[str, object]].ok({"status": "checks-passed"})],
        )
        self._setup(monkeypatch, _args(action="checks", number="42"), mgr)
        assert main() == 0

    def test_checks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            checks_returns=[r[dict[str, object]].fail("checks failed")],
        )
        self._setup(
            monkeypatch,
            _args(action="checks", number="42", checks_strict=1),
            mgr,
        )
        assert main() == 1

    def test_merge_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            merge_returns=[r[dict[str, object]].ok({"status": "merged"})],
        )
        self._setup(monkeypatch, _args(action="merge", number="42"), mgr)
        assert main() == 0

    def test_merge_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(
            merge_returns=[r[dict[str, object]].fail("merge failed")],
        )
        self._setup(monkeypatch, _args(action="merge", number="42"), mgr)
        assert main() == 1

    def test_close_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(close_returns=[r[bool].ok(True)])
        self._setup(monkeypatch, _args(action="close", number="42"), mgr)
        assert main() == 0

    def test_close_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager(close_returns=[r[bool].fail("close failed")])
        self._setup(monkeypatch, _args(action="close", number="42"), mgr)
        assert main() == 1

    def test_unknown_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = StubPrManager()
        self._setup(monkeypatch, _args(action="invalid_action"), mgr)
        with pytest.raises(RuntimeError, match="unknown action"):
            main()


class TestParseArgs:
    """Test _parse_args function."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["prog"])
        args = _parse_args()
        assert args.action == "status"
        assert args.base == "main"
        assert args.head == ""
        assert args.number == ""
        assert args.title == ""
        assert args.body == ""
        assert args.draft == 0
        assert args.merge_method == "squash"
        assert args.auto == 0
        assert args.delete_branch == 0
        assert args.checks_strict == 0
        assert args.release_on_merge == 1

    def test_custom_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--action",
                "create",
                "--base",
                "develop",
                "--head",
                "feature/test",
                "--number",
                "42",
                "--title",
                "Test PR",
                "--body",
                "Test body",
                "--draft",
                "1",
                "--merge-method",
                "rebase",
                "--auto",
                "1",
                "--delete-branch",
                "1",
                "--checks-strict",
                "1",
                "--release-on-merge",
                "0",
            ],
        )
        args = _parse_args()
        assert args.action == "create"
        assert args.base == "develop"
        assert args.head == "feature/test"
        assert args.number == "42"
        assert args.title == "Test PR"
        assert args.body == "Test body"
        assert args.draft == 1
        assert args.merge_method == "rebase"
        assert args.auto == 1
        assert args.delete_branch == 1
        assert args.checks_strict == 1
        assert args.release_on_merge == 0


class TestSelectorFunction:
    """Test _selector module function."""

    def test_with_pr_number(self) -> None:
        assert _selector("42", "feature") == "42"

    def test_with_head_only(self) -> None:
        assert _selector("", "feature") == "feature"
