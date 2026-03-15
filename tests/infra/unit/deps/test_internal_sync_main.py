from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from flext_core import r
from flext_infra.deps import internal_sync
from flext_infra.deps.internal_sync import main
from flext_tests import tm
from tests.infra import h, u


class TestMain:
    def test_main_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cli_args = u.Infra.CliArgs(workspace=tmp_path)

        monkeypatch.setattr(
            argparse.ArgumentParser,
            "parse_args",
            lambda _self, _args=None, _ns=None: argparse.Namespace(workspace=tmp_path),
        )
        monkeypatch.setattr(u.Infra, "resolve", staticmethod(lambda _a: cli_args))
        monkeypatch.setattr(
            internal_sync.FlextInfraInternalDependencySyncService,
            "sync",
            lambda _self, _root: r[int].ok(0),
        )
        tm.that(main(), eq=0)

    def test_main_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cli_args = u.Infra.CliArgs(workspace=tmp_path)

        monkeypatch.setattr(
            argparse.ArgumentParser,
            "parse_args",
            lambda _self, _args=None, _ns=None: argparse.Namespace(workspace=tmp_path),
        )
        monkeypatch.setattr(u.Infra, "resolve", staticmethod(lambda _a: cli_args))
        monkeypatch.setattr(
            internal_sync.FlextInfraInternalDependencySyncService,
            "sync",
            lambda _self, _root: r[int].fail("sync failed"),
        )
        tm.that(main(), eq=1)
        tm.that(hasattr(h, "assert_ok"), eq=True)
