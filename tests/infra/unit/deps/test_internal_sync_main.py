from __future__ import annotations

import argparse
import types
from pathlib import Path

import pytest

from flext_core import r
from flext_infra.deps import internal_sync
from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService, main
from flext_tests import tm
from tests.infra import h


class TestMain:
    def test_main_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _parse_args(self: argparse.ArgumentParser) -> types.SimpleNamespace:
            _ = self
            return types.SimpleNamespace(project_root=Path("/tmp/test"))

        def _sync(
            self: FlextInfraInternalDependencySyncService,
            _project_root: Path,
        ) -> r[int]:
            _ = self
            _ = _project_root
            return r[int].ok(0)

        monkeypatch.setattr(
            internal_sync.argparse.ArgumentParser,
            "parse_args",
            _parse_args,
        )
        monkeypatch.setattr(
            internal_sync.FlextInfraInternalDependencySyncService,
            "sync",
            _sync,
        )
        tm.that(main(), eq=0)

    def test_main_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _parse_args(self: argparse.ArgumentParser) -> types.SimpleNamespace:
            _ = self
            return types.SimpleNamespace(project_root=Path("/tmp/test"))

        def _sync(
            self: FlextInfraInternalDependencySyncService,
            _project_root: Path,
        ) -> r[int]:
            _ = self
            _ = _project_root
            return r[int].fail("sync failed")

        monkeypatch.setattr(
            internal_sync.argparse.ArgumentParser,
            "parse_args",
            _parse_args,
        )
        monkeypatch.setattr(
            internal_sync.FlextInfraInternalDependencySyncService,
            "sync",
            _sync,
        )
        tm.that(main(), eq=1)
        tm.that(hasattr(h, "assert_ok"), eq=True)
