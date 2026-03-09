from __future__ import annotations

import types
from pathlib import Path

from flext_core import r
from flext_infra.deps import internal_sync
from flext_infra.deps.internal_sync import main
from flext_tests import tm
from tests.infra import h


class TestMain:
    def test_main_success(self, monkeypatch) -> None:
        monkeypatch.setattr(
            internal_sync.argparse.ArgumentParser,
            "parse_args",
            lambda self: types.SimpleNamespace(project_root=Path("/tmp/test")),
        )
        monkeypatch.setattr(
            internal_sync.FlextInfraInternalDependencySyncService,
            "sync",
            lambda self, _project_root: r[int].ok(0),
        )
        tm.that(main(), eq=0)

    def test_main_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(
            internal_sync.argparse.ArgumentParser,
            "parse_args",
            lambda self: types.SimpleNamespace(project_root=Path("/tmp/test")),
        )
        monkeypatch.setattr(
            internal_sync.FlextInfraInternalDependencySyncService,
            "sync",
            lambda self, _project_root: r[int].fail("sync failed"),
        )
        tm.that(main(), eq=1)
        tm.that(h is not None, eq=True)
