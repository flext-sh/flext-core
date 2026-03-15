from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraInternalDependencySyncService
from flext_infra.deps import internal_sync
from flext_tests import tm
from tests.infra import h


class TestEnsureCheckoutEdgeCases:
    def test_ensure_checkout_cleanup_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_path = tmp_path / "dep"
        dep_path.mkdir()
        (dep_path / "file.txt").write_text("content")

        def _raise_rmtree(_path: Path) -> None:
            msg = "Permission denied"
            raise OSError(msg)

        monkeypatch.setattr(internal_sync.shutil, "rmtree", _raise_rmtree)
        error = tm.fail(
            FlextInfraInternalDependencySyncService().ensure_checkout(
                dep_path,
                "https://github.com/test/repo.git",
                "main",
            ),
        )
        tm.that(error, contains="cleanup failed")
        tm.that(hasattr(h, "assert_ok"), eq=True)
