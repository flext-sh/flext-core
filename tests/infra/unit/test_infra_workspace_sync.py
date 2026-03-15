from __future__ import annotations

import fcntl
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import override

import pytest

from flext_core import r
from flext_infra import m, t
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator
from flext_infra.workspace.sync import FlextInfraSyncService, main
from flext_tests import tf, tm

_S = FlextInfraSyncService
SetupFn = Callable[[_S, pytest.MonkeyPatch], None]


def _stub_gen(content: str, *, fail: bool = False) -> FlextInfraBaseMkGenerator:
    class _Gen(FlextInfraBaseMkGenerator):
        def __init__(self) -> None:
            super().__init__()

        @override
        def generate(
            self,
            config: m.Infra.Basemk.BaseMkConfig | Mapping[str, t.Scalar] | None = None,
        ) -> r[str]:
            _ = self
            _ = config
            return r[str].fail(content) if fail else r[str].ok(content)

    return _Gen()


def _setup_lock_fail(_svc: _S, monkeypatch: pytest.MonkeyPatch) -> None:
    def _flock(_fd: int, _op: int) -> None:
        msg = "Lock failed"
        raise OSError(msg)

    monkeypatch.setattr(fcntl, "flock", _flock)


def _setup_gen_fail(svc: _S, _monkeypatch: pytest.MonkeyPatch) -> None:
    svc._generator = _stub_gen("Generation failed", fail=True)


def _setup_gitignore_fail(_svc: _S, monkeypatch: pytest.MonkeyPatch) -> None:
    def _open(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "Write failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "open", _open)


@pytest.fixture
def svc(tmp_path: Path) -> _S:
    return _S(canonical_root=tmp_path)


@pytest.mark.parametrize(
    ("base_mk", "gitignore"),
    [("", ""), ("# Old content\n", ""), ("", "*.pyc\n")],
    ids=["empty", "existing-base-mk", "existing-gitignore"],
)
def test_sync_success_scenarios(
    svc: _S, tmp_path: Path, base_mk: str, gitignore: str
) -> None:
    if base_mk:
        tf.create_in(base_mk, "base.mk", tmp_path)
    if gitignore:
        tf.create_in(gitignore, ".gitignore", tmp_path)
    tm.ok(svc.sync(workspace_root=tmp_path))
    tm.that((tmp_path / "base.mk").exists(), eq=True)


@pytest.mark.parametrize(
    ("project_root", "expected_error"),
    [
        (None, "workspace_root is required"),
        (Path("/nonexistent/path"), "does not exist"),
    ],
    ids=["missing-project-root", "project-root-not-found"],
)
def test_sync_root_validation(project_root: Path | None, expected_error: str) -> None:
    tm.fail(_S().sync(workspace_root=project_root), has=expected_error)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["sync", "--workspace", "{tmp}"], 0),
        (["sync", "--workspace", "/nonexistent/path"], 1),
    ],
    ids=["cli-success", "cli-failure"],
)
def test_cli_result_by_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: int
) -> None:
    monkeypatch.setattr(
        sys, "argv", [str(tmp_path) if part == "{tmp}" else part for part in argv]
    )
    tm.that(main(), eq=expected)


@pytest.mark.parametrize(
    ("setup_fn", "expected_error"),
    [
        (_setup_lock_fail, "lock acquisition failed"),
        (_setup_gen_fail, "Generation failed"),
        (_setup_gitignore_fail, "Write failed"),
    ],
    ids=["lock-fail", "gen-fail", "gitignore-fail"],
)
def test_sync_error_scenarios(
    svc: _S,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup_fn: SetupFn,
    expected_error: str,
) -> None:
    setup_fn(svc, monkeypatch)
    tm.fail(svc.sync(workspace_root=tmp_path), has=expected_error)


def test_gitignore_sync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _S()

    def _ensure(*_args: t.Scalar, **_kwargs: t.Scalar) -> r[bool]:
        return r[bool].fail(".gitignore sync failed")

    monkeypatch.setattr(
        service,
        "_ensure_gitignore_entries",
        _ensure,
    )
    tm.fail(service.sync(workspace_root=tmp_path), has=".gitignore sync failed")


def test_atomic_write_ok(tmp_path: Path) -> None:
    target = tmp_path / "test.txt"
    tm.ok(_S._atomic_write(target, "test content"), eq=True)
    tm.that(target.read_text(encoding="utf-8"), eq="test content")


def test_atomic_write_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _temp(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "Temp file failed"
        raise OSError(msg)

    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        _temp,
    )
    tm.fail(_S._atomic_write(tmp_path / "t.txt", "c"), has="atomic write failed")


@pytest.mark.parametrize(
    ("generated", "ok_result", "expected"),
    [
        ("# Same content\n", True, True),
        ("Generation failed", False, "Generation failed"),
    ],
    ids=["no-change", "generation-failure"],
)
def test_sync_basemk_scenarios(
    tmp_path: Path, generated: str, ok_result: bool, expected: bool | str
) -> None:
    service = _S()
    tf.create_in("# Same content\n", "base.mk", tmp_path)
    service._generator = _stub_gen(generated, fail=not ok_result)
    result = service._sync_basemk(tmp_path, None)
    if ok_result:
        tm.ok(result, eq=expected)
        return
    tm.fail(result, has=str(expected))


@pytest.mark.parametrize(
    ("initial_content", "entries", "expected"),
    [
        ("*.pyc\n", [".reports/", ".venv/"], True),
        (".reports/\n.venv/\n__pycache__/\n", [".reports/", ".venv/"], False),
    ],
    ids=["missing-entries", "entries-present"],
)
def test_gitignore_entry_scenarios(
    tmp_path: Path, initial_content: str, entries: list[str], expected: bool
) -> None:
    tf.create_in(initial_content, ".gitignore", tmp_path)
    tm.ok(_S()._ensure_gitignore_entries(tmp_path, entries), eq=expected)
    content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    for entry in entries:
        tm.that(content, has=entry)


def test_gitignore_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tf.create_in("*.pyc\n", ".gitignore", tmp_path)

    def _open(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "Write failed"
        raise OSError(msg)

    monkeypatch.setattr(
        Path,
        "open",
        _open,
    )
    tm.fail(
        _S()._ensure_gitignore_entries(tmp_path, [".reports/"]),
        has=".gitignore update failed",
    )
