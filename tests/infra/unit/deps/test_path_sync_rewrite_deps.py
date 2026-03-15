from __future__ import annotations

from pathlib import Path

import pytest
from tomlkit.toml_document import TOMLDocument

from flext_core import r
from flext_infra.deps.path_sync import FlextInfraDependencyPathSync
from flext_tests import tm


def rewrite_dep_paths(
    pyproject_path: Path,
    *,
    mode: str,
    internal_names: set[str],
    is_root: bool = False,
    dry_run: bool = False,
) -> r[list[str]]:
    return FlextInfraDependencyPathSync().rewrite_dep_paths(
        pyproject_path,
        mode=mode,
        internal_names=internal_names,
        is_root=is_root,
        dry_run=dry_run,
    )


class TestRewriteDepPaths:
    def test_rewrite_dep_paths_success(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\ndependencies = ["flext-core @ file://.flext-deps/flext-core"]\n',
        )
        result = rewrite_dep_paths(
            pyproject,
            mode="workspace",
            internal_names={"flext-core"},
            is_root=True,
        )
        tm.ok(result)
        tm.that(len(result.value) > 0, eq=True)

    def test_rewrite_dep_paths_dry_run(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        original = (
            '[project]\ndependencies = ["flext-core @ file://.flext-deps/flext-core"]\n'
        )
        pyproject.write_text(original)
        result = rewrite_dep_paths(
            pyproject,
            mode="workspace",
            internal_names={"flext-core"},
            is_root=True,
            dry_run=True,
        )
        tm.ok(result)
        tm.that(pyproject.read_text(), eq=original)

    def test_rewrite_dep_paths_no_changes(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\ndependencies = ["requests>=2.0.0"]\n')
        result = rewrite_dep_paths(
            pyproject,
            mode="workspace",
            internal_names={"flext-core"},
            is_root=True,
        )
        tm.that(tm.ok(result), eq=[])

    def test_rewrite_dep_paths_read_failure(self, tmp_path: Path) -> None:
        result = rewrite_dep_paths(
            tmp_path / "pyproject.toml",
            mode="workspace",
            internal_names={"flext-core"},
            is_root=True,
        )
        tm.fail(result)

    def test_rewrite_dep_paths_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\ndependencies = ["flext-core @ file://.flext-deps/flext-core"]\n',
        )

        def fail_write(
            _path: Path,
            _doc: TOMLDocument,
        ) -> r[bool]:
            _ = _path, _doc
            return r[bool].fail("write failed")

        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesToml.write_document", fail_write
        )
        tm.fail(
            rewrite_dep_paths(
                pyproject,
                mode="workspace",
                internal_names={"flext-core"},
                is_root=True,
            )
        )


def test_rewrite_dep_paths_with_internal_names(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\ndependencies = ["flext-core @ file:.flext-deps/flext-core"]\n'
    )
    result = rewrite_dep_paths(
        pyproject,
        mode="workspace",
        internal_names={"flext-core"},
        is_root=False,
        dry_run=False,
    )
    tm.ok(result)
    tm.that(len(result.value) > 0, eq=True)


def test_rewrite_dep_paths_dry_run(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    original = '[project]\ndependencies = ["flext-core @ file:../flext-core"]\n'
    pyproject.write_text(original)
    tm.ok(
        rewrite_dep_paths(
            pyproject,
            mode="workspace",
            internal_names={"flext-core"},
            is_root=False,
            dry_run=True,
        )
    )
    tm.that(pyproject.read_text(), eq=original)


def test_rewrite_dep_paths_read_failure(tmp_path: Path) -> None:
    tm.fail(
        rewrite_dep_paths(
            tmp_path / "pyproject.toml",
            mode="workspace",
            internal_names={"flext-core"},
            is_root=False,
            dry_run=False,
        )
    )


def test_rewrite_dep_paths_with_no_deps(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.poetry.dependencies]\npython = "^3.13"')
    tm.ok(
        rewrite_dep_paths(
            pyproject,
            mode="poetry",
            internal_names=set(),
            dry_run=True,
        )
    )
