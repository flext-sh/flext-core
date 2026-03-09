from __future__ import annotations

from pathlib import Path

import pytest
import tomlkit

from flext_core import r, t
from flext_infra.deps._constants import FlextInfraDepsConstants
from flext_infra.deps.extra_paths import (
    FlextInfraExtraPathsManager,
    get_dep_paths,
    sync_one,
)
from flext_tests import tm
from ...helpers import h


class TestFlextInfraExtraPathsManager:
    def test_manager_initialization(self) -> None:
        manager = FlextInfraExtraPathsManager()
        tm.that(isinstance(manager, FlextInfraExtraPathsManager), eq=True)

    def test_manager_has_required_services(self) -> None:
        manager = FlextInfraExtraPathsManager()
        tm.that(hasattr(manager, "resolver"), eq=True)
        tm.that(hasattr(manager, "toml"), eq=True)


class TestGetDepPaths:
    def test_get_dep_paths_empty_doc(self) -> None:
        tm.that(get_dep_paths(tomlkit.document(), is_root=False), eq=[])

    def test_get_dep_paths_with_pep621_deps(self) -> None:
        doc = tomlkit.document()
        doc["project"] = {"dependencies": ["flext-core @ file:../flext-core"]}
        paths = get_dep_paths(doc, is_root=False)
        tm.that(any("flext-core" in item for item in paths), eq=True)

    def test_get_dep_paths_with_poetry_deps(self) -> None:
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {"dependencies": {"flext-core": {"path": "../flext-core"}}}
        }
        paths = get_dep_paths(doc, is_root=False)
        tm.that(any("flext-core" in item for item in paths), eq=True)

    def test_get_dep_paths_is_root_true(self) -> None:
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {"dependencies": {"flext-core": {"path": "../flext-core"}}}
        }
        tm.that(
            all(
                not item.startswith("../") for item in get_dep_paths(doc, is_root=True)
            ),
            eq=True,
        )

    def test_get_dep_paths_is_root_false(self) -> None:
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {"dependencies": {"flext-core": {"path": "../flext-core"}}}
        }
        tm.that(
            all(item.startswith("../") for item in get_dep_paths(doc, is_root=False)),
            eq=True,
        )

    def test_get_dep_paths_combined_sources(self) -> None:
        doc = tomlkit.document()
        doc["project"] = {"dependencies": ["flext-api @ file:../flext-api"]}
        doc["tool"] = {
            "poetry": {"dependencies": {"flext-core": {"path": "../flext-core"}}}
        }
        tm.that(len(get_dep_paths(doc, is_root=False)) >= 2, eq=True)

    def test_get_dep_paths_with_is_root_true(self) -> None:
        doc = tomlkit.document()
        project = tomlkit.table()
        project["dependencies"] = ["flext-core @ file:flext-core"]
        doc["project"] = project
        result = get_dep_paths(doc, is_root=True)
        tm.that(any("flext-core/src" in item for item in result), eq=True)

    def test_get_dep_paths_with_is_root_false(self) -> None:
        doc = tomlkit.document()
        project = tomlkit.table()
        project["dependencies"] = ["flext-core @ file:../flext-core"]
        doc["project"] = project
        result = get_dep_paths(doc, is_root=False)
        tm.that(any("../flext-core/src" in item for item in result), eq=True)


class TestSyncOne:
    def test_sync_one_missing_file(self, tmp_path: Path) -> None:
        tm.that(h.assert_ok(sync_one(tmp_path / "nonexistent.toml")), eq=False)

    def test_sync_one_no_tool_section(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        pyproject.write_text(doc.as_string(), encoding="utf-8")
        tm.that(h.assert_ok(sync_one(pyproject)), eq=False)

    def test_sync_one_no_pyright_section(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {"other": {}}
        pyproject.write_text(doc.as_string(), encoding="utf-8")
        tm.that(h.assert_ok(sync_one(pyproject)), eq=False)

    @pytest.mark.parametrize(
        "tool_doc",
        [
            {"pyright": {"extraPaths": ["src"]}},
            {"pyright": {"extraPaths": []}, "mypy": {"mypy_path": ["src"]}},
            {"pyright": {"extraPaths": []}, "pyrefly": {"search-path": ["."]}},
        ],
    )
    def test_sync_one_success_cases(
        self,
        tmp_path: Path,
        tool_doc: dict[str, t.ContainerValue],
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = tool_doc
        pyproject.write_text(doc.as_string(), encoding="utf-8")
        result = sync_one(pyproject, is_root="pyrefly" not in tool_doc)
        tm.that(result.is_success, eq=True)

    def test_sync_one_dry_run(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {"pyright": {"extraPaths": ["old"]}}
        pyproject.write_text(doc.as_string(), encoding="utf-8")
        tm.ok(sync_one(pyproject, dry_run=True, is_root=True))
        tm.that(pyproject.read_text(encoding="utf-8"), contains="old")

    def test_sync_one_write_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")

        def _broken_write(
            *_args: t.ContainerValue,
            **_kwargs: t.ContainerValue,
        ) -> r[bool]:
            return r[bool].fail("write error")

        monkeypatch.setattr(
            "flext_infra.deps.extra_paths.FlextInfraUtilitiesToml.write_document",
            _broken_write,
        )
        tm.fail(sync_one(pyproject, is_root=True), has="write error")


class TestConstants:
    def test_base_constants(self) -> None:
        tm.that(len(FlextInfraDepsConstants.PYRIGHT_BASE_ROOT) > 0, eq=True)
        tm.that("scripts" in FlextInfraDepsConstants.PYRIGHT_BASE_ROOT, eq=True)
        tm.that("src" in FlextInfraDepsConstants.PYRIGHT_BASE_ROOT, eq=True)
        tm.that(len(FlextInfraDepsConstants.MYPY_BASE_ROOT) > 0, eq=True)
        tm.that("src" in FlextInfraDepsConstants.MYPY_BASE_ROOT, eq=True)
        tm.that(len(FlextInfraDepsConstants.PYRIGHT_BASE_PROJECT) > 0, eq=True)
        tm.that("." in FlextInfraDepsConstants.PYRIGHT_BASE_PROJECT, eq=True)
        tm.that("src" in FlextInfraDepsConstants.PYRIGHT_BASE_PROJECT, eq=True)
        tm.that(len(FlextInfraDepsConstants.MYPY_BASE_PROJECT) > 0, eq=True)
        tm.that("." in FlextInfraDepsConstants.MYPY_BASE_PROJECT, eq=True)
