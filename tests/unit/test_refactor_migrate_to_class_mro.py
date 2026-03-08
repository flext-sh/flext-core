"""Tests for migrate-to-mro automation."""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.refactor._utilities import FlextInfraUtilitiesRefactor
from flext_infra.refactor.migrate_to_class_mro import (
    FlextInfraRefactorMigrateToClassMRO,
)
from flext_infra.refactor.mro_migrator import FlextInfraRefactorMROMigrationScanner


def test_migrate_to_mro_moves_constant_and_rewrites_reference(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\nfrom typing import Final\n\nVALUE: Final[int] = 42\n\nclass SampleConstants:\n    pass\n\nc = SampleConstants\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import VALUE\n\nresult = VALUE\n", encoding="utf-8"
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants", apply_changes=True
    )
    constants_source = (src_pkg / "constants.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    assert report.errors == ()
    assert (
        "VALUE: Final[int] = 42"
        not in constants_source.split("class SampleConstants:")[0]
    )
    assert (
        "VALUE: Final[int] = 42"
        in constants_source.split("class SampleConstants:", maxsplit=1)[1]
    )
    assert "from sample_pkg.constants import c" in consumer_source
    assert "result = c.VALUE" in consumer_source


def test_migrate_to_mro_inlines_alias_constant_into_constants_class(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\nfrom typing import Final\n\n_TIMEOUT: Final[int] = 30\n\nclass SampleConstants:\n    TIMEOUT = _TIMEOUT\n\nc = SampleConstants\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import _TIMEOUT\n\nvalue = _TIMEOUT\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants", apply_changes=True
    )
    constants_source = (src_pkg / "constants.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    assert report.errors == ()
    assert (
        "_TIMEOUT: Final[int] = 30"
        not in constants_source.split("class SampleConstants:", maxsplit=1)[0]
    )
    assert "TIMEOUT: Final[int] = 30" in constants_source
    assert "TIMEOUT = _TIMEOUT" not in constants_source
    assert "from sample_pkg.constants import c" in consumer_source
    assert "value = c.TIMEOUT" in consumer_source


def test_migrate_to_mro_normalizes_facade_alias_to_c(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\nfrom typing import Final\n\nVALUE: Final[int] = 42\n\nclass SampleConstants:\n    pass\n\nc = SampleConstants\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import VALUE\nfrom sample_pkg.constants import c as constants\n\nresult = VALUE\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants", apply_changes=True
    )
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    assert report.errors == ()
    assert "from sample_pkg.constants import VALUE" not in consumer_source
    assert "from sample_pkg.constants import c as constants" not in consumer_source
    assert "from sample_pkg.constants import c" in consumer_source
    assert "result = c.VALUE" in consumer_source


def test_migrate_to_mro_rejects_unknown_target(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    project_root.mkdir(parents=True)
    migrator = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root)
    with pytest.raises(ValueError, match="unsupported target"):
        _ = migrator.run(target="unknown", apply_changes=False)


def test_migrate_typings_rewrites_references_with_t_alias(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "typings.py").write_text(
        "from __future__ import annotations\nfrom typing import TypeAlias\n\nValueType: TypeAlias = str | int\n\nclass SampleTypes:\n    pass\n\nt = SampleTypes\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.typings import ValueType\n\nvalue: ValueType = 1\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="typings", apply_changes=True
    )
    typings_source = (src_pkg / "typings.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    assert report.errors == ()
    assert (
        "ValueType: TypeAlias = str | int"
        not in typings_source.split("class SampleTypes:", maxsplit=1)[0]
    )
    assert (
        "ValueType: TypeAlias = str | int"
        in typings_source.split("class SampleTypes:", maxsplit=1)[1]
    )
    assert "from sample_pkg.typings import t" in consumer_source
    assert "value: t.ValueType = 1" in consumer_source


def test_migrate_protocols_rewrites_references_with_p_alias(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "protocols.py").write_text(
        "from __future__ import annotations\n"
        "from typing import Protocol\n\n"
        "class SampleProtocols:\n"
        "    pass\n\n"
        "class GreeterProtocol(Protocol):\n"
        "    def greet(self) -> str:\n"
        "        ...\n\n"
        "p = SampleProtocols\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.protocols import GreeterProtocol\n\n"
        "def call_greet(protocol: GreeterProtocol) -> str:\n"
        "    return protocol.greet()\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="protocols", apply_changes=True
    )
    protocols_source = (src_pkg / "protocols.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    assert report.errors == ()
    assert (
        "class GreeterProtocol(Protocol):"
        not in protocols_source.split("class SampleProtocols:", maxsplit=1)[0]
    )
    assert (
        "class GreeterProtocol(Protocol):"
        in protocols_source.split("class SampleProtocols:", maxsplit=1)[1]
    )
    assert "from sample_pkg.protocols import p" in consumer_source
    assert "def call_greet(protocol: p.GreeterProtocol) -> str:" in consumer_source


def test_mro_scanner_includes_constants_variants_in_all_scopes(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    project_root.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    file_paths = [
        project_root / "src" / "sample_pkg" / "constants.py",
        project_root / "src" / "sample_pkg" / "_constants.py",
        project_root / "src" / "sample_pkg" / "constants" / "domain.py",
        project_root / "examples" / "constants.py",
        project_root / "scripts" / "constants.py",
        project_root / "tests" / "constants.py",
    ]
    for file_path in file_paths:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "from __future__ import annotations\nfrom typing import Final\nVALUE: Final[str] = 'x'\n",
            encoding="utf-8",
        )
    scanned = FlextInfraRefactorMROMigrationScanner._iter_constants_files(
        project_root=project_root
    )
    assert set(scanned) == set(file_paths)


def test_refactor_utilities_iter_python_files_includes_examples_and_scripts(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "sample"
    project_root.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (project_root / ".git").mkdir(parents=True)
    expected_paths = [
        project_root / "src" / "sample_pkg" / "module.py",
        project_root / "tests" / "test_module.py",
        project_root / "examples" / "demo.py",
        project_root / "scripts" / "sync.py",
    ]
    for file_path in expected_paths:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("from __future__ import annotations\n", encoding="utf-8")
    discovered = FlextInfraUtilitiesRefactor.iter_python_files(workspace_root=tmp_path)
    assert set(discovered) == set(expected_paths)


def test_discover_project_roots_without_nested_git_dirs(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)

    project_root = workspace_root / "proj-a"
    project_root.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='proj-a'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (project_root / "src").mkdir(parents=True)

    discovered = FlextInfraUtilitiesRefactor.discover_project_roots(
        workspace_root=workspace_root
    )
    assert discovered == [project_root]


def test_migrate_to_mro_moves_manual_uppercase_assignment(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\n\nVALUE = 42\n\nclass SampleConstants:\n    pass\n\nc = SampleConstants\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import VALUE\n\nresult = VALUE\n", encoding="utf-8"
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants", apply_changes=True
    )
    constants_source = (src_pkg / "constants.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    assert report.errors == ()
    assert "VALUE = 42" not in constants_source.split("class SampleConstants:")[0]
    assert (
        "VALUE = 42" in constants_source.split("class SampleConstants:", maxsplit=1)[1]
    )
    assert "from sample_pkg.constants import c" in consumer_source
    assert "result = c.VALUE" in consumer_source
