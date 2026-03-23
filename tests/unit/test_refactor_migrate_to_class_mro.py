"""Tests for migrate-to-mro automation."""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra import (
    FlextInfraRefactorMigrateToClassMRO,
    FlextInfraUtilitiesIteration,
)
from flext_tests import tm


def test_migrate_to_mro_moves_constant_and_rewrites_reference(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    _ = (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\nfrom typing import Final\n\nVALUE: Final[int] = 42\n\nclass SampleConstants:\n    pass\n\nc = SampleConstants\n",
        encoding="utf-8",
    )
    _ = (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import VALUE\n\nresult = VALUE\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants",
        apply=True,
    )
    constants_source = (src_pkg / "constants.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    tm.that(report.errors, eq=())
    tm.that(
        "VALUE: Final[int] = 42" in constants_source.split("class SampleConstants:")[0],
        eq=False,
    )
    tm.that(
        "VALUE: Final[int] = 42"
        in constants_source.split("class SampleConstants:", maxsplit=1)[1],
        eq=True,
    )
    tm.that(consumer_source, has="from sample_pkg.constants import c")
    tm.that(consumer_source, has="result = c.VALUE")


def test_migrate_to_mro_inlines_alias_constant_into_constants_class(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    _ = (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\nfrom typing import Final\n\n_TIMEOUT: Final[int] = 30\n\nclass SampleConstants:\n    TIMEOUT = _TIMEOUT\n\nc = SampleConstants\n",
        encoding="utf-8",
    )
    _ = (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import _TIMEOUT\n\nvalue = _TIMEOUT\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants",
        apply=True,
    )
    constants_source = (src_pkg / "constants.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    tm.that(report.errors, eq=())
    tm.that(
        "_TIMEOUT: Final[int] = 30"
        in constants_source.split("class SampleConstants:", maxsplit=1)[0],
        eq=False,
    )
    tm.that(constants_source, has="TIMEOUT: Final[int] = 30")
    tm.that("TIMEOUT = _TIMEOUT" in constants_source, eq=False)
    tm.that(consumer_source, has="from sample_pkg.constants import c")
    tm.that(consumer_source, has="value = c.TIMEOUT")


def test_migrate_to_mro_normalizes_facade_alias_to_c(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    _ = (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\nfrom typing import Final\n\nVALUE: Final[int] = 42\n\nclass SampleConstants:\n    pass\n\nc = SampleConstants\n",
        encoding="utf-8",
    )
    _ = (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import VALUE\nfrom sample_pkg.constants import c as constants\n\nresult = VALUE\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants",
        apply=True,
    )
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    tm.that(report.errors, eq=())
    tm.that("from sample_pkg.constants import VALUE\n" in consumer_source, eq=False)
    tm.that(consumer_source, has="from sample_pkg.constants import c")
    tm.that(consumer_source, has="result = c.VALUE")


def test_migrate_to_mro_rejects_unknown_target(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    project_root.mkdir(parents=True)
    migrator = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root)
    with pytest.raises(ValueError, match="unsupported target"):
        _ = migrator.run(target="unknown", apply=False)


def test_migrate_typings_rewrites_references_with_t_alias(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    _ = (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (src_pkg / "typings.py").write_text(
        "from __future__ import annotations\nfrom typing import TypeAlias\n\nValueType: TypeAlias = str | int\n\nclass SampleTypes:\n    pass\n\nt = SampleTypes\n",
        encoding="utf-8",
    )
    _ = (src_pkg / "consumer.py").write_text(
        "from sample_pkg.typings import ValueType\n\nvalue: ValueType = 1\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="typings",
        apply=True,
    )
    typings_source = (src_pkg / "typings.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    tm.that(report.errors, eq=())
    tm.that(
        "ValueType: TypeAlias = str | int"
        in typings_source.split("class SampleTypes:", maxsplit=1)[0],
        eq=False,
    )
    tm.that(
        "ValueType: TypeAlias = str | int"
        in typings_source.split("class SampleTypes:", maxsplit=1)[1],
        eq=True,
    )
    tm.that(consumer_source, has="from sample_pkg.typings import t")
    tm.that(consumer_source, has="value: t.ValueType = 1")


def test_migrate_protocols_rewrites_references_with_p_alias(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    _ = (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (src_pkg / "protocols.py").write_text(
        "from __future__ import annotations\nfrom typing import Protocol\n\nclass SampleProtocols:\n    pass\n\nclass Greeter(Protocol):\n    def greet(self) -> str:\n        ...\n\np = SampleProtocols",
        encoding="utf-8",
    )
    _ = (src_pkg / "consumer.py").write_text(
        "from sample_pkg.protocols import Greeter\n\ndef call_greet(protocol: Greeter) -> str:\n    return protocol.greet()",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="protocols",
        apply=True,
    )
    protocols_source = (src_pkg / "protocols.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    tm.that(report.errors, eq=())
    tm.that(
        "class Greeter(Protocol):"
        in protocols_source.split("class SampleProtocols:", maxsplit=1)[0],
        eq=False,
    )
    tm.that(
        "class Greeter(Protocol):"
        in protocols_source.split("class SampleProtocols:", maxsplit=1)[1],
        eq=True,
    )
    tm.that(consumer_source, has="from sample_pkg.protocols import p")
    tm.that(consumer_source, has="def call_greet(protocol: p.Greeter) -> str:")


def test_refactor_utilities_iter_python_files_includes_examples_and_scripts(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "sample"
    project_root.mkdir(parents=True)
    _ = (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (project_root / ".git").mkdir(parents=True)
    expected_paths = [
        project_root / "src" / "sample_pkg" / "module.py",
        project_root / "tests" / "test_module.py",
        project_root / "examples" / "demo.py",
        project_root / "scripts" / "sync.py",
    ]
    for file_path in expected_paths:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _ = file_path.write_text(
            "from __future__ import annotations\n", encoding="utf-8"
        )
    discovered = FlextInfraUtilitiesIteration.iter_python_files(workspace_root=tmp_path)
    tm.ok(discovered)
    tm.that(set(discovered.value) == set(expected_paths), eq=True)


def test_discover_project_roots_without_nested_git_dirs(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)

    project_root = workspace_root / "proj-a"
    project_root.mkdir(parents=True)
    _ = (project_root / "pyproject.toml").write_text(
        "[project]\nname='proj-a'\n",
        encoding="utf-8",
    )
    _ = (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (project_root / "src").mkdir(parents=True)

    discovered = FlextInfraUtilitiesIteration.discover_project_roots(
        workspace_root=workspace_root,
    )
    tm.that(discovered, eq=[project_root])


def test_migrate_to_mro_moves_manual_uppercase_assignment(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    _ = (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\n\nVALUE = 42\n\nclass SampleConstants:\n    pass\n\nc = SampleConstants\n",
        encoding="utf-8",
    )
    _ = (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import VALUE\n\nresult = VALUE\n",
        encoding="utf-8",
    )
    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants",
        apply=True,
    )
    constants_source = (src_pkg / "constants.py").read_text(encoding="utf-8")
    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")
    tm.that(report.errors, eq=())
    tm.that(
        "VALUE = 42" in constants_source.split("class SampleConstants:")[0], eq=False
    )
    tm.that(
        "VALUE = 42" in constants_source.split("class SampleConstants:", maxsplit=1)[1],
        eq=True,
    )
    tm.that(consumer_source, has="from sample_pkg.constants import c")
    tm.that(consumer_source, has="result = c.VALUE")
