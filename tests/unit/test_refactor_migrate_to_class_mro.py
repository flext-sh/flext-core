"""Tests for migrate-to-mro automation."""

from __future__ import annotations

from pathlib import Path

from flext_infra.refactor.migrate_to_class_mro import (
    FlextInfraRefactorMigrateToClassMRO,
)


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
        "from __future__ import annotations\n"
        "from typing import Final\n\n"
        "VALUE: Final[int] = 42\n\n"
        "class SampleConstants:\n"
        "    pass\n\n"
        "c = SampleConstants\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import VALUE\n\nresult = VALUE\n",
        encoding="utf-8",
    )

    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants",
        apply_changes=True,
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
        "from __future__ import annotations\n"
        "from typing import Final\n\n"
        "_TIMEOUT: Final[int] = 30\n\n"
        "class SampleConstants:\n"
        "    TIMEOUT = _TIMEOUT\n\n"
        "c = SampleConstants\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import _TIMEOUT\n\nvalue = _TIMEOUT\n",
        encoding="utf-8",
    )

    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants",
        apply_changes=True,
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


def test_migrate_to_mro_preserves_existing_facade_alias(tmp_path: Path) -> None:
    project_root = tmp_path / "sample"
    src_pkg = project_root / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project_root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "constants.py").write_text(
        "from __future__ import annotations\n"
        "from typing import Final\n\n"
        "VALUE: Final[int] = 42\n\n"
        "class SampleConstants:\n"
        "    pass\n\n"
        "c = SampleConstants\n",
        encoding="utf-8",
    )
    (src_pkg / "consumer.py").write_text(
        "from sample_pkg.constants import VALUE\n"
        "from sample_pkg.constants import c as constants\n\n"
        "result = VALUE\n",
        encoding="utf-8",
    )

    report = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_root).run(
        target="constants",
        apply_changes=True,
    )

    consumer_source = (src_pkg / "consumer.py").read_text(encoding="utf-8")

    assert report.errors == ()
    assert "from sample_pkg.constants import VALUE" not in consumer_source
    assert "from sample_pkg.constants import c as constants" in consumer_source
    assert "result = constants.VALUE" in consumer_source
