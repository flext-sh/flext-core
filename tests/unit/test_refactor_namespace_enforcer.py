from __future__ import annotations

from pathlib import Path

from flext_infra.refactor.namespace_enforcer import FlextInfraNamespaceEnforcer


def test_namespace_enforcer_creates_missing_facades_and_rewrites_imports(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    pkg.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "service.py").write_text(
        "from flext_core.constants import System\n"
        "from flext_infra.constants import Infra\n"
        "\n"
        "VALUE = 1\n",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=True
    )

    assert report.total_facades_missing == 0
    assert report.total_import_violations == 0
    assert (pkg / "constants.py").exists()
    assert (pkg / "typings.py").exists()
    assert (pkg / "protocols.py").exists()
    assert (pkg / "models.py").exists()
    assert (pkg / "utilities.py").exists()

    service_source = (pkg / "service.py").read_text(encoding="utf-8")
    assert "from flext_core import c, m, r, t, u, p, d, e, h, s, x" in service_source
    assert "from flext_infra import c, m, t, u, p" in service_source
