"""Unit tests for namespace enforcer detection and auto-fix behaviors."""

from __future__ import annotations

from pathlib import Path

from flext_infra.refactor.namespace_enforcer import (
    FlextInfraNamespaceEnforcer,
)


def test_namespace_enforcer_creates_missing_facades_and_rewrites_imports(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    pkg.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (pkg / "service.py").write_text(
        "from flext_core.constants import System\nfrom flext_infra.constants import Infra\n\nVALUE = 1",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=True,
    )

    assert report.total_facades_missing == 0
    assert report.total_import_violations == 0
    assert (pkg / "constants.py").exists()
    assert (pkg / "typings.py").exists()
    assert (pkg / "protocols.py").exists()
    assert (pkg / "models.py").exists()
    assert (pkg / "utilities.py").exists()

    service_source = (pkg / "service.py").read_text(encoding="utf-8")
    assert "from flext_core import c, m, r, t, u, p" in service_source
    assert "from flext_infra import c, m, t, u, p" in service_source


def test_namespace_enforcer_detects_manual_typings_and_compat_aliases(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    pkg.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (pkg / "service.py").write_text(
        "from __future__ import annotations\nfrom typing import TypeAlias\n\nPayloadMap: TypeAlias = dict[str, str]\nLegacyResult = ModernResult",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=False,
    )

    assert report.total_manual_typing_violations >= 1
    assert report.total_compatibility_alias_violations >= 1


def test_namespace_enforcer_detects_manual_protocol_outside_canonical_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    pkg.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (pkg / "service.py").write_text(
        "from __future__ import annotations\nfrom typing import Protocol\n\nclass ServiceContract(Protocol):\n    def run(self) -> str:\n        ...",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=False,
    )

    assert report.total_manual_protocol_violations == 1
    project_report = report.projects[0]
    violations = project_report.manual_protocol_violations
    assert len(violations) == 1
    violation = violations[0]
    assert violation.name == "ServiceContract"
    rendered = FlextInfraNamespaceEnforcer.render_text(report)
    assert "Manual protocols: 1" in rendered


def test_namespace_enforcer_detects_internal_private_imports(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    pkg.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (pkg / "service.py").write_text(
        "from __future__ import annotations\nfrom flext_core._utilities.guards import FlextUtilitiesGuards\nfrom sample_pkg.protocols import _InternalContract\n\n_ = FlextUtilitiesGuards\n_ = _InternalContract",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=False,
    )

    assert report.total_internal_import_violations >= 1
    rendered = FlextInfraNamespaceEnforcer.render_text(report)
    assert "Internal imports:" in rendered


def test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    pkg.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    service_file = pkg / "service.py"
    _ = service_file.write_text(
        "from __future__ import annotations\nfrom typing import Protocol\n\nclass ServiceContract(Protocol):\n    def run(self) -> str:\n        ...\n\nclass ServiceImpl:\n    def run(self) -> str:\n        return 'ok'",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=True,
    )

    assert report.total_manual_protocol_violations == 0
    protocols_file = pkg / "protocols.py"
    assert protocols_file.exists()

    service_source = service_file.read_text(encoding="utf-8")
    protocols_source = protocols_file.read_text(encoding="utf-8")
    assert "class ServiceContract(Protocol):" not in service_source
    assert "class ServiceContract(Protocol):" in protocols_source
    assert "from __future__ import annotations" in protocols_source
    assert "from typing import Protocol" in protocols_source


def test_namespace_enforcer_detects_cyclic_imports_in_tests_directory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    test_pkg = project / "tests"
    pkg.mkdir(parents=True)
    test_pkg.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (test_pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (test_pkg / "a.py").write_text(
        "from __future__ import annotations\nfrom tests.b import value_b\nvalue_a = value_b\n",
        encoding="utf-8",
    )
    _ = (test_pkg / "b.py").write_text(
        "from __future__ import annotations\nfrom tests.a import value_a\nvalue_b = value_a\n",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=False,
    )

    assert report.total_cyclic_imports >= 1


def test_namespace_enforcer_detects_missing_runtime_alias_outside_src(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    examples_dir = project / "examples"
    pkg.mkdir(parents=True)
    examples_dir.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    _ = (examples_dir / "constants.py").write_text(
        "from __future__ import annotations\n\nclass DemoConstants:\n    pass\n",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=False,
    )

    assert report.total_runtime_alias_violations >= 1


def test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    scripts_dir = project / "scripts"
    pkg.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    script_file = scripts_dir / "run.py"
    _ = script_file.write_text(
        "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nprint('ok')\n",
        encoding="utf-8",
    )

    _ = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply_changes=True,
    )

    rewritten_lines = script_file.read_text(encoding="utf-8").splitlines()
    assert rewritten_lines[0] == "#!/usr/bin/env python3"
    assert rewritten_lines[1] == "# -*- coding: utf-8 -*-"
    assert "from __future__ import annotations" in rewritten_lines
