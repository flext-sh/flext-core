"""Unit tests for namespace enforcer detection and auto-fix behaviors."""

from __future__ import annotations

from pathlib import Path

from flext_infra import (
    FlextInfraNamespaceEnforcer,
)
from flext_tests import tm


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
        "from flext_core import c, m, r, t, u, p\nfrom flext_infra import c, m, t, u, p\n\nVALUE = 1",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply=True,
    )

    tm.that(report.total_facades_missing, eq=0)
    tm.that(report.total_import_violations, eq=0)
    tm.that((pkg / "constants.py").exists(), eq=True)
    tm.that((pkg / "typings.py").exists(), eq=True)
    tm.that((pkg / "protocols.py").exists(), eq=True)
    tm.that((pkg / "models.py").exists(), eq=True)
    tm.that((pkg / "utilities.py").exists(), eq=True)

    service_source = (pkg / "service.py").read_text(encoding="utf-8")
    tm.that(service_source, has="from flext_core import c, m, r, t, u, p")
    tm.that(service_source, has="from flext_infra import c, m, t, u, p")


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
        apply=False,
    )

    tm.that(report.total_manual_typing_violations, gt=0)
    tm.that(report.total_compatibility_alias_violations, gt=0)


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
        apply=False,
    )

    tm.that(report.total_manual_protocol_violations, eq=1)
    project_report = report.projects[0]
    violations = project_report.manual_protocol_violations
    tm.that(len(violations), eq=1)
    violation = violations[0]
    tm.that(violation.name, eq="ServiceContract")
    rendered = FlextInfraNamespaceEnforcer.render_text(report)
    tm.that(rendered, has="Manual protocols: 1")


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
        "from __future__ import annotations\nfrom flext_core import FlextUtilitiesGuards\nfrom sample_pkg.protocols import _InternalContract\n\n_ = FlextUtilitiesGuards\n_ = _InternalContract",
        encoding="utf-8",
    )

    report = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply=False,
    )

    tm.that(report.total_internal_import_violations, gt=0)
    rendered = FlextInfraNamespaceEnforcer.render_text(report)
    tm.that(rendered, has="Internal imports:")


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
        apply=True,
    )

    tm.that(report.total_manual_protocol_violations, eq=0)
    protocols_file = pkg / "protocols.py"
    tm.that(protocols_file.exists(), eq=True)

    service_source = service_file.read_text(encoding="utf-8")
    protocols_source = protocols_file.read_text(encoding="utf-8")
    tm.that("class ServiceContract(Protocol):" in service_source, eq=False)
    tm.that(protocols_source, has="class ServiceContract(Protocol):")
    tm.that(protocols_source, has="from __future__ import annotations")
    tm.that(protocols_source, has="from typing import Protocol")


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
        apply=False,
    )

    tm.that(report.total_cyclic_imports, gt=0)


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
        apply=False,
    )

    tm.that(report.total_runtime_alias_violations, gt=0)


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
        apply=True,
    )

    rewritten_lines = script_file.read_text(encoding="utf-8").splitlines()
    tm.that(rewritten_lines[0], eq="#!/usr/bin/env python3")
    tm.that(rewritten_lines[1], eq="# -*- coding: utf-8 -*-")
    tm.that(rewritten_lines, has="from __future__ import annotations")


def test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-proj"
    pkg = project / "src" / "sample_pkg"
    tests_dir = project / "tests"
    pkg.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n",
        encoding="utf-8",
    )
    _ = (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    _ = (pkg / "__init__.py").write_text("", encoding="utf-8")
    target_file = tests_dir / "base_improved.py"
    _ = target_file.write_text(
        '"""Improved test base with high automation and real functionality."""\n'
        "from pathlib import Path\n"
        "\n"
        "class AlgarOudMigTestBase:\n"
        '    """Highly automated test base with real functionality patterns."""\n'
        "    temp_dir: Path\n",
        encoding="utf-8",
    )

    _ = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply=True,
    )

    rewritten_lines = target_file.read_text(encoding="utf-8").splitlines()
    tm.that(rewritten_lines[0].startswith('"""Improved test base'), eq=True)
    tm.that(rewritten_lines[2], eq="from __future__ import annotations")


def test_namespace_enforcer_does_not_rewrite_indented_import_aliases(
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
        "from __future__ import annotations\n\n"
        "def runner() -> None:\n"
        "    from flext_core import System\n"
        "    _ = System\n",
        encoding="utf-8",
    )

    _ = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply=True,
    )

    service_source = service_file.read_text(encoding="utf-8")
    tm.that(service_source, has="    from flext_core import System")


def test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks(
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
    module_file = pkg / "constants.py"
    _ = module_file.write_text(
        "from __future__ import annotations\n"
        "from flext_infra import (\n"
        "    FlextInfraCoreConstants,\n"
        "    FlextInfraSharedInfraConstants,\n"
        ")\n"
        "\n"
        "class DemoConstants:\n"
        "    pass\n",
        encoding="utf-8",
    )

    _ = FlextInfraNamespaceEnforcer(workspace_root=workspace).enforce(
        apply=True,
    )

    module_source = module_file.read_text(encoding="utf-8")
    tm.that(module_source, has="from flext_infra import (")
    tm.that(module_source, has="FlextInfraCoreConstants")
