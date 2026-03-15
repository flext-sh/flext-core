"""Stub supply chain service.

Manages typing stubs and typing dependencies for workspace projects,
including stub generation, types-package installation, and idempotency checks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra import FlextInfraUtilitiesSubprocess, c, m, p


class FlextInfraStubSupplyChain:
    """Manages typing stub supply chain for workspace projects.

    Coordinates mypy stub hints, pyrefly missing imports, stubgen
    generation, and types-package installation.
    """

    def __init__(self) -> None:
        """Initialize the stub supply chain."""
        self._runner: p.Infra.CommandRunner = FlextInfraUtilitiesSubprocess()

    def _discover_stub_projects(self, workspace_root: Path) -> list[Path]:
        """Discover projects that should participate in stub checks."""
        _ = self
        projects: list[Path] = []
        for entry in sorted(workspace_root.iterdir(), key=lambda v: v.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if (entry / c.Infra.Files.PYPROJECT_FILENAME).exists() and (
                entry / c.Infra.Paths.DEFAULT_SRC_DIR
            ).is_dir():
                projects.append(entry)
        return projects

    def _is_internal(self, module_name: str, project_name: str) -> bool:
        """Check if a module is an internal project module."""
        _ = self
        root_mod = module_name.split(".", 1)[0]
        project_root = project_name.replace("-", "_")
        if root_mod.startswith(c.Infra.Core.INTERNAL_PREFIXES):
            return True
        return root_mod == project_root

    def _stub_exists(self, module_name: str, workspace_root: Path) -> bool:
        """Check if a stub file exists for a module."""
        _ = self
        rel = module_name.replace(".", "/")
        for base in (
            workspace_root / c.Infra.Directories.TYPINGS,
            workspace_root / c.Infra.Directories.TYPINGS / "generated",
        ):
            candidates = [base / f"{rel}.pyi", base / rel / "__init__.pyi"]
            if any(c.exists() for c in candidates):
                return True
        return False

    def analyze(
        self,
        project_dir: Path,
        workspace_root: Path,
    ) -> r[m.Infra.Core.StubAnalysisReport]:
        """Analyze a project for missing stubs and type packages.

        Runs mypy for hints and pyrefly for missing imports, then
        classifies each as internal, resolved, or unresolved.

        Args:
            project_dir: Path to the project directory.
            workspace_root: Root of the workspace for stub lookup.

        Returns:
            r with analysis report dict.

        """
        try:
            root = workspace_root.resolve()
            proj = project_dir.resolve()
            mypy_hints = self._run_mypy_hints(proj)
            missing_imports = self._run_pyrefly_missing(proj)
            internal = [m for m in missing_imports if self._is_internal(m, proj.name)]
            external = [
                m for m in missing_imports if not self._is_internal(m, proj.name)
            ]
            unresolved = [m for m in external if not self._stub_exists(m, root)]
            result = m.Infra.Core.StubAnalysisReport(
                project=proj.name,
                mypy_hints=mypy_hints,
                internal_missing=internal,
                unresolved_missing=unresolved,
                total_missing=len(missing_imports),
            )
            return r[m.Infra.Core.StubAnalysisReport].ok(result)
        except (OSError, TypeError, ValueError) as exc:
            return r[m.Infra.Core.StubAnalysisReport].fail(
                f"stub analysis failed for {project_dir.name}: {exc}",
            )

    def validate(
        self,
        workspace_root: Path,
        project_dirs: list[Path] | None = None,
    ) -> r[m.Infra.Core.ValidationReport]:
        """Validate stub supply chain across projects.

        Args:
            workspace_root: Root directory of the workspace.
            project_dirs: Optional specific projects; discovers all if None.

        Returns:
            r with ValidationReport indicating overall status.

        """
        try:
            root = workspace_root.resolve()
            projects = project_dirs or self._discover_stub_projects(root)
            violations: list[str] = []
            for proj in projects:
                result = self.analyze(proj, root)
                if result.is_failure:
                    violations.append(f"{proj.name}: {result.error}")
                    continue
                data = result.value
                internal = data.internal_missing
                unresolved = data.unresolved_missing
                if internal:
                    violations.append(
                        f"{proj.name}: {len(internal)} internal missing imports",
                    )
                if unresolved:
                    violations.append(
                        f"{proj.name}: {len(unresolved)} unresolved imports",
                    )
            passed = len(violations) == 0
            summary = f"stub chain: {len(projects)} projects, {len(violations)} issues"
            return r[m.Infra.Core.ValidationReport].ok(
                m.Infra.Core.ValidationReport(
                    passed=passed,
                    violations=violations,
                    summary=summary,
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            return r[m.Infra.Core.ValidationReport].fail(
                f"stub validation failed: {exc}",
            )

    def _run_mypy_hints(self, project_dir: Path) -> list[str]:
        """Run mypy and extract types-package hints."""
        result = self._runner.run(
            [
                c.Infra.Cli.POETRY,
                c.Infra.Verbs.RUN,
                c.Infra.Cli.MYPY,
                c.Infra.Paths.DEFAULT_SRC_DIR,
                "--config-file",
                c.Infra.Files.PYPROJECT_FILENAME,
                "--no-error-summary",
            ],
            cwd=project_dir,
        )
        output = ""
        if result.is_success:
            cmd_output: p.Infra.CommandOutput = result.value
            output = cmd_output.stdout
        return sorted({
            m.group(1).strip()
            for m in c.Infra.Core.MYPY_HINT_RE.finditer(output)
            if m.group(1).strip()
        })

    def _run_pyrefly_missing(self, project_dir: Path) -> list[str]:
        """Run pyrefly check and extract missing imports."""
        result = self._runner.run(
            [
                c.Infra.Cli.POETRY,
                c.Infra.Verbs.RUN,
                c.Infra.Cli.PYREFLY,
                c.Infra.Cli.RuffCmd.CHECK,
                c.Infra.Paths.DEFAULT_SRC_DIR,
                "--config",
                c.Infra.Files.PYPROJECT_FILENAME,
            ],
            cwd=project_dir,
        )
        output = ""
        if result.is_success:
            cmd_output: p.Infra.CommandOutput = result.value
            output = cmd_output.stdout
        seen: set[str] = set()
        ordered: list[str] = []
        for match in c.Infra.Core.MISSING_IMPORT_RE.finditer(output):
            name = match.group(1).strip()
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered


__all__ = ["FlextInfraStubSupplyChain"]
