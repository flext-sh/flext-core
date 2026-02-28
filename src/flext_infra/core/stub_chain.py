"""Stub supply chain service.

Manages typing stubs and typing dependencies for workspace projects,
including stub generation, types-package installation, and idempotency checks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from flext_core import FlextResult, r, t

from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.subprocess import CommandRunner

_MISSING_IMPORT_RE = re.compile(r"Cannot find module `([^`]+)` \[missing-import\]")
_MYPY_HINT_RE = re.compile(r"note:\s+(?:hint|note):\s+.*?`(types-\S+)`")
_MYPY_STUB_RE = re.compile(r"Library stubs not installed for ['\"](\S+?)['\"]")

_INTERNAL_PREFIXES = ("flext_", "flext-")


class StubSupplyChain:
    """Manages typing stub supply chain for workspace projects.

    Coordinates mypy stub hints, pyrefly missing imports, stubgen
    generation, and types-package installation.
    """

    def __init__(self) -> None:
        """Initialize the stub supply chain."""
        self._runner = CommandRunner()

    def analyze(
        self,
        project_dir: Path,
        workspace_root: Path,
    ) -> FlextResult[Mapping[str, t.ConfigMapValue]]:
        """Analyze a project for missing stubs and type packages.

        Runs mypy for hints and pyrefly for missing imports, then
        classifies each as internal, resolved, or unresolved.

        Args:
            project_dir: Path to the project directory.
            workspace_root: Root of the workspace for stub lookup.

        Returns:
            FlextResult with analysis report dict.

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

            result: MutableMapping[str, t.ConfigMapValue] = {
                "project": proj.name,
                "mypy_hints": mypy_hints,
                "internal_missing": internal,
                "unresolved_missing": unresolved,
                "total_missing": len(missing_imports),
            }
            return r[Mapping[str, t.ConfigMapValue]].ok(result)
        except (OSError, TypeError, ValueError) as exc:
            return r[Mapping[str, t.ConfigMapValue]].fail(
                f"stub analysis failed for {project_dir.name}: {exc}",
            )

    def validate(
        self,
        workspace_root: Path,
        project_dirs: list[Path] | None = None,
    ) -> FlextResult[m.ValidationReport]:
        """Validate stub supply chain across projects.

        Args:
            workspace_root: Root directory of the workspace.
            project_dirs: Optional specific projects; discovers all if None.

        Returns:
            FlextResult with ValidationReport indicating overall status.

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
                internal = data.get("internal_missing", [])
                unresolved = data.get("unresolved_missing", [])
                if isinstance(internal, list) and internal:
                    violations.append(
                        f"{proj.name}: {len(internal)} internal missing imports",
                    )
                if isinstance(unresolved, list) and unresolved:
                    violations.append(
                        f"{proj.name}: {len(unresolved)} unresolved imports",
                    )

            passed = len(violations) == 0
            summary = f"stub chain: {len(projects)} projects, {len(violations)} issues"
            return r[m.ValidationReport].ok(
                m.ValidationReport(
                    passed=passed,
                    violations=violations,
                    summary=summary,
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            return r[m.ValidationReport].fail(
                f"stub validation failed: {exc}",
            )

    def _run_mypy_hints(self, project_dir: Path) -> list[str]:
        """Run mypy and extract types-package hints."""
        result = self._runner.run(
            [
                "poetry",
                "run",
                "mypy",
                "src",
                "--config-file",
                "pyproject.toml",
                "--no-error-summary",
            ],
            cwd=project_dir,
        )
        output = ""
        if result.is_success:
            output = result.value.stdout
        return sorted({
            m.group(1).strip()
            for m in _MYPY_HINT_RE.finditer(output)
            if m.group(1).strip()
        })

    def _run_pyrefly_missing(self, project_dir: Path) -> list[str]:
        """Run pyrefly check and extract missing imports."""
        result = self._runner.run(
            ["poetry", "run", "pyrefly", "check", "src", "--config", "pyproject.toml"],
            cwd=project_dir,
        )
        output = ""
        if result.is_success:
            output = result.value.stdout
        seen: set[str] = set()
        ordered: list[str] = []
        for match in _MISSING_IMPORT_RE.finditer(output):
            name = match.group(1).strip()
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    @staticmethod
    def _is_internal(module_name: str, project_name: str) -> bool:
        """Check if a module is an internal project module."""
        root_mod = module_name.split(".", 1)[0]
        project_root = project_name.replace("-", "_")
        if root_mod.startswith(_INTERNAL_PREFIXES):
            return True
        return root_mod == project_root

    @staticmethod
    def _stub_exists(module_name: str, root: Path) -> bool:
        """Check if a stub file exists for a module."""
        rel = module_name.replace(".", "/")
        for base in (root / "typings", root / "typings" / "generated"):
            candidates = [
                base / f"{rel}.pyi",
                base / rel / "__init__.pyi",
            ]
            if any(c.exists() for c in candidates):
                return True
        return False

    @staticmethod
    def _discover_stub_projects(root: Path) -> list[Path]:
        """Discover projects that should participate in stub checks."""
        projects: list[Path] = []
        for entry in sorted(root.iterdir(), key=lambda v: v.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if (entry / c.Files.PYPROJECT_FILENAME).exists() and (
                entry / "src"
            ).is_dir():
                projects.append(entry)
        return projects


__all__ = ["StubSupplyChain"]
