"""Guard 7 — fresh-process import smoke validator.

Imports each advertised package in a subprocess (fresh Python process)
and reports any ImportError. Catches circular-import cycles that the
lazy-loading machinery in ``flext_core.lazy`` would otherwise mask
until first attribute access.

Subprocess calls are routed through ``u.Cli.run_raw``.

Architecture: flext-infra validate layer — depends on ``m.Infra.ValidationReport``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from typing import Annotated, override

from flext_infra import c, m, p, r, s, t, u


class FlextInfraValidateFreshImport(s[bool]):
    """Validates that advertised packages import cleanly in fresh processes.

    Guard 7 of the circular-import defense-in-depth suite. Each package
    is imported via ``python -c 'import <pkg>'`` in a subprocess; any
    non-zero exit is reported as a violation.
    """

    packages: Annotated[
        t.StrSequence,
        m.Field(description="Package names to import-smoke in fresh subprocesses"),
    ] = (c.Infra.PKG_CORE_UNDERSCORE, "flext_infra", "flext_tests")

    def build_report(
        self, packages: t.StrSequence = ()
    ) -> p.Result[m.Infra.ValidationReport]:
        """Import each package in a fresh subprocess, collect failures.

        Args:
            packages: Package names to import-smoke. Empty sequence yields
                a passing empty report.

        Returns:
            r with ValidationReport listing packages that failed to import.

        """
        violations: t.MutableSequenceOf[str] = []
        env = self._workspace_import_env()
        for package in packages:
            smoke_result = u.Cli.run_raw(
                [sys.executable, "-c", f"import {package}"],
                cwd=self.repository_root,
                env=env,
            )
            if smoke_result.failure:
                violations.append(
                    f"{package}: {smoke_result.error or 'execution error'}"
                )
                continue
            output = smoke_result.value
            rc = output.exit_code
            lines = (output.stderr.strip() or output.stdout.strip()).splitlines()
            last_line = lines[-1] if lines else ""
            if rc != 0:
                reason = last_line or "ImportError"
                violations.append(f"{package}: {reason}")
        passed = not violations
        total = len(violations)
        summary = (
            f"{len(packages)} package(s) imported cleanly"
            if passed
            else f"{total} package(s) failed to import"
        )
        return r[m.Infra.ValidationReport].ok(
            m.Infra.ValidationReport(
                passed=passed, violations=list(violations), summary=summary
            )
        )

    def _workspace_import_env(self) -> t.StrMapping:
        """Return subprocess env that can import the selected workspace package."""
        inherited_env = u.Cli.process_env()
        import_roots = (
            str(self.repository_root),
            str(self.repository_root / c.Infra.DEFAULT_SRC_DIR),
        )
        existing_pythonpath = inherited_env.get(c.Infra.ORCHESTRATOR_ENV_PYTHONPATH, "")
        pythonpath = c.Infra.ORCHESTRATOR_ENV_PATH_SEPARATOR.join(
            part for part in (*import_roots, existing_pythonpath) if part
        )
        return u.Cli.process_env(
            overrides={c.Infra.ORCHESTRATOR_ENV_PYTHONPATH: pythonpath}
        )

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the fresh-import validation CLI flow."""
        report_result = self.build_report(packages=self.packages)
        if report_result.failure:
            return r[bool].fail(report_result.error or "fresh-import validation failed")
        report = report_result.unwrap()
        return r[bool].ok(True) if report.passed else r[bool].fail(report.summary)


__all__: t.StrSequence = ("FlextInfraValidateFreshImport",)
