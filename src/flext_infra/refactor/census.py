"""Usage census orchestrator logic.

Delegates core file crawling and parsing to `u.Infra` and LibCST visitors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from pathlib import Path

from flext_core import r
from flext_infra import c, m, u
from flext_infra._utilities.output import output
from flext_infra.refactor.output import render_census_report
from flext_infra.refactor.transformers.census_visitors import (
    CensusImportDiscoveryVisitor,
    CensusUsageCollector,
)

__all__ = ["FlextInfraRefactorCensus"]

type RCensusReport = r[m.Infra.Refactor.CensusReport]
CI = c.Infra.Refactor.Census


class FlextInfraRefactorCensus:
    """Census execution engine resolving family usage patterns."""

    @staticmethod
    def render_text(report: m.Infra.Refactor.CensusReport) -> str:
        """Render the census report cleanly."""
        return render_census_report(report)

    def run(
        self, root: Path, *, target: m.Infra.Refactor.MROFamilyTarget | None = None
    ) -> RCensusReport:
        """Execute the workspace census."""
        target = target or u.Infra.build_mro_target(CI.DEFAULT_FAMILY)
        t0 = time.monotonic()
        output.header(f"Usage Census — family={target.family} ({target.class_suffix})")

        pkg = (
            root
            / target.core_project
            / c.Infra.Paths.DEFAULT_SRC_DIR
            / target.package_dir
        )
        facade = (
            root
            / target.core_project
            / c.Infra.Paths.DEFAULT_SRC_DIR
            / target.facade_module
        )

        # 1-3. Metadata & Discovery
        output.progress(1, 5, "Metadata gathering", "metadata")
        parsed = (
            u.Infra.extract_public_methods_from_dir(pkg)
            if pkg.is_dir()
            else u.Infra.extract_public_methods_from_file(pkg)
        )
        methods = {
            cls: [
                m.Infra.Refactor.CensusMethodInfo(name=n, method_type=t, source_file=s)
                for n, t, s in lst
            ]
            for cls, lst in parsed.items()
        }
        index = {cls: {mi.name for mi in ms} for cls, ms in methods.items()}

        flat = u.Infra.build_facade_alias_map(facade, target.facade_class_prefix)
        inner = u.Infra.build_facade_inner_class_map(facade, target.facade_class_prefix)
        roots = u.Infra.discover_project_roots(workspace_root=root)

        # 4. Scanning & Visitors
        output.progress(4, 5, "scan-files", "libcst")
        pf = target.package_dir.replace("/", ".").replace("\\", ".")
        files = [
            f
            for f in u.Infra.iter_python_files(workspace_root=root)
            if pf not in f.as_posix() and "__pycache__" not in f.as_posix()
        ]

        recs: list[m.Infra.Refactor.CensusUsageRecord] = []
        errs = usage = 0
        for i, fp in enumerate(files, 1):
            if i % 500 == 0:
                output.info(f"  [{i}/{len(files)}] scanned...")

            project = u.Infra.identify_project_by_roots(fp, roots)
            imp = CensusImportDiscoveryVisitor(
                family_alias=target.family,
                facade_class_prefix=target.facade_class_prefix,
            )
            col = CensusUsageCollector(
                method_index=index,
                flat_aliases=flat,
                inner_class_map=inner,
                alias_locals=imp.alias_locals,
                direct_imports=imp.direct_imports,
                file_path=fp,
                project_name=project,
            )
            tree = u.Infra.scan_cst_with_visitors(fp, imp, col)
            if not tree:
                errs += 1
                continue
            if col.records:
                usage += 1
                recs.extend(col.records)

        output.info(f"Files with usage: {usage}, parse errors: {errs}")

        # 5. Rollup and format
        output.progress(5, 5, "aggregate", "report")
        rep = u.Infra.aggregate_usage_metrics(methods, recs, len(files), errs)
        output.summary(
            "census",
            total=rep.total_methods,
            success=rep.total_methods - rep.total_unused,
            failed=rep.total_unused,
            skipped=errs,
            elapsed=time.monotonic() - t0,
        )
        return r[m.Infra.Refactor.CensusReport].ok(rep)
