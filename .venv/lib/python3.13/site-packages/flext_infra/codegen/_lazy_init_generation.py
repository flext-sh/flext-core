"""Lazy-init per-directory generation service — extracted concern."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u
from flext_infra.codegen._lazy_init_generation_io import (
    FlextInfraCodegenLazyInitGenerationIOMixin,
)
from flext_infra.codegen._lazy_init_generation_registry import (
    FlextInfraCodegenLazyInitGenerationRegistryMixin,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

    from flext_infra import m, t
    from flext_infra.codegen.lazy_init_planner import FlextInfraCodegenLazyInitPlanner


# flext-i6nq.10: Root manifests and initializers are synchronized as one artifact set.
class FlextInfraCodegenLazyInitGenerationMixin(
    FlextInfraCodegenLazyInitGenerationIOMixin,
    FlextInfraCodegenLazyInitGenerationRegistryMixin,
):
    """Generate/remove ``__init__.py`` per package directory."""

    if TYPE_CHECKING:
        repository_root: Path
        _modified_files: t.Infra.StrSet

    def _generate_all_inits(
        self,
        pkg_dirs: t.SequenceOf[Path],
        *,
        check_only: bool,
        planner: FlextInfraCodegenLazyInitPlanner,
        target_package_dir: Path | None = None,
    ) -> tuple[int, int, int, MutableMapping[str, t.LazyAliasMap]]:
        total = ok = 0
        dir_exports: MutableMapping[str, t.LazyAliasMap] = {}
        planned: list[m.Infra.LazyInitPlan] = []
        progress_interval = max(1, len(pkg_dirs) // 20) if pkg_dirs else 1
        for idx, pkg_dir in enumerate(pkg_dirs, start=1):
            total += 1
            if idx == 1 or idx == len(pkg_dirs) or idx % progress_interval == 0:
                rel_path = (
                    pkg_dir.relative_to(self.repository_root)
                    if self.repository_root in pkg_dir.parents
                    else pkg_dir
                )
                u.Cli.info(f"lazy-init: progress {idx}/{len(pkg_dirs)} — {rel_path}")
            plan = planner.build_plan(pkg_dir, dir_exports=dir_exports)
            if plan.lazy_map:
                dir_exports[str(pkg_dir.resolve())] = dict(plan.lazy_map)
            should_process = (
                target_package_dir is None
                or pkg_dir.resolve() == target_package_dir.resolve()
                or (
                    target_package_dir.name == c.Infra.DIR_TESTS
                    and pkg_dir.resolve().is_relative_to(target_package_dir.resolve())
                )
            )
            if should_process:
                planned.append(plan)
            if (
                target_package_dir is not None
                and pkg_dir.resolve() == target_package_dir.resolve()
            ):
                break
        u.Cli.info(f"lazy-init: applying {len(planned)} preflighted package plans")
        for plan in planned:
            result, _exports = self._process_plan(plan, check_only=check_only)
            if result is None:
                continue
            if result < 0:
                return total, ok, 1, dir_exports
            ok += 1
        return total, ok, 0, dir_exports

    def _process_plan(
        self, plan: m.Infra.LazyInitPlan, *, check_only: bool
    ) -> t.Infra.LazyInitProcessResult:
        """Process a resolved lazy-init plan."""
        if plan.action == c.Infra.LazyInitAction.SKIP:
            return (None, dict(plan.lazy_map))
        if plan.action == c.Infra.LazyInitAction.REMOVE:
            return (
                self._check_remove_init(plan) if check_only else self._remove_init(plan)
            )
        if check_only:
            return self._check_write_init(plan)
        return self._write_init(plan)


__all__: list[str] = ["FlextInfraCodegenLazyInitGenerationMixin"]
