"""Lazy-init ``__init__.py`` generator (PEP 562).

Auto-discovers exports from sibling ``.py`` files and generates clean
lazy-loading ``__init__.py`` files using ``flext_core``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, override

from flext_core import r
from flext_infra import c, config, u
from flext_infra._utilities._sort_keys import path_depth
from flext_infra.base import s
from flext_infra.codegen._lazy_init_generation import (
    FlextInfraCodegenLazyInitGenerationMixin,
)
from flext_infra.codegen.lazy_init_planner import FlextInfraCodegenLazyInitPlanner
from flext_infra.workspace.rope import FlextInfraRopeWorkspace

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraCodegenLazyInit(s[bool], FlextInfraCodegenLazyInitGenerationMixin):
    """Generates ``__init__.py`` with PEP 562 lazy imports.

    Scans sibling ``.py`` files in each package directory, discovers their
    exports, and generates clean lazy-loading ``__init__.py`` files.
    Processes bottom-up so child packages are generated before parents.
    """

    _modified_files: t.Infra.StrSet = u.PrivateAttr(default_factory=set)
    _duplicate_class_names: int = u.PrivateAttr(default_factory=lambda: 0)

    @property
    def modified_files(self) -> t.StrSequence:
        """Generated __init__.py files that changed on disk."""
        return tuple(sorted(self._modified_files))

    @override
    def execute(self) -> p.Result[bool]:
        """Execute lazy-init directly from the validated CLI service model."""
        # NOTE (multi-agent, flext-wkii.17.15): one normalized mode controls every write.
        effective_dry_run = self.effective_dry_run
        errors = self.generate_inits(check_only=effective_dry_run)
        if self._duplicate_class_names > 0:
            return r[bool].fail(
                f"init aborted: {self._duplicate_class_names} "
                "duplicate class name(s) detected (see errors above); "
                "rename one side before regenerating"
            )
        if errors > 0:
            return r[bool].fail(f"init failed in {errors} package directories")
        if effective_dry_run and self._modified_files:
            drifted_files = ", ".join(sorted(self._modified_files))
            return r[bool].fail(
                f"init drift detected in {len(self._modified_files)} "
                f"generated artifacts: {drifted_files}"
            )
        return r[bool].ok(True)

    def generate_inits(self, *, check_only: bool = False) -> int:
        """Process all package directories bottom-up and generate PEP 562 inits."""
        self._modified_files.clear()
        self._duplicate_class_names = 0
        if not self.repository_root.exists():
            u.Cli.info("Lazy-init summary: 0 generated, 0 errors (0 dirs scanned)")
            return 0
        started_at = perf_counter()
        u.Cli.info(
            "lazy-init: starting "
            f"({'check' if check_only else 'apply'}) for {self.repository_root}"
        )
        lazy_init = config.Infra.tooling.lazy_init
        with FlextInfraRopeWorkspace.open_workspace(
            self.repository_root, rope_repository_root=self.repository_root
        ) as rope:
            workspace_index = rope.workspace_index
            resolved_repository_root = self.repository_root.resolve()
            indexed_package_dirs = tuple(
                sorted(
                    (
                        package_dir.resolve()
                        for package_dir in workspace_index.package_dirs
                        if package_dir.is_relative_to(resolved_repository_root)
                    ),
                    key=path_depth,
                    reverse=True,
                )
            )
            target_package_dir: Path | None = None
            if self.target_module:
                mapped_package_dir = workspace_index.package_dir_by_name.get(
                    self.target_module
                )
                target_module_dirs = frozenset(
                    entry.package_dir.resolve()
                    for entry in workspace_index.modules_by_path.values()
                    if entry.module_name == self.target_module
                )
                if mapped_package_dir is not None:
                    target_module_dirs = frozenset((
                        *target_module_dirs,
                        mapped_package_dir.resolve(),
                    ))
                sorted_target_dirs = tuple(sorted(target_module_dirs))
                if not sorted_target_dirs:
                    u.Cli.error(
                        f"lazy-init target module not found: {self.target_module}"
                    )
                    return 1
                if sorted_target_dirs[1:]:
                    u.Cli.error(
                        f"lazy-init target module is ambiguous: {self.target_module}"
                    )
                    return 1
                target_package_dir = sorted_target_dirs[0]
            if target_package_dir is None:
                # ALL_SCAN_PATTERNS is the contract: every Python surface
                # (src, tests, examples, scripts) is generated, no exceptions.
                package_dirs = indexed_package_dirs
            else:
                target_parts = target_package_dir.relative_to(
                    resolved_repository_root
                ).parts
                boundary_names = frozenset({
                    c.Infra.DEFAULT_SRC_DIR,
                    *c.Infra.NON_PUBLIC_LAZY_ROOTS,
                })
                boundary_index = next(
                    (
                        index
                        for index, part in enumerate(target_parts)
                        if part in boundary_names
                    ),
                    len(target_parts) - 1,
                )
                scope_prefix = target_parts[: boundary_index + 1]
                project_prefix = target_parts[:boundary_index]
                production_prefix = (*project_prefix, c.Infra.DEFAULT_SRC_DIR)
                package_dirs = tuple(
                    package_dir
                    for package_dir in indexed_package_dirs
                    if (
                        package_dir.relative_to(resolved_repository_root).parts[
                            : len(scope_prefix)
                        ]
                        == scope_prefix
                        # flext-pulj (codex): wrapper aliases depend on the same
                        # project's production plans, consumed read-only.
                        or package_dir.relative_to(resolved_repository_root).parts[
                            : len(production_prefix)
                        ]
                        == production_prefix
                    )
                )
            duplicates = self._detect_duplicate_class_names(
                rope, package_dirs=package_dirs
            )
            if duplicates:
                self._duplicate_class_names = len(duplicates)
                for class_name, locations in duplicates.items():
                    joined = ", ".join(locations)
                    u.Cli.error(
                        f"duplicate class name {class_name!r} in: {joined}; "
                        "rename one before regenerating __init__.py"
                    )
                u.Cli.info(
                    "Lazy-init summary: 0 generated, "
                    f"{len(duplicates)} duplicate class name(s) "
                    "(aborted before codegen)"
                )
                return len(duplicates)
            planner = FlextInfraCodegenLazyInitPlanner(
                rope_workspace=rope, lazy_init=lazy_init
            )
            u.Cli.info(f"lazy-init: planning {len(package_dirs)} package dirs")
            total, ok, errors, _dir_exports = self._generate_all_inits(
                package_dirs,
                check_only=check_only,
                planner=planner,
                target_package_dir=target_package_dir,
            )
        # flext-96j2.4 (agent: claude): Ruff check runs once over the changed
        # artifact set instead of per rendered template. Apply mode only:
        # check mode never writes, so the on-disk files still hold the OLD
        # content and linting them would report drift as false lint errors.
        if not check_only:
            errors += self.batch_lint_generated(self.modified_files)
        warnings = planner.collision_count
        u.Cli.info(
            f"Lazy-init summary: {ok} generated, {errors} errors, {warnings} warnings"
            f" ({total} dirs scanned, {perf_counter() - started_at:.2f}s)"
        )
        return errors

    def batch_lint_generated(self, generated_files: t.StrSequence) -> int:
        """Ruff-check every changed initializer in one invocation.

        flext-96j2.4 (agent: claude): the renderer emits byte-canonical output via
        ``ruff format`` per template; validation (``ruff check``) is deferred to
        this single batched stage over the changed artifact set so generation
        spawns one Ruff check subprocess instead of one per generated file.
        Returns the number of artifacts that failed the check.
        """
        targets = tuple(
            path
            for path in generated_files
            if path.endswith(c.Infra.EXT_PYTHON) and Path(path).is_file()
        )
        if not targets:
            return 0
        check = u.Cli.run_raw(
            [c.Infra.RUFF, c.Infra.CHECK, "--no-fix", *targets],
            cwd=self.repository_root,
        )
        if check.failure:
            u.Cli.error(f"batched Ruff check failed to run: {check.error}")
            return len(targets)
        result = check.value
        if result.exit_code == 0:
            return 0
        detail = (result.stdout or result.stderr).strip()
        u.Cli.error(
            f"generated initializers failed Ruff ({result.exit_code}): {detail}"
        )
        return 1

    @staticmethod
    def _detect_duplicate_class_names(
        rope: FlextInfraRopeWorkspace, *, package_dirs: t.SequenceOf[Path]
    ) -> t.MappingKV[str, t.StrSequence]:
        """Return class-name collisions.

        Scope rules:
        - ``src/`` modules: duplicates forbidden across the entire workspace.
        - ``tests/``/``scripts/``/``examples/``/``docs/`` modules: duplicates
          forbidden only within the same owning project (they do not escape).
        """
        scoped_modules: defaultdict[t.StrPair, set[str]] = defaultdict(set)
        selected_package_dirs = frozenset(path.resolve() for path in package_dirs)
        for entry in rope.workspace_index.modules_by_path.values():
            if (
                entry.package_dir.resolve() not in selected_package_dirs
                or entry.is_package_init
                or not entry.module_name
            ):
                continue
            module_segments = frozenset(entry.module_name.split("."))
            is_private_scope = bool(module_segments & c.Infra.NON_PUBLIC_LAZY_ROOTS)
            scope_key = (
                str(entry.project_root)
                if is_private_scope and entry.project_root is not None
                else ""
            )
            for obj in rope.objects(
                entry.file_path, include_local_scopes=False, include_references=False
            ):
                if obj.kind != "class" or obj.scope_path:
                    continue
                name = obj.name
                if len(name) < c.Infra.DUPLICATE_CLASS_MIN_LEN or not name[0].isupper():
                    continue
                scoped_modules[name, scope_key].add(entry.module_name)
        return {
            f"[{Path(scope_key).name}] {name}"
            if scope_key
            else f"[workspace] {name}": tuple(sorted(modules))
            for (name, scope_key), modules in scoped_modules.items()
            if len(modules) > 1
        }


__all__: list[str] = ["FlextInfraCodegenLazyInit"]
