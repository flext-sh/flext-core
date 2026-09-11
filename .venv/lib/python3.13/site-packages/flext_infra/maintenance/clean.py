"""Disposable-artifact census and removal for the generated ``clean`` verb.

Stale caches and traces are not cosmetic: they produce FALSE DIAGNOSES. A stale
``__pycache__`` once raised a pydantic ``ValidationError`` for source that was
already correct, and a stale ``.reports`` directory made a green suite look
failed. Removing them is therefore a first-class maintenance operation, not a
shell loop embedded in a Makefile recipe.

The disposable set is DATA (``config.Infra.codegen.make.clean``): every project
cleans exactly the same things, and a new artifact kind is one config row rather
than an edit to 31 generated Makefiles.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import config, m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p

logger = u.fetch_logger(__name__)


class FlextInfraCleanService(s[int]):
    """Report or remove the declared disposable artifacts of one project."""

    apply_changes: Annotated[
        bool, m.Field(description="Remove the artifacts instead of reporting them")
    ] = False

    def _cache_dirs(self) -> tuple[Path, ...]:
        """Return every cache directory found anywhere under the project."""
        spec = config.Infra.codegen.make.clean
        return tuple(
            path
            for name in spec.cache_dirs
            for path in self.repository_root.rglob(name)
            if path.is_dir()
        )

    def _root_entries(self) -> tuple[Path, ...]:
        """Return the root-level directories and files that exist."""
        spec = config.Infra.codegen.make.clean
        return tuple(
            path
            for name in (*spec.root_dirs, *spec.root_files)
            if (path := self.repository_root / name).exists()
        )

    def _trace_files(self) -> tuple[Path, ...]:
        """Return every trace or profile artifact found under the project."""
        spec = config.Infra.codegen.make.clean
        return tuple(
            path
            for pattern in spec.trace_globs
            for path in self.repository_root.rglob(pattern)
            if path.is_file()
        )

    @override
    def execute(self) -> p.Result[int]:
        """Report the disposable residue, removing it when apply is requested."""
        if not self.repository_root.is_dir():
            return r[int].fail(f"workspace is not a directory: {self.repository_root}")
        targets = (*self._cache_dirs(), *self._root_entries(), *self._trace_files())
        if not targets:
            u.Cli.info("clean: no disposable artifacts")
            return r[int].ok(0)
        if not self.apply_changes:
            for target in targets:
                u.Cli.info(f"  {target.relative_to(self.repository_root)}")
            apply_token = config.Infra.codegen.make.apply_variable
            apply_value = config.Infra.codegen.make.apply_value
            u.Cli.info(
                f"clean: {len(targets)} disposable artifact(s); "
                f"remove with make clean WHAT=generated {apply_token}={apply_value}"
            )
            return r[int].ok(0)
        removed = 0
        for target in targets:
            # A cache directory can vanish when its parent is removed first;
            # a missing target is the requested end state, never an error.
            outcome = (
                u.Cli.files_remove_directory(target)
                if target.is_dir()
                else u.Cli.files_delete(target)
            )
            if outcome.failure and target.exists():
                return r[int].fail(outcome.error or f"failed to remove {target}")
            removed += 1
        u.Cli.info(f"clean: removed {removed} disposable artifact(s)")
        return r[int].ok(0)


__all__: list[str] = ["FlextInfraCleanService"]
