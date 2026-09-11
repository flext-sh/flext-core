"""Session binding of an external consumer onto one flext worktree.

An external project declares flext packages by pinned git URL, so it always
validates PUBLISHED code. A cross-project change therefore could not be reviewed
until it was published, which inverts the order of work.

``FLEXT=<worktree>`` rebinds the consumer's environment onto that checkout for
the session. It is deliberately NOT a declaration: the consumer's
``pyproject.toml`` is never modified, so a local path can never be committed and
running setup without the flag restores the pinned resolution. Persistent,
declared sources remain owned by ``pyproject_conform._sync_uv_sources``; this
service owns only the session override, so the two never overlap.

The rebind set is derived from the intersection of what the consumer declares
and what the worktree actually provides, read from the worktree's own manifest —
never a hardcoded package list.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, u
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector

if TYPE_CHECKING:
    from flext_infra import p

logger = u.fetch_logger(__name__)


class FlextInfraFlextBindingService:
    """Resolve and apply one session binding onto a flext worktree."""

    @staticmethod
    def _declared_distributions(consumer_root: Path) -> p.Result[tuple[str, ...]]:
        """Return the distribution names the consumer declares as dependencies."""
        manifest = consumer_root / c.Infra.PYPROJECT_FILENAME
        if not manifest.is_file():
            return r[tuple[str, ...]].fail(
                f"consumer has no {c.Infra.PYPROJECT_FILENAME}: {consumer_root}"
            )
        payload = tomllib.loads(manifest.read_text(encoding="utf-8"))
        project = payload.get("project")
        declared = project.get("dependencies", []) if isinstance(project, dict) else []
        names: list[str] = []
        for entry in declared:
            # A requirement is "<name>" or "<name> @ <url>" or "<name>>=<spec>";
            # the distribution name is whatever precedes the first delimiter.
            text = str(entry).strip()
            name = text.split("@", 1)[0].split(";", 1)[0]
            for delimiter in ("[", ">", "<", "=", "!", "~", " "):
                name = name.split(delimiter, 1)[0]
            if name:
                names.append(name.strip())
        return r[tuple[str, ...]].ok(tuple(names))

    @classmethod
    def plan_targets(
        cls, *, consumer_root: Path, flext_root: Path
    ) -> p.Result[tuple[str, ...]]:
        """Return the distributions this worktree can supply to the consumer.

        Fails closed when ``flext_root`` is not a flext workspace, so a mistyped
        path can never silently bind nothing and leave the consumer on its pins.
        """
        workspace = FlextInfraWorkspaceDetector.load_workspace_spec(flext_root)
        if workspace.failure:
            return r[tuple[str, ...]].fail(
                f"FLEXT is not a flext workspace: {flext_root}: "
                f"{workspace.error or 'manifest unreadable'}"
            )
        available = {
            declared_repository.distribution: declared_repository
            for declared_repository in workspace.value.declared_repositories
            if declared_repository.package
        }
        declared = cls._declared_distributions(consumer_root)
        if declared.failure:
            return r[tuple[str, ...]].fail(
                declared.error or "cannot read consumer dependencies"
            )
        return r[tuple[str, ...]].ok(
            tuple(sorted(name for name in declared.value if name in available))
        )

    @classmethod
    def apply(
        cls, *, consumer_root: Path, flext_root: Path, python: Path
    ) -> p.Result[int]:
        """Rebind the consumer environment onto the worktree for this session."""
        planned = cls.plan_targets(consumer_root=consumer_root, flext_root=flext_root)
        if planned.failure:
            return r[int].fail(planned.error or "failed to plan the flext binding")
        targets = planned.value
        if not targets:
            u.Cli.info("flext binding: consumer declares no flext packages")
            return r[int].ok(0)
        workspace = FlextInfraWorkspaceDetector.load_workspace_spec(flext_root)
        if workspace.failure:
            return r[int].fail(workspace.error or "workspace topology unreadable")
        paths = {
            declared_repository.distribution: (
                flext_root / declared_repository.path
            ).resolve()
            for declared_repository in workspace.value.declared_repositories
            if declared_repository.package
        }
        editables: list[str] = []
        for name in targets:
            editables.extend(("--editable", str(paths[name])))
        installed = u.Cli.run_checked(
            (c.Infra.UV, "pip", "install", "--python", str(python), *editables),
            cwd=consumer_root,
        )
        if installed.failure:
            return r[int].fail(installed.error or "failed to bind the flext worktree")
        u.Cli.info(
            f"flext binding: {len(targets)} package(s) bound to {flext_root} "
            f"({', '.join(targets)})"
        )
        return r[int].ok(0)


__all__: list[str] = ["FlextInfraFlextBindingService"]
