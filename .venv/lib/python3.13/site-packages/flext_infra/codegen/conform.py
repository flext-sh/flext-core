"""Unified, fail-closed conformance for new and existing repositories.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import re
import stat
import time
from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Annotated, override

from filelock import UnixFileLock
from flext_core import r
from flext_infra import config, p, settings
from flext_infra.base import s
from flext_infra.codegen._mise_artifacts_transaction import (
    FlextInfraCodegenMiseArtifactTransaction,
)
from flext_infra.codegen.mise_artifacts import FlextInfraCodegenMiseArtifacts
from flext_infra.constants import c
from flext_infra.deps.modernizer import FlextInfraPyprojectModernizer
from flext_infra.deps.phases.ensure_ruff import FlextInfraEnsureRuffConfigPhase
from flext_infra._utilities.project_managed_artifacts import (
    FlextInfraUtilitiesProjectManagedArtifacts,
)
from flext_infra.models import m
from flext_infra.services.codegen import FlextInfraCodegen
from flext_infra.typings import t
from flext_infra.utilities import u
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector


class FlextInfraCodegenConform(s[m.Infra.CodegenResult]):
    """Plan every selected output, then atomically write only a clean plan."""

    @staticmethod
    def _link_mode(
        repository: m.Infra.RepositoryRef, toolchain: m.Infra.ToolchainSpec
    ) -> str:
        """Resolve the repository override through one codegen authority."""
        link_mode = repository.uv_link_mode or toolchain.uv_link_mode
        if not isinstance(link_mode, str):
            msg = "resolved uv link mode must be a string"
            raise TypeError(msg)
        return link_mode

    @staticmethod
    def _dependency_cooldown_policy(
        repository: m.Infra.RepositoryRef, toolchain: m.Infra.ToolchainSpec
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        """Compose fleet defaults with the repository's narrower policy."""
        exclusions = dict.fromkeys(toolchain.dependency_cooldown_exclusions)
        overrides = dict(toolchain.dependency_cooldown_overrides)
        for package in repository.dependency_cooldown_exclusions:
            overrides.pop(package, None)
            exclusions[package] = None
        for package, cutoff in repository.dependency_cooldown_overrides.items():
            exclusions.pop(package, None)
            overrides[package] = cutoff
        return tuple(exclusions), overrides

    @staticmethod
    def _beads_projection_is_inherited(destination: str, repository_root: Path) -> bool:
        """Return whether a tracked ledger route disowns this Beads projection.

        Governed members track ``.beads`` as a symlink into the superproject
        ledger. The route is the declaration: it holds in the workspace, where
        it resolves, and in a standalone clone of the same member, where it
        dangles outside the checkout. Either way the member owns no Beads
        projection of its own, so every conform route asks this one question
        here rather than repeating the pair of conditions.
        """
        if destination not in {
            c.Infra.BEADS_CONFIG_RELPATH,
            c.Infra.BEADS_METADATA_RELPATH,
        }:
            return False
        route: Path = repository_root / c.Infra.BEADS_DIRNAME
        return route.is_symlink()

    @classmethod
    def _surface_contract(
        cls, surface: c.Infra.CodegenConformSurface
    ) -> m.Infra.CodegenConformSurfaceContract:
        match surface:
            case c.Infra.CodegenConformSurface.ALL:
                return m.Infra.CodegenConformSurfaceContract(complete_governed=True)
            case c.Infra.CodegenConformSurface.DEPENDENCIES:
                return m.Infra.CodegenConformSurfaceContract(
                    destinations=frozenset({c.Infra.PYPROJECT_FILENAME}),
                    dependencies_only=True,
                    delegates=False,
                    templates=False,
                    custom=False,
                )
            case c.Infra.CodegenConformSurface.PYPROJECT:
                return m.Infra.CodegenConformSurfaceContract(
                    destinations=frozenset({c.Infra.PYPROJECT_FILENAME}),
                    delegates=False,
                    templates=False,
                    custom=False,
                )
            case c.Infra.CodegenConformSurface.MAKEFILE:
                return m.Infra.CodegenConformSurfaceContract(
                    destinations=frozenset({c.Infra.MAKEFILE_FILENAME}),
                    pyproject=False,
                    custom=False,
                )
            case _:
                msg = f"Unsupported codegen conform surface: {surface}"
                raise ValueError(msg)

    # This is the only
    # orchestrator for Make/toolchain/source conformance. Rendering stays in
    # flext-cli; Git-source TOML policy and attached detection are composed from
    # their separately owned u.Infra/workspace services.
    request: Annotated[
        m.Infra.CodegenConformRequest | None,
        m.Field(default=None, exclude=True, description="Validated conform request"),
    ] = None
    initial_workspace: Annotated[
        m.Infra.WorkspaceSpec | None,
        m.Field(
            default=None,
            exclude=True,
            description="Validated scaffold specification included in the atomic plan",
        ),
    ] = None

    @classmethod
    def execute_request(
        cls,
        request: m.Infra.CodegenConformRequest,
        initial_workspace: m.Infra.WorkspaceSpec | None = None,
    ) -> p.Result[m.Infra.CodegenResult]:
        """Execute one already validated public CLI request."""
        service = cls(
            repository_root=request.root.expanduser().resolve(),
            request=request,
            initial_workspace=initial_workspace,
        )
        return service.execute()

    @staticmethod
    def conform_transaction_lock_path(repository_root: Path) -> Path:
        """Return the XDG runtime coordination path for one repository owner."""
        root = repository_root.expanduser().resolve()
        identity = u.Cli.sha256_content(str(root))
        return settings.runtime_dir / "codegen-conform" / f"{identity}.lock"

    @staticmethod
    def _authenticated_managed_file(
        repository_root: Path, target: Path, *, allow_missing_parent: bool = False
    ) -> p.Result[tuple[str, tuple[int, int] | None]]:
        """Read one nominal regular file without following links or sharing inodes."""
        try:
            root = repository_root.expanduser().absolute()
            nominal = target.expanduser().absolute()
            nominal.relative_to(root)
        except (OSError, ValueError) as exc:
            return r[tuple[str, tuple[int, int] | None]].fail_op(
                f"managed destination resolution ({target})", exc
            )
        if allow_missing_parent and not nominal.exists() and not nominal.is_symlink():
            return r[tuple[str, tuple[int, int] | None]].ok(("", None))
        try:
            resolved_root = root.resolve(strict=True)
            parent = nominal.parent.resolve(strict=True)
            parent.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            return r[tuple[str, tuple[int, int] | None]].fail_op(
                f"managed destination parent authentication ({target})", exc
            )
        if parent != nominal.parent:
            return r[tuple[str, tuple[int, int] | None]].fail(
                f"managed destination has a linked parent: {nominal}"
            )
        try:
            before = os.lstat(nominal)
        except FileNotFoundError:
            return r[tuple[str, tuple[int, int] | None]].ok(("", None))
        except OSError as exc:
            return r[tuple[str, tuple[int, int] | None]].fail_op(
                f"managed destination inspection ({target})", exc
            )
        if not stat.S_ISREG(before.st_mode):
            return r[tuple[str, tuple[int, int] | None]].fail(
                f"managed destination is not a regular file: {nominal}"
            )
        if before.st_nlink != 1:
            return r[tuple[str, tuple[int, int] | None]].fail(
                f"managed destination has shared hard-link identity: {nominal}"
            )
        nofollow = getattr(os, "O_NOFOLLOW", None)
        nonblock = getattr(os, "O_NONBLOCK", None)
        if nofollow is None or nonblock is None:
            return r[tuple[str, tuple[int, int] | None]].fail(
                "authenticated managed-file reads require O_NOFOLLOW and O_NONBLOCK"
            )
        identity = (before.st_dev, before.st_ino)
        try:
            descriptor = os.open(nominal, os.O_RDONLY | nofollow | nonblock)
            with os.fdopen(descriptor, encoding=c.Infra.ENCODING_DEFAULT) as stream:
                opened = os.fstat(stream.fileno())
                if (
                    (opened.st_dev, opened.st_ino) != identity
                    or not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                ):
                    return r[tuple[str, tuple[int, int] | None]].fail(
                        f"managed destination changed during authentication: {nominal}"
                    )
                content = stream.read()
        except (OSError, UnicodeError) as exc:
            return r[tuple[str, tuple[int, int] | None]].fail_op(
                f"managed destination authenticated read ({target})", exc
            )
        try:
            after = os.lstat(nominal)
        except OSError as exc:
            return r[tuple[str, tuple[int, int] | None]].fail_op(
                f"managed destination post-read inspection ({target})", exc
            )
        if (
            (after.st_dev, after.st_ino) != identity
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
        ):
            return r[tuple[str, tuple[int, int] | None]].fail(
                f"managed destination changed during authentication: {nominal}"
            )
        return r[tuple[str, tuple[int, int] | None]].ok((content, identity))

    def _apply_makefile_locked(
        self, request: m.Infra.CodegenConformRequest, planned: m.Infra.CodegenPlan
    ) -> p.Result[m.Infra.CodegenResult]:
        """Promote one unchanged projection while its strict owner lock is held."""
        u.Cli.info("stage=verify-before-publish")
        replanned = self.plan(request)
        if replanned.failure:
            return r[m.Infra.CodegenResult].fail(
                "stage=verify-before-publish: "
                f"{replanned.error or 'Makefile replan failed'}"
            )
        if replanned.value != planned:
            return r[m.Infra.CodegenResult].fail(
                "stage=verify-before-publish: Makefile projection changed"
            )
        files = replanned.value.files
        if len(files) != 1 or files[0].path.name != c.Infra.MAKEFILE_FILENAME:
            return r[m.Infra.CodegenResult].fail(
                "Makefile selection did not resolve exactly one canonical target"
            )
        file = files[0]
        authenticated = self._authenticated_managed_file(request.root, file.path)
        if authenticated.failure:
            return r[m.Infra.CodegenResult].fail(
                authenticated.error or "Makefile authentication failed"
            )
        current, identity = authenticated.value
        current_sha = u.Cli.sha256_content(current) if identity is not None else ""
        if current_sha != file.current_sha256:
            return r[m.Infra.CodegenResult].fail(
                f"managed file changed after planning: {file.path}"
            )
        written: tuple[Path, ...] = ()
        u.Cli.info(f"stage=apply changed={int(file.changed)}")
        if file.changed:
            u.Cli.emit_raw(f"  write [1/1] {file.path}\n")
            promoted = u.Cli.atomic_write_text_file(file.path, file.rendered)
            if promoted.failure:
                return r[m.Infra.CodegenResult].fail(
                    promoted.error
                    or f"stage=apply path={file.path}: atomic write failed"
                )
            written = (file.path,)
        materialized = self._authenticated_managed_file(request.root, file.path)
        if materialized.failure:
            return r[m.Infra.CodegenResult].fail(
                materialized.error or "Makefile readback failed"
            )
        materialized_content, _materialized_identity = materialized.value
        if (
            materialized_content != file.rendered
            or u.Cli.sha256_content(materialized_content) != file.expected_sha256
        ):
            return r[m.Infra.CodegenResult].fail(
                f"stage=verify-fixed-point: material Makefile differs: {file.path}"
            )
        u.Cli.info("stage=verify-fixed-point")
        verified = self.plan(request)
        if verified.failure:
            return r[m.Infra.CodegenResult].fail(
                verified.error
                or "stage=verify-fixed-point: Makefile verification failed"
            )
        residual = tuple(item for item in verified.value.files if item.changed)
        if residual:
            paths = ", ".join(str(item.path) for item in residual)
            return r[m.Infra.CodegenResult].fail(
                f"codegen apply did not reach a fixed point: {paths}"
            )
        return r[m.Infra.CodegenResult].ok(
            m.Infra.CodegenResult(plan=verified.value, written_files=written)
        )

    def _apply_makefile_plan(
        self, request: m.Infra.CodegenConformRequest, planned: m.Infra.CodegenPlan
    ) -> p.Result[m.Infra.CodegenResult]:
        """Authenticate, promote and materially verify one Makefile under its lock."""
        lock_path = self.conform_transaction_lock_path(request.root)
        ensured = u.Cli.ensure_dir(lock_path.parent)
        if ensured.failure:
            return r[m.Infra.CodegenResult].fail(
                ensured.error
                or f"codegen transaction directory failed: {lock_path.parent}"
            )
        try:
            u.Cli.info(f"stage=wait-transaction-lock path={lock_path}")
            with UnixFileLock(
                lock_path,
                timeout=c.Infra.TIMEOUT_DEFAULT,
                fallback_to_soft=False,
                context_error_policy="group",
                close_error_policy="raise",
            ):
                return self._apply_makefile_locked(request, planned)
        except Exception as exc:  # ruff:ignore[blind-except]  Why: public Result boundary preserves any lock protocol or cleanup failure.
            return r[m.Infra.CodegenResult].fail_op(
                f"Makefile transaction lock ({lock_path})", exc
            )

    @override
    def execute(self) -> p.Result[m.Infra.CodegenResult]:
        """Run check or apply and require a verified fixed point."""
        request = self.request or m.Infra.CodegenConformRequest(
            root=self.repository_root
        )
        if (
            c.Infra.CodegenConformSurface(request.what)
            is c.Infra.CodegenConformSurface.ALL
            and self.initial_workspace is None
        ):
            return self._execute_managed(request)
        return self._execute_plan(request)

    def _execute_plan(
        self, request: m.Infra.CodegenConformRequest
    ) -> p.Result[m.Infra.CodegenResult]:
        """Execute a non-toolchain conform surface without widening its scope."""
        u.Cli.header("Codegen Conform")
        u.Cli.info(
            f"stage=plan mode={request.mode} scope={request.scope} "
            f"what={request.what} root={request.root}"
        )
        planned = self.plan(request)
        if planned.failure:
            return r[m.Infra.CodegenResult].fail(
                planned.error or "codegen conform planning failed"
            )
        plan = planned.value
        mode = c.Infra.CodegenConformMode(request.mode)
        ancestry = self._validate_ancestry(plan)
        if ancestry.failure:
            return r[m.Infra.CodegenResult].from_failure(ancestry)
        makefile_only = (
            c.Infra.CodegenConformSurface(request.what)
            is c.Infra.CodegenConformSurface.MAKEFILE
        )
        if mode is c.Infra.CodegenConformMode.APPLY and makefile_only:
            # The dispatcher is promoted only under its transaction lock with a
            # verify-before-publish replan, on every route -- the bootstrap
            # `make setup` refresh included, not just the managed ALL surface.
            return self._apply_makefile_plan(request, plan)
        changed = tuple(file for file in plan.files if file.changed)
        if mode is c.Infra.CodegenConformMode.CHECK:
            if changed:
                paths = ", ".join(str(file.path) for file in changed)
                return r[m.Infra.CodegenResult].fail(f"codegen drift detected: {paths}")
            return r[m.Infra.CodegenResult].ok(m.Infra.CodegenResult(plan=plan))
        written = self._apply_files(plan, changed)
        if written.failure:
            return r[m.Infra.CodegenResult].from_failure(written)
        u.Cli.info("stage=verify-fixed-point")
        verified = self.plan(request)
        if verified.failure:
            return r[m.Infra.CodegenResult].fail(
                verified.error
                or "stage=verify-fixed-point: post-apply conform verification failed"
            )
        verified_plan = verified.value
        residual = tuple(file for file in verified_plan.files if file.changed)
        if residual:
            paths = ", ".join(str(file.path) for file in residual)
            return r[m.Infra.CodegenResult].fail(
                f"codegen apply did not reach a fixed point: {paths}"
            )
        return r[m.Infra.CodegenResult].ok(
            m.Infra.CodegenResult(plan=verified_plan, written_files=written.value)
        )

    def _execute_managed(
        self, request: m.Infra.CodegenConformRequest
    ) -> p.Result[m.Infra.CodegenResult]:
        """Run complete conformance inside the sole generation lock."""
        mode = c.Infra.CodegenConformMode(request.mode)
        mise_owner = FlextInfraCodegenMiseArtifacts(
            repository_root=request.root,
            apply_changes=mode is c.Infra.CodegenConformMode.APPLY,
            check_only=mode is c.Infra.CodegenConformMode.CHECK,
        )
        transaction = FlextInfraCodegenMiseArtifactTransaction(mise_owner)
        return transaction.run_locked(
            prepare=mode is c.Infra.CodegenConformMode.APPLY,
            operation=lambda scope_root: self._execute_managed_locked(
                request, scope_root, transaction
            ),
        )

    def _execute_managed_locked(
        self,
        request: m.Infra.CodegenConformRequest,
        scope_root: Path,
        transaction: FlextInfraCodegenMiseArtifactTransaction,
    ) -> p.Result[m.Infra.CodegenResult]:
        """Plan, publish, and validate one complete locked generation cycle."""
        u.Cli.header("Codegen Conform")
        u.Cli.info(
            f"stage=plan mode={request.mode} scope={request.scope} "
            f"what={request.what} root={request.root} lock_scope={scope_root}"
        )
        planned = self.plan(request)
        if planned.failure:
            return r[m.Infra.CodegenResult].fail(
                planned.error or "codegen conform planning failed"
            )
        plan = planned.value
        mode = c.Infra.CodegenConformMode(request.mode)
        ancestry = self._validate_ancestry(plan)
        if ancestry.failure:
            return r[m.Infra.CodegenResult].from_failure(ancestry)
        config_plans = self._mise_config_plans(plan)
        if config_plans.failure:
            return r[m.Infra.CodegenResult].from_failure(config_plans)
        makefile_only = (
            c.Infra.CodegenConformSurface(request.what)
            is c.Infra.CodegenConformSurface.MAKEFILE
        )
        mode = c.Infra.CodegenConformMode(request.mode)
        if mode is c.Infra.CodegenConformMode.APPLY and makefile_only:
            return self._apply_makefile_plan(request, plan)
        changed = tuple(file for file in plan.files if file.changed)
        if mode is c.Infra.CodegenConformMode.CHECK:
            reality = transaction.validate_locked(scope_root, config_plans.value)
            if reality.failure:
                return r[m.Infra.CodegenResult].from_failure(reality)
            if changed:
                paths = ", ".join(str(file.path) for file in changed)
                return r[m.Infra.CodegenResult].fail(f"codegen drift detected: {paths}")
            return r[m.Infra.CodegenResult].ok(m.Infra.CodegenResult(plan=plan))
        config_paths = {item.path for item in config_plans.value}
        ordinary = tuple(file for file in changed if file.path not in config_paths)
        written = self._apply_files(plan, ordinary)
        if written.failure:
            return r[m.Infra.CodegenResult].from_failure(written)
        published = transaction.publish_locked(scope_root, config_plans.value)
        if published.failure:
            return r[m.Infra.CodegenResult].from_failure(published)
        u.Cli.info("stage=verify-fixed-point")
        verified = self.plan(request)
        if verified.failure:
            return r[m.Infra.CodegenResult].fail(
                verified.error
                or "stage=verify-fixed-point: post-apply conform verification failed"
            )
        verified_plan = verified.value
        verified_configs = self._mise_config_plans(verified_plan)
        if verified_configs.failure:
            return r[m.Infra.CodegenResult].from_failure(verified_configs)
        reality = transaction.validate_locked(scope_root, verified_configs.value)
        if reality.failure:
            return r[m.Infra.CodegenResult].from_failure(reality)
        residual = tuple(file for file in verified_plan.files if file.changed)
        if residual:
            paths = ", ".join(str(file.path) for file in residual)
            return r[m.Infra.CodegenResult].fail(
                f"codegen apply did not reach a fixed point: {paths}"
            )
        return r[m.Infra.CodegenResult].ok(
            m.Infra.CodegenResult(
                plan=verified_plan, written_files=(*written.value, *published.value)
            )
        )

    @staticmethod
    def _mise_config_plans(
        plan: m.Infra.CodegenPlan,
    ) -> p.Result[tuple[m.Infra.CodegenFilePlan, ...]]:
        """Select one planned Mise configuration for each selected repository."""
        expected = tuple(
            environment.project_root / c.Infra.MISE_TOML_FILENAME
            for environment in plan.uv_environments
        )
        by_path = {item.path: item for item in plan.files if item.path in expected}
        if (
            len(expected) != len(plan.repositories)
            or len(set(expected)) != len(expected)
            or len(by_path) != len(expected)
        ):
            return r[tuple[m.Infra.CodegenFilePlan, ...]].fail(
                "conform plan must contain one Mise configuration per repository"
            )
        return r[tuple[m.Infra.CodegenFilePlan, ...]].ok(
            tuple(by_path[path] for path in expected)
        )

    @staticmethod
    def _validate_ancestry(plan: m.Infra.CodegenPlan) -> p.Result[bool]:
        """Reject every governed ref outside its repository integration line."""
        violations = tuple(
            (ancestry, reference)
            for ancestry in plan.branch_ancestry
            for reference in ancestry.references
            if reference.ancestor is False
        )
        if not violations:
            return r[bool].ok(True)
        details = "; ".join(
            (
                f"{reference.reference}@{reference.sha} does not descend from "
                f"{ancestry.baseline_reference}@{ancestry.baseline_sha}"
            )
            for ancestry, reference in violations
        )
        return r[bool].fail(f"governed branch ancestry violations: {details}")

    @staticmethod
    def _apply_files(
        plan: m.Infra.CodegenPlan, changed: tuple[m.Infra.CodegenFilePlan, ...]
    ) -> p.Result[tuple[Path, ...]]:
        """Apply one prevalidated set of ordinary non-toolchain files."""
        written: list[Path] = []
        total_changed = len(changed)
        u.Cli.info(f"stage=apply changed={total_changed}")
        for write_index, file in enumerate(changed, start=1):
            u.Cli.emit_raw(f"  write [{write_index}/{total_changed}] {file.path}\n")
            target = file.path.expanduser().resolve()
            if not target.is_relative_to(plan.request.root.expanduser().resolve()):
                return r[tuple[Path, ...]].fail(
                    f"managed path escapes repository root: {file.path}"
                )
            current_sha = ""
            if target.is_file():
                current = u.Cli.files_read_text(target)
                if current.failure:
                    return r[tuple[Path, ...]].fail(
                        current.error or f"managed file authentication failed: {target}"
                    )
                current_sha = u.Cli.sha256_content(current.value)
            elif target.exists():
                return r[tuple[Path, ...]].fail(
                    f"managed destination is not a regular file: {target}"
                )
            if current_sha != file.current_sha256:
                return r[tuple[Path, ...]].fail(
                    f"managed file changed after planning: {target}"
                )
            if file.absent:
                if target.exists():
                    if not target.is_file():
                        return r[tuple[Path, ...]].fail(
                            f"absent path is not a regular file: {target}"
                        )
                    removed = r.create_from_callable(
                        lambda path=target: (path.unlink(), path)[1],
                        error_code="E_CODEGEN_ABSENT_UNLINK",
                    )
                    if removed.failure:
                        return r[tuple[Path, ...]].fail(
                            removed.error or f"absent path unlink failed: {target}"
                        )
                written.append(file.path)
                continue
            result = u.Cli.atomic_write_text_file(file.path, file.rendered)
            if result.failure:
                return r[tuple[Path, ...]].fail(
                    result.error
                    or (
                        f"stage=apply position={write_index}/{total_changed} "
                        f"path={file.path}: atomic write failed"
                    )
                )
            if file.executable is not None:
                current_mode = file.path.stat().st_mode
                executable_bits = 0o111
                file.path.chmod(
                    current_mode | executable_bits
                    if file.executable
                    else current_mode & ~executable_bits
                )
            written.append(file.path)
        return r[tuple[Path, ...]].ok(tuple(written))

    def plan(
        self, request: m.Infra.CodegenConformRequest
    ) -> p.Result[m.Infra.CodegenPlan]:
        """Build and validate the complete selection without writing."""
        config_spec = config.Infra.codegen
        surface = c.Infra.CodegenConformSurface(request.what)
        contract = self._surface_contract(surface)
        if surface is c.Infra.CodegenConformSurface.MAKEFILE:
            return self._plan_declared_makefile(request, config_spec, contract)
        root = request.root.expanduser().resolve()
        repository_root = root
        workspace = self.initial_workspace
        if workspace is None:
            workspace_result = FlextInfraWorkspaceDetector.load_workspace_spec(
                repository_root
            )
            if workspace_result.failure:
                return r[m.Infra.CodegenPlan].fail(
                    workspace_result.error or "workspace topology load failed"
                )
            workspace = workspace_result.value
        current_repository = workspace.repository
        if self.initial_workspace is None:
            current_target_result = FlextInfraWorkspaceDetector.conform_target(
                root, workspace
            )
            if current_target_result.failure:
                return r[m.Infra.CodegenPlan].fail(
                    current_target_result.error
                    or "repository conformance target resolution failed"
                )
            current_target = current_target_result.value
            current_repository = current_target.repository
        else:
            providers = tuple(
                item
                for item in config_spec.providers
                if item.name == current_repository.provider
            )
            if len(providers) != 1:
                return r[m.Infra.CodegenPlan].fail(
                    "repository provider must resolve exactly once: "
                    f"{current_repository.provider}"
                )
            (provider,) = providers
            # The provider default is the fallback, never the answer: this
            # repository's own published integration branch decides.
            baseline_result = u.Infra.repository_baseline_branch(
                root,
                fallback=provider.branch,
                preference=config_spec.branch_policy.integration_branch_preference,
            )
            if baseline_result.failure:
                return r[m.Infra.CodegenPlan].fail(
                    baseline_result.error
                    or f"integration baseline resolution failed: {root}"
                )
            current_target = m.Infra.RepositoryConformTarget(
                repository=current_repository,
                root=root,
                make_profile=current_repository.role,
                beads=workspace.beads,
                canonical_project_name=current_repository.distribution,
                baseline_branch=baseline_result.value,
                ci_enabled=True,
                external_dependency_paths=workspace.external_dependency_paths,
                technical_branch_patterns=(
                    config_spec.branch_policy.technical_branch_patterns
                ),
                governed_branch_patterns=(
                    config_spec.branch_policy.governed_branch_patterns
                ),
            )
        selected_result = self._select_repositories(
            request, workspace, current_repository
        )
        if selected_result.failure:
            return r[m.Infra.CodegenPlan].fail(
                selected_result.error or "repository selection failed"
            )
        selected = selected_result.value
        files: list[m.Infra.CodegenFilePlan] = []
        environments: list[m.Infra.UvEnvironmentPlan] = []
        ancestry_plans: list[m.Infra.BranchAncestryPlan] = []
        total_repositories = len(selected)
        u.Cli.info(f"stage=plan repositories={total_repositories}")
        for repository_index, repository in enumerate(selected, start=1):
            repository_started = time.monotonic()
            u.Cli.progress(
                repository_index, total_repositories, repository.name, "conform"
            )
            is_current_repository = repository.name == current_target.repository.name
            if is_current_repository:
                repository_root = current_target.root
                if repository_root != root:
                    return r[m.Infra.CodegenPlan].fail(
                        "current conformance target differs from the requested root: "
                        f"{repository_root} != {root}"
                    )
            else:
                # The governing root is the requested checkout, never the
                # previous iteration's member: resolving the second declared
                # repository against the first produced <root>/alpha/beta.
                repository_root_result = self._repository_root(
                    root, workspace, repository
                )
                if repository_root_result.failure:
                    return r[m.Infra.CodegenPlan].fail(
                        repository_root_result.error
                        or f"invalid repository root: {repository.name}"
                    )
                repository_root = repository_root_result.value
            if repository_root.exists() and not repository_root.is_dir():
                return r[m.Infra.CodegenPlan].fail(
                    f"declared repository path is not a directory: {repository_root}"
                )
            if not repository_root.is_dir() and self.initial_workspace is None:
                return r[m.Infra.CodegenPlan].fail(
                    f"declared repository checkout is missing: {repository_root}"
                )
            if is_current_repository:
                target = current_target
                local_workspace = workspace
            else:
                local_workspace_result = (
                    FlextInfraWorkspaceDetector.load_workspace_spec(repository_root)
                )
                if local_workspace_result.failure:
                    return r[m.Infra.CodegenPlan].fail(
                        local_workspace_result.error
                        or f"repository topology load failed: {repository_root}"
                    )
                local_workspace = local_workspace_result.value
                target_result = FlextInfraWorkspaceDetector.conform_target(
                    repository_root, local_workspace
                )
                if target_result.failure:
                    return r[m.Infra.CodegenPlan].fail(
                        target_result.error
                        or f"repository target resolution failed: {repository_root}"
                    )
                target = target_result.value
            if (
                self.initial_workspace is not None
                and repository.name == workspace.repository.name
            ):
                repository_plan = self._plan_scaffold_repository(
                    root=repository_root,
                    repository=target.repository,
                    target=target,
                    workspace=local_workspace,
                    codegen=config_spec,
                    contract=contract,
                )
            else:
                repository_plan = self._plan_existing_repository(
                    root=repository_root,
                    repository_root=repository_root,
                    repository=target.repository,
                    target=target,
                    workspace=local_workspace,
                    codegen=config_spec,
                    contract=contract,
                )
            if repository_plan.failure:
                return r[m.Infra.CodegenPlan].fail(
                    repository_plan.error
                    or (
                        f"stage=plan position={repository_index}/"
                        f"{total_repositories} repository={repository.name}: "
                        f"repository planning failed: {repository_root}"
                    )
                )
            governed = self._complete_governed_plans(
                repository_root,
                repository_plan.value,
                config_spec,
                contract,
                profile=target.make_profile,
                workspace=local_workspace,
            )
            if governed.failure:
                return r[m.Infra.CodegenPlan].fail(
                    governed.error
                    or (
                        f"stage=plan position={repository_index}/"
                        f"{total_repositories} repository={repository.name}: "
                        f"artifact ownership planning failed: {repository_root}"
                    )
                )
            files.extend(governed.value)
            if contract.complete_governed:
                retired = self.retired_projection_plans(
                    repository_root, target.make_profile
                )
                if retired.failure:
                    return r[m.Infra.CodegenPlan].fail(
                        retired.error
                        or (
                            f"stage=plan position={repository_index}/"
                            f"{total_repositories} repository={repository.name}: "
                            "profile-excluded projection planning failed"
                        )
                    )
                governed_paths = {item.path for item in governed.value}
                files.extend(
                    item for item in retired.value if item.path not in governed_paths
                )
            environments.append(
                self._uv_environment_plan(
                    root=repository_root,
                    repository_root=repository_root,
                    target=target,
                    workspace=local_workspace,
                    config=config_spec,
                )
            )
            if self.initial_workspace is None or (repository_root / ".git").exists():
                ancestry_result = self._branch_ancestry_plan(target)
                if ancestry_result.failure:
                    return r[m.Infra.CodegenPlan].fail(
                        ancestry_result.error
                        or (
                            f"stage=plan position={repository_index}/"
                            f"{total_repositories} repository={repository.name}: "
                            f"branch ancestry inventory failed: {repository_root}"
                        )
                    )
                ancestry_plans.append(ancestry_result.value)
            u.Cli.status(
                "conform",
                repository.name,
                result=True,
                elapsed=time.monotonic() - repository_started,
            )
        return r[m.Infra.CodegenPlan].ok(
            m.Infra.CodegenPlan(
                request=request,
                repositories=selected,
                workspace=workspace,
                make_spec=config_spec.make,
                uv_environments=tuple(environments),
                branch_ancestry=tuple(ancestry_plans),
                files=tuple(files),
            )
        )

    def _plan_declared_makefile(
        self,
        request: m.Infra.CodegenConformRequest,
        codegen: m.Infra.CodegenConfigSpec,
        contract: m.Infra.CodegenConformSurfaceContract,
    ) -> p.Result[m.Infra.CodegenPlan]:
        """Plan one Makefile from declared inputs without operational discovery."""
        if (
            c.Infra.CodegenConformScope(request.scope)
            is not c.Infra.CodegenConformScope.SELF
        ):
            return r[m.Infra.CodegenPlan].fail(
                "makefile conformance requires scope=self"
            )
        root = request.root.expanduser().resolve()
        # The declared Makefile projection reads declarations only: the
        # repository's own config/workspace.yaml when it ships one, else its
        # PEP 621 metadata. Git, Beads, mise and uv are never consulted here,
        # so `make setup` can refresh a stale dispatcher in a checkout that has
        # no initialized topology yet.
        workspace_result = (
            r[m.Infra.WorkspaceSpec].ok(self.initial_workspace)
            if self.initial_workspace is not None
            else FlextInfraWorkspaceDetector.load_projection_workspace_spec(root)
        )
        if workspace_result.failure:
            return r[m.Infra.CodegenPlan].fail(
                workspace_result.error or "declared workspace manifest load failed"
            )
        workspace = workspace_result.value
        target_result = FlextInfraWorkspaceDetector.declared_conform_target(
            root, workspace
        )
        if target_result.failure:
            return r[m.Infra.CodegenPlan].fail(
                target_result.error or "declared conformance target is invalid"
            )
        target = target_result.value
        repository = target.repository
        u.Cli.info("stage=plan repositories=1")
        u.Cli.progress(1, 1, repository.name, "conform")
        managed_artifacts = FlextInfraUtilitiesProjectManagedArtifacts.snapshot_project_managed_artifacts(
            root
        )
        if managed_artifacts.failure:
            return r[m.Infra.CodegenPlan].from_failure(managed_artifacts)
        planned = self._plan_existing_templates(
            root=root,
            repository=repository,
            target=target,
            workspace=workspace,
            codegen=codegen,
            tooling_runtime=None,
            contract=contract,
            managed_artifacts=managed_artifacts.value,
        )
        if planned.failure:
            return r[m.Infra.CodegenPlan].fail(
                planned.error or f"Makefile planning failed: {root}"
            )
        governed = self._complete_governed_plans(
            root,
            planned.value,
            codegen,
            contract,
            profile=target.make_profile,
            workspace=workspace,
        )
        if governed.failure:
            return r[m.Infra.CodegenPlan].fail(
                governed.error or f"Makefile ownership planning failed: {root}"
            )
        return r[m.Infra.CodegenPlan].ok(
            m.Infra.CodegenPlan(
                request=request,
                repositories=(repository,),
                workspace=workspace,
                make_spec=codegen.make,
                uv_environments=(),
                branch_ancestry=(),
                files=tuple(governed.value),
            )
        )

    @staticmethod
    def _complete_governed_plans(
        root: Path,
        planned: t.SequenceOf[m.Infra.CodegenFilePlan],
        codegen: m.Infra.CodegenConfigSpec,
        contract: m.Infra.CodegenConformSurfaceContract,
        *,
        profile: c.Infra.MakeProfile,
        workspace: m.Infra.WorkspaceSpec | None = None,
    ) -> p.Result[t.SequenceOf[m.Infra.CodegenFilePlan]]:
        """Attach ownership metadata and represent every governed root artifact.

        Only the ``ALL`` surface completes the full governed set; the
        pyproject-scoped surfaces (``DEPENDENCIES``/``PYPROJECT``) keep the plan
        restricted to what their own planners already produced.
        """
        governed_by_path = {item.path: item for item in codegen.managed_files}
        completed: list[m.Infra.CodegenFilePlan] = []
        represented: set[Path] = set()
        represented_indexes: dict[Path, int] = {}
        for file in planned:
            relative = file.path.relative_to(root)
            governed = governed_by_path.get(relative)
            if governed is None:
                completed.append(file)
                continue
            represented.add(relative)
            governed_file = file.model_copy(
                update={
                    "owner": governed.owner,
                    "policy": governed.policy,
                    "executable": governed.executable,
                }
            )
            if relative in represented_indexes:
                completed[represented_indexes[relative]] = governed_file
            else:
                represented_indexes[relative] = len(completed)
                completed.append(governed_file)
        if not contract.complete_governed:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(tuple(completed))
        for relative, governed in governed_by_path.items():
            if relative in represented:
                continue
            path = root / relative
            if path.exists() and not path.is_file():
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"governed artifact is not a regular file: {path}"
                )
            entry_profiles = tuple(
                entry.profiles
                for entry in codegen.templates.entries
                if entry.destination == relative.as_posix()
            )
            if entry_profiles:
                allowed = {item for profiles in entry_profiles for item in profiles}
                if profile not in allowed:
                    continue
            current = ""
            if path.is_file():
                read = u.Cli.files_read_text(path)
                if read.failure:
                    return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                        read.error or f"governed artifact read failed: {path}"
                    )
                current = read.value
            digest = u.Cli.sha256_content(current) if path.is_file() else ""
            if (
                governed.policy == "merge"
                and governed.owner == c.Infra.CODEGEN_OWNER_VSCODE
            ):
                # Owner-merge dispatch: owners with a canonical document merge
                # (vscode settings today) produce their rendered content here.
                merged = FlextInfraCodegen.render_vscode_settings(root)
                if merged.failure:
                    return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                        merged.error or f"vscode settings merge failed: {path}"
                    )
                if merged.value != current:
                    completed.append(
                        m.Infra.CodegenFilePlan(
                            path=path,
                            owner=governed.owner,
                            policy=governed.policy,
                            rendered=merged.value,
                            expected_sha256=u.Cli.sha256_content(merged.value),
                            current_sha256=digest,
                            changed=True,
                        )
                    )
                    continue
            if governed.policy == "merge" and relative.as_posix() == c.Infra.GITIGNORE:
                # The canonical .gitignore body is rendered
                # from the same base/gitignore.j2 + computed
                # CodegenConfigSpec.gitignore_sections used by `codegen new` —
                # ONE render mechanism derived from the artifact SSOT.
                # Per-project exception fields land in their typed owner.
                rendered_gitignore = FlextInfraCodegenConform._render_gitignore(
                    codegen,
                    profile=profile,
                    project_name=root.name,
                    workspace=workspace,
                    project_dir=root,
                )
                if rendered_gitignore.failure:
                    return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                        rendered_gitignore.error or f"gitignore render failed: {path}"
                    )
                if rendered_gitignore.value != current:
                    completed.append(
                        m.Infra.CodegenFilePlan(
                            path=path,
                            owner=governed.owner,
                            policy=governed.policy,
                            rendered=rendered_gitignore.value,
                            expected_sha256=u.Cli.sha256_content(
                                rendered_gitignore.value
                            ),
                            current_sha256=digest,
                            changed=True,
                        )
                    )
                    continue
            completed.append(
                m.Infra.CodegenFilePlan(
                    path=path,
                    owner=governed.owner,
                    policy=governed.policy,
                    rendered=current,
                    expected_sha256=digest or u.Cli.sha256_content(current),
                    current_sha256=digest,
                    changed=False,
                )
            )
        return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(tuple(completed))

    @staticmethod
    def _package_root() -> Path:
        """Return the installed flext-infra package root."""
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def render_project_gitignore(
        codegen: m.Infra.CodegenConfigSpec,
        *,
        profile: c.Infra.MakeProfile,
        project_name: str,
        workspace: m.Infra.WorkspaceSpec | None = None,
        project_dir: Path | None = None,
    ) -> p.Result[str]:
        """Render the canonical ``.gitignore`` for one named project.

        Public seam consumed by the layout engine: per-project
        layout ``gitignore_additions`` from the layout SSOT and the
        repository-owned ``ManagedArtifacts.Gitignore.patterns`` read from
        ``project_dir`` are appended as trailing derived sections so conform
        and layout never diverge.
        """
        return FlextInfraCodegenConform._render_gitignore(
            codegen,
            profile=profile,
            project_name=project_name,
            workspace=workspace,
            project_dir=project_dir,
        )

    @staticmethod
    def _render_gitignore(
        codegen: m.Infra.CodegenConfigSpec,
        *,
        profile: c.Infra.MakeProfile,
        project_name: str | None = None,
        workspace: m.Infra.WorkspaceSpec | None = None,
        project_dir: Path | None = None,
    ) -> p.Result[str]:
        """Render the canonical ``.gitignore`` body via the single template.

        ``codegen new`` renders ``base/gitignore.j2`` with
        the full project context; conform renders the same template with the
        codegen config — both consume the same computed ``gitignore_sections``
        projection, so the body is byte-identical.
        """
        entry = next(
            (
                item
                for item in codegen.templates.entries
                if item.destination == c.Infra.GITIGNORE
            ),
            None,
        )
        if entry is None:
            return r[str].fail(
                "gitignore template is missing from codegen configuration"
            )
        templates_root = (
            FlextInfraCodegenConform._package_root()
            / "templates"
            / codegen.templates.root
        ).resolve()
        sections = [
            section
            for section in codegen.gitignore_sections
            if not section.profiles or profile in section.profiles
        ]
        # The deny-all root policy (`/*` + `/*/`) would swallow every governed
        # declared_repository directory, so their whitelist is DERIVED from the live workspace
        # topology instead of a hardcoded name glob: declaring a declared_repository in
        # local .gitmodules is the single source that makes it trackable.
        # Nested paths need every ancestor unignored, otherwise git never
        # descends far enough to reach the declared_repository itself.
        member_patterns: list[str] = []
        if workspace is not None:
            for declared_repository in workspace.declared_repositories:
                parts = declared_repository.path.as_posix().strip("/").split("/")
                # Every ancestor is unignored so git can descend into the
                # declared_repository, then its contents are unignored with the `/**` form.
                prefixes = [
                    "/".join(parts[:depth]) for depth in range(1, len(parts) + 1)
                ]
                candidates = [f"!/{prefix}/" for prefix in prefixes]
                candidates.append(f"!/{prefixes[-1]}/**")
                for pattern in candidates:
                    if pattern not in member_patterns:
                        member_patterns.append(pattern)
        if member_patterns:
            sections.append(
                m.Infra.ScaffoldGitignoreSectionSpec(
                    name="WHITELIST: governed workspace declared_repositories (derived)",
                    patterns=tuple(member_patterns),
                )
            )
        if project_name is not None:
            override = codegen.layout.project_overrides.get(project_name)
            if override is not None and override.gitignore_additions:
                sections.append(
                    m.Infra.ScaffoldGitignoreSectionSpec(
                        name=c.Infra.GITIGNORE_LAYOUT_SECTION_NAME,
                        patterns=override.gitignore_additions,
                    )
                )
        if project_dir is not None:
            # The repository owns the ignore patterns the fleet scaffold cannot
            # know (local caches, generated runtime state); they are declared in
            # its own config/*.yaml and appended as one derived section.
            resolved = FlextInfraUtilitiesProjectManagedArtifacts.load_project_managed_artifacts(
                project_dir
            )
            if resolved.failure:
                return r[str].fail(
                    resolved.error or f"project artifact load failed: {project_dir}"
                )
            project_patterns = resolved.value.artifacts.Gitignore.patterns
            if project_patterns:
                sections.append(
                    m.Infra.ScaffoldGitignoreSectionSpec(
                        name=c.Infra.GITIGNORE_PROJECT_SECTION_NAME,
                        patterns=project_patterns,
                    )
                )
        context = m.Infra.GitignoreRenderSpec(gitignore_sections=tuple(sections))
        return u.Cli.template_render(templates_root / entry.source, context)

    @staticmethod
    def _select_repositories(
        request: m.Infra.CodegenConformRequest,
        workspace: m.Infra.WorkspaceSpec,
        current_repository: m.Infra.RepositoryRef,
    ) -> p.Result[tuple[m.Infra.RepositoryRef, ...]]:
        """Resolve self/declared_repositories/all from the local read-only topology."""
        scope = c.Infra.CodegenConformScope(request.scope)
        if scope is c.Infra.CodegenConformScope.SELF:
            selected = (current_repository,)
        elif scope is c.Infra.CodegenConformScope.DECLARED:
            if not workspace.declared_repositories:
                return r[tuple[m.Infra.RepositoryRef, ...]].fail(
                    "declared_repositories scope requires local .gitmodules entries"
                )
            selected = tuple(workspace.declared_repositories)
        else:
            selected = (workspace.repository, *workspace.declared_repositories)
        mutable = tuple(
            repository
            for repository in selected
            if repository.codegen is not c.Infra.CodegenKind.NONE
            and not repository.read_only
        )
        if not mutable:
            return r[tuple[m.Infra.RepositoryRef, ...]].fail(
                "selected repositories do not permit code generation"
            )
        return r[tuple[m.Infra.RepositoryRef, ...]].ok(mutable)

    @staticmethod
    def _repository_root(
        root: Path, workspace: p.Infra.WorkspaceSpec, repository: p.Infra.RepositoryRef
    ) -> p.Result[Path]:
        """Resolve one declared checkout without escaping its workspace owner."""
        if repository.name == workspace.repository.name:
            return r[Path].ok(root)
        resolved_root = root.resolve()
        resolved: Path = (resolved_root / repository.path).resolve()
        if not resolved.is_relative_to(resolved_root):
            return r[Path].fail(
                "declared repository path escapes repository root: "
                f"{repository.path.as_posix()}"
            )
        return r[Path].ok(resolved)

    @staticmethod
    def _scaffold_python_dirs(
        entries: t.SequenceOf[p.Infra.TemplateEntrySpec], profile: c.Infra.MakeProfile
    ) -> t.StrSequence:
        """Return Python roots the selected scaffold manifest actually creates."""
        # Derive future roots from both
        # declarative owners so scaffold and existing-tree discovery converge.
        generated_roots = {
            Path(entry.destination).parts[0]
            for entry in entries
            if profile in entry.profiles
            and entry.delegate == "render"
            and Path(entry.destination).parts
        }
        return tuple(
            directory
            for directory in config.Infra.tooling.tools.pyright.path_rules.env_dirs
            if directory in generated_roots
        )

    def _plan_scaffold_repository(
        self,
        *,
        root: Path,
        repository: m.Infra.RepositoryRef,
        target: m.Infra.RepositoryConformTarget,
        workspace: m.Infra.WorkspaceSpec,
        codegen: m.Infra.CodegenConfigSpec,
        contract: m.Infra.CodegenConformSurfaceContract,
    ) -> p.Result[t.SequenceOf[m.Infra.CodegenFilePlan]]:
        """Render the complete scaffold for ``codegen new`` only."""
        project = workspace.project
        if project is None:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                f"scaffold workspace has no project metadata: {workspace.name}"
            )
        profile = target.make_profile
        pyproject = root / c.Infra.PYPROJECT_FILENAME
        # New and existing repositories share the exact same
        # root-scoped modernizer pipeline, so first generation is a fixed point.
        # A declared declared_repository consumes the repository root
        # tooling profile even before the atomic scaffold creates files on disk.
        tooling_root = target.root
        modernizer = FlextInfraPyprojectModernizer(
            repository_root=tooling_root, skip_check=True
        )
        analysis_exclusions = tuple(
            path.as_posix()
            for path in (
                *target.external_dependency_paths,
                *workspace.external_dependency_paths,
            )
        )
        declared_python_dirs = self._scaffold_python_dirs(
            codegen.templates.entries, profile
        )
        # Why (flext-6itas.4): a scaffold's declared roots are the complete
        # future topology only for a declared_repository/standalone target; a workspace
        # root aggregates declared_repository trees it has not declared here.
        declared_python_dirs_are_complete = profile is not c.Infra.MakeProfile.WORKSPACE
        tooling_result = modernizer.resolve_tooling_context(
            project_name=repository.distribution,
            package_name=project.package_name,
            path=pyproject,
            declared_python_dirs=declared_python_dirs,
            declared_python_dirs_are_complete=declared_python_dirs_are_complete,
            analysis_exclusions=analysis_exclusions,
        )
        if tooling_result.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                tooling_result.error or f"tooling render failed: {pyproject}"
            )
        context_result = self._project_render_context(
            repository,
            target,
            workspace,
            codegen,
            tooling_runtime=tooling_result.value,
            repository_root=pyproject.parent,
        )
        if context_result.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                context_result.error or "project render context is invalid"
            )
        context = context_result.value
        uv_exclude_dependencies = self._routed_uv_exclude_dependencies(
            repository, target, workspace, codegen
        )
        planned: list[m.Infra.CodegenFilePlan] = []
        templates_root = (
            self._package_root() / "templates" / codegen.templates.root
        ).resolve()
        seen_destinations: set[str] = set()
        for entry in codegen.templates.entries:
            if profile not in entry.profiles:
                continue
            if (
                contract.destinations is not None
                and entry.destination not in contract.destinations
            ):
                continue
            source = (templates_root / entry.source).resolve()
            if not source.is_relative_to(templates_root) or not source.is_file():
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"template source is missing or escapes its root: {entry.source}"
                )
            destination = entry.destination.format(
                package_name=context.package_name, ns=context.ns
            )
            relative = Path(destination)
            if relative.is_absolute() or ".." in relative.parts:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"template destination escapes repository root: {destination}"
                )
            if destination in seen_destinations:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"duplicate template destination: {destination}"
                )
            seen_destinations.add(destination)
            path = root / relative
            if path.exists() and not path.is_file():
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"template destination is not a regular file: {path}"
                )
            for parent in path.parents:
                if parent == root:
                    break
                if parent.exists() and not parent.is_dir():
                    return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                        f"template destination parent is not a directory: {parent}"
                    )
        for entry in codegen.templates.entries:
            if profile not in entry.profiles:
                continue
            if (
                contract.destinations is not None
                and entry.destination not in contract.destinations
            ):
                continue
            if not contract.delegates:
                continue
            # One formatted path governs validation and planning.
            destination = entry.destination.format(
                package_name=context.package_name, ns=context.ns
            )
            if entry.delegate != "render":
                continue
            if destination == c.Infra.PYPROJECT_FILENAME:
                continue
            if self._beads_projection_is_inherited(destination, root):
                # A tracked ledger route inherits the workspace ledger; planning
                # a member-local projection would create a second identity, and
                # in a standalone clone the route dangles outside the checkout.
                continue
            if (
                destination == c.Infra.BEADS_METADATA_RELPATH
                and not (root / destination).is_file()
            ):
                # Why (flext-l2296 family): the ledger metadata is minted by
                # Beads at first use, so a fresh clone legitimately lacks it.
                # Planning the absent runtime artifact failed the gen check
                # gate on every clean checkout. When present, the render below
                # stays identity-preserving.
                continue
            artifact_context = self._artifact_render_context(
                dist=context.dist,
                repository=repository,
                repository_root=root,
                target=target,
                workspace=workspace,
                codegen=codegen,
                destination=destination,
                tooling_runtime=tooling_result.value,
                project_context=context,
            )
            if artifact_context.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    artifact_context.error
                    or f"artifact render context failed: {destination}"
                )
            rendered = u.Cli.template_render(
                templates_root / entry.source, artifact_context.value
            )
            if rendered.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    rendered.error
                    or (
                        f"stage=templates repository={repository.name} "
                        f"template={entry.source}: template render failed"
                    )
                )
            rendered_content = self._compose_project_artifact(
                root, destination, rendered.value
            )
            if rendered_content.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    rendered_content.error
                    or f"managed artifact composition failed: {destination}"
                )
            file_plan = self._file_plan(
                root,
                destination,
                rendered_content.value.rendered,
                source_states=rendered_content.value.source_states,
            )
            if file_plan.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    file_plan.error
                    or f"managed file planning failed: {entry.destination}"
                )
            planned.append(file_plan.value)
        if not contract.pyproject:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(tuple(planned))
        pyproject_entry = next(
            (
                item
                for item in codegen.templates.entries
                if item.destination == c.Infra.PYPROJECT_FILENAME
                and profile in item.profiles
                and item.delegate == "render"
            ),
            None,
        )
        if pyproject_entry is None:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                "pyproject template is missing from codegen configuration"
            )
        pyproject_render = u.Cli.template_render(
            templates_root / pyproject_entry.source, context
        )
        if pyproject_render.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                pyproject_render.error or "pyproject template render failed"
            )
        initial_tooling = modernizer.conform_source(
            pyproject_render.value,
            path=pyproject,
            format_source=False,
            declared_python_dirs=declared_python_dirs,
            declared_python_dirs_are_complete=declared_python_dirs_are_complete,
            analysis_exclusions=analysis_exclusions,
        )
        if initial_tooling.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                initial_tooling.error or f"initial tooling conform failed: {pyproject}"
            )
        cooldown_exclusions, cooldown_overrides = self._dependency_cooldown_policy(
            repository, codegen.toolchain
        )
        prepared_result = u.Infra.pyproject_conform(
            initial_tooling.value,
            providers=codegen.providers,
            workspace=workspace,
            workspace_mode=c.Infra.MakeProfile.STANDALONE,
            toolchain=codegen.toolchain,
            required_dev_dependencies=codegen.scaffold.project.dev,
            uv_link_mode=repository.uv_link_mode,
            dependency_cooldown_exclusions=cooldown_exclusions,
            dependency_cooldown_overrides=cooldown_overrides,
            uv_exclude_dependencies=uv_exclude_dependencies,
        )
        if prepared_result.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                prepared_result.error or f"pyproject prepare failed: {pyproject}"
            )
        final_tooling = modernizer.conform_source(
            prepared_result.value,
            path=pyproject,
            declared_python_dirs=declared_python_dirs,
            declared_python_dirs_are_complete=declared_python_dirs_are_complete,
            analysis_exclusions=analysis_exclusions,
        )
        if final_tooling.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                final_tooling.error or f"final tooling conform failed: {pyproject}"
            )
        pyproject_plan = self._file_plan(
            root, c.Infra.PYPROJECT_FILENAME, final_tooling.value
        )
        if pyproject_plan.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                pyproject_plan.error or f"pyproject planning failed: {pyproject}"
            )
        planned.append(pyproject_plan.value)
        return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(tuple(planned))

    @staticmethod
    def _routed_uv_exclude_dependencies(
        repository: m.Infra.RepositoryRef,
        target: m.Infra.RepositoryConformTarget,
        workspace: m.Infra.WorkspaceSpec,
        codegen: m.Infra.CodegenConfigSpec,
    ) -> tuple[m.Infra.UvScopedDependencyExclusionSpec, ...]:
        """Route the official uv exclusions to the distributions this root governs.

        uv reads ``exclude-dependencies`` only from the repository root, so a
        workspace root aggregates the exclusions of every distribution it
        declares while a standalone clone keeps only its own. The list itself
        describes the FLEXT reverse edges: a workspace whose declared
        repositories are not those distributions (a content workspace with
        non-FLEXT submodules, flext-cee4z) receives none of them instead of
        dropping a dependency its own venv still needs.
        """
        governed = {repository.distribution}
        if target.make_profile is c.Infra.MakeProfile.WORKSPACE:
            governed.update(
                declared.distribution for declared in workspace.declared_repositories
            )
        return tuple(
            item for item in codegen.uv_exclude_dependencies if item.project in governed
        )

    def _plan_existing_repository(
        self,
        *,
        root: Path,
        repository_root: Path,
        repository: m.Infra.RepositoryRef,
        target: m.Infra.RepositoryConformTarget,
        workspace: m.Infra.WorkspaceSpec,
        codegen: m.Infra.CodegenConfigSpec,
        contract: m.Infra.CodegenConformSurfaceContract,
    ) -> p.Result[t.SequenceOf[m.Infra.CodegenFilePlan]]:
        """Conform every declared managed surface in an existing repository."""
        u.Cli.info(f"  stage=pyproject repository={repository.name}")
        pyproject = root / c.Infra.PYPROJECT_FILENAME
        if not pyproject.is_file():
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                f"existing repository has no pyproject.toml: {root}; "
                "scaffold templates are available only through codegen new"
            )
        metadata = u.read_project_metadata(root)
        if metadata.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                metadata.error or f"project metadata load failed: {root}"
            )
        dist = metadata.value.project.name
        if dist != repository.distribution:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                "PEP 621 project name does not match catalog distribution: "
                f"{dist} != {repository.distribution}"
            )
        managed_artifacts = FlextInfraUtilitiesProjectManagedArtifacts.snapshot_project_managed_artifacts(
            root
        )
        if managed_artifacts.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].from_failure(
                managed_artifacts
            )
        pyproject_read = u.Cli.files_read_text(pyproject)
        if pyproject_read.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                pyproject_read.error or f"pyproject read failed: {pyproject}"
            )
        uv_exclude_dependencies = self._routed_uv_exclude_dependencies(
            repository, target, workspace, codegen
        )
        if contract.dependencies_only:
            dependency_result = u.Infra.pyproject_dependencies_conform(
                pyproject_read.value,
                providers=codegen.providers,
                workspace=workspace,
                workspace_mode=target.make_profile,
            )
            if dependency_result.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    dependency_result.error
                    or f"pyproject dependency conform failed: {pyproject}"
                )
            dependency_plan = self._file_plan(
                root, c.Infra.PYPROJECT_FILENAME, dependency_result.value
            )
            if dependency_plan.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    dependency_plan.error
                    or f"pyproject dependency planning failed: {pyproject}"
                )
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok((dependency_plan.value,))
        modernizer = FlextInfraPyprojectModernizer(
            repository_root=repository_root,
            skip_check=True,
            managed_artifacts=managed_artifacts.value.resolution,
        )
        analysis_exclusions = tuple(
            path.as_posix() for path in target.external_dependency_paths
        )
        generated_python_roots = self._scaffold_python_dirs(
            codegen.templates.entries, target.make_profile
        )
        tooling_context = modernizer.resolve_tooling_context(
            project_name=repository.distribution,
            package_name=metadata.value.package_name,
            path=pyproject,
            declared_python_dirs=generated_python_roots,
            analysis_exclusions=analysis_exclusions,
        )
        if tooling_context.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                tooling_context.error or f"tooling render failed: {pyproject}"
            )
        if not contract.pyproject:
            return self._plan_existing_templates(
                root=root,
                repository=repository,
                target=target,
                workspace=workspace,
                codegen=codegen,
                tooling_runtime=tooling_context.value,
                contract=contract,
                managed_artifacts=managed_artifacts.value,
            )
        cooldown_exclusions, cooldown_overrides = self._dependency_cooldown_policy(
            repository, codegen.toolchain
        )
        prepared_result = u.Infra.pyproject_conform(
            pyproject_read.value,
            providers=codegen.providers,
            workspace=workspace,
            workspace_mode=target.make_profile,
            toolchain=codegen.toolchain,
            required_dev_dependencies=codegen.scaffold.project.dev,
            uv_link_mode=repository.uv_link_mode,
            dependency_cooldown_exclusions=cooldown_exclusions,
            dependency_cooldown_overrides=cooldown_overrides,
            uv_exclude_dependencies=uv_exclude_dependencies,
        )
        if prepared_result.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                prepared_result.error or f"pyproject preparation failed: {pyproject}"
            )
        # Dependency topology is conformed before tooling so the modernizer is
        # the final owner of TOML ordering, comments, and type-checker settings.
        # It preserves the already canonical dependency source declarations.
        # Managed roots this plan materializes (tests/) count as analyzer roots
        # already in the plan, so apply and its verification are one fixed point.
        tooling_result = modernizer.conform_source(
            prepared_result.value,
            path=pyproject,
            generated_python_roots=generated_python_roots,
            analysis_exclusions=analysis_exclusions,
        )
        if tooling_result.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                tooling_result.error or f"tooling conform failed: {pyproject}"
            )
        pyproject_plan = self._file_plan(
            root, c.Infra.PYPROJECT_FILENAME, tooling_result.value
        )
        if pyproject_plan.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                pyproject_plan.error or f"pyproject planning failed: {pyproject}"
            )
        planned = [pyproject_plan.value]
        if not contract.templates:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(tuple(planned))
        # managed_files is the existing-tree
        # ownership SSOT; templates.entries remains the single render manifest.
        managed_result = self._plan_existing_templates(
            root=root,
            repository=repository,
            target=target,
            workspace=workspace,
            codegen=codegen,
            tooling_runtime=tooling_context.value,
            contract=contract,
            managed_artifacts=managed_artifacts.value,
        )
        if managed_result.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                managed_result.error or f"managed template planning failed: {root}"
            )
        planned.extend(managed_result.value)
        if contract.custom:
            custom_result = self._plan_existing_custom(
                root, codegen, profile=target.make_profile.value
            )
            if custom_result.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    custom_result.error or f"custom Make validation failed: {root}"
                )
            planned.extend(custom_result.value)
        return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(tuple(planned))

    def _plan_existing_templates(
        self,
        *,
        root: Path,
        repository: m.Infra.RepositoryRef,
        target: m.Infra.RepositoryConformTarget,
        workspace: m.Infra.WorkspaceSpec,
        codegen: m.Infra.CodegenConfigSpec,
        tooling_runtime: m.Infra.ToolingRuntimeContext | None,
        contract: m.Infra.CodegenConformSurfaceContract,
        managed_artifacts: m.Infra.ProjectManagedArtifactsSnapshot,
    ) -> p.Result[t.SequenceOf[m.Infra.CodegenFilePlan]]:
        """Render configured overwrite-owned templates for an existing tree."""
        u.Cli.info(f"  stage=templates repository={repository.name}")
        profile = target.make_profile
        templates_root = (
            self._package_root() / "templates" / codegen.templates.root
        ).resolve()
        planned: list[m.Infra.CodegenFilePlan] = []
        for managed in codegen.managed_files:
            if not target.ci_enabled and managed.path.parts[:2] == (
                ".github",
                "workflows",
            ):
                continue
            if (
                contract.destinations is not None
                and managed.path.as_posix() not in contract.destinations
            ):
                continue
            if (
                managed.policy in {"delegated", "manual"}
                or managed.path == Path(c.Infra.PYPROJECT_FILENAME)
                or managed.path == Path(c.Infra.CUSTOM_MAKE_FILENAME)
            ):
                continue
            entries = tuple(
                entry
                for entry in codegen.templates.entries
                if entry.destination == managed.path.as_posix()
                and entry.delegate == "render"
            )
            if not entries:
                continue
            if len(entries) != 1:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"managed file requires exactly one render template: {managed.path}"
                )
            entry = entries[0]
            u.Cli.emit_raw(f"  template {entry.source} -> {entry.destination}\n")
            relative = Path(entry.destination)
            if relative.is_absolute() or ".." in relative.parts:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"managed destination escapes repository root: {entry.destination}"
                )
            if self._beads_projection_is_inherited(entry.destination, root):
                # Mirror the delegated-template path so both conform routes
                # preserve the same inherited-ledger ownership rule.
                continue
            path = (root / relative).resolve()
            if (
                entry.destination == c.Infra.BEADS_METADATA_RELPATH
                and not path.is_file()
            ):
                # Why (flext-l2296 family): the ledger metadata is minted by
                # Beads at first use, so a fresh clone legitimately lacks it.
                # Planning an absent runtime artifact made the gen check gate
                # fail on every clean checkout. When the file exists, the
                # identity-preserving refresh below still applies.
                continue
            try:
                path.relative_to(root.absolute())
            except ValueError:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"managed destination escapes repository root: {entry.destination}"
                )
            if profile not in entry.profiles:
                # Why: profile-excluded managed workflows must not keep firing
                # (ci-matrix on standalone). Prune the orphan projection.
                if (
                    managed.path.parts[:2] == (".github", "workflows")
                    and path.is_file()
                ):
                    # Keep the typed read result distinct from its string payload.
                    orphan_read = u.Cli.files_read_text(path)
                    if orphan_read.failure:
                        return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                            orphan_read.error or f"orphan workflow read failed: {path}"
                        )
                    planned.append(
                        m.Infra.CodegenFilePlan(
                            path=path,
                            owner=managed.owner,
                            policy=managed.policy,
                            rendered="",
                            expected_sha256=u.Cli.sha256_content(""),
                            current_sha256=u.Cli.sha256_content(orphan_read.value),
                            changed=True,
                            absent=True,
                        )
                    )
                continue
            if managed.policy == "create-only" and path.is_file():
                current = u.Cli.files_read_text(path)
                if current.failure:
                    return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                        current.error or f"managed file read failed: {path}"
                    )
                file_plan = self._file_plan(
                    root,
                    entry.destination,
                    current.value,
                    executable=managed.executable,
                )
                if file_plan.failure:
                    return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                        file_plan.error
                        or f"managed file planning failed: {entry.destination}"
                    )
                planned.append(file_plan.value)
                continue
            if managed.policy == "create-only" and path.exists():
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    f"create-only destination is not a regular file: {path}"
                )
            if managed.policy == "create-only":
                continue
            artifact_context = self._artifact_render_context(
                dist=repository.distribution,
                repository=repository,
                repository_root=root,
                target=target,
                workspace=workspace,
                codegen=codegen,
                destination=entry.destination,
                tooling_runtime=tooling_runtime,
                project_context=None,
                managed_artifacts=managed_artifacts.resolution,
            )
            if artifact_context.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    artifact_context.error
                    or f"managed artifact context failed: {entry.destination}"
                )
            rendered = u.Cli.template_render(
                templates_root / entry.source, artifact_context.value
            )
            if rendered.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    rendered.error or f"template render failed: {entry.source}"
                )
            rendered_content = rendered.value
            composed = self._compose_project_artifact(
                root,
                entry.destination,
                rendered_content,
                managed_artifacts=managed_artifacts,
            )
            if composed.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    composed.error
                    or f"managed artifact composition failed: {entry.destination}"
                )
            rendered_content = composed.value.rendered
            conflict_marker = next(
                (
                    line
                    for line in rendered_content.splitlines()
                    if line.startswith(("<<<<<<< ", "||||||| ", ">>>>>>> "))
                ),
                None,
            )
            if conflict_marker is not None:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    "rendered template contains a merge conflict marker: "
                    f"source={entry.source}; target={path}; root={root}; "
                    f"marker={conflict_marker}"
                )
            file_plan = self._file_plan(
                root,
                entry.destination,
                rendered_content,
                executable=managed.executable,
                source_states=composed.value.source_states,
            )
            if file_plan.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    file_plan.error
                    or f"managed file planning failed: {entry.destination}"
                )
            planned.append(file_plan.value)
        return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(tuple(planned))

    @staticmethod
    def _compose_project_artifact(
        repository_root: Path,
        destination: str,
        rendered: str,
        *,
        managed_artifacts: m.Infra.ProjectManagedArtifactsSnapshot | None = None,
    ) -> p.Result[m.Infra.CodegenArtifactComposition]:
        """Apply typed project overlays after canonical template rendering."""
        if destination != c.Infra.MISE_TOML_FILENAME:
            return r[m.Infra.CodegenArtifactComposition].ok(
                m.Infra.CodegenArtifactComposition(rendered=rendered)
            )
        resolved_artifacts = managed_artifacts
        if resolved_artifacts is None and not repository_root.is_dir():
            # A scaffold materializes this root in the same plan, so it owns no
            # declared overlay yet. An existing root that cannot be inspected
            # still fails loud through the snapshot below.
            return r[m.Infra.CodegenArtifactComposition].ok(
                m.Infra.CodegenArtifactComposition(rendered=rendered)
            )
        if resolved_artifacts is None:
            snapshot = FlextInfraUtilitiesProjectManagedArtifacts.snapshot_project_managed_artifacts(
                repository_root
            )
            if snapshot.failure:
                return r[m.Infra.CodegenArtifactComposition].from_failure(snapshot)
            resolved_artifacts = snapshot.value
        if resolved_artifacts is None:
            return r[m.Infra.CodegenArtifactComposition].fail(
                f"project managed-artifact snapshot is absent: {repository_root}"
            )
        composed = (
            FlextInfraUtilitiesProjectManagedArtifacts.compose_mise_toml_from_snapshot(
                resolved_artifacts.sources, rendered
            )
        )
        if composed.failure:
            return r[m.Infra.CodegenArtifactComposition].from_failure(composed)
        return r[m.Infra.CodegenArtifactComposition].ok(
            m.Infra.CodegenArtifactComposition(
                rendered=composed.value, source_states=resolved_artifacts.sources
            )
        )

    @staticmethod
    def _repository_root_rel(workspace: m.Infra.WorkspaceSpec) -> str:
        """Return the environment root owned by the inferred target."""
        if workspace.project is not None:
            project_root_rel: str = workspace.project.repository_root_rel
            return project_root_rel
        return "."

    @staticmethod
    def _repository_provider(
        repository: m.Infra.RepositoryRef, codegen: m.Infra.CodegenConfigSpec
    ) -> p.Result[m.Infra.ProviderSpec]:
        """Resolve one repository to exactly one provider-owned policy."""
        resolved: p.Result[m.Infra.ProviderSpec] = u.Infra.repository_provider(
            repository, codegen.providers
        )
        return resolved

    @classmethod
    def _managed_gitlinks(
        cls, workspace: m.Infra.WorkspaceSpec, codegen: m.Infra.CodegenConfigSpec
    ) -> p.Result[tuple[m.Infra.ManagedGitlinkSpec, ...]]:
        """Resolve provider baselines only for mutable governed declared_repositories."""
        resolved: list[m.Infra.ManagedGitlinkSpec] = []
        for repository in workspace.declared_repositories:
            provider = cls._repository_provider(repository, codegen)
            if provider.failure:
                return r[tuple[m.Infra.ManagedGitlinkSpec, ...]].fail(
                    provider.error
                    or f"declared_repository provider is invalid: {repository.name}"
                )
            resolved.append(
                m.Infra.ManagedGitlinkSpec(
                    repository=repository,
                    branch=u.Infra.resolve_integration_branch(
                        workspace, provider.value
                    ),
                )
            )
        return r[tuple[m.Infra.ManagedGitlinkSpec, ...]].ok(tuple(resolved))

    @staticmethod
    def _beads_project_id(repository_root: Path) -> str | None:
        """Return the checkout's own ledger identity, or None if unminted.

        `.beads/identity.toml` is the canonical owner (`[project] id`); the
        generated marker is its projection. An absent file is not a failure —
        it means Beads has not minted an identity for this checkout yet.
        """
        identity = repository_root / ".beads" / "identity.toml"
        if not identity.is_file():
            return None
        source = u.Cli.files_read_text(identity)
        if source.failure:
            return None
        payload = u.Cli.toml_mapping_from_text(source.value)
        if payload is None:
            return None
        project = payload.get("project")
        if not isinstance(project, Mapping):
            return None
        value = project.get("id")
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _artifact_render_context(
        self,
        *,
        dist: str,
        repository: m.Infra.RepositoryRef,
        repository_root: Path,
        target: m.Infra.RepositoryConformTarget,
        workspace: m.Infra.WorkspaceSpec,
        codegen: m.Infra.CodegenConfigSpec,
        destination: str,
        tooling_runtime: m.Infra.ToolingRuntimeContext | None,
        project_context: m.Infra.ProjectRenderContext | None,
        managed_artifacts: m.Infra.ProjectManagedArtifactsResolution | None = None,
    ) -> p.Result[p.Model]:
        """Resolve one governed artifact to its canonical typed render input."""
        if self._beads_projection_is_inherited(destination, repository_root):
            return r[p.Model].fail(
                "linked workspace member cannot own a Beads projection: "
                f"{repository.name}; ledger is inherited from the workspace root"
            )
        if destination == c.Infra.GITIGNORE:
            profile = target.make_profile
            sections = [
                section
                for section in codegen.gitignore_sections
                if not section.profiles or profile in section.profiles
            ]
            if managed_artifacts is not None:
                project_patterns = managed_artifacts.artifacts.Gitignore.patterns
                if project_patterns:
                    sections.append(
                        m.Infra.ScaffoldGitignoreSectionSpec(
                            name=c.Infra.GITIGNORE_PROJECT_SECTION_NAME,
                            patterns=project_patterns,
                        )
                    )
            return r[p.Model].ok(
                m.Infra.GitignoreRenderSpec(gitignore_sections=tuple(sections))
            )
        if destination == "sgconfig.yml":
            # Why (ai-hub-qwoc): the ast-grep contract is identical for every
            # governed repository, so it renders straight from the codegen SSOT.
            return r[p.Model].ok(codegen.sgconfig)
        if destination == ".pre-commit-config.yaml":
            return r[p.Model].ok(
                m.Infra.MakeWorkflowRenderSpec(dist=dist, make=codegen.make)
            )
        if destination in {".envrc", ".mise.toml", ".python-version"}:
            return r[p.Model].ok(codegen.toolchain)
        if destination == c.Infra.BEADS_CONFIG_RELPATH:
            if target.beads is None:
                return r[p.Model].fail(
                    "ledger projection requires the observed conformance target, "
                    f"never a declaration-only one: {destination}"
                )
            project_types = target.beads.custom_issue_types
            required_types = codegen.toolchain.beads.required_custom_types
            return r[p.Model].ok(
                m.Infra.BeadsConfigRenderSpec(
                    issue_prefix=target.beads.issue_prefix,
                    endpoint_origin=codegen.toolchain.beads.endpoint_origin,
                    endpoint_status=codegen.toolchain.beads.endpoint_status,
                    custom_issue_types=tuple(
                        dict.fromkeys((*project_types, *required_types))
                    ),
                )
            )
        if destination == c.Infra.BEADS_METADATA_RELPATH:
            # Why: this marker is regenerated on every `make gen`, but the
            # ledger identity inside it is owned by the checkout, not by the
            # fleet SSOT. Rendering without it stripped the key, and Beads then
            # minted a NEW identity on next access — rig gmn lost
            # 2b1a0582-… that way (commit 3e7ba1e). Read it back so a
            # regeneration is identity-preserving.
            if target.beads is None:
                return r[p.Model].fail(
                    "ledger projection requires the observed conformance target, "
                    f"never a declaration-only one: {destination}"
                )
            return r[p.Model].ok(
                m.Infra.BeadsMetadataRenderSpec(
                    database=target.beads.database,
                    project_id=self._beads_project_id(repository_root),
                )
            )
        if destination.startswith(".github/"):
            provider = self._repository_provider(repository, codegen)
            if provider.failure:
                return r[p.Model].fail(
                    provider.error or "workflow provider resolution failed"
                )
            workspace_repositories = (
                tuple(workspace.declared_repositories)
                if target.make_profile is c.Infra.MakeProfile.WORKSPACE
                else ()
            )
            # Why: ci.yml.j2 iterates this to build its push/pull_request branch
            # filters, so an unsupplied value fails the render outright. The
            # repository's own integration branch is the only branch this layer
            # can name from resolved data; a fleet-wide list hardcoded here would
            # make every repository trigger on branches it does not have.
            branch = u.Infra.resolve_integration_branch(workspace, provider.value)
            return r[p.Model].ok(
                m.Infra.GithubWorkflowRenderSpec(
                    dist=dist,
                    make_profile=target.make_profile,
                    repository_branch=branch,
                    ci_trigger_branches=tuple(
                        dict.fromkeys((
                            *codegen.branch_policy.ci_trigger_branches,
                            branch,
                        ))
                    ),
                    python_version=codegen.toolchain.python_version,
                    dependency_cooldown_days=(
                        codegen.toolchain.dependency_cooldown_days
                    ),
                    github_actions=codegen.github_actions,
                    make=codegen.make,
                    workspace_repositories=workspace_repositories,
                    # Why: dependabot.yml.j2 branches on this and the model
                    # declares it, but the .github/ spec never supplied it, so
                    # every render died with "'has_devcontainer' is undefined".
                    # Dependabot rejects its ENTIRE config when an ecosystem
                    # names a directory that is absent, so this is read from the
                    # checkout rather than declared: a stale flag would silently
                    # disable Dependabot for the repository.
                    has_devcontainer=(repository_root / ".devcontainer").is_dir(),
                    checkout_submodules=codegen.checkout_submodules_overrides.get(
                        dist, codegen.checkout_submodules
                    ),
                    private_submodules=codegen.ci_private_submodules.get(dist),
                    system_packages=tuple(codegen.ci_system_packages.get(dist, ())),
                )
            )
        destination_path = Path(destination)
        if (
            destination_path.parent.as_posix() == "tests/fixtures/ci/docker"
            and destination_path.suffix == ".Dockerfile"
        ):
            return r[p.Model].ok(
                m.Infra.DistroDockerRenderSpec(
                    package_name=dist.replace("-", "_"),
                    python_version=codegen.toolchain.python_version,
                    make=codegen.make,
                )
            )
        if destination in {
            c.Infra.RELEASE_BUILD_CONSTRAINTS_PATH,
            c.Infra.RELEASE_GITLEAKS_CONFIG_PATH,
        }:
            # Why (flext-to3n7): the release build phase snapshots these two
            # policies from the repository; they are fleet policy owned by
            # config/infra.yaml, never scaffold-only project metadata.
            return r[p.Model].ok(
                m.Infra.ReleasePolicyRenderSpec(
                    build_constraints=config.Infra.release.build_constraints
                )
            )
        if destination == c.Infra.MAKEFILE_FILENAME:
            profile = target.make_profile
            declared_repositories = (
                tuple(workspace.declared_repositories)
                if profile is c.Infra.MakeProfile.WORKSPACE
                else ()
            )
            gitlinks = self._managed_gitlinks(workspace, codegen)
            if gitlinks.failure:
                return r[p.Model].fail(
                    gitlinks.error or "managed Gitlink resolution failed"
                )
            cooldown_exclusions, cooldown_overrides = self._dependency_cooldown_policy(
                repository, codegen.toolchain
            )
            return r[p.Model].ok(
                m.Infra.MakefileRenderSpec(
                    pytest=config.Infra.tooling.tools.pytest,
                    dist=dist,
                    infra_cli=config.Infra.name,
                    make_profile=profile,
                    makefile_custom_include=c.Infra.MAKEFILE_CUSTOM_INCLUDE,
                    repository_root_rel=FlextInfraCodegenConform._repository_root_rel(
                        workspace
                    ),
                    declared_repositories=tuple(
                        item.path.as_posix() for item in workspace.declared_repositories
                    ),
                    workspace_repositories=declared_repositories,
                    workspace_gitlinks=gitlinks.value,
                    uv_link_mode=FlextInfraCodegenConform._link_mode(
                        repository, codegen.toolchain
                    ),
                    uv_version=codegen.toolchain.uv_version,
                    uv_exclude_newer=codegen.toolchain.uv_exclude_newer,
                    dependency_cooldown_exclusions=cooldown_exclusions,
                    dependency_cooldown_overrides=cooldown_overrides,
                    make=codegen.make,
                    extra_verbs=repository.extra_verbs,
                    script_dispatch=repository.script_dispatch,
                    orchestrated_verbs=c.Infra.ORCHESTRATED_PROJECT_VERBS,
                    workspace_cli_group=c.Infra.CLI_GROUP_WORKSPACE,
                    project_selection_conflict_error=(
                        c.Infra.PROJECT_SELECTION_CONFLICT_ERROR
                    ),
                    mypy_memory_limit_mb=c.Infra.MYPY_MEMORY_LIMIT_MB_DEFAULT,
                    mypy_timeout_seconds=c.Infra.MYPY_TIMEOUT_SECONDS_DEFAULT,
                    mypy_timeout_exit_code=c.Infra.PROCESS_TIMEOUT_EXIT_CODE,
                    mypy_signal_exit_offset=c.Infra.PROCESS_SIGNAL_EXIT_OFFSET,
                    prlimit_command=c.Infra.PRLIMIT_COMMAND,
                    prlimit_address_space_option=(c.Infra.PRLIMIT_ADDRESS_SPACE_OPTION),
                    timeout_command=c.Infra.TIMEOUT_COMMAND,
                    timeout_kill_after_seconds=c.Infra.TIMEOUT_KILL_AFTER_SECONDS,
                    pytest_process_timeout_seconds=(
                        config.Infra.tooling.tools.pytest.process_timeout_seconds
                    ),
                )
            )
        if destination == c.Infra.CUSTOM_MAKE_FILENAME:
            # Existing repositories project custom routes from the same typed
            # Make contract as Makefile; they do not require scaffold-only
            # project metadata. They do require the resolved tooling runtime,
            # which the declaration-only Makefile surface never resolves.
            if tooling_runtime is None:
                return r[p.Model].fail(
                    f"tooling runtime is required for managed artifact: {destination}"
                )
            make_context = FlextInfraCodegenConform.make_render_context(
                repository, target, workspace, codegen, tooling_runtime=tooling_runtime
            )
            if make_context.failure:
                return r[p.Model].fail(
                    make_context.error or "custom Make render context failed"
                )
            return r[p.Model].ok(make_context.value)
        if project_context is not None:
            return r[p.Model].ok(project_context)
        if tooling_runtime is None:
            return r[p.Model].fail(
                f"tooling runtime is required for managed artifact: {destination}"
            )
        context_result = self._project_render_context(
            repository,
            target,
            workspace,
            codegen,
            tooling_runtime=tooling_runtime,
            repository_root=repository_root,
            managed_artifacts=managed_artifacts,
        )
        if context_result.failure:
            return r[p.Model].fail(
                context_result.error
                or f"managed artifact context failed: {destination}"
            )
        return r[p.Model].ok(context_result.value)

    @staticmethod
    def make_render_context(
        repository: m.Infra.RepositoryRef,
        target: m.Infra.RepositoryConformTarget,
        workspace: m.Infra.WorkspaceSpec,
        codegen: m.Infra.CodegenConfigSpec,
        *,
        tooling_runtime: m.Infra.ToolingRuntimeContext,
    ) -> p.Result[m.Infra.MakeRenderContext]:
        """Build the typed context consumed by the generated Makefile."""
        profile = target.make_profile
        declared_repositories = (
            tuple(workspace.declared_repositories)
            if profile is c.Infra.MakeProfile.WORKSPACE
            else ()
        )
        gitlinks = FlextInfraCodegenConform._managed_gitlinks(workspace, codegen)
        if gitlinks.failure:
            return r[m.Infra.MakeRenderContext].fail(
                gitlinks.error or "managed Gitlink resolution failed"
            )
        cooldown_exclusions, cooldown_overrides = (
            FlextInfraCodegenConform._dependency_cooldown_policy(
                repository, codegen.toolchain
            )
        )
        return r[m.Infra.MakeRenderContext].ok(
            m.Infra.MakeRenderContext(
                pytest=config.Infra.tooling.tools.pytest,
                make=codegen.make,
                mypy_memory_limit_mb=c.Infra.MYPY_MEMORY_LIMIT_MB_DEFAULT,
                mypy_timeout_seconds=c.Infra.MYPY_TIMEOUT_SECONDS_DEFAULT,
                mypy_timeout_exit_code=c.Infra.PROCESS_TIMEOUT_EXIT_CODE,
                mypy_signal_exit_offset=c.Infra.PROCESS_SIGNAL_EXIT_OFFSET,
                prlimit_command=c.Infra.PRLIMIT_COMMAND,
                prlimit_address_space_option=c.Infra.PRLIMIT_ADDRESS_SPACE_OPTION,
                timeout_command=c.Infra.TIMEOUT_COMMAND,
                timeout_kill_after_seconds=c.Infra.TIMEOUT_KILL_AFTER_SECONDS,
                tooling_runtime=tooling_runtime,
                dist=repository.distribution,
                infra_cli=config.Infra.name,
                python_version=codegen.toolchain.python_version,
                uv_link_mode=FlextInfraCodegenConform._link_mode(
                    repository, codegen.toolchain
                ),
                uv_exclude_newer=codegen.toolchain.uv_exclude_newer,
                dependency_cooldown_exclusions=cooldown_exclusions,
                dependency_cooldown_overrides=cooldown_overrides,
                # ProjectRenderContext replaces this with the composed map.
                # Pass the neutral value explicitly so Pydantic never deep-copies
                # the MappingProxyType model default while building the base.
                ruff_per_file_ignores={},
                make_profile=profile,
                orchestrated_verbs=c.Infra.ORCHESTRATED_PROJECT_VERBS,
                workspace_cli_group=c.Infra.CLI_GROUP_WORKSPACE,
                project_selection_conflict_error=(
                    c.Infra.PROJECT_SELECTION_CONFLICT_ERROR
                ),
                repository_root_rel=FlextInfraCodegenConform._repository_root_rel(
                    workspace
                ),
                makefile_custom_include=c.Infra.MAKEFILE_CUSTOM_INCLUDE,
                declared_repositories=tuple(
                    item.path.as_posix() for item in workspace.declared_repositories
                ),
                workspace_repositories=declared_repositories,
                workspace_gitlinks=gitlinks.value,
                extra_verbs=repository.extra_verbs,
                script_dispatch=repository.script_dispatch,
            )
        )

    @staticmethod
    def _project_render_context(
        repository: m.Infra.RepositoryRef,
        target: m.Infra.RepositoryConformTarget,
        workspace: m.Infra.WorkspaceSpec,
        codegen: m.Infra.CodegenConfigSpec,
        *,
        tooling_runtime: m.Infra.ToolingRuntimeContext,
        repository_root: Path,
        managed_artifacts: m.Infra.ProjectManagedArtifactsResolution | None = None,
    ) -> p.Result[m.Infra.ProjectRenderContext]:
        """Build the complete typed context consumed by project templates."""
        if workspace.project is None:
            return r[m.Infra.ProjectRenderContext].fail(
                f"workspace has no project metadata: {workspace.name}"
            )
        project = workspace.project
        dependency_profile = next(
            (
                item
                for item in codegen.scaffold.project.dependency_profiles
                if item.upstream == project.upstream
            ),
            None,
        )
        if dependency_profile is None:
            return r[m.Infra.ProjectRenderContext].fail(
                f"unsupported scaffold upstream: {project.upstream}"
            )
        if project.license not in codegen.scaffold.project.supported_licenses:
            supported = ", ".join(codegen.scaffold.project.supported_licenses)
            return r[m.Infra.ProjectRenderContext].fail(
                f"unsupported scaffold license: {project.license}; "
                f"supported licenses: {supported}"
            )
        profile = target.make_profile
        make_context = FlextInfraCodegenConform.make_render_context(
            repository, target, workspace, codegen, tooling_runtime=tooling_runtime
        )
        if make_context.failure:
            return r[m.Infra.ProjectRenderContext].fail(
                make_context.error or "Make render context resolution failed"
            )
        repository_provider = FlextInfraCodegenConform._repository_provider(
            repository, codegen
        )
        if repository_provider.failure:
            return r[m.Infra.ProjectRenderContext].fail(
                repository_provider.error or "repository provider resolution failed"
            )
        flext_provider = repository_provider.value
        packaged_data_dirs = (
            tuple(
                data_dir
                for data_dir in config.Infra.tooling.tools.hatch.packaged_data_dirs
                if any(
                    profile in entry.profiles
                    and Path(entry.destination).parts
                    and Path(entry.destination).parts[0] == data_dir
                    for entry in codegen.templates.entries
                )
            )
            if profile is not c.Infra.MakeProfile.WORKSPACE
            else ()
        )
        # Emit only the .gitignore sections that apply to this profile: a
        # section with no declared profiles is universal; a workspace-only
        # section (declared-repository directory allowlist and submodule/Beads
        # coordination) never reaches a declared repository or standalone
        # .gitignore.
        profile_gitignore_sections = [
            section
            for section in codegen.gitignore_sections
            if not section.profiles or profile in section.profiles
        ]
        if managed_artifacts is not None:
            project_patterns = managed_artifacts.artifacts.Gitignore.patterns
            if project_patterns:
                profile_gitignore_sections.append(
                    m.Infra.ScaffoldGitignoreSectionSpec(
                        name=c.Infra.GITIGNORE_PROJECT_SECTION_NAME,
                        patterns=project_patterns,
                    )
                )
        # The repository's own pyproject.toml is the version SSOT; the release
        # protocol is its only writer, so conform reads it and never syncs it.
        # A tree that has no pyproject yet is being created: it starts at the
        # typed initial version and the protocol owns every change after that.
        version_result = (
            u.Infra.current_workspace_version(repository_root)
            if (repository_root / c.Infra.PYPROJECT_FILENAME).is_file()
            else r[str].ok(config.Infra.initial_project_version)
        )
        if version_result.failure:
            return r[m.Infra.ProjectRenderContext].fail(
                version_result.error or f"project version unresolved: {repository_root}"
            )
        return r[m.Infra.ProjectRenderContext].ok(
            m.Infra.ProjectRenderContext(
                **make_context.value.model_dump(
                    by_alias=True,
                    exclude={"ruff_per_file_ignores"},
                    exclude_computed_fields=True,
                ),
                scaffold=codegen.scaffold,
                gitignore_sections=tuple(profile_gitignore_sections),
                dependency_profile=dependency_profile,
                tooling=config.Infra.tooling,
                # Why: the fleet policy alone is not the effective Ruff contract.
                # A repository may carry an operator-authorized exemption in its
                # own config/*.yaml ManagedArtifacts block, and ensure_ruff
                # composes the two when it edits a pyproject in place. The
                # template rendered only the fleet map, so a full render silently
                # dropped the local overlay -- flext-infra's own _rope exemption
                # disappeared on every conform and returned 12 SLF001 findings
                # the operator had already ruled on. Compose here so both paths
                # produce the same effective map.
                ruff_per_file_ignores=(
                    FlextInfraEnsureRuffConfigPhase.compose_per_file_ignores(
                        repository_root, managed_artifacts=managed_artifacts
                    )
                ),
                environment_path_prepends=(codegen.toolchain.environment_path_prepends),
                beads_tool_selector=codegen.toolchain.beads.selector,
                beads_tool_version=codegen.toolchain.beads.version,
                # prerelease is load-bearing: every fork release of bd carries a
                # suffixed tag (-fdN) and mise refuses to resolve one unless
                # told the release is a prerelease. Omitting it silently pinned
                # every rig to upstream, which lacks the bd list cycle guard.
                beads_tool_prerelease=codegen.toolchain.beads.prerelease,
                beads_tool_minimum_release_age=(
                    codegen.toolchain.beads.minimum_release_age
                ),
                beads=workspace.beads,
                canonical_project_name=target.canonical_project_name,
                const_name=project.constant_name,
                package_name=project.package_name,
                packaged_data_dirs=packaged_data_dirs,
                class_stem=project.class_stem,
                ns=project.namespace,
                ns_attr=project.namespace_attribute,
                alias=project.alias,
                env_prefix=project.environment_prefix,
                upstream=project.upstream,
                inherited_facets=project.inherited_facets,
                description=project.description,
                version=version_result.value,
                license=project.license,
                python_required_version=codegen.toolchain.python_required_version,
                kubectl_version=codegen.toolchain.kubectl_version,
                helm_version=codegen.toolchain.helm_version,
                kind_version=codegen.toolchain.kind_version,
                direnv_version=codegen.toolchain.direnv_version,
                uv_version=codegen.toolchain.uv_version,
                qlty_version=codegen.toolchain.qlty_version,
                node_version=codegen.toolchain.node_version,
                jscpd_version=codegen.toolchain.jscpd_version,
                taplo_version=codegen.toolchain.taplo_version,
                ast_grep_version=codegen.toolchain.ast_grep_version,
                gitleaks_version=codegen.toolchain.gitleaks_version,
                scc_version=codegen.toolchain.scc_version,
                kubeconform_version=codegen.toolchain.kubeconform_version,
                go_version=codegen.toolchain.go_version,
                author_name=project.author_name,
                author_email=project.author_email,
                repository=project.homepage,
                homepage=project.homepage,
                documentation=project.documentation,
                flext_git_base_url=flext_provider.base_url,
                flext_git_branch=flext_provider.branch,
                repository_provider=repository.provider,
                repository_git_url=repository.url,
                repository_branch=u.Infra.resolve_integration_branch(
                    workspace, repository_provider.value
                ),
                year=project.year,
            )
        )

    def _plan_existing_custom(
        self,
        root: Path,
        config: m.Infra.CodegenConfigSpec,
        *,
        profile: str | None = None,
    ) -> p.Result[t.SequenceOf[m.Infra.CodegenFilePlan]]:
        """Validate the handwritten Make surface against its profile contract."""
        policy = config.make.custom_handler_policies.get(
            profile or "", config.make.custom_handler_policy
        )
        path = root / policy.filename
        if path.exists() and not path.is_file():
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                f"custom Make destination is not a regular file: {path}"
            )
        if not path.is_file():
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(())
        read = u.Cli.files_read_text(path)
        if read.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                read.error or f"custom Make read failed: {path}"
            )
        validation = self.validate_custom_make(read.value, policy)
        if validation.failure:
            return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                validation.error or f"invalid custom Make handlers: {path}"
            )
        digest = u.Cli.sha256_content(read.value)
        return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok((
            m.Infra.CodegenFilePlan(
                path=path,
                rendered=read.value,
                expected_sha256=digest,
                current_sha256=digest,
                changed=False,
            ),
        ))

    @staticmethod
    def validate_custom_make(
        content: str, policy: m.Infra.CustomHandlerPolicy
    ) -> p.Result[bool]:
        """Reject public targets, aliases, includes, and toolchain declarations."""
        target_re = re.compile(policy.target_pattern)
        in_define = False
        # Collapse backslash continuation lines before validating so that
        # directives like `.PHONY` can span multiple physical lines. Only
        # collapse non-recipe lines (recipe lines start with whitespace and are
        # skipped below); the reported line number is the first physical line.
        logical_lines: list[tuple[int, str]] = []
        pending_line: str | None = None
        pending_number: int = 0
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            if (
                raw_line
                and not raw_line[0].isspace()
                and raw_line.rstrip().endswith("\\")
            ):
                trimmed = raw_line.rstrip()[:-1].rstrip()
                if pending_line is None:
                    pending_line = trimmed
                    pending_number = line_number
                else:
                    pending_line += " " + trimmed
                continue
            # A continuation collapses several physical lines into one logical
            # line, which is reported at the line the continuation STARTED on,
            # not the line it ended on. Assigning back onto the loop variables
            # made the two indistinguishable and left the next iteration reading
            # a value the iterator never produced.
            logical_line = raw_line
            logical_number = line_number
            if pending_line is not None:
                joined = pending_line + " " + raw_line.strip()
                if joined.rstrip().endswith("\\"):
                    pending_line = joined.rstrip()[:-1].rstrip()
                    continue
                logical_line = joined
                logical_number = pending_number
                pending_line = None
            logical_lines.append((logical_number, logical_line))
        if pending_line is not None:
            if pending_line.startswith(".PHONY:"):
                return r[bool].fail(
                    f"{policy.filename} has an unterminated .PHONY continuation"
                )
            logical_lines.append((pending_number, pending_line))
        for line_number, raw_line in logical_lines:
            if in_define:
                in_define = not raw_line.startswith("endef")
                continue
            if raw_line.startswith("define "):
                if not policy.allow_toolchain_declarations:
                    return r[bool].fail(
                        f"{policy.filename} line {line_number} "
                        "declares a macro, which this profile forbids"
                    )
                in_define = True
                continue
            if not raw_line or raw_line.lstrip().startswith("#"):
                continue
            if raw_line[0].isspace():
                continue
            if c.Infra.MAKE_CONDITIONAL_RE.match(raw_line):
                continue
            if raw_line.startswith(".PHONY:"):
                declaration = raw_line.partition(":")[2].strip()
                names = declaration.split()
                if names and all(target_re.fullmatch(name) for name in names):
                    continue
            target = raw_line.partition(":")[0].strip() if ":" in raw_line else ""
            if target and target_re.fullmatch(target):
                continue
            if c.Infra.MAKE_ASSIGNMENT_RE.match(
                raw_line
            ) or c.Infra.MAKE_DIRECTIVE_RE.match(raw_line):
                if policy.allow_toolchain_declarations:
                    continue
                return r[bool].fail(
                    f"{policy.filename} line {line_number} "
                    "declares a variable, which this profile forbids"
                )
            if target and policy.allow_public_targets:
                continue
            return r[bool].fail(
                f"{policy.filename} line {line_number} is not a private custom handler"
            )
        return r[bool].ok(True)

    @classmethod
    def _file_plan(
        cls,
        root: Path,
        relative_path: str,
        rendered: str,
        *,
        executable: bool | None = None,
        source_states: tuple[m.Cli.AtomicFileState, ...] = (),
    ) -> p.Result[m.Infra.CodegenFilePlan]:
        """Compare one expected output and mark whether it changed."""
        path = root / relative_path
        conflict_marker = next(
            (
                line
                for line in rendered.splitlines()
                if line.startswith(("<<<<<<< ", "||||||| ", ">>>>>>> "))
            ),
            None,
        )
        if conflict_marker is not None:
            return r[m.Infra.CodegenFilePlan].fail(
                "rendered managed file contains a merge-conflict marker "
                f"({conflict_marker}) in the declared content of {relative_path}: "
                f"{path}"
            )
        authenticated = cls._authenticated_managed_file(
            root, path, allow_missing_parent=True
        )
        if authenticated.failure:
            return r[m.Infra.CodegenFilePlan].fail(
                authenticated.error or f"managed file read failed: {path}"
            )
        # The destination identity is re-authenticated under the transaction
        # lock immediately before promotion, so the plan keeps only the content
        # digest proving the destination did not move after planning.
        current, _identity = authenticated.value
        expected_sha = u.Cli.sha256_content(rendered)
        current_sha = u.Cli.sha256_content(current) if path.is_file() else ""
        mode_changed = (
            path.is_file()
            and executable is not None
            and bool(path.stat().st_mode & 0o111) is not executable
        )
        changed = current != rendered or mode_changed
        return r[m.Infra.CodegenFilePlan].ok(
            m.Infra.CodegenFilePlan(
                path=path,
                rendered=rendered,
                expected_sha256=expected_sha,
                source_states=source_states,
                current_sha256=current_sha,
                executable=executable,
                changed=changed,
                blocked=False,
                reason="",
            )
        )

    @staticmethod
    def _technical_branch(reference: str, patterns: t.StrSequence) -> bool:
        """Match one Git ref against the typed technical-branch policy."""
        short = reference
        for prefix in ("refs/heads/", "refs/remotes/origin/"):
            if short.startswith(prefix):
                short = short.removeprefix(prefix)
                break
        return any(
            fnmatchcase(short, pattern) or fnmatchcase(reference, pattern)
            for pattern in patterns
        )

    @classmethod
    def _branch_ancestry_plan(
        cls, target: m.Infra.RepositoryConformTarget
    ) -> p.Result[m.Infra.BranchAncestryPlan]:
        """Inventory local governed refs and prove descent from the baseline."""
        root = target.root
        baseline_reference = f"refs/remotes/origin/{target.baseline_branch}"
        baseline_command = (c.Infra.GIT, "rev-parse", "--verify", baseline_reference)
        baseline_result = u.Cli.run_raw(baseline_command, cwd=root)
        if baseline_result.failure:
            return r[m.Infra.BranchAncestryPlan].fail(
                "provider baseline command failed: "
                f"command={' '.join(baseline_command)}; error={baseline_result.error}"
            )
        if baseline_result.value.exit_code != 0:
            return r[m.Infra.BranchAncestryPlan].fail(
                "provider baseline ref is missing: "
                f"{baseline_reference}; command={' '.join(baseline_command)}; "
                f"exit={baseline_result.value.exit_code}; "
                f"stderr={baseline_result.value.stderr.strip() or '<empty>'}"
            )
        baseline_sha = baseline_result.value.stdout.strip()
        # flext-9ehwb: `refs/remotes/origin/<lane>` is the remote's LIVE tip.
        # `actions/checkout` fetches at job start, so that tip advances whenever
        # another actor publishes to the same lane while this run waits in the
        # queue. Gating against it asks "has this commit already absorbed work
        # published after it was written?" -- unanswerable by construction, and
        # a perfectly linear commit fails with "does not descend from"
        # (run 31218338222). The question the gate actually owns is "was this
        # commit written on top of the lane?", so the baseline is pinned to the
        # merge base between the live tip and the commit that triggered the run.
        # That point is immutable for a given commit: concurrent publishers move
        # the tip, never the merge base. Outside CI GITHUB_SHA is unset and the
        # live tip remains the baseline, which is correct for a local checkout.
        triggering_sha = os.environ.get(c.Infra.ENV_VAR_GITHUB_SHA, "").strip()
        if triggering_sha:
            # GITHUB_SHA is the superproject's triggering commit (the PR
            # merge commit actions/checkout synthesizes). Inside a submodule
            # repo that SHA does not exist, so `git merge-base` fails with
            # exit 128 ("Not a valid commit name") and breaks `gen check` in
            # CI. Verify membership before anchoring the baseline through it:
            # only pin to the merge base when triggering_sha actually resolves
            # in the current repository (the workspace root). In a submodule
            # context the gitlink position is the immutable reference and the
            # live baseline tip remains the correct anchor.
            verify_command = (c.Infra.GIT, "cat-file", "-t", triggering_sha)
            verify_result = u.Cli.run_raw(verify_command, cwd=root)
            if verify_result.success and verify_result.value.exit_code == 0:
                merge_base_command = (
                    c.Infra.GIT,
                    "merge-base",
                    baseline_sha,
                    triggering_sha,
                )
                merge_base_result = u.Cli.run_raw(merge_base_command, cwd=root)
                if merge_base_result.failure:
                    return r[m.Infra.BranchAncestryPlan].fail(
                        "cannot anchor ancestry baseline to the triggering commit: "
                        f"command={' '.join(merge_base_command)}; "
                        f"error={merge_base_result.error}"
                    )
                if merge_base_result.value.exit_code != 0:
                    return r[m.Infra.BranchAncestryPlan].fail(
                        "triggering commit shares no history with the baseline: "
                        f"{c.Infra.ENV_VAR_GITHUB_SHA}={triggering_sha}; "
                        f"command={' '.join(merge_base_command)}; "
                        f"exit={merge_base_result.value.exit_code}; "
                        f"stderr={merge_base_result.value.stderr.strip() or '<empty>'}"
                    )
                baseline_sha = merge_base_result.value.stdout.strip()
        pending_merge_result = u.Cli.run_raw(
            (
                c.Infra.GIT,
                "merge-base",
                "--is-ancestor",
                baseline_sha,
                c.Infra.GIT_MERGE_HEAD,
            ),
            cwd=root,
        )
        pending_merge_includes_baseline = (
            pending_merge_result.success and pending_merge_result.value.exit_code == 0
        )
        current_branch_result = u.Cli.run_raw(
            (c.Infra.GIT, "rev-parse", "--abbrev-ref", "HEAD"), cwd=root
        )
        current_branch_ref = ""
        if current_branch_result.success and current_branch_result.value.exit_code == 0:
            current_branch = current_branch_result.value.stdout.strip()
            if current_branch != "HEAD":
                current_branch_ref = f"refs/heads/{current_branch}"
        refs_command = (
            c.Infra.GIT,
            "for-each-ref",
            "--format=%(refname)%09%(objectname)",
            "refs/heads",
            "refs/remotes/origin",
        )
        refs_result = u.Cli.run_raw(refs_command, cwd=root)
        if refs_result.failure:
            return r[m.Infra.BranchAncestryPlan].fail(
                "cannot enumerate governed refs: "
                f"command={' '.join(refs_command)}; error={refs_result.error}"
            )
        if refs_result.value.exit_code != 0:
            return r[m.Infra.BranchAncestryPlan].fail(
                "cannot enumerate governed refs: "
                f"command={' '.join(refs_command)}; "
                f"exit={refs_result.value.exit_code}; "
                f"stderr={refs_result.value.stderr.strip() or '<empty>'}"
            )
        observations: list[tuple[str, str]] = []
        for line in refs_result.value.stdout.splitlines():
            reference, separator, sha = line.partition("\t")
            if not separator or not reference or not sha:
                return r[m.Infra.BranchAncestryPlan].fail(
                    f"malformed Git ref inventory entry: {line}"
                )
            if reference == "refs/remotes/origin/HEAD":
                continue
            observations.append((reference, sha))
        worktrees_command = (c.Infra.GIT, "worktree", "list", "--porcelain")
        worktrees_result = u.Cli.run_raw(worktrees_command, cwd=root)
        if worktrees_result.failure:
            return r[m.Infra.BranchAncestryPlan].fail(
                "cannot enumerate registered worktrees: "
                f"command={' '.join(worktrees_command)}; "
                f"error={worktrees_result.error}"
            )
        if worktrees_result.value.exit_code != 0:
            return r[m.Infra.BranchAncestryPlan].fail(
                "cannot enumerate registered worktrees: "
                f"command={' '.join(worktrees_command)}; "
                f"exit={worktrees_result.value.exit_code}; "
                f"stderr={worktrees_result.value.stderr.strip() or '<empty>'}"
            )
        worktree_path = ""
        worktree_sha = ""
        worktree_branch = "detached"
        worktree_bare = False
        for line in (*worktrees_result.value.stdout.splitlines(), ""):
            if line.startswith("worktree "):
                worktree_path = line.removeprefix("worktree ")
                worktree_bare = False
            elif line == "bare":
                # The main worktree of a bare repository (e.g. a Gas Town rig
                # .repo.git) lists itself with a `bare` attribute and no HEAD;
                # it owns refs but is never a governed branch checkout.
                worktree_bare = True
            elif line.startswith("HEAD "):
                worktree_sha = line.removeprefix("HEAD ")
            elif line.startswith("branch "):
                worktree_branch = line.removeprefix("branch ")
            elif not line and worktree_path:
                if worktree_bare:
                    worktree_path = ""
                    worktree_sha = ""
                    worktree_branch = "detached"
                    continue
                if not worktree_sha:
                    return r[m.Infra.BranchAncestryPlan].fail(
                        f"worktree has no HEAD: {worktree_path}"
                    )
                if Path(worktree_path).resolve() != root.resolve():
                    worktree_path = ""
                    worktree_sha = ""
                    worktree_branch = "detached"
                    continue
                if worktree_branch == "detached":
                    # Detached checkouts (e.g., temporary CI/worktree transactions)
                    # are not governed branch refs; skip them.
                    worktree_path = ""
                    worktree_sha = ""
                    worktree_branch = "detached"
                    continue
                observations.append((
                    f"worktree:{worktree_path}:{worktree_branch}",
                    worktree_sha,
                ))
                worktree_path = ""
                worktree_sha = ""
                worktree_branch = "detached"
        references: list[m.Infra.BranchAncestryRef] = []
        for reference, sha in sorted(observations):
            policy_reference = (
                reference.rpartition(":")[2]
                if reference.startswith("worktree:")
                else reference
            )
            # Ancestry is a development-line rule. Only refs on the
            # governed allowlist are gated; parked releases (0.10/0.11), snapshots
            # and lane branches are inventoried but must never block conform.
            excluded = cls._technical_branch(
                policy_reference, target.technical_branch_patterns
            ) or not cls._technical_branch(
                policy_reference, target.governed_branch_patterns
            )
            # Only enforce ancestry on active checkouts: the current branch and
            # registered worktrees. Shared local/remote branches that are not
            # currently checked out are excluded from this repository-local gate.
            if not excluded and not reference.startswith("worktree:"):
                is_remote = reference.startswith("refs/remotes/")
                is_other_local = (
                    reference.startswith("refs/heads/")
                    and reference != current_branch_ref
                )
                if is_remote or is_other_local:
                    excluded = True
            ancestor: bool | None = None
            if not excluded:
                ancestry_command = (
                    c.Infra.GIT,
                    "merge-base",
                    "--is-ancestor",
                    baseline_sha,
                    sha,
                )
                ancestry_result = u.Cli.run_raw(ancestry_command, cwd=root)
                if ancestry_result.failure:
                    return r[m.Infra.BranchAncestryPlan].fail(
                        "cannot validate branch ancestry: "
                        f"{reference}; command={' '.join(ancestry_command)}; "
                        f"error={ancestry_result.error}"
                    )
                if ancestry_result.value.exit_code not in {0, 1}:
                    return r[m.Infra.BranchAncestryPlan].fail(
                        "Git ancestry validation failed: "
                        f"{reference}; command={' '.join(ancestry_command)}; "
                        f"exit={ancestry_result.value.exit_code}; "
                        f"stderr={ancestry_result.value.stderr.strip() or '<empty>'}"
                    )
                ancestor = ancestry_result.value.exit_code == 0
                if not ancestor and policy_reference == current_branch_ref:
                    ancestor = pending_merge_includes_baseline
            references.append(
                m.Infra.BranchAncestryRef(
                    reference=reference, sha=sha, excluded=excluded, ancestor=ancestor
                )
            )
        return r[m.Infra.BranchAncestryPlan].ok(
            m.Infra.BranchAncestryPlan(
                repository_root=root,
                baseline_reference=baseline_reference,
                baseline_sha=baseline_sha,
                references=tuple(references),
            )
        )

    @staticmethod
    def _uv_environment_plan(
        *,
        root: Path,
        repository_root: Path,
        target: m.Infra.RepositoryConformTarget,
        workspace: m.Infra.WorkspaceSpec,
        config: m.Infra.CodegenConfigSpec,
    ) -> m.Infra.UvEnvironmentPlan:
        """Describe the exact setup overlay without executing uv."""
        del repository_root
        workspace_environment = target.make_profile is c.Infra.MakeProfile.WORKSPACE
        environment_root = target.root
        groups: tuple[str, ...] = ("dev", "codegen")
        editable_repositories: tuple[m.Infra.RepositoryRef, ...] = ()
        if workspace_environment:
            groups = (*groups, "workspace")
            editable_repositories = tuple(
                item
                for item in (workspace.repository, *workspace.declared_repositories)
                if item.package and item.editable and not item.read_only
            )
        return m.Infra.UvEnvironmentPlan(
            project_root=root,
            environment_root=environment_root,
            lock_path=environment_root / c.Infra.UV_LOCK_FILENAME,
            python_version=config.toolchain.python_version,
            groups=groups,
            editable_repositories=editable_repositories,
        )

    @staticmethod
    def _absent_file_plan(path: Path, current: str) -> m.Infra.CodegenFilePlan:
        """Plan the removal of one retired projection."""
        return m.Infra.CodegenFilePlan(
            path=path,
            rendered="",
            expected_sha256=u.Cli.sha256_content(""),
            current_sha256=u.Cli.sha256_content(current),
            changed=True,
            absent=True,
        )

    @classmethod
    def retired_projection_plans(
        cls, root: Path, profile: c.Infra.MakeProfile
    ) -> p.Result[t.SequenceOf[m.Infra.CodegenFilePlan]]:
        """Plan removal of generated projections excluded from this profile."""
        codegen = config.Infra.codegen
        planned: list[m.Infra.CodegenFilePlan] = []
        for entry in codegen.templates.entries:
            if profile in entry.profiles or "{" in entry.destination:
                continue
            path = root / Path(entry.destination)
            if not path.is_file():
                continue
            current = u.Cli.files_read_text(path)
            if current.failure:
                return r[t.SequenceOf[m.Infra.CodegenFilePlan]].fail(
                    current.error or f"profile-excluded projection read failed: {path}"
                )
            if c.Infra.TEMPLATE_GENERATED_MARKER not in current.value:
                continue
            planned.append(cls._absent_file_plan(path, current.value))
        return r[t.SequenceOf[m.Infra.CodegenFilePlan]].ok(tuple(planned))


__all__: list[str] = ["FlextInfraCodegenConform"]
