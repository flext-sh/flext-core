"""Refactor orchestration scope mixin (project/workspace + safety flow)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, t
    from flext_infra.refactor.loader import FlextInfraRefactorRuleLoader
    from flext_infra.refactor.safety import FlextInfraRefactorSafetyManager


class FlextInfraRefactorOrchestratorScopeMixin:
    """Provide project/workspace refactor scopes with checkpoint/rollback flow."""

    if TYPE_CHECKING:
        loader: FlextInfraRefactorRuleLoader
        safety_manager: FlextInfraRefactorSafetyManager

        def refactor_files(
            self,
            file_paths: t.SequenceOf[Path],
            *,
            dry_run: bool = False,
            gates: t.StrSequence | None = None,
        ) -> t.SequenceOf[m.Infra.Result]: ...

        @staticmethod
        def _error_result(fp: Path, error: str) -> m.Infra.Result: ...

        @staticmethod
        def _refactor_header(message: str) -> None: ...

    def _try_safety_checkpoint(
        self, target: Path, *, apply_safety: bool, dry_run: bool
    ) -> t.Pair[str, t.SequenceOf[m.Infra.Result] | None]:
        """Try safety checkpoint."""
        if not apply_safety or dry_run:
            return "", None
        checkpoint = self.safety_manager.create_pre_transformation_checkpoint(target)
        if checkpoint.failure:
            msg = checkpoint.error or "pre-transformation checkpoint failed"
            u.Cli.error(msg)
            return "", [self._error_result(target, msg)]
        return checkpoint.value, None

    def _finalize_safety(
        self,
        *,
        target: Path,
        checkpoint_ref: str,
        processed_targets: t.StrSequence,
        results: t.MutableSequenceOf[m.Infra.Result],
    ) -> None:
        """Finalize safety."""
        checkpoint = self.safety_manager.save_checkpoint_state(
            target,
            status="post-transform",
            checkpoint_ref=checkpoint_ref,
            processed_targets=processed_targets,
        )
        if checkpoint.failure:
            self._abort_with_rollback(
                target=target,
                checkpoint_ref=checkpoint_ref,
                results=results,
                msg=checkpoint.error or "checkpoint save failed",
            )
            return
        validation = self.safety_manager.run_semantic_validation(target)
        if validation.failure:
            self._abort_with_rollback(
                target=target,
                checkpoint_ref=checkpoint_ref,
                results=results,
                msg=validation.error or "semantic validation failed",
            )
            return
        cleared = self.safety_manager.clear_checkpoint(
            keep=[
                result.file_path
                for result in results
                if result.success and result.modified
            ]
        )
        if cleared.failure:
            u.Cli.error(cleared.error or "checkpoint clear failed")

    def _abort_with_rollback(
        self,
        *,
        target: Path,
        checkpoint_ref: str,
        results: t.MutableSequenceOf[m.Infra.Result],
        msg: str,
    ) -> None:
        """Stop processing, log, attempt rollback, and append a failure result.

        Centralizes the "fail + rollback + record" pattern shared by every
        post-transform safety failure path so each failure branch stays one
        line at the call site (DRY + fail-loud — rollback errors are also
        logged, never silently swallowed).
        """
        self.safety_manager.request_emergency_stop(msg)
        u.Cli.error(msg)
        rollback = self.safety_manager.rollback(target, checkpoint_ref)
        if rollback.failure:
            u.Cli.error(rollback.error or "rollback failed")
        results.append(self._error_result(target, msg))

    def refactor_project(
        self,
        project_path: Path,
        *,
        dry_run: bool = False,
        pattern: str = c.Infra.EXT_PYTHON_GLOB,
        apply_safety: bool = True,
        gates: t.StrSequence | None = None,
    ) -> t.SequenceOf[m.Infra.Result]:
        """Refactor files under configured project directories."""
        checkpoint_ref, error_results = self._try_safety_checkpoint(
            project_path, apply_safety=apply_safety, dry_run=dry_run
        )
        if error_results is not None:
            results_out: t.SequenceOf[m.Infra.Result] = error_results
            return results_out
        collected = u.Infra.collect_refactor_project_files(
            self.loader.settings, project_path, pattern=pattern
        )
        if collected is None:
            return [
                self._error_result(
                    project_path, f"File iteration failed for {project_path}"
                )
            ]
        u.Cli.info(f"Found {len(collected)} files to process")
        results: t.MutableSequenceOf[m.Infra.Result] = []
        results.extend(self.refactor_files(collected, dry_run=dry_run, gates=gates))
        if apply_safety and not dry_run:
            self._finalize_safety(
                target=project_path,
                checkpoint_ref=checkpoint_ref,
                processed_targets=[str(project_path)],
                results=results,
            )
        return results

    def refactor_workspace(
        self,
        repository_root: Path,
        *,
        dry_run: bool = False,
        pattern: str = c.Infra.EXT_PYTHON_GLOB,
        apply_safety: bool = True,
        gates: t.StrSequence | None = None,
    ) -> t.SequenceOf[m.Infra.Result]:
        """Refactor all discoverable workspace projects."""
        root = repository_root.resolve()
        if not root.exists() or not root.is_dir():
            u.Cli.error(f"Invalid repository root: {repository_root}")
            return []
        projects = u.Infra.discover_refactor_projects(self.loader.settings, root)
        if not projects:
            u.Cli.error(f"No projects discovered under: {repository_root}")
            return []
        u.Cli.info(f"Discovered {len(projects)} projects in workspace")
        checkpoint_ref, error_results = self._try_safety_checkpoint(
            root, apply_safety=apply_safety, dry_run=dry_run
        )
        if error_results is not None:
            results_out: t.SequenceOf[m.Infra.Result] = error_results
            return results_out
        results: t.MutableSequenceOf[m.Infra.Result] = []
        processed: t.MutableSequenceOf[str] = []
        for project in projects:
            if apply_safety and self.safety_manager.emergency_stop_requested:
                break
            self._refactor_header(f"Project: {project}")
            results.extend(
                self.refactor_project(
                    project,
                    dry_run=dry_run,
                    pattern=pattern,
                    apply_safety=False,
                    gates=gates,
                )
            )
            if apply_safety and not dry_run:
                processed.append(str(project))
        if apply_safety and not dry_run:
            self._finalize_safety(
                target=root,
                checkpoint_ref=checkpoint_ref,
                processed_targets=processed,
                results=results,
            )
        return results


__all__: list[str] = ["FlextInfraRefactorOrchestratorScopeMixin"]
