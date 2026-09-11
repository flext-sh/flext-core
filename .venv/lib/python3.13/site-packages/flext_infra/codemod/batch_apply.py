"""Batch ast-grep rule application guarded by the operator safety circuit.

Circuit contract (``make mod APPLY=Y``): measure ruff + pyrefly error counts
before the batch apply, checkpoint the tree (commit when dirty), apply every
rule discovered through the package cascade (``flext_infra.codemod.discovery``)
plus the project's own hand-written ``ast-grep-rules/``, then re-measure. Any
count increase rolls the tree back to the checkpoint and fails loud; equal or
lower counts keep the applied fixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, override

from flext_cli import cli
from flext_infra import config, m, p, r, t, u
from flext_infra.base import FlextInfraServiceBase
from flext_infra.codemod.batch_gates import FlextInfraModGateEngine
from flext_infra.codemod.discovery import discover_rules

_CHECKPOINT_MESSAGE: Final[str] = "chore(git): checkpoint before ast-grep batch apply"


class FlextInfraCodemodBatchApply(FlextInfraServiceBase[t.Cli.ResultValue]):
    """Apply the discovered ast-grep rule batch under the rollback circuit."""

    @staticmethod
    def _checkpoint(root: Path) -> p.Result[str]:
        """Record the pre-apply state: checkpoint commit when dirty, else HEAD."""
        status = u.Infra.git_status(m.Infra.GitStatusRequest(repo_root=root))
        if status.failure:
            return r[str].fail(status.error or "git status failed")
        if status.value.dirty:
            checkpoint = u.Infra.git_checkpoint_worktree(
                root, message=_CHECKPOINT_MESSAGE
            )
            if checkpoint.failure:
                return r[str].fail(checkpoint.error or "checkpoint commit failed")
            return r[str].ok(checkpoint.value)
        head = u.Infra.git_repository_head(m.Infra.GitRepoRequest(repo_root=root))
        if head.failure:
            return r[str].fail(head.error or "failed to resolve HEAD")
        return r[str].ok(head.value.oid)

    @staticmethod
    def _rollback(root: Path, checkpoint_sha: str) -> str | None:
        """Restore tracked paths to the checkpoint; return an error on failure."""
        restored = u.Infra.git_checkout_restore(m.Infra.GitRepoRequest(repo_root=root))
        if restored.failure:
            return (
                restored.error
                or f"rollback failed; restore manually to {checkpoint_sha}"
            )
        return None

    def _rules(self) -> p.Result[t.SequenceOf[Path]]:
        """Resolve the batch: packaged cascade plus project-local own rules.

        Project-local ``ast-grep-rules/`` files are applied last so a
        hand-written rule overrides a packaged rule with the same rule ID.
        """
        rules = {rule.stem: rule for rule in discover_rules()}
        for rule_dir_name in config.Infra.codegen.sgconfig.rule_dirs:
            rule_dir = self.repository_root / rule_dir_name
            if not rule_dir.is_dir():
                continue
            for rule_file in sorted(rule_dir.rglob("*.yml")):
                rules[rule_file.stem] = rule_file
        if not rules:
            return r[t.SequenceOf[Path]].fail(
                f"no ast-grep rules discovered for {self.repository_root}"
            )
        return r[t.SequenceOf[Path]].ok(tuple(sorted(rules.values())))

    @override
    def execute(self) -> p.Result[t.Cli.ResultValue]:
        """Run check mode (report pending fixes) or the guarded apply circuit."""
        root = self.repository_root
        rules_result = self._rules()
        if rules_result.failure:
            return r[t.Cli.ResultValue].fail(
                rules_result.error or "ast-grep rule discovery failed"
            )
        rules = rules_result.value
        effective_dry_run: bool = self.effective_dry_run
        if effective_dry_run:
            pending = FlextInfraModGateEngine.scan(root, rules, fix=False)
            if pending.failure:
                return r[t.Cli.ResultValue].fail(
                    pending.error or "ast-grep scan failed"
                )
            if pending.value.nodes:
                return r[t.Cli.ResultValue].fail(
                    f"{pending.value.nodes} pending actionable ast-grep fix(es) "
                    f"across {len(rules)} discovered rule file(s)"
                )
            cli.display_text("mod: no pending ast-grep fixes")
            return r[t.Cli.ResultValue].ok(True)
        return self._execute_apply(root, rules)

    def _execute_apply(
        self, root: Path, rules: t.SequenceOf[Path]
    ) -> p.Result[t.Cli.ResultValue]:
        """Measure, checkpoint, batch-apply, re-measure, roll back on regression."""
        baseline = FlextInfraModGateEngine.measure(root)
        if baseline.failure:
            return r[t.Cli.ResultValue].fail(
                baseline.error or "baseline measure failed"
            )
        checkpoint = self._checkpoint(root)
        if checkpoint.failure:
            return r[t.Cli.ResultValue].fail(checkpoint.error or "checkpoint failed")
        checkpoint_sha = checkpoint.value
        pending = FlextInfraModGateEngine.scan(root, rules, fix=False)
        if pending.failure:
            return r[t.Cli.ResultValue].fail(pending.error or "ast-grep scan failed")
        applied = FlextInfraModGateEngine.scan(root, rules, fix=True)
        if applied.failure:
            return self._fail_with_rollback(
                root, checkpoint_sha, applied.error or "ast-grep fix pass failed"
            )
        remaining = FlextInfraModGateEngine.scan(root, rules, fix=False)
        if remaining.failure:
            return self._fail_with_rollback(
                root,
                checkpoint_sha,
                remaining.error or "ast-grep verification scan failed",
            )
        if remaining.value.nodes:
            return self._fail_with_rollback(
                root,
                checkpoint_sha,
                f"{remaining.value.nodes} actionable finding(s) remained after apply",
            )
        verified_nodes = pending.value.nodes - remaining.value.nodes
        changed_files = len(applied.value.files)
        final = FlextInfraModGateEngine.measure(root)
        if final.failure:
            return r[t.Cli.ResultValue].fail(final.error or "final measure failed")
        if FlextInfraModGateEngine.circuit_broken(baseline.value, final.value):
            regression = (
                f"mod circuit: gates regressed "
                f"(ruff {baseline.value.ruff_errors}→{final.value.ruff_errors}, "
                f"pyrefly {baseline.value.pyrefly_errors}→{final.value.pyrefly_errors})"
            )
            return self._fail_with_rollback(root, checkpoint_sha, regression)
        cli.display_text(
            f"mod: applied {verified_nodes} node(s) across {changed_files} file(s); "
            f"ruff {baseline.value.ruff_errors}→{final.value.ruff_errors}, "
            f"pyrefly {baseline.value.pyrefly_errors}→{final.value.pyrefly_errors}; "
            f"checkpoint {checkpoint_sha}"
        )
        return r[t.Cli.ResultValue].ok(True)

    @staticmethod
    def _fail_with_rollback(
        root: Path, checkpoint_sha: str, detail: str
    ) -> p.Result[t.Cli.ResultValue]:
        """Roll back to the checkpoint and fail loud with the circuit detail."""
        rollback_error = FlextInfraCodemodBatchApply._rollback(root, checkpoint_sha)
        if rollback_error is not None:
            return r[t.Cli.ResultValue].fail(f"{detail}; {rollback_error}")
        return r[t.Cli.ResultValue].fail(f"{detail}; rolled back to {checkpoint_sha}")


__all__: list[str] = ["FlextInfraCodemodBatchApply"]
