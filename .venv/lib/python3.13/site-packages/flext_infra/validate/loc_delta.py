"""Net-LOC-delta validator (AGENTS.md §3.5).

A commit whose subject is labelled ``refactor``/``deduplicate``/``cleanup``/
``yagni``/``simplify`` MUST show ``insertions - deletions <= 0``. Non-labelled
commits (feat/fix/docs/…) are exempt — they may legitimately add lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_core import r
from flext_infra import c, m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraLocDeltaValidator(s[bool]):
    """Fail refactor/cleanup commits that grow the codebase (net positive LOC)."""

    @classmethod
    def evaluate(
        cls, *, subject: str, insertions: int, deletions: int
    ) -> p.Result[bool]:
        """Pure rule: net positive delta on a labelled commit is a violation."""
        lowered = subject.lower()
        if not any(label in lowered for label in c.Infra.REFACTOR_COMMIT_LABELS):
            return r[bool].ok(True)
        delta = insertions - deletions
        if delta > 0:
            return r[bool].fail(
                f"net-LOC-delta violation (§3.5): '{subject}' adds +{delta} "
                f"(insertions={insertions}, deletions={deletions}); refactor/cleanup "
                "commits must be net non-positive"
            )
        return r[bool].ok(True)

    @staticmethod
    def _sum_numstat(numstat: str) -> tuple[int, int]:
        """Sum insertions/deletions from `git diff --numstat` output (skip binary)."""
        insertions = 0
        deletions = 0
        for line in numstat.splitlines():
            match line.split("\t"):
                case [added, removed, *_] if added.isdigit() and removed.isdigit():
                    insertions += int(added)
                    deletions += int(removed)
                case _:
                    continue
        return insertions, deletions

    @override
    def execute(self) -> p.Result[bool]:
        """Evaluate the workspace HEAD commit's labelled net-LOC delta."""
        report = u.Infra.git_head_numstat(
            m.Infra.GitRepoRequest(repo_root=self.repository_root)
        )
        if report.failure:
            return r[bool].fail(report.error or "git numstat read failed")
        insertions, deletions = self._sum_numstat(report.value.numstat)
        verdict = self.evaluate(
            subject=report.value.subject, insertions=insertions, deletions=deletions
        )
        if verdict.failure:
            return r[bool].fail(verdict.error or "net-LOC-delta violation")
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraLocDeltaValidator"]
