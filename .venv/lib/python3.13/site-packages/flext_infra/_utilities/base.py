"""Base utilities for flext-infra project.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra.constants import c
from flext_infra.typings import t

if TYPE_CHECKING:
    from flext_infra.protocols import p


class FlextInfraUtilitiesBase:
    """Base utilities for flext-infra project.

    Provides primitive helpers used across all infra utility subclasses.
    Generic ``validate`` and ``deep`` methods use PEP 695 type parameters
    so callers can validate ANY shape with a single SSOT helper.
    """

    @staticmethod
    def resolve_repository_root_or_cwd(repository_root: Path | None = None) -> Path:
        """Resolve the root a verb operates on from its invocation point.

        Scope follows where the verb is invoked: run it at the workspace and it
        works on the whole active workspace; run it inside a project and it
        works on that project alone. The checkout is therefore the root, and a
        member is never escalated to its enclosing superproject.

        Escalating inverted that rule. A verb invoked inside one member
        resolved every relative path against the superproject shared by all
        sibling worktrees, so `FILE=` selectors rejected files that exist and
        `.reports/tests/latest.txt` -- the canonical evidence artifact -- was
        written to the shared root, where each project's run overwrote the
        previous one's result.
        """
        target = repository_root or Path.cwd()
        if target.is_file():
            target = target.parent
        return target.resolve()

    @staticmethod
    def normalize_optional_path(value: str | Path | None) -> Path | None:
        """Resolve one optional path-like value when present."""
        if value is None:
            return None
        path = value if isinstance(value, Path) else Path(value)
        return path.resolve()

    @staticmethod
    def normalize_cli_values(*values: str | None) -> t.StrSequence:
        """Normalize comma-separated or whitespace-separated CLI selectors."""
        return tuple(
            item.strip()
            for value in values
            for group in (value or "").split(",")
            for item in group.split()
            if item.strip()
        )

    @staticmethod
    def normalize_sequence_values(values: t.StrSequence | None) -> t.StrSequence | None:
        """Normalize repeated CLI sequence fields into a compact selector list."""
        names = FlextInfraUtilitiesBase.normalize_cli_values(*(values or ()))
        return names or None

    @staticmethod
    def normalize_make_args(values: t.StrSequence) -> t.StrSequence:
        """Return trimmed make arguments without blank entries."""
        return tuple(item.strip() for item in values if item.strip())

    @staticmethod
    def strongly_connected_components(
        graph: t.MappingKV[str, set[str]],
    ) -> t.SequenceOf[t.StrSequence]:
        """Return every strongly connected component in one directed graph."""
        next_index = 0
        stack: list[str] = []
        indexes: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        on_stack: set[str] = set()
        components: list[t.StrSequence] = []

        def visit(node: str) -> None:
            nonlocal next_index
            indexes[node] = next_index
            lowlinks[node] = next_index
            next_index += 1
            stack.append(node)
            on_stack.add(node)
            for successor in graph.get(node, set()):
                if successor not in indexes:
                    visit(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indexes[successor])
            if lowlinks[node] != indexes[node]:
                return
            component: list[str] = []
            while stack:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(tuple(component))

        for node in graph:
            if node not in indexes:
                visit(node)
        return tuple(components)

    @staticmethod
    def classify_process_exit(exit_code: int) -> str:
        """Classify a nonzero process status as timeout, signal, or failure."""
        if exit_code == c.Infra.PROCESS_TIMEOUT_EXIT_CODE:
            return "timeout"
        if exit_code < 0:
            return f"signal={-exit_code}"
        if exit_code >= c.Infra.PROCESS_SIGNAL_EXIT_OFFSET:
            return f"signal={exit_code - c.Infra.PROCESS_SIGNAL_EXIT_OFFSET}"
        return "failure"

    @staticmethod
    def normalize_process_exit_code(raw_exit_code: int) -> int:
        """Map a subprocess signal return code into the portable shell domain."""
        if raw_exit_code < 0:
            normalized_exit_code: int = (
                c.Infra.PROCESS_SIGNAL_EXIT_OFFSET - raw_exit_code
            )
            return normalized_exit_code
        return raw_exit_code

    @staticmethod
    def resolve_what(verb: str, phase: str) -> p.Result[t.StrSequence]:
        """Resolve a ``WHAT=`` phase against ``c.Infra.WHAT_PHASES`` (single SSOT).

        Empty ``phase`` expands to every phase of ``verb`` (sorted); a non-empty
        unknown phase is a usage failure listing the valid phases. Shared by the
        orchestrator, check and validate groups so WHAT resolution lives in one
        place.
        """
        phases = c.Infra.WHAT_PHASES.get(verb, frozenset())
        if not phase:
            return r[t.StrSequence].ok(tuple(sorted(phases)))
        if phase not in phases:
            valid = ", ".join(sorted(phases)) or "(none)"
            return r[t.StrSequence].fail(
                f"unknown WHAT '{phase}' for verb '{verb}' (valid: {valid})"
            )
        return r[t.StrSequence].ok((phase,))


__all__: list[str] = ["FlextInfraUtilitiesBase"]
