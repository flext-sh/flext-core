"""Base class and registry for automatic smell fixers.

Every fixer is AST-based, opt-in, and records every change it makes.
Fixers never silently change semantics: they only rewrite patterns that
are provably equivalent under FLEXT law.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Final

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, t


class FlextInfraSmellFixer:
    """Abstract base for a smell-specific automatic fixer.

    A fixer targets exactly one smell ``tag`` (e.g. ``smell_boolean_logic``)
    and rewrites source files in-place when asked. It reports whether it
    actually changed anything and a human-readable list of changes.
    """

    tag: ClassVar[str]

    def __init__(self, *, on_change: t.Infra.ChangeCallback = None) -> None:
        """Initialize change tracking."""
        self._on_change = on_change
        self.changes: list[str] = []

    def can_fix(self, issue: m.Infra.Issue) -> bool:
        """Return True when this fixer handles the given issue code."""
        issue_code: str = issue.code
        return issue_code == self.tag

    @abstractmethod
    def fix(self, project_dir: Path, issue: m.Infra.Issue) -> tuple[bool, list[str]]:
        """Attempt to fix ``issue`` in ``project_dir / issue.file``.

        Returns ``(fixed, changes)``. ``fixed`` is True only when the source
        file was actually rewritten. ``changes`` is a list of human-readable
        descriptions of the edits.
        """
        ...

    def _record_change(self, message: str) -> None:
        """Record one change and optionally notify a callback."""
        self.changes.append(message)
        if self._on_change is not None:
            self._on_change(message)


_SMELL_FIXERS: Final[dict[str, type[FlextInfraSmellFixer]]] = {}


def register_smell_fixer(
    fixer_class: type[FlextInfraSmellFixer],
) -> type[FlextInfraSmellFixer]:
    """Register a smell fixer under its class ``tag``."""
    _SMELL_FIXERS[fixer_class.tag] = fixer_class
    return fixer_class


def smell_fixer_for(code: str) -> FlextInfraSmellFixer | None:
    """Return a fresh fixer instance for ``code``, or None when absent."""
    fixer_class = _SMELL_FIXERS.get(code)
    return None if fixer_class is None else fixer_class()


def auto_fixable_smell_tags() -> tuple[str, ...]:
    """Return tags of all registered smell fixers."""
    return tuple(_SMELL_FIXERS.keys())


__all__: list[str] = [
    "FlextInfraSmellFixer",
    "auto_fixable_smell_tags",
    "register_smell_fixer",
    "smell_fixer_for",
]
