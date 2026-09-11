"""Private GitPython mixins for the ``u.Infra`` git facet.

External consumers never import this package — use ``from flext_infra import u``.
"""

from __future__ import annotations

from flext_infra._utilities._git.scope import FlextInfraUtilitiesGitScopeMixin
from flext_infra._utilities._git.semantic import FlextInfraUtilitiesGitSemanticMixin
from flext_infra._utilities._git.worktree import FlextInfraUtilitiesGitWorktreeMixin

__all__: list[str] = [
    "FlextInfraUtilitiesGitScopeMixin",
    "FlextInfraUtilitiesGitSemanticMixin",
    "FlextInfraUtilitiesGitWorktreeMixin",
]
