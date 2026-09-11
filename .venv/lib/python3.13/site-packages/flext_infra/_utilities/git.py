"""Public Git utilities facet for ``u.Infra`` (composed into utilities FLEXT).

Private GitPython parts live under ``_utilities/_git/``. Consumers use
``from flext_infra import u`` only — never import this module or ``_git``.
"""

from __future__ import annotations

from flext_infra._utilities._git.attestation import (
    FlextInfraUtilitiesGitAttestationMixin,
)
from flext_infra._utilities._git.scope import FlextInfraUtilitiesGitScopeMixin
from flext_infra._utilities._git.semantic_submodule import (
    FlextInfraUtilitiesGitSemanticSubmoduleMixin,
)


class FlextInfraUtilitiesGit(
    FlextInfraUtilitiesGitAttestationMixin,
    FlextInfraUtilitiesGitScopeMixin,
    FlextInfraUtilitiesGitSemanticSubmoduleMixin,
):
    """Canonical Git owner for flext-infra: scope + worktree + checkpoint/patch.

    The private mixins form TWO chains, and this facet is where they meet:

      scope -> semantic -> worktree -> ... -> repo
      submodule -> identity -> semantic_worktree -> index -> paths -> publish
                -> refs -> worktree -> ... -> repo

    Only the first was composed, so everything the second chain owns —
    ``git_submodule_init``, ``git_submodule_sections``,
    ``git_submodule_config_value``, ``git_staged_gitlink_oid`` — was absent
    from ``u.Infra`` even though the modules defining them shipped and were
    exported. ``worktree_provisioning.py`` calls all four, so the facade
    advertised an API that resolved to nothing at runtime.

    Composing at the facet, rather than inserting the submodule mixin into
    ``GitSemanticMixin``, is what keeps the two chains from colliding: they
    share ``worktree`` as a base, so joining them mid-chain re-derives the same
    methods through two paths and every shared member becomes an override.
    """


__all__: list[str] = ["FlextInfraUtilitiesGit"]
