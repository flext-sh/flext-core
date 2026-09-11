"""Fail-closed recovery for owner-declared managed document conflicts."""

from __future__ import annotations

from flext_infra import p, r, t
from flext_infra.codegen.managed_conflicts_core import (
    ManagedConflictError,
    recover_managed_toml,
)


class FlextInfraCodegenManagedConflicts:
    """Recover only merge blocks explicitly owned by document configuration."""

    @classmethod
    def recover_toml(
        cls, content: str, *, conflict_sections: t.StrSequence
    ) -> p.Result[str]:
        """Choose the current projection inside configured TOML sections only."""
        try:
            recovered = recover_managed_toml(
                content, conflict_sections=conflict_sections
            )
        except ManagedConflictError as exc:
            return r[str].fail(str(exc))
        return r[str].ok(recovered)


__all__: tuple[str, ...] = ("FlextInfraCodegenManagedConflicts",)
