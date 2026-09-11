"""Write and restore flows facade for protected edit workflows."""

from __future__ import annotations

from flext_infra._utilities.protected_edit_apply import (
    FlextInfraUtilitiesProtectedEditApply,
)


class FlextInfraUtilitiesProtectedEditWrites(FlextInfraUtilitiesProtectedEditApply):
    """Shared write, rollback, and backup helpers for protected edits."""


__all__: list[str] = ["FlextInfraUtilitiesProtectedEditWrites"]
