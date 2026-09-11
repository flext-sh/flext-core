"""Protected file edit helpers facade."""

from __future__ import annotations

from flext_infra._utilities.protected_edit_writes import (
    FlextInfraUtilitiesProtectedEditWrites,
)


class FlextInfraUtilitiesProtectedEdit(FlextInfraUtilitiesProtectedEditWrites):
    """Shared safety helpers for protected file edits in refactor workflows."""


__all__: list[str] = ["FlextInfraUtilitiesProtectedEdit"]
