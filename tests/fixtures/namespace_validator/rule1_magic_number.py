"""Rule 1 violation: loose collection constant at module level."""

from __future__ import annotations


class FlextTestUtilities:
    """Utilities namespace."""

    pass


VALID_STATUSES = frozenset({"active", "inactive"})  # VIOLATION — loose frozenset
DEFAULT_HEADERS = {"content_type": "json"}  # VIOLATION — loose dict
u = FlextTestUtilities
__all__ = ["FlextTestUtilities", "u"]
