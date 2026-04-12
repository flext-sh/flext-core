"""Rule 1 violation: loose collection constant at module level."""

from __future__ import annotations


class FlextTestUtilities:
    """Utilities namespace."""


VALID_STATUSES = frozenset({"active", "inactive"})
DEFAULT_HEADERS = {"content_type": "json"}
u = FlextTestUtilities
__all__ = []
