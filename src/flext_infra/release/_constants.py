"""Centralized constants for the release subpackage."""

from __future__ import annotations

import re
from typing import Final


class FlextInfraReleaseConstants:
    """Release infrastructure constants."""

    VALID_PHASES: Final[frozenset[str]] = frozenset({
        "validate",
        "version",
        "build",
        "publish",
    })
    VERSION_RE: Final[re.Pattern[str]] = re.compile(
        r'^version\s*=\s*"(.+?)"', re.MULTILINE
    )


__all__ = ["FlextInfraReleaseConstants"]
