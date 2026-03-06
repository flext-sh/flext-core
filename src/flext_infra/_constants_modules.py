"""Centralized constants for standalone flext_infra modules.

Groups constants from modules that are NOT subpackages (paths, versioning,
reporting) into classes for MRO composition in FlextInfraConstants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import Final


class FlextInfraPathsConstants:
    """Path resolution constants for workspace navigation."""

    WORKSPACE_MARKERS: Final[frozenset[str]] = frozenset({
        ".git",
        "Makefile",
        "pyproject.toml",
    })
    """Filesystem markers used to detect workspace root directories."""


class FlextInfraVersioningConstants:
    """Semantic versioning constants for version management."""

    SEMVER_RE: Final[re.Pattern[str]] = re.compile(
        r"^(\d+)\.(\d+)\.(\d+)(?:-dev)?$",
    )
    """Regex pattern for parsing semantic version strings."""

    DEV_BRANCH_RE: Final[re.Pattern[str]] = re.compile(
        r"^(\d+\.\d+\.\d+)-dev$",
    )
    """Regex pattern for matching development branch names."""

    VALID_BUMP_TYPES: Final[frozenset[str]] = frozenset({
        "major",
        "minor",
        "patch",
    })
    """Allowed version bump type identifiers."""


class FlextInfraReportingConstants:
    """Reporting service constants for .reports/ path management."""

    REPORTS_DIR_NAME: Final[str] = ".reports"
    """Standard directory name for report output."""


__all__ = [
    "FlextInfraPathsConstants",
    "FlextInfraReportingConstants",
    "FlextInfraVersioningConstants",
]
