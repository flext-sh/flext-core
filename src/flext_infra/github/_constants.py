"""Centralized constants for the github subpackage."""

from __future__ import annotations

from typing import Final


class FlextInfraGithubConstants:
    """Github infrastructure constants."""

    MANAGED_FILES: Final[frozenset[str]] = frozenset({"ci.yml"})


__all__ = ["FlextInfraGithubConstants"]
