"""Centralized constants for the github subpackage."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextInfraConstantsGithub:
    """Github infrastructure constants."""

    @unique
    class WorkflowLintStatus(StrEnum):
        """GitHub workflow lint status enumeration (single source of truth).

        DRY Pattern:
            StrEnum is the single source of truth. Use WorkflowLintStatus.OK.value
            or WorkflowLintStatus.OK directly - no base strings needed.

        Represents all possible outcomes of GitHub workflow linting operations.
        """

        OK = "ok"
        SKIPPED = "skipped"
        FAIL = "fail"

    MANAGED_FILES: Final[frozenset[str]] = frozenset({"ci.yml"})


__all__: list[str] = ["FlextInfraConstantsGithub"]
