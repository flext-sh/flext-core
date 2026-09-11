"""Git constants for flext-infra project.

Centralizes git mode/stage/remote/sha constants consumed by the
``_utilities/_git/`` facet so the GitPython-backed operations never
hardcode magic numbers or strings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextInfraConstantsGit:
    """Git-specific constants for the GitPython-backed facet."""

    # --- Index entry modes (git mode field, not POSIX st_mode) ---

    GIT_MODE_FILE: Final[int] = 0o100644
    "Normal tracked file index mode."
    GIT_MODE_EXECUTABLE: Final[int] = 0o100755
    "Executable tracked file index mode."
    GIT_MODE_SYMLINK: Final[int] = 0o120000
    "Symbolic link index mode."
    GIT_MODE_GITLINK: Final[int] = 0o160000
    "Submodule gitlink index mode (used by BaseIndexEntry)."
    GIT_CACHEINFO_GITLINK: Final[str] = "160000"
    "Submodule gitlink mode string for ``git update-index --cacheinfo``."

    # --- Index stage values ---

    GIT_STAGE_NORMAL: Final[int] = 0
    "Normal (non-conflict) index stage."

    # --- SHA lengths ---

    GIT_OID_HEX_LENGTH_SHA1: Final[int] = 40
    "Hex length of a SHA-1 Git object id."
    GIT_OID_HEX_LENGTH_SHA256: Final[int] = 64
    "Hex length of a SHA-256 Git object id."

    # --- Remote defaults ---

    GIT_DEFAULT_REMOTE: Final[str] = "origin"
    "Canonical upstream remote name."

    # --- Ref prefixes ---

    GIT_REFS_HEADS: Final[str] = "refs/heads/"
    "Local branch ref prefix."


__all__: list[str] = ["FlextInfraConstantsGit"]
