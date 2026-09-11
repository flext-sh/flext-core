"""Generic file pattern matching iteration mixin.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

from flext_infra._utilities.git import FlextInfraUtilitiesGit

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraUtilitiesIterationMatching:
    """Static helpers for matching files by include/exclude glob patterns."""

    @classmethod
    def iter_matching_files(
        cls, root: Path, *, includes: t.StrSequence, excludes: t.StrSequence = ()
    ) -> t.SequenceOf[Path]:
        """Return files in one scope through the canonical git-aware selection path."""
        if not root.is_dir():
            return []
        root = root.resolve()
        tracked_files = FlextInfraUtilitiesGit.git_tracked_scope_paths(root)
        candidates = (
            tracked_files
            if tracked_files is not None
            else [path for path in root.rglob("*") if path.is_file()]
        )
        return sorted({
            path
            for path in candidates
            if path.is_file()
            if (
                not includes
                or any(
                    fnmatch.fnmatch(path.relative_to(root).as_posix(), pattern)
                    for pattern in includes
                )
            )
            if not any(
                fnmatch.fnmatch(path.relative_to(root).as_posix(), pattern)
                for pattern in excludes
            )
        })


__all__: list[str] = ["FlextInfraUtilitiesIterationMatching"]
