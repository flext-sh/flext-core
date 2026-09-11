"""Detect hardcoded ``__version__`` assignments and report them.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, override

from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorHardcodedVersion(FlextInfraRopeTransformer):
    r"""Detect hardcoded ``__version__ = \"...\"`` assignments.

    This transformer is **not safe**: it only reports the violation because
    the canonical fix requires reading the version from ``importlib.metadata``
    (or the project's package metadata). Consumers should apply the reported
    change manually or via an unsafe enforcement run.
    """

    _description = "detect hardcoded __version__ assignment"

    # Matches `__version__ = "..."` or `__version__ = '...'`.
    _HARDCODED_VERSION_RE: re.Pattern[str] = re.compile(
        r"^__version__\s*=\s*(['\"])[^'\"]*\1", re.MULTILINE
    )

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Record a change for each hardcoded __version__ assignment."""
        for match in self._HARDCODED_VERSION_RE.finditer(source):
            self._record_change(
                f"Hardcoded __version__ assignment found: {match.group(0).strip()!r}. "
                "Use importlib.metadata.version(__name__.split('.')[0]) instead."
            )
        return source, list(self.changes)


__all__: list[str] = ["FlextInfraRefactorHardcodedVersion"]
