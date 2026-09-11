"""Rewrite ``typing.Dict[K, V]`` annotations to canonical ``t.MappingKV``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, override

from flext_infra.transformers._canonical_t_import import (
    FlextInfraEnsureCanonicalTImportMixin,
)
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraRefactorTypingDictAttr(
    FlextInfraEnsureCanonicalTImportMixin, FlextInfraRopeTransformer
):
    """Rewrite ``typing.Dict[K, V]`` to ``t.MappingKV[K, V]``.

    Safe deterministic rewrite for attribute-style typing references.
    The ``typing`` module import is left untouched; only the annotation form
    is canonicalized and the ``t`` import is injected when needed.
    """

    _description = "rewrite typing.Dict to t.MappingKV"

    # Matches typing.Dict[...] usage.
    _TYPING_DICT_ATTR_RE: re.Pattern[str] = re.compile(r"\btyping\s*\.\s*Dict\s*\[")

    def __init__(self, *, file_path: Path | None = None) -> None:
        """Initialize with optional file path for canonical t import resolution."""
        super().__init__()
        self._file_path = file_path

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Rewrite typing.Dict usages and ensure t import."""
        updated = self._rewrite_dict_annotations(source)
        if updated != source:
            added, did_add = self._ensure_t_import(
                updated, self._canonical_import_module(self._file_path)
            )
            if did_add:
                self._record_change(
                    "Added canonical t import from "
                    f"{self._canonical_import_module(self._file_path)}"
                )
            updated = added
        return updated, list(self.changes)

    def _rewrite_dict_annotations(self, source: str) -> str:
        """Rewrite every ``typing.Dict[K, V]`` occurrence to ``t.MappingKV[K, V]``."""

        def replacer(_match: re.Match[str]) -> str:
            self._record_change(
                "Rewrote typing.Dict[...] annotation to t.MappingKV[...]"
            )
            return "t.MappingKV["

        return self._TYPING_DICT_ATTR_RE.sub(replacer, source)


__all__: list[str] = ["FlextInfraRefactorTypingDictAttr"]
