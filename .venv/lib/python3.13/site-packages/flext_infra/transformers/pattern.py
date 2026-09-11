"""Generic pattern-based transformer driven by catalog rules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c
from flext_infra.transformers._canonical_t_import import (
    FlextInfraEnsureCanonicalTImportMixin,
)
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraRefactorPatternTransformer(
    FlextInfraEnsureCanonicalTImportMixin, FlextInfraRopeTransformer
):
    """Apply declarative regex substitutions declared in enforcement catalog rules.

    Each catalog entry declares a list of ``patterns`` (regex + replacement +
    change message) and an optional canonical ``required_alias`` to import when
    the transformation introduces uses of that alias. This replaces a fleet of
    tiny single-purpose transformers with one rule-driven engine.
    """

    _description = "apply declarative regex patterns from catalog"

    _FLAG_MAP: ClassVar[t.MappingKV[str, int]] = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }

    def __init__(
        self,
        *,
        patterns: t.SequenceOf[t.MappingKV[str, t.JsonValue]],
        required_alias: str = "",
        alias_module: str = "",
        file_path: Path | None = None,
    ) -> None:
        """Initialize with catalog patterns and optional alias injection.

        Args:
            patterns: Sequence of pattern specs. Each spec must contain
                ``regex`` and ``replacement`` strings and may contain
                ``change_message`` and ``flags`` (list of flag names).
            required_alias: Alias to import if used after transformation
                (e.g. ``t`` or ``u``).
            alias_module: Explicit module to import the alias from. When empty,
                inferred from ``file_path`` or the canonical FLEXT core module.
            file_path: Optional file path for module inference.

        """
        super().__init__()
        self._patterns = tuple(patterns)
        self._required_alias = required_alias
        self._alias_module = (
            alias_module
            or self._canonical_import_module(file_path)
            or c.Infra.PKG_CORE_UNDERSCORE
        )

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply all declared patterns and ensure required alias import."""
        updated = source
        for pattern_spec in self._patterns:
            updated = self._apply_pattern(updated, pattern_spec)
        if updated != source and self._required_alias:
            updated, _did_add = self._ensure_alias_import(
                source=updated,
                module_name=self._alias_module,
                alias=self._required_alias,
            )
        return updated, list(self.changes)

    def _apply_pattern(
        self, source: str, pattern_spec: t.MappingKV[str, t.JsonValue]
    ) -> str:
        """Apply one regex substitution spec to source."""
        raw_regex = pattern_spec.get("regex")
        raw_replacement = pattern_spec.get("replacement")
        if not isinstance(raw_regex, str) or not isinstance(raw_replacement, str):
            msg = "pattern specs require string regex and replacement fields"
            raise TypeError(msg)
        flags = self._compile_flags(pattern_spec.get("flags"))
        try:
            compiled = re.compile(raw_regex, flags)
        except c.Infra.REGEX_ERROR as exc:
            msg = f"invalid enforcement pattern regex {raw_regex!r}: {exc}"
            raise ValueError(msg) from exc

        raw_change_message = pattern_spec.get("change_message")
        if raw_change_message is None:
            change_message = "Applied pattern"
        elif isinstance(raw_change_message, str):
            change_message = raw_change_message
        else:
            msg = "pattern change_message must be a string when provided"
            raise TypeError(msg)

        def replacer(match: re.Match[str]) -> str:
            self._record_change(change_message)
            return match.expand(raw_replacement)

        return compiled.sub(replacer, source)

    @classmethod
    def _compile_flags(cls, flags_value: t.JsonValue | None) -> int:
        """Return combined regex flags from a list of flag names."""
        if flags_value is None:
            return 0
        if not isinstance(flags_value, (list, tuple)):
            msg = "pattern flags must be a list of flag names"
            raise TypeError(msg)
        combined = 0
        for name in flags_value:
            if not isinstance(name, str):
                msg = "pattern flag names must be strings"
                raise TypeError(msg)
            if name not in cls._FLAG_MAP:
                msg = f"unknown pattern flag {name!r}"
                raise ValueError(msg)
            combined |= cls._FLAG_MAP[name]
        return combined


__all__: list[str] = ["FlextInfraRefactorPatternTransformer"]
