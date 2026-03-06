"""Type aliases for flext-infra.

Re-exports and extends flext_core typings for infrastructure services.
Infra-specific type aliases live inside ``FlextInfraTypes`` so they are
accessed via ``t.Infra.Payload``, ``t.Infra.PayloadMap``, etc.

Non-recursive aliases MUST use ``X: TypeAlias = ...`` (isinstance-safe).
See CLAUDE.md §3 AXIOMATIC rule.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import TypeAlias

from flext_core import FlextTypes


class FlextInfraTypes(FlextTypes):
    """Type namespace for flext-infra; extends FlextTypes via MRO.

    Infra-specific types are nested under the ``Infra`` inner class to
    keep the namespace explicit (``t.Infra.Payload``, ``t.Infra.StrMap``).
    Parent types (``t.Scalar``, ``t.Container``, etc.) are inherited
    transparently from ``FlextTypes`` via MRO.
    """

    # ── Infra-specific type layers ───────────────────────────────────
    class Infra:
        """Infrastructure-domain type aliases.

        These aliases compose ``FlextTypes.Scalar`` and collection generics
        for infrastructure payload contracts and common patterns.
        """

        # ── Payload types ────────────────────────────────────────────
        Payload: TypeAlias = (
            FlextTypes.Scalar
            | Mapping[str, FlextTypes.Scalar]
            | Sequence[FlextTypes.Scalar]
        )
        """Infrastructure payload: scalar, scalar mapping, or scalar sequence."""

        PayloadMap: TypeAlias = Mapping[str, Payload]
        """Infrastructure payload map: string-keyed mapping of payloads."""

        # ── String collection aliases ────────────────────────────────
        Lines: TypeAlias = list[str]
        """List of string lines (log output, violation messages, etc.)."""

        StrMap: TypeAlias = dict[str, str]
        """Mutable string-to-string mapping (symbol replacements, renames)."""

        StrMapping: TypeAlias = Mapping[str, str]
        """Immutable string-to-string mapping (env vars, keyword renames)."""

        MutableStrMap: TypeAlias = MutableMapping[str, str]
        """Mutable string-to-string mapping for accumulation patterns."""

        # ── Config / rule types ──────────────────────────────────────
        RuleConfig: TypeAlias = dict[str, object]
        """A single rule configuration dict (parsed from TOML/YAML)."""

        RuleConfigList: TypeAlias = list[RuleConfig]
        """List of rule configuration dicts."""

        ContainerDict: TypeAlias = dict[str, FlextTypes.ContainerValue]
        """Dict with string keys and container values (project reports, etc.)."""

        ContainerReport: TypeAlias = dict[str, ContainerDict]
        """Nested container dict (project-level reports)."""

        # ── Lazy import registry ─────────────────────────────────────
        LazyImportEntry: TypeAlias = tuple[str, str]
        """A (module_path, attr_name) pair for lazy imports."""

        LazyImportMap: TypeAlias = dict[str, LazyImportEntry]
        """Mapping of export names to (module_path, attr_name) pairs."""

        # ── Callable patterns ────────────────────────────────────────
        ChangeCallback: TypeAlias = Callable[[str], None]
        """Callback invoked when a refactoring change is applied."""

        EnvMap: TypeAlias = Mapping[str, str] | None
        """Optional environment variable mapping for subprocess execution."""

        # ── Path types ───────────────────────────────────────────────
        PathLike: TypeAlias = str | Path
        """Flexible path representation (str or Path)."""

        # ── TOML types ───────────────────────────────────────────────
        TomlScalar: TypeAlias = FlextTypes.Primitives
        """TOML scalar value (str | int | float | bool). Add ``| None`` at usage sites."""

        type TomlValue = (
            FlextInfraTypes.Infra.TomlScalar
            | None
            | list[FlextInfraTypes.Infra.TomlScalar | None]
            | list[FlextInfraTypes.Infra.TomlValue]
            | MutableMapping[str, FlextInfraTypes.Infra.TomlValue]
        )
        """Recursive TOML value: scalar, null, scalar list, nested list, or mapping."""

        TomlMap: TypeAlias = MutableMapping[str, TomlValue]
        """TOML mapping: string-keyed mutable mapping of TOML values."""

        TomlMutableMap: TypeAlias = MutableMapping[str, TomlValue]
        """TOML mutable mapping (alias for accumulation/modification patterns)."""


t = FlextInfraTypes

__all__ = ["FlextInfraTypes", "t"]
