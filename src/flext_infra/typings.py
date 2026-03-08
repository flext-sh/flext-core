"""Type aliases for flext-infra.

Re-exports and extends flext_core typings for infrastructure services.
Infra-specific type aliases live inside ``FlextInfraTypes`` so they are
accessed via ``t.Infra.Payload``, ``t.Infra.PayloadMap``, etc.

Non-recursive aliases MUST use ``X: TypeAlias = ...`` (isinstance-safe).
See AGENTS.md §3 AXIOMATIC rule.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

from flext_core import FlextTypes


class FlextInfraTypes(FlextTypes):
    """Type namespace for flext-infra; extends FlextTypes via MRO.

    Infra-specific types are nested under the ``Infra`` inner class to
    keep the namespace explicit (``t.Infra.Payload``, ``t.Infra.StrMap``).
    Parent types (``t.Scalar``, ``t.Container``, etc.) are inherited
    transparently from ``FlextTypes`` via MRO.
    """

    class Infra:
        """Infrastructure-domain type aliases.

        These aliases compose ``FlextTypes.Scalar`` and collection generics
        for infrastructure payload contracts and common patterns.
        """

        Payload: TypeAlias = (
            FlextTypes.Scalar
            | Mapping[str, FlextTypes.Scalar]
            | Sequence[FlextTypes.Scalar]
        )
        "Infrastructure payload: scalar, scalar mapping, or scalar sequence."
        PayloadMap: TypeAlias = Mapping[str, Payload]
        "Infrastructure payload map: string-keyed mapping of payloads."
        Lines: TypeAlias = list[str]
        "List of string lines (log output, violation messages, etc.)."
        StrMap: TypeAlias = dict[str, str]
        "Mutable string-to-string mapping (symbol replacements, renames)."
        StrMapping: TypeAlias = Mapping[str, str]
        "Immutable string-to-string mapping (env vars, keyword renames)."
        MutableStrMap: TypeAlias = MutableMapping[str, str]
        "Mutable string-to-string mapping for accumulation patterns."
        ContainerDict: TypeAlias = dict[str, FlextTypes.ContainerValue]
        "Dict with string keys and container values (project reports, etc.)."
        ContainerReport: TypeAlias = dict[str, ContainerDict]
        "Nested container dict (project-level reports)."
        LazyImportEntry: TypeAlias = tuple[str, str]
        "A (module_path, attr_name) pair for lazy imports."
        LazyImportMap: TypeAlias = dict[str, LazyImportEntry]
        "Mapping of export names to (module_path, attr_name) pairs."
        ChangeCallback: TypeAlias = Callable[[str], None]
        "Callback invoked when a refactoring change is applied."
        EnvMap: TypeAlias = Mapping[str, str] | None
        "Optional environment variable mapping for subprocess execution."
        PathLike: TypeAlias = str | Path
        "Flexible path representation (str or Path)."
        InfraValue: TypeAlias = FlextTypes.ContainerValue
        "Recursive infrastructure value: primitive, nested list/mapping, or null."
        IssueMap: TypeAlias = Mapping[str, InfraValue]
        "Dependency issue mapping: string-keyed mapping of infra values."
        RuleConfig: TypeAlias = dict[str, InfraValue]
        "A single rule configuration dict (parsed from TOML/YAML)."
        RuleConfigList: TypeAlias = list[RuleConfig]
        "List of rule configuration dicts."
        OrchestrationSummary: TypeAlias = Mapping[
            str, int | list[Mapping[str, FlextTypes.Scalar]],
        ]
        "Workspace PR orchestration summary."
        FacadeFamily: TypeAlias = Literal["c", "t", "p", "m", "u"]
        "Facade family identifier for MRO chain resolution."
        ExpectedBase: TypeAlias = type | str
        "Expected MRO base: a class or its qualified name."
        PolicyContext: TypeAlias = Mapping[str, ContainerDict]
        "Class-nesting policy matrix keyed by module family."
        ClassFamilyMap: TypeAlias = Mapping[str, str]
        "Mapping from symbol name to resolved module family."


t = FlextInfraTypes
__all__ = ["FlextInfraTypes", "t"]
