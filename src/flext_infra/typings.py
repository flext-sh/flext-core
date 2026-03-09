"""Type aliases for flext-infra.

Re-exports and extends flext_core typings for infrastructure services.
Infra-specific type aliases live inside ``FlextInfraTypes`` so they are
accessed via ``t.Infra.Payload``, ``t.Infra.PayloadMap``, etc.

Non-recursive aliases use ``type X = ...`` (PEP 695 Python 3.13+ syntax).
See AGENTS.md §3 AXIOMATIC rule.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Literal

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

        type Payload = (
            FlextTypes.Scalar
            | Mapping[str, FlextTypes.Scalar]
            | Sequence[FlextTypes.Scalar]
        )
        "Infrastructure payload: scalar, scalar mapping, or scalar sequence."
        type PayloadMap = Mapping[str, Payload]
        "Infrastructure payload map: string-keyed mapping of payloads."
        type Lines = list[str]
        "List of string lines (log output, violation messages, etc.)."
        type StrMap = dict[str, str]
        "Mutable string-to-string mapping (symbol replacements, renames)."
        type StrMapping = Mapping[str, str]
        "Immutable string-to-string mapping (env vars, keyword renames)."
        type MutableStrMap = MutableMapping[str, str]
        "Mutable string-to-string mapping for accumulation patterns."
        type ContainerDict = dict[str, FlextTypes.ContainerValue]
        "Dict with string keys and container values (project reports, etc.)."
        type ContainerReport = dict[str, ContainerDict]
        "Nested container dict (project-level reports)."
        type LazyImportEntry = tuple[str, str]
        "A (module_path, attr_name) pair for lazy imports."
        type LazyImportMap = dict[str, LazyImportEntry]
        "Mapping of export names to (module_path, attr_name) pairs."
        type ChangeCallback = Callable[[str], None]
        "Callback invoked when a refactoring change is applied."
        type EnvMap = Mapping[str, str] | None
        "Optional environment variable mapping for subprocess execution."
        type PathLike = str | Path
        "Flexible path representation (str or Path)."
        type InfraValue = FlextTypes.ContainerValue
        "Recursive infrastructure value: primitive, nested list/mapping, or null."
        type IssueMap = Mapping[str, InfraValue]
        "Dependency issue mapping: string-keyed mapping of infra values."
        type RuleConfig = dict[str, InfraValue]
        "A single rule configuration dict (parsed from TOML/YAML)."
        type RuleConfigList = list[RuleConfig]
        "List of rule configuration dicts."
        type OrchestrationSummary = Mapping[
            str,
            int | list[Mapping[str, FlextTypes.Scalar]],
        ]
        "Workspace PR orchestration summary."
        type FacadeFamily = Literal["c", "t", "p", "m", "u"]
        "Facade family identifier for MRO chain resolution."
        type ExpectedBase = type | str
        "Expected MRO base: a class or its qualified name."
        type PolicyContext = Mapping[str, ContainerDict]
        "Class-nesting policy matrix keyed by module family."
        type ClassFamilyMap = Mapping[str, str]
        "Mapping from symbol name to resolved module family."


t = FlextInfraTypes
__all__ = ["FlextInfraTypes", "t"]
