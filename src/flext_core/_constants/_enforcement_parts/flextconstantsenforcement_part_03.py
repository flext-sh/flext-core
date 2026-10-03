"""Namespace enforcement constants for FlextConstantsEnforcement.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_01 import (
    FlextConstantsEnforcementEnums,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flext_core._typings.base import FlextTypingBase as t

class FlextConstantsEnforcementNamespace:
    """MRO namespace and violation-shape constants."""

    ENFORCEMENT_NAMESPACE_MODE: ClassVar[
        FlextConstantsEnforcementEnums.EnforcementMode
    ] = FlextConstantsEnforcementEnums.EnforcementMode.WARN
    """Separate mode for namespace checks — see EnforcementMode."""

    # SSOT seed: the five canonical facade layers. Used to derive
    # ENFORCEMENT_NAMESPACE_FACADE_ROOTS (Flext{Name}) and
    # ENFORCEMENT_NAMESPACE_LAYER_MAP ((Name, name.lower())) below — adding
    # a layer requires editing only this tuple.
    NAMESPACE_LAYER_NAMES: ClassVar[t.VariadicTuple[str]] = (
        "Constants",
        "Models",
        "Protocols",
        "Types",
        "Utilities",
    )

    NAMESPACE_FAMILY_PREFIX: ClassVar[str] = "flext_"
    """Declared family namespace prefix (import-name grammar).

    Discovery seed for runtime family-surface derivation: candidate
    distributions whose normalized name starts with this prefix are family
    members when their root publishes the lazy export contract. The prefix
    narrows discovery only; membership is proven by the published contract,
    never by an enumerated roster.
    """

    FAMILY_SURFACE_MIN_PUBLISHED: ClassVar[int] = 1
    """Minimum family roots that must publish the lazy export contract.

    Threshold for the family-surface derivation: a runtime where no
    distribution publishes the contract is a broken installation and fails
    loud instead of deriving an empty surface.
    """

    ENFORCEMENT_NAMESPACE_FACADE_ROOTS: ClassVar[frozenset[str]] = frozenset(
        {f"Flext{name}" for name in NAMESPACE_LAYER_NAMES}
        | {"FlextModelsBase", "FlextModelsNamespace", "EnforcedModel"},
    )
    """Root facade class names — skip namespace prefix check on these."""

    ENFORCEMENT_NAMESPACE_LAYER_MAP: ClassVar[t.StrPairTuple] = tuple(
        (name, name.lower()) for name in NAMESPACE_LAYER_NAMES
    )
    """Class name suffix → layer name mapping for cross-layer detection."""

    # --- Violation message shape (single parameterized template) ---
    #
    # One template covers every violation: the check supplies the
    # ``location`` (field / attribute / path / class qualname), the
    # ``problem`` (what is wrong), and the ``fix`` (remediation). Adding
    # a new check never requires editing this constant.

    ENFORCEMENT_MSG_VIOLATION: ClassVar[str] = "{location}: {problem}. {fix}"
    """Single message shape — location + problem + fix."""

    ENFORCEMENT_CANONICAL_ALIASES: ClassVar[frozenset[str]] = frozenset({
        "c",
        "m",
        "p",
        "t",
        "u",
        "d",
        "e",
        "h",
        "r",
        "s",
        "x",
    })
    """Canonical short aliases exposed by FLEXT facade namespaces."""

    ENFORCEMENT_PROJECT_ALIAS_OWNERS: ClassVar[Mapping[str, t.VariadicTuple[str]]] = (
        MappingProxyType(
            dict.fromkeys(
                (
                    "flext_api",
                    "flext_auth",
                    "flext_cli",
                    "flext_core",
                    "flext_db_oracle",
                    "flext_dbt_ldap",
                    "flext_dbt_ldif",
                    "flext_dbt_oracle",
                    "flext_dbt_oracle_wms",
                    "flext_grpc",
                    "flext_infra",
                    "flext_ldap",
                    "flext_ldif",
                    "flext_meltano",
                    "flext_observability",
                    "flext_oracle_oic",
                    "flext_oracle_wms",
                    "flext_plugin",
                    "flext_quality",
                    "flext_tap_ldap",
                    "flext_tap_ldif",
                    "flext_tap_oracle",
                    "flext_tap_oracle_oic",
                    "flext_tap_oracle_wms",
                    "flext_target_ldap",
                    "flext_target_ldif",
                    "flext_target_oracle",
                    "flext_target_oracle_oic",
                    "flext_target_oracle_wms",
                    "flext_tests",
                    "flext_web",
                ),
                ("c", "m", "p", "t", "u"),
            ),
        )
    )
    """SSOT: project package name → canonical aliases it re-exports locally.

    Used by runtime census and flext-infra detectors to flag
    ``from flext_core import c`` inside a project that owns ``c`` locally.
    """


__all__: list[str] = ["FlextConstantsEnforcementNamespace"]
