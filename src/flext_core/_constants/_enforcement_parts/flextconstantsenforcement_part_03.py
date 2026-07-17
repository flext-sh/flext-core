"""Namespace enforcement constants for FlextConstantsEnforcement."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, TYPE_CHECKING

from .flextconstantsenforcement_part_01 import FlextConstantsEnforcementEnums


if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as t
    from collections.abc import Mapping


class FlextConstantsEnforcementNamespace:
    """MRO namespace and violation-shape constants."""

    ENFORCEMENT_NAMESPACE_MODE: Final[
        FlextConstantsEnforcementEnums.EnforcementMode
    ] = FlextConstantsEnforcementEnums.EnforcementMode.WARN
    """Separate mode for namespace checks — see EnforcementMode."""

    # SSOT seed: the five canonical facade layers. Used to derive
    # ENFORCEMENT_NAMESPACE_FACADE_ROOTS (Flext{Name}) and
    # ENFORCEMENT_NAMESPACE_LAYER_MAP ((Name, name.lower())) below — adding
    # a layer requires editing only this tuple.
    NAMESPACE_LAYER_NAMES: Final[tuple[str, ...]] = (
        "Constants",
        "Models",
        "Protocols",
        "Types",
        "Utilities",
    )

    ENFORCEMENT_NAMESPACE_FACADE_ROOTS: Final[frozenset[str]] = frozenset(
        {f"Flext{name}" for name in NAMESPACE_LAYER_NAMES}
        | {"FlextModelsBase", "FlextModelsNamespace", "EnforcedModel"}
    )
    """Root facade class names — skip namespace prefix check on these."""

    ENFORCEMENT_NAMESPACE_LAYER_MAP: Final[t.StrPairTuple] = tuple(
        (name, name.lower()) for name in NAMESPACE_LAYER_NAMES
    )
    """Class name suffix → layer name mapping for cross-layer detection."""

    NAMESPACE_CLASS_TO_MODULE_OVERRIDES: Final[Mapping[str, str]] = MappingProxyType({})
    """Class-name → owning-package overrides for facade-layer classes that
    do not follow the ``Flext<Project><Layer><Concern>`` convention.

    Consumed by ``FlextUtilitiesEnforcement.class_name_to_module`` for both
    detection (rules that flag a wrong import path) and correction (refactor
    verbs that emit the right ``from <module> import <Class>`` line). Keep
    this empty until a real exception is encountered — adding an entry is a
    declaration that the workspace genuinely deviates from the convention,
    and that deviation must be justified at the call site that needs it."""

    ENFORCEMENT_LAYER_ALLOWS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
        "constants": frozenset({"StrEnum"}),
        "models": frozenset(),
        "protocols": frozenset({"Protocol"}),
        "types": frozenset(),
        "utilities": frozenset(),
    })
    """SSOT: per-layer inner-class kinds that cross-layer checks permit.

    Every canonical facade layer MUST be enumerated here so the
    ``v_class_placement`` visitor disambiguates the cross-layer branch
    from the name-prefix branch via membership lookup. Empty frozensets
    are deliberate — they declare *no* allowed exception for that layer.

    ``check_cross_strenum`` / ``check_cross_protocol`` resolve their
    ``layer_allows`` argument via ``"StrEnum" in ENFORCEMENT_LAYER_ALLOWS.get(layer, ())``.
    """

    # --- Violation message shape (single parameterized template) ---
    #
    # One template covers every violation: the check supplies the
    # ``location`` (field / attribute / path / class qualname), the
    # ``problem`` (what is wrong), and the ``fix`` (remediation). Adding
    # a new check never requires editing this constant.

    ENFORCEMENT_MSG_VIOLATION: Final[str] = "{location}: {problem}. {fix}"
    """Single message shape — location + problem + fix."""

    ENFORCEMENT_VALUE_OBJECT_BASES: Final[frozenset[str]] = frozenset({
        "FrozenValueModel",
        "ImmutableValueModel",
    })
    """Base-class names that require ``frozen=True`` configuration."""

    ENFORCEMENT_INLINE_UNION_MAX: Final[int] = 2
    """Inline union arms allowed before centralization is required."""

    ENFORCEMENT_NESTED_MRO_MIN_DEPTH: Final[int] = 2
    """Minimum qualname depth for a class to count as nested inside a container."""

    ENFORCEMENT_CANONICAL_ALIASES: Final[frozenset[str]] = frozenset({
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

    ENFORCEMENT_PROJECT_ALIAS_OWNERS: Final[Mapping[str, tuple[str, ...]]] = (
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
            )
        )
    )
    """SSOT: project package name → canonical aliases it re-exports locally.

    Used by runtime census and flext-infra detectors to flag
    ``from flext_core import c`` inside a project that owns ``c`` locally.
    """

    ENFORCEMENT_CLASSVAR_EXEMPT_NAMES: Final[frozenset[str]] = frozenset({
        "model_config",
        "logger",
    })
    """ClassVar attribute names that are framework idioms and stay in place."""


__all__: list[str] = ["FlextConstantsEnforcementNamespace"]
