"""Enforcement constants for Pydantic v2 runtime governance.

Constants used by FlextModelsBase.Enforcement to validate
class definitions at import time. Accessed via c.ENFORCEMENT_*.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import (
    Mapping,
)
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as _me


class FlextMroViolation(UserWarning):
    """Runtime governance violation emitted by the FLEXT enforcement engine.

    The historical name is preserved for API stability, but the warning is used
    across model, constants, typing, and namespace enforcement layers.

    Route with: `python -W error::FlextMroViolation` to fail on enforcement
    violations.
    """


class FlextConstantsEnforcement:
    """Constants governing Pydantic v2 enforcement behavior."""

    class EnforcementMode(StrEnum):
        """Controls whether violations raise, warn, or are ignored."""

        STRICT = "strict"
        WARN = "warn"
        OFF = "off"

    class EnforcementLayer(StrEnum):
        """Facade layer where a violation is raised."""

        MODEL = "Model"
        CONSTANTS = "Constants"
        TYPES = "Types"
        UTILITIES = "Utilities"
        PROTOCOLS = "Protocols"
        NAMESPACE = "namespace"

    class EnforcementSeverity(StrEnum):
        """Severity label attached to every violation."""

        HARD_RULES = "HARD rules"
        BEST_PRACTICES = "best practices"
        NAMESPACE_RULES = "namespace rules"

    ENFORCEMENT_MODE: Final[EnforcementMode] = EnforcementMode.WARN
    """Controls behavior: strict (TypeError), warn (UserWarning), off."""

    BEARTYPE_MODE: Final[EnforcementMode] = EnforcementMode.OFF
    """Controls flext_core beartype.claw bootstrap: strict, warn, or off."""

    BEARTYPE_CLAW_SKIP_PACKAGES: Final[tuple[str, ...]] = (
        "flext_core._typings",
        "flext_core.runtime",
        "flext_core.loggings",
        "flext_core._models.context",
        "flext_core._utilities.logging_config",
        "flext_core._utilities.parser",
        "flext_core._utilities.reliability",
    )
    """Package paths skipped by the flext_core beartype bootstrap."""

    ENFORCEMENT_EXEMPT_MODULE_FRAGMENTS: Final[tuple[str, ...]] = (
        "tests.",
        "test_",
        "conftest",
        "examples.",
        "scripts.",
        "_utilities/adapters",
        "adapters.py",
    )
    """Module path fragments that auto-exempt classes from enforcement."""

    ENFORCEMENT_RELAXED_EXTRA_BASES: Final[frozenset[str]] = frozenset({
        "DynamicModel",
        "FlexibleModel",
        "FlexibleInternalModel",
        "FrozenDynamicModel",
    })
    """Base model names allowed to have relaxed extra= policies."""

    ENFORCEMENT_INFRASTRUCTURE_BASES: Final[frozenset[str]] = frozenset({
        "ManagedModel",
        "EnumManagedModel",
        "NormalizedModel",
        "StrictManagedModel",
        "ArbitraryTypesModel",
        "StrictBoundaryModel",
        "FlexibleInternalModel",
        "ImmutableValueModel",
        "TaggedModel",
        "FlexibleModel",
        "ContractModel",
        "FrozenValueModel",
        "MutableConfiguredMixin",
        "NormalizedMutableConfiguredMixin",
        "Metadata",
        "TimestampableMixin",
        "VersionableMixin",
        "IdentifiableMixin",
        "TimestampedModel",
        "RetryConfigurationMixin",
        "ValidOutcome",
        "InvalidOutcome",
        "WarningOutcome",
    })
    """FLEXT infrastructure base class names exempt from enforcement checks."""

    ENFORCEMENT_FORBIDDEN_COLLECTIONS: Final[Mapping[type, str]] = MappingProxyType({
        dict: "Mapping[K, V] or t.JsonMapping",
        list: "Sequence[X] or t.JsonList",
        set: "frozenset[X] or AbstractSet[X]",
    })
    """SSOT: forbidden mutable-collection types mapped to replacement hints.

    Downstream constants (``ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS`` as
    name set, ``ENFORCEMENT_MUTABLE_RUNTIME_TYPES`` as runtime tuple) are
    derived from this single mapping — do not maintain parallel lists.
    """

    ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS: Final[frozenset[str]] = frozenset(
        kind.__name__ for kind in ENFORCEMENT_FORBIDDEN_COLLECTIONS
    )
    """Derived view: collection names used by annotation-origin checks."""

    ENFORCEMENT_MUTABLE_RUNTIME_TYPES: Final[tuple[type, ...]] = tuple(
        ENFORCEMENT_FORBIDDEN_COLLECTIONS,
    )
    """Derived view: concrete types used by ``isinstance`` checks."""

    # --- Per-layer metadata (single SSOT mappings keyed by EnforcementLayer) ---

    ENFORCEMENT_CONSTANTS_SKIP_ATTRS: Final[frozenset[str]] = frozenset({
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__init_subclass__",
        "__subclasshook__",
        "__abstractmethods__",
        "__orig_bases__",
        "__class_getitem__",
        "__pydantic_complete__",
        # Pydantic v2 class-level contract attributes — NOT constants,
        # they are framework metadata owned by the BaseModel machinery.
        "model_config",
        "model_fields",
        "model_computed_fields",
        "model_extra",
        "model_post_init",
    })
    """Class-level attributes to skip during constants enforcement."""

    ENFORCEMENT_UTILITIES_EXEMPT_METHODS: Final[frozenset[str]] = frozenset({
        "__init__",
        "__init_subclass__",
        "__new__",
        "__class_getitem__",
    })
    """Methods exempt from static/classmethod enforcement on utilities."""

    # --- MRO Namespace enforcement ---

    ENFORCEMENT_NAMESPACE_MODE: Final[EnforcementMode] = EnforcementMode.WARN
    """Separate mode for namespace checks — see EnforcementMode."""

    ENFORCEMENT_NAMESPACE_FACADE_ROOTS: Final[frozenset[str]] = frozenset({
        "FlextConstants",
        "FlextProtocols",
        "FlextTypes",
        "FlextUtilities",
        "FlextModels",
        "FlextModelsBase",
        "FlextModelsNamespace",
        "EnforcedModel",
    })
    """Root facade class names — skip namespace prefix check on these."""

    ENFORCEMENT_NAMESPACE_LAYER_MAP: Final[tuple[tuple[str, str], ...]] = (
        ("Constants", "constants"),
        ("Models", "models"),
        ("Protocols", "protocols"),
        ("Types", "types"),
        ("Utilities", "utilities"),
    )
    """Class name suffix → layer name mapping for cross-layer detection."""

    ENFORCEMENT_LAYER_ALLOWS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
        "constants": frozenset({"StrEnum"}),
        "protocols": frozenset({"Protocol"}),
    })
    """SSOT: per-layer inner-class kinds that cross-layer checks permit.

    ``check_cross_strenum`` / ``check_cross_protocol`` resolve their
    ``layer_allows`` argument via ``"StrEnum" in ENFORCEMENT_LAYER_ALLOWS.get(layer, ())``.
    """

    ENFORCEMENT_FACADE_PREFIXES: Final[Mapping[str, tuple[str, ...]]] = (
        MappingProxyType({
            EnforcementLayer.CONSTANTS.lower(): ("FlextConstants",),
            EnforcementLayer.PROTOCOLS.lower(): ("FlextProtocols",),
            EnforcementLayer.TYPES.lower(): ("FlextTypes", "FlextTyping"),
            EnforcementLayer.UTILITIES.lower(): ("FlextUtilities", "FlextLogger"),
        })
    )
    """SSOT: per-layer class-name prefixes that mark a facade root."""

    # --- Violation message shape (single parameterized template) ---
    #
    # One template covers every violation: the check supplies the
    # ``location`` (field / attribute / path / class qualname), the
    # ``problem`` (what is wrong), and the ``fix`` (remediation). Adding
    # a new check never requires editing this constant.

    ENFORCEMENT_MSG_VIOLATION: Final[str] = "{location}: {problem}. {fix}"
    """Single message shape — location + problem + fix."""

    ENFORCEMENT_VALUE_OBJECT_BASES: Final[frozenset[str]] = frozenset({
        "ImmutableValueModel",
        "FrozenValueModel",
        "FrozenStrictModel",
    })
    """Base-class names that require ``frozen=True`` configuration."""

    ENFORCEMENT_INLINE_UNION_MAX: Final[int] = 2
    """Inline union arms allowed before centralization is required."""

    ENFORCEMENT_NESTED_MRO_MIN_DEPTH: Final[int] = 2
    """Minimum qualname depth for a class to count as nested inside a container."""

    # --- Rule registry (single source of truth) ---
    #
    # Each row: tag → (EnforcementCategory, layer, severity, problem, fix).
    #
    #   EnforcementCategory = one of the ``EnforcementCategory`` StrEnum members below.
    #   layer    = facade layer for the violation message; for EnforcementCategory.ATTR
    #              rules the lowercase form doubles as the dispatch layer
    #              ("Constants" → runs when target layer == constants).
    #
    # Engine derives per-category tag groups directly from this table — do
    # not maintain parallel lists. Adding a new rule is one row.

    class EnforcementCategory(StrEnum):
        """Rule category — dispatches engine behaviour per row."""

        FIELD = "field"
        MODEL_CLASS = "model_class"
        ATTR = "attr"
        NAMESPACE = "namespace"
        PROTOCOL_TREE = "protocol_tree"

    # --- Lane A-PT detection inputs (ENFORCE-039/041/043/044) ---
    # Centralized SSOT for the regex/path/builtin sentinels consumed by the
    # corresponding ``check_<tag>`` predicates on ``FlextUtilitiesBeartypeEngine``.
    # Keep regex constants compiled as Final[re.Pattern[str]] here so the
    # engine module never carries loose detection inputs.

    ENFORCE_CAST_TYPING_IMPORT_RE: Final[re.Pattern[str]] = re.compile(
        r"\bfrom\s+typing\s+import\b[^\n]*\bcast\b"
    )
    """ENFORCE-039: detects ``from typing import ... cast ...`` import lines."""

    ENFORCE_CAST_CALL_RE: Final[re.Pattern[str]] = re.compile(r"\bcast\s*\(")
    """ENFORCE-039: detects ``cast(...)`` call sites in source text."""

    ENFORCE_MODEL_REBUILD_CALL_RE: Final[re.Pattern[str]] = re.compile(
        r"\bmodel_rebuild\s*\("
    )
    """ENFORCE-041: detects ``model_rebuild(...)`` call sites in source text."""

    ENFORCE_FLEXT_CORE_PATH_MARKERS: Final[frozenset[str]] = frozenset({
        "flext_core",
        "flext-core",
    })
    """Path fragments identifying flext-core source files (ENFORCE-039 exemption)."""

    ENFORCE_PRIVATE_PROBE_BUILTINS: Final[frozenset[str]] = frozenset({
        "hasattr",
        "getattr",
        "setattr",
    })
    """ENFORCE-044: builtins that probe attributes by name."""

    ENFORCE_PRIVATE_PROBE_MIN_ARGS: Final[int] = 2
    """ENFORCE-044: minimum positional args required to inspect ``args[1]`` as the attribute-name literal."""

    ENFORCEMENT_RULES: Final[
        Mapping[
            str,
            tuple[
                EnforcementCategory,
                EnforcementLayer,
                EnforcementSeverity,
                str,
                str,
            ],
        ]
    ] = MappingProxyType({
        "no_any": (
            EnforcementCategory.FIELD,
            EnforcementLayer.MODEL,
            EnforcementSeverity.HARD_RULES,
            "Any is FORBIDDEN (detected recursively)",
            "Use a t.* type contract.",
        ),
        "no_bare_collection": (
            EnforcementCategory.FIELD,
            EnforcementLayer.MODEL,
            EnforcementSeverity.HARD_RULES,
            "bare {kind}[...] annotation FORBIDDEN",
            "Use {replacement}.",
        ),
        "no_mutable_default": (
            EnforcementCategory.FIELD,
            EnforcementLayer.MODEL,
            EnforcementSeverity.HARD_RULES,
            "mutable default {kind}() is FORBIDDEN",
            "Use m.Field(default_factory={kind}).",
        ),
        "no_raw_collections_field_default": (
            EnforcementCategory.FIELD,
            EnforcementLayer.MODEL,
            EnforcementSeverity.HARD_RULES,
            "Field(default_factory={kind}) conflicts with a read-only field contract",
            "Use the immutable equivalent (tuple, MappingProxyType, frozenset) or declare an explicit MutableSequence/MutableMapping/MutableSet contract when in-place mutation is part of the model API.",
        ),
        "no_str_none_empty": (
            EnforcementCategory.FIELD,
            EnforcementLayer.MODEL,
            EnforcementSeverity.BEST_PRACTICES,
            'str | None with default="" is wrong',
            'Use str with default="" (None has no business meaning here).',
        ),
        "no_inline_union": (
            EnforcementCategory.FIELD,
            EnforcementLayer.MODEL,
            EnforcementSeverity.BEST_PRACTICES,
            "complex inline union with {arms} arms",
            "Centralize as a t.* type alias in typings.py.",
        ),
        "missing_description": (
            EnforcementCategory.FIELD,
            EnforcementLayer.MODEL,
            EnforcementSeverity.BEST_PRACTICES,
            "m.Field() missing description",
            'Provide description="...".',
        ),
        "no_v1_config": (
            EnforcementCategory.MODEL_CLASS,
            EnforcementLayer.MODEL,
            EnforcementSeverity.HARD_RULES,
            "class Config is Pydantic v1",
            "Use model_config: ClassVar[ConfigDict] = ConfigDict(...).",
        ),
        "extra_missing": (
            EnforcementCategory.MODEL_CLASS,
            EnforcementLayer.MODEL,
            EnforcementSeverity.BEST_PRACTICES,
            'model_config missing extra="forbid"',
            "Inherit a configured FLEXT base (ArbitraryTypesModel, etc.).",
        ),
        "extra_wrong": (
            EnforcementCategory.MODEL_CLASS,
            EnforcementLayer.MODEL,
            EnforcementSeverity.BEST_PRACTICES,
            'model_config extra="{extra}" not allowed',
            "Use FlexibleModel or FlexibleInternalModel.",
        ),
        "value_not_frozen": (
            EnforcementCategory.MODEL_CLASS,
            EnforcementLayer.MODEL,
            EnforcementSeverity.BEST_PRACTICES,
            "value objects must be frozen=True",
            "Inherit from ImmutableValueModel or FrozenValueModel.",
        ),
        "const_mutable": (
            EnforcementCategory.ATTR,
            EnforcementLayer.CONSTANTS,
            EnforcementSeverity.HARD_RULES,
            "mutable constant value FORBIDDEN",
            "Use frozenset, tuple, or MappingProxyType.",
        ),
        "const_lowercase": (
            EnforcementCategory.ATTR,
            EnforcementLayer.CONSTANTS,
            EnforcementSeverity.BEST_PRACTICES,
            "constant names must be UPPER_CASE",
            "Rename to UPPER_SNAKE_CASE.",
        ),
        "alias_any": (
            EnforcementCategory.ATTR,
            EnforcementLayer.TYPES,
            EnforcementSeverity.HARD_RULES,
            "Any in type alias FORBIDDEN",
            "Use t.* contracts.",
        ),
        "typeadapter_name": (
            EnforcementCategory.ATTR,
            EnforcementLayer.TYPES,
            EnforcementSeverity.BEST_PRACTICES,
            'TypeAdapter "{name}" needs UPPER_CASE naming',
            'Rename to "ADAPTER_{upper_name}".',
        ),
        "utility_not_static": (
            EnforcementCategory.ATTR,
            EnforcementLayer.UTILITIES,
            EnforcementSeverity.BEST_PRACTICES,
            "utility must be @staticmethod or @classmethod",
            "Utilities must be stateless.",
        ),
        "class_prefix": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.NAMESPACE_RULES,
            'class name missing project prefix "{expected}"',
            'Rename to start with "{expected}".',
        ),
        "cross_strenum": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.NAMESPACE_RULES,
            "StrEnum in wrong layer",
            "Move to constants (c.*).",
        ),
        "cross_protocol": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.NAMESPACE_RULES,
            "Protocol in wrong layer",
            "Move to protocols (p.*).",
        ),
        "nested_mro": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.NAMESPACE_RULES,
            'must be nested inside a "{expected}*" container class',
            'Wrap in a container whose name starts with "{expected}".',
        ),
        "proto_inner_kind": (
            EnforcementCategory.PROTOCOL_TREE,
            EnforcementLayer.PROTOCOLS,
            EnforcementSeverity.BEST_PRACTICES,
            "inner class must be Protocol / namespace / ABC",
            "Declare a Protocol subclass, namespace holder, or nominal contract.",
        ),
        "proto_not_runtime": (
            EnforcementCategory.PROTOCOL_TREE,
            EnforcementLayer.PROTOCOLS,
            EnforcementSeverity.BEST_PRACTICES,
            "Protocol must be @runtime_checkable",
            "Decorate the Protocol with @runtime_checkable.",
        ),
        "no_accessor_methods": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            'accessor method "{name}" FORBIDDEN (AGENTS.md §3.1)',
            'Rename "{name}" to a domain verb ({suggestion}) or expose as a field/@u.computed_field.',
        ),
        "settings_inheritance": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            '"{name}" must inherit FlextSettings (AGENTS.md §2.6)',
            "Add FlextSettings to the MRO; remove BaseModel/BaseSettings bases.",
        ),
        # --- Lane A-PT detection rows (ENFORCE-039/041/043/044) ---
        "cast_outside_core": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            "cast() call in {file} is outside flext-core (AGENTS.md §3.2)",
            "Replace cast() with FlextResult narrowing or explicit isinstance().",
        ),
        "model_rebuild_call": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            "model_rebuild() invocation in {file} (AGENTS.md §3.4)",
            "Resolve forward refs via proper imports / __future__ annotations.",
        ),
        "pass_through_wrapper": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.BEST_PRACTICES,
            'pass-through wrapper "{name}" in {file} (AGENTS.md §3.5)',
            "Inline the wrapper at every call site and delete the function.",
        ),
        "private_attr_probe": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            '{probe}(obj, "{name}") probes private attribute in {file}',
            "Refactor the consumer to use the public surface (AGENTS.md §3.6).",
        ),
        # R1–R10 MRO compliance rules
        "no_concrete_namespace_import": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            "bare Flext* class import FORBIDDEN (R1, R3)",
            "Import alias (t, m, c, u, p) from parent; use in class bases.",
        ),
        "no_pydantic_consumer_import": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            "bare pydantic import FORBIDDEN (R2)",
            "Use u.Field(), m.BaseModel, m.ConfigDict, m.TypeAdapter, u.model_validator, u.field_validator, u.computed_field, u.PrivateAttr from parent facade.",
        ),
        "facade_base_is_alias_or_peer": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            "facade class base must be alias or peer concrete class (R4, R5)",
            "Use class Base(t): for Pattern A; class Base(t, FlextPeerXxx): for Pattern B.",
        ),
        "alias_first_multi_parent": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            "multi-parent facade must have alias first in MRO (R5)",
            "Order bases: alias first (t), then concrete peer (FlextPeerXxx).",
        ),
        "alias_rebound_at_module_end": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            "module must rebind alias at end (R6)",
            "Add {rebind_form} as final statement (e.g., t = FlextxxxTypes).",
        ),
        "no_redundant_inner_namespace": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.BEST_PRACTICES,
            "redundant inner namespace re-inheritance (R8)",
            "Remove empty inner class — MRO already exposes it from parent.",
        ),
        "no_self_root_import_in_core_files": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.HARD_RULES,
            "same-package root import in canonical file (R7)",
            "Import alias from parent package, not own package.",
        ),
        "sibling_models_type_checking": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.NAMESPACE,
            EnforcementSeverity.BEST_PRACTICES,
            "sibling models/* import used only in annotation must be TYPE_CHECKING (R9)",
            "Wrap annotation-only imports under `if TYPE_CHECKING:`.",
        ),
        "utilities_explicit_class_when_self_ref": (
            EnforcementCategory.NAMESPACE,
            EnforcementLayer.UTILITIES,
            EnforcementSeverity.BEST_PRACTICES,
            "utilities.py with self-referencing method must use explicit class base (R10)",
            "Use class FlextXxxUtilities(FlextParentUtilities, FlextPeerUtilities): for parent (not alias).",
        ),
    })
    """Rule registry: tag → (category, layer, severity, problem, fix)."""

    ENFORCEMENT_RECURSIVE_TAGS: Final[frozenset[str]] = frozenset({
        "const_mutable",
    })
    """Tags that must recurse into inner namespace classes during scanning."""

    ENFORCEMENT_CANONICAL_FILES: Final[frozenset[str]] = frozenset({
        "typings.py",
        "models.py",
        "protocols.py",
        "utilities.py",
        "constants.py",
    })
    """The five canonical facade files per project (AGENTS.md §2.2)."""

    ENFORCEMENT_PYDANTIC_ALLOWED_MODULES: Final[frozenset[str]] = frozenset({
        "flext_core.models",
        "flext_core._utilities",
        "flext_core._protocols",
        "flext_core._constants",
        "flext_core._typings",
    })
    """Whitelist of modules permitted to import pydantic directly (R2 exemption)."""

    ENFORCEMENT_MRO_ALIAS_MAP: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
        "flext_core": frozenset({
            "c",
            "m",
            "t",
            "u",
            "p",
            "r",
            "s",
            "x",
            "d",
            "e",
            "h",
        }),
        "flext_cli": frozenset({"c", "m", "t", "u", "p", "r", "s"}),
        "flext_web": frozenset({"c", "m", "t", "u", "p", "r", "s"}),
        "flext_meltano": frozenset({"c", "m", "t", "u", "p", "r", "s"}),
        "flext_api": frozenset({"c", "m", "t", "u", "p"}),
        "flext_auth": frozenset({"c", "m", "t", "u", "p"}),
        "flext_grpc": frozenset({"c", "m", "t", "u", "p"}),
        "flext_infra": frozenset({"c", "m", "t", "u", "p"}),
        "flext_tests": frozenset({"c", "m", "t", "u", "p"}),
        "flext_plugin": frozenset({"c", "m", "t", "u", "p"}),
        "flext_observability": frozenset({"c", "m", "t", "u", "p"}),
        "flext_ldap": frozenset({"c", "m", "t", "u", "p"}),
        "flext_ldif": frozenset({"c", "m", "t", "u", "p"}),
        "flext_db_oracle": frozenset({"c", "m", "t", "u", "p"}),
        "flext_oracle_oic": frozenset({"c", "m", "t", "u", "p"}),
        "flext_oracle_wms": frozenset({"c", "m", "t", "u", "p"}),
        "flext_quality": frozenset({"c", "m", "t", "u", "p"}),
        "tests": frozenset({"c", "m", "t", "u", "p"}),
    })
    """Expected canonical aliases per project package."""

    ENFORCEMENT_PATTERN_B_UTILITIES_WHITELIST: Final[frozenset[str]] = frozenset({
        "flext_quality",
    })
    """Projects where utilities.py legitimately uses explicit-class base (R10)."""

    # --- Cross-layer enforcement catalog (SSOT for the pytest dispatcher) ---
    #
    # Every rule is addressed by a stable ``ENFORCE-NNN`` identifier and
    # carries its origin as a typed ``Source`` variant. The catalog is the
    # source of truth consumed by
    # ``flext_tests._fixtures.enforcement`` at collection time, so adding or
    # retiring a rule is a single entry here. Documentation-only kinds
    # (``ast_grep``, ``ruff``, ``skill_pointer``) are indexed but not
    # dispatched — they cross-link to existing authoritative layers
    # (``sgconfig.yml`` / ``make lint`` / ``.agents/skills/``).

    # Populated by ``_hydrate_enforcement_catalog()`` at module bottom after
    # the class body finishes — a deliberate late-binding to break the
    # beartype ↔ models ↔ constants import cycle.
    ENFORCEMENT_CATALOG: ClassVar[_me.EnforcementCatalog]


def _hydrate_enforcement_catalog() -> None:
    """Construct and attach the enforcement catalog to the constants class.

    Importing ``FlextModelsEnforcement`` at module top triggers the
    ``flext_core._models`` package init → beartype bootstrap, which re-enters
    this module before ``FlextConstantsEnforcement`` has been bound. Deferring
    the import to this bottom-of-module hook breaks the cycle.
    """
    # the beartype ↔ models ↔ constants cycle this avoids.
    from flext_core._models.enforcement import (  # noqa: PLC0415
        FlextModelsEnforcement as _me,
    )

    FlextConstantsEnforcement.ENFORCEMENT_CATALOG = _me.EnforcementCatalog(
        rules=(
            # --- FLEXT_INFRA_DETECTOR (14 rules, one per ProjectEnforcementReport field) ---
            _me.EnforcementRuleSpec(
                id="ENFORCE-001",
                description=(
                    "Loose object detected at module level — every public "
                    "symbol must be nested inside its facade family."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="loose_objects",
                ),
                agents_md_anchor="2-architecture-law",
                skills=("flext-mro-namespace-rules",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-002",
                description=(
                    "Import alias source is wrong — alias imported from a "
                    "non-canonical module."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="import_violations",
                ),
                skills=("flext-import-rules",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-003",
                description=(
                    "Namespace source violation — canonical alias imported "
                    "from a project that does not own that slot."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="namespace_source_violations",
                ),
                skills=("flext-import-rules", "flext-mro-namespace-rules"),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-004",
                description=(
                    "Internal (private) module import reaches outside its "
                    "owning package boundary."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="internal_import_violations",
                ),
                skills=("flext-import-rules",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-005",
                description=(
                    "Manual Protocol class declared outside protocols.py / "
                    "_protocols/ tree."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="manual_protocol_violations",
                ),
                skills=("flext-patterns",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-006",
                description="Cyclic import detected between modules.",
                severity=_me.EnforcementRuleSeverity.CRITICAL,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="cyclic_imports",
                ),
                skills=("flext-import-rules",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-007",
                description=(
                    "Runtime alias (c/p/t/m/u/r/s/x/d/e/h) rebound in a "
                    "module that should not own it."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="runtime_alias_violations",
                ),
                skills=("flext-mro-namespace-rules",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-008",
                description=(
                    "Python module missing `from __future__ import annotations`."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="future_violations",
                ),
                skills=("rules-examples",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-009",
                description=(
                    "Manual typing alias declared outside typings.py / _typings/ tree."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="manual_typing_violations",
                ),
                skills=("flext-type-system",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-010",
                description=(
                    "Backwards-compatibility alias retained after refactor "
                    "— should be removed."
                ),
                severity=_me.EnforcementRuleSeverity.LOW,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="compatibility_alias_violations",
                ),
                skills=("flext-refactoring-workflow",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-011",
                description=(
                    "Class placed in the wrong facade layer (e.g. Protocol "
                    "in models.py)."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="class_placement_violations",
                ),
                skills=("flext-mro-namespace-rules",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-012",
                description=(
                    "MRO composition incomplete — facade does not compose "
                    "all its domain mixin trees."
                ),
                severity=_me.EnforcementRuleSeverity.CRITICAL,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="mro_completeness_violations",
                ),
                skills=("flext-mro-namespace-rules",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-013",
                description="Source file failed to parse during enforcement.",
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="parse_failures",
                ),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-014",
                description=(
                    "Canonical facade family is missing (no constants.py / "
                    "models.py / typings.py / protocols.py / utilities.py)."
                ),
                severity=_me.EnforcementRuleSeverity.CRITICAL,
                source=_me.EnforcementInfraDetectorSource(
                    violation_field="facade_statuses",
                    match_missing=True,
                ),
                skills=("flext-mro-namespace-rules",),
            ),
            # --- FLEXT_TESTS_VALIDATOR (7 rules, one per public tv.* method) ---
            _me.EnforcementRuleSpec(
                id="ENFORCE-015",
                description=(
                    "Import discipline violation — lazy imports, "
                    "TYPE_CHECKING misuse, sys.path manipulation, "
                    "tech-lib leaks, or non-root flext-* imports."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementTestsValidatorSource(
                    method="imports",
                    rule_ids=(
                        "IMPORT-001",
                        "IMPORT-002",
                        "IMPORT-003",
                        "IMPORT-004",
                        "IMPORT-005",
                        "IMPORT-006",
                    ),
                ),
                skills=("flext-import-rules",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-016",
                description=(
                    "Type-system violation — Any/object/legacy typing or "
                    "type: ignore bypass."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementTestsValidatorSource(
                    method="types",
                    rule_ids=("TYPE-001", "TYPE-002", "TYPE-003"),
                ),
                skills=("flext-strict-typing", "flext-type-system"),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-017",
                description=(
                    "Bypass pattern — noqa, pragma: no cover (unapproved), "
                    "or exception swallowing."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementTestsValidatorSource(
                    method="bypass",
                    rule_ids=("BYPASS-001", "BYPASS-002", "BYPASS-003"),
                ),
                skills=("flext-patterns",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-018",
                description="Layer violation — lower layer importing an upper layer.",
                severity=_me.EnforcementRuleSeverity.CRITICAL,
                source=_me.EnforcementTestsValidatorSource(
                    method="layer",
                    rule_ids=("LAYER-001",),
                ),
                skills=("flext-architecture-layers",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-019",
                description=(
                    "Test pattern violation — monkeypatch, Mock/MagicMock, "
                    "or @patch usage."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementTestsValidatorSource(
                    method="tests",
                    rule_ids=("TEST-001", "TEST-002", "TEST-003"),
                ),
                skills=("testing-patterns",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-020",
                description=(
                    "pyproject.toml deviation — mypy ignore_errors, unapproved "
                    "ruff ignores, or incomplete type strictness."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementTestsValidatorSource(
                    method="validate_config",
                    rule_ids=(
                        "CONFIG-001",
                        "CONFIG-002",
                        "CONFIG-003",
                        "CONFIG-004",
                        "CONFIG-005",
                    ),
                ),
                skills=("flext-development-workflow",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-021",
                description=(
                    "Markdown code block validation — syntax, forbidden "
                    "typings, missing future annotations, object as type."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementTestsValidatorSource(
                    method="markdown",
                    rule_ids=("MD-001", "MD-002", "MD-003", "MD-004"),
                ),
                skills=("testing-patterns",),
            ),
            # --- RUNTIME_WARNING (1 rule) ---
            _me.EnforcementRuleSpec(
                id="ENFORCE-022",
                description=(
                    "FlextMroViolation emitted by the flext-core enforcement "
                    "engine at class-definition time."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementRuntimeWarningSource(
                    category="flext_core._constants.enforcement.FlextMroViolation",
                ),
                skills=("flext-mro-namespace-rules", "pydantic-v2-governance"),
            ),
            # --- RUFF (3 rules — delegated to `make lint`, documentation-only) ---
            _me.EnforcementRuleSpec(
                id="ENFORCE-023",
                description=(
                    "Dynamic Any usage (ruff ANN401) — enforced at lint time "
                    "by `make lint`; listed here for cross-reference."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementRuffSource(rule_code="ANN401"),
                skills=("flext-strict-typing",),
                notes="Dispatched by ruff via make lint; catalog entry is documentation-only.",
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-024",
                description=(
                    "Missing specific rule code on pyright/pygrep suppressions "
                    "(ruff PGH003)."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementRuffSource(rule_code="PGH003"),
                skills=("flext-strict-typing",),
                notes="Dispatched by ruff via make lint; catalog entry is documentation-only.",
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-025",
                description="Relative import (ruff TID252) — prefer absolute imports.",
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementRuffSource(rule_code="TID252"),
                skills=("flext-import-rules",),
                notes="Dispatched by ruff via make lint; catalog entry is documentation-only.",
            ),
            # --- AST_GREP (8 rules — cross-reference sgconfig.yml, no dispatch) ---
            _me.EnforcementRuleSpec(
                id="ENFORCE-026",
                description="Bare `except:` clause swallows all exceptions including SystemExit.",
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementAstGrepSource(
                    skill="flext-patterns",
                    rule_id="ban-bare-except",
                ),
                skills=("flext-patterns",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-027",
                description="`print()` call in source code — use structured logging.",
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementAstGrepSource(
                    skill="flext-patterns",
                    rule_id="ban-print-in-src",
                ),
                skills=("flext-patterns",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-028",
                description="`breakpoint()` / pdb left in code.",
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementAstGrepSource(
                    skill="flext-patterns",
                    rule_id="no-breakpoint",
                ),
                skills=("flext-patterns",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-029",
                description="`open()` without explicit encoding.",
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementAstGrepSource(
                    skill="flext-patterns",
                    rule_id="ban-open-no-encoding",
                ),
                skills=("flext-patterns",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-030",
                description="`dict` in type annotation — prefer Mapping / MutableMapping / TypedDict.",
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementAstGrepSource(
                    skill="flext-strict-typing",
                    rule_id="ban-dict-type-annotation",
                ),
                skills=("flext-strict-typing",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-031",
                description="`typing.Dict` attribute usage — use collections.abc.Mapping family.",
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementAstGrepSource(
                    skill="flext-strict-typing",
                    rule_id="ban-typing-dict-attribute",
                ),
                skills=("flext-strict-typing",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-032",
                description="`from typing import Dict` — banned in favor of dict / Mapping.",
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementAstGrepSource(
                    skill="flext-strict-typing",
                    rule_id="ban-typing-dict-import",
                ),
                skills=("flext-strict-typing",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-033",
                description="Hardcoded `__version__` string — use importlib.metadata.",
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementAstGrepSource(
                    skill="flext-patterns",
                    rule_id="no-hardcoded-version-string",
                ),
                skills=("flext-patterns",),
            ),
            # --- SKILL_POINTER (5 rules — narrative, no automation) ---
            _me.EnforcementRuleSpec(
                id="ENFORCE-034",
                description=(
                    "Accessor method (get_*, set_*) forbidden — expose as "
                    "field or @u.computed_field (AGENTS.md §3.1)."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementSkillPointerSource(
                    skill="flext-patterns",
                    anchor="no-accessor-methods",
                ),
                agents_md_anchor="3-code-law",
                skills=("flext-patterns",),
                enabled=False,
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-035",
                description=(
                    "Settings models must inherit FlextSettings, not BaseModel"
                    " or BaseSettings (AGENTS.md §2.6)."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementSkillPointerSource(
                    skill="lib-pydantic-settings",
                    anchor="settings-baseline",
                ),
                agents_md_anchor="2-architecture-law",
                skills=("lib-pydantic-settings",),
                enabled=False,
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-036",
                description=(
                    "Never call `model_rebuild()` as a fix strategy — resolve "
                    "forward refs via proper imports/annotations."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementSkillPointerSource(
                    skill="pydantic-v2-governance",
                ),
                agents_md_anchor="0-quick-reference-must-read",
                skills=("pydantic-v2-governance",),
                enabled=False,
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-037",
                description=(
                    "No `os.environ` / `os.getenv` in src/ — use settings + "
                    "constants contracts."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementSkillPointerSource(
                    skill="lib-pydantic-settings",
                ),
                agents_md_anchor="0-quick-reference-must-read",
                skills=("lib-pydantic-settings", "flext-constants-discipline"),
                enabled=False,
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-038",
                description=(
                    "Never flatten organic namespace paths — preserve "
                    "`m.TargetOracle.ExecuteResult` etc., don't rebind to "
                    "`m.ExecuteResult`."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementSkillPointerSource(
                    skill="flext-mro-namespace-rules",
                ),
                agents_md_anchor="0-quick-reference-must-read",
                skills=("flext-mro-namespace-rules",),
                enabled=False,
            ),
            # Lane A-PT (peppy-thacker) catalog entries — ENFORCE-039..044.
            # Per AGENT_COORDINATION.md §4.1, A-PT owns this range.
            # ENFORCE-040 delegates to ruff PGH003; ENFORCE-042 reuses the
            # existing check_settings_inheritance hook (zero new code for
            # those two — pure SSOT/DRY/YAGNI). The other four point at
            # check_<tag> hooks added in
            # flext-core/_utilities/beartype_engine.py and dispatched via
            # the cast_outside_core / model_rebuild_call /
            # pass_through_wrapper / private_attr_probe rows in
            # ENFORCEMENT_RULES.
            _me.EnforcementRuleSpec(
                id="ENFORCE-039",
                description=(
                    "cast() call outside flext-core result internals "
                    "violates AGENTS.md §3.2 (Strict Types)."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementSkillPointerSource(
                    skill="flext-strict-typing",
                    anchor="cast-outside-core",
                ),
                agents_md_anchor="3-2-types-and-contracts",
                skills=("flext-strict-typing", "flext-patterns"),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-040",
                description=(
                    "Linter ignore directive without inline justification "
                    "violates AGENTS.md §3.5 (Linter Zero Tolerance + "
                    "Suppressions)."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementRuffSource(
                    rule_code="PGH003",
                ),
                agents_md_anchor="3-5-integrity",
                skills=("flext-strict-typing", "flext-quality-gates"),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-041",
                description=(
                    "model_rebuild() call indicates unresolved forward refs "
                    "and violates AGENTS.md §3.4 (Tools/Modules/Env)."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementSkillPointerSource(
                    skill="flext-patterns",
                    anchor="model-rebuild-ban",
                ),
                agents_md_anchor="3-4-tools-and-modules",
                skills=("flext-patterns",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-042",
                description=(
                    "Settings class missing FlextSettings base or wrong "
                    "env_prefix violates AGENTS.md §2.6 (Settings Law). "
                    "Reuses the existing check_settings_inheritance hook — "
                    "no new detection code per SSOT/DRY."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementSkillPointerSource(
                    skill="flext-patterns",
                    anchor="settings-inheritance",
                ),
                agents_md_anchor="2-6-settings-law",
                skills=("flext-patterns", "lib-pydantic-settings"),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-043",
                description=(
                    "Pass-through wrapper (single-statement return delegating "
                    "to another callable with identical args) violates "
                    "AGENTS.md §3.5."
                ),
                severity=_me.EnforcementRuleSeverity.MEDIUM,
                source=_me.EnforcementSkillPointerSource(
                    skill="flext-refactoring-workflow",
                    anchor="pass-through-wrapper",
                ),
                agents_md_anchor="3-5-integrity",
                skills=("flext-refactoring-workflow",),
            ),
            _me.EnforcementRuleSpec(
                id="ENFORCE-044",
                description=(
                    "hasattr/getattr/setattr probing of private attributes "
                    "(single-underscore names) violates AGENTS.md §3.6."
                ),
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementSkillPointerSource(
                    skill="flext-strict-typing",
                    anchor="private-attr-probe",
                ),
                agents_md_anchor="3-6-test-standardization",
                skills=("flext-strict-typing", "flext-patterns"),
            ),
        ),
    )


_hydrate_enforcement_catalog()
del _hydrate_enforcement_catalog
