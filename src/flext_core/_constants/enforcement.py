"""Enforcement constants for Pydantic v2 runtime governance.

Constants used by FlextModelsBase.Enforcement to validate
class definitions at import time. Accessed via c.ENFORCEMENT_*.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


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
        dict: "Mapping[K, V] or t.RecursiveContainerMapping",
        list: "Sequence[X] or t.RecursiveContainerList",
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
        ENFORCEMENT_FORBIDDEN_COLLECTIONS
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
    })
    """Rule registry: tag → (category, layer, severity, problem, fix)."""

    ENFORCEMENT_RECURSIVE_TAGS: Final[frozenset[str]] = frozenset({
        "const_mutable",
    })
    """Tags that must recurse into inner namespace classes during scanning."""
