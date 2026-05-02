"""Enforcement constants for Pydantic v2 runtime governance.

Constants used by FlextModelsBase.Enforcement to validate
class definitions at import time. Accessed via c.ENFORCEMENT_*.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from enum import StrEnum, unique
from types import MappingProxyType
from typing import Final


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

        OFF = "off"
        STRICT = "strict"
        WARN = "warn"

    class EnforcementLayer(StrEnum):
        """Facade layer where a violation is raised."""

        CONSTANTS = "Constants"
        MODEL = "Model"
        NAMESPACE = "namespace"
        PROTOCOLS = "Protocols"
        TYPES = "Types"
        UTILITIES = "Utilities"

    class EnforcementSeverity(StrEnum):
        """Severity label attached to every violation."""

        BEST_PRACTICES = "best practices"
        HARD_RULES = "HARD rules"
        NAMESPACE_RULES = "namespace rules"

    @unique
    class EnforcementPredicateKind(StrEnum):
        """Generic detection predicate keyed by AST/typing shape.

        Replaces legacy 1:1 ``check_<tag>`` dispatch — many rules share one
        predicate kind and differ only in their typed predicate-params
        payload (``m.Enforcement.*Params``). The engine selects a single
        thin beartype-driven visitor by this kind, then reads parameters
        from the rule row.
        """

        ALIAS_REBIND = "alias_rebind"
        ATTR_SHAPE = "attr_shape"
        CLASS_PLACEMENT = "class_placement"
        DEPRECATED_SYNTAX = "deprecated_syntax"
        DUPLICATE_SYMBOL = "duplicate_symbol"
        FIELD_SHAPE = "field_shape"
        IMPORT_BLACKLIST = "import_blacklist"
        LIBRARY_IMPORT = "library_import"
        LOC_CAP = "loc_cap"
        LOOSE_SYMBOL = "loose_symbol"
        METHOD_SHAPE = "method_shape"
        MODEL_CONFIG = "model_config"
        MRO_SHAPE = "mro_shape"
        PROTOCOL_TREE = "protocol_tree"
        WRAPPER = "wrapper"

    ENFORCEMENT_MODE: Final[EnforcementMode] = EnforcementMode.WARN
    """Controls behavior: strict (TypeError), warn (UserWarning), off."""

    BEARTYPE_MODE: Final[EnforcementMode] = EnforcementMode.OFF
    """Controls flext_core beartype.claw bootstrap: strict, warn, or off.

    Override at process start with the ``BEARTYPE_MODE`` env var
    (``strict`` / ``warn`` / ``off``). Default is ``off`` to keep regular
    runs free of runtime overhead; CI / strict gates set ``strict``.
    """

    BEARTYPE_CLAW_SKIP_PACKAGES: Final[tuple[str, ...]] = (
        "flext_core._models.context",
        "flext_core._typings",
        "flext_core._utilities.logging_config",
        "flext_core._utilities.parser",
        "flext_core._utilities.reliability",
        "flext_core.loggings",
        "flext_core.runtime",
    )
    """Package paths skipped by the flext_core beartype bootstrap."""

    ENFORCEMENT_RELAXED_EXTRA_BASES: Final[frozenset[str]] = frozenset({
        "DynamicModel",
        "FlexibleModel",
        "FlexibleInternalModel",
        "FrozenDynamicModel",
    })
    """Base model names allowed to have relaxed extra= policies."""

    ENFORCEMENT_INFRASTRUCTURE_BASES: Final[frozenset[str]] = frozenset({
        "ArbitraryTypesModel",
        "ContractModel",
        "EnumManagedModel",
        "FlexibleInternalModel",
        "FlexibleModel",
        "FrozenValueModel",
        "IdentifiableMixin",
        "ImmutableValueModel",
        "InvalidOutcome",
        "ManagedModel",
        "Metadata",
        "MutableConfiguredMixin",
        "NormalizedModel",
        "NormalizedMutableConfiguredMixin",
        "RetryConfigurationMixin",
        "StrictBoundaryModel",
        "StrictManagedModel",
        "TaggedModel",
        "TimestampableMixin",
        "TimestampedModel",
        "ValidOutcome",
        "VersionableMixin",
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
        "__abstractmethods__",
        "__class_getitem__",
        "__dict__",
        "__doc__",
        "__init_subclass__",
        "__module__",
        "__orig_bases__",
        "__pydantic_complete__",
        "__qualname__",
        "__subclasshook__",
        "__weakref__",
        # Pydantic v2 class-level contract attributes — NOT constants,
        # they are framework metadata owned by the BaseModel machinery.
        "model_computed_fields",
        "model_config",
        "model_extra",
        "model_fields",
        "model_post_init",
    })
    """Class-level attributes to skip during constants enforcement."""

    ENFORCEMENT_UTILITIES_EXEMPT_METHODS: Final[frozenset[str]] = frozenset({
        "__class_getitem__",
        "__init__",
        "__init_subclass__",
        "__new__",
    })
    """Methods exempt from static/classmethod enforcement on utilities."""

    # --- MRO Namespace enforcement ---

    ENFORCEMENT_NAMESPACE_MODE: Final[EnforcementMode] = EnforcementMode.WARN
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
        | {"FlextModelsBase", "FlextModelsNamespace", "EnforcedModel"},
    )
    """Root facade class names — skip namespace prefix check on these."""

    ENFORCEMENT_NAMESPACE_LAYER_MAP: Final[tuple[tuple[str, str], ...]] = tuple(
        (name, name.lower()) for name in NAMESPACE_LAYER_NAMES
    )
    """Class name suffix → layer name mapping for cross-layer detection."""

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

    # --- Rule category dispatch ---

    class EnforcementCategory(StrEnum):
        """Rule category — dispatches engine behaviour per row."""

        ATTR = "attr"
        FIELD = "field"
        MODEL_CLASS = "model_class"
        NAMESPACE = "namespace"
        PROTOCOL_TREE = "protocol_tree"

    # --- ENFORCE-039 / 041 / 043 / 044 detection inputs ---
    # Centralized SSOT for the AST-name / path / builtin sentinels consumed by the
    # corresponding ``check_<tag>`` predicates on ``FlextUtilitiesBeartypeEngine``.

    class EnforceAstHookSymbol(StrEnum):
        """AST identifier names matched by A-PT enforcement hooks."""

        CAST_CALL = "cast"
        """ENFORCE-039: ``ast.Name.id`` matched as the ``typing.cast`` call."""

        MODEL_REBUILD_ATTR = "model_rebuild"
        """ENFORCE-041: ``ast.Attribute.attr`` matched as ``BaseModel.model_rebuild``."""

    ENFORCE_FLEXT_CORE_PATH_MARKERS: Final[frozenset[str]] = frozenset({
        "flext_core",
        "flext-core",
    })
    """Path fragments identifying flext-core source files (ENFORCE-039 exemption)."""

    ENFORCE_NON_WORKSPACE_PATH_MARKERS: Final[frozenset[str]] = frozenset({
        "/usr/lib/",
        "/usr/local/lib/",
        "dist-packages",
        "site-packages",
    })
    """Filesystem path fragments identifying third-party source."""

    ENFORCE_PRIVATE_PROBE_BUILTINS: Final[frozenset[str]] = frozenset({
        "getattr",
        "hasattr",
        "setattr",
    })
    """ENFORCE-044: builtins that probe attributes by name."""

    ENFORCE_PRIVATE_PROBE_MIN_ARGS: Final[int] = 2
    """ENFORCE-044: minimum positional args required to inspect ``args[1]`` as the attribute-name literal."""

    # --- Legacy: tag metadata for old enforcement API ---
    # Mapping tags to their (problem_template, fix_template, category).
    # New code should use m.EnforcementCatalog instead.

    ENFORCEMENT_TAG_CATEGORY: Final[Mapping[str, EnforcementCategory]] = (
        MappingProxyType({
            "alias_any": EnforcementCategory.ATTR,
            "alias_first_multi_parent": EnforcementCategory.NAMESPACE,
            "alias_rebound_at_module_end": EnforcementCategory.NAMESPACE,
            "cast_outside_core": EnforcementCategory.NAMESPACE,
            "class_prefix": EnforcementCategory.NAMESPACE,
            "const_lowercase": EnforcementCategory.ATTR,
            "const_mutable": EnforcementCategory.ATTR,
            "cross_project_duplicate": EnforcementCategory.NAMESPACE,
            "cross_protocol": EnforcementCategory.NAMESPACE,
            "cross_strenum": EnforcementCategory.NAMESPACE,
            "deprecated_typealias_syntax": EnforcementCategory.NAMESPACE,
            "extra_missing": EnforcementCategory.MODEL_CLASS,
            "extra_wrong": EnforcementCategory.MODEL_CLASS,
            "facade_base_is_alias_or_peer": EnforcementCategory.NAMESPACE,
            "library_abstraction": EnforcementCategory.NAMESPACE,
            "loc_cap": EnforcementCategory.NAMESPACE,
            "missing_description": EnforcementCategory.FIELD,
            "model_rebuild_call": EnforcementCategory.NAMESPACE,
            "nested_layer_misplacement": EnforcementCategory.NAMESPACE,
            "nested_mro": EnforcementCategory.NAMESPACE,
            "no_accessor_methods": EnforcementCategory.NAMESPACE,
            "no_any": EnforcementCategory.FIELD,
            "no_bare_collection": EnforcementCategory.FIELD,
            "no_concrete_namespace_import": EnforcementCategory.NAMESPACE,
            "no_core_tests_namespace": EnforcementCategory.NAMESPACE,
            "no_inline_union": EnforcementCategory.FIELD,
            "no_mutable_default": EnforcementCategory.FIELD,
            "no_pydantic_consumer_import": EnforcementCategory.NAMESPACE,
            "no_raw_collections_field_default": EnforcementCategory.FIELD,
            "no_redundant_inner_namespace": EnforcementCategory.NAMESPACE,
            "no_self_root_import_in_core_files": EnforcementCategory.NAMESPACE,
            "no_str_none_empty": EnforcementCategory.FIELD,
            "no_v1_config": EnforcementCategory.MODEL_CLASS,
            "no_wrapper_root_alias_import": EnforcementCategory.NAMESPACE,
            "pass_through_wrapper": EnforcementCategory.NAMESPACE,
            "private_attr_probe": EnforcementCategory.NAMESPACE,
            "proto_inner_kind": EnforcementCategory.PROTOCOL_TREE,
            "proto_not_runtime": EnforcementCategory.PROTOCOL_TREE,
            "settings_inheritance": EnforcementCategory.NAMESPACE,
            "sibling_models_type_checking": EnforcementCategory.NAMESPACE,
            "typeadapter_name": EnforcementCategory.ATTR,
            "utilities_explicit_class_when_self_ref": EnforcementCategory.NAMESPACE,
            "utility_not_static": EnforcementCategory.ATTR,
            "value_not_frozen": EnforcementCategory.MODEL_CLASS,
        })
    )
    """Tag → category mapping for old enforcement API."""

    ENFORCEMENT_TAG_LAYER: Final[Mapping[str, str]] = MappingProxyType({
        "alias_any": "Types",
        "const_lowercase": "Constants",
        "const_mutable": "Constants",
        "typeadapter_name": "Types",
        "utility_not_static": "Utilities",
    })
    """Per-tag layer for ATTR-category rules (layer guard in _items_for)."""

    ENFORCEMENT_RULES_TEXT: Final[Mapping[str, tuple[str, str]]] = MappingProxyType({
        "no_any": (
            "Any is FORBIDDEN (detected recursively)",
            "Use a t.* type contract.",
        ),
        "no_bare_collection": (
            "bare {kind}[...] annotation FORBIDDEN",
            "Use {replacement}.",
        ),
        "no_mutable_default": (
            "mutable default {kind}() is FORBIDDEN",
            "Use m.Field(default_factory={kind}).",
        ),
        "no_raw_collections_field_default": (
            "Field(default_factory={kind}) conflicts with a read-only field contract",
            "Use the immutable equivalent (tuple, MappingProxyType, frozenset) or declare an explicit MutableSequence/MutableMapping/MutableSet contract when in-place mutation is part of the model API.",
        ),
        "no_str_none_empty": (
            'str | None with default="" is wrong',
            'Use str with default="" (None has no business meaning here).',
        ),
        "no_inline_union": (
            "complex inline union with {arms} arms",
            "Centralize as a t.* type alias in typings.py.",
        ),
        "missing_description": (
            "m.Field() missing description",
            'Provide description="...".',
        ),
        "no_v1_config": (
            "class Config is Pydantic v1",
            "Use model_config: ClassVar[ConfigDict] = ConfigDict(...).",
        ),
        "extra_missing": (
            'model_config missing extra="forbid"',
            "Inherit a configured FLEXT base (ArbitraryTypesModel, etc.).",
        ),
        "extra_wrong": (
            'model_config extra="{extra}" not allowed',
            "Use FlexibleModel or FlexibleInternalModel.",
        ),
        "value_not_frozen": (
            "value objects must be frozen=True",
            "Inherit from ImmutableValueModel or FrozenValueModel.",
        ),
        "const_mutable": (
            "mutable constant value FORBIDDEN",
            "Use frozenset, tuple, or MappingProxyType.",
        ),
        "const_lowercase": (
            "constant names must be UPPER_CASE",
            "Rename to UPPER_SNAKE_CASE.",
        ),
        "alias_any": ("Any in type alias FORBIDDEN", "Use t.* contracts."),
        "typeadapter_name": (
            'TypeAdapter "{name}" needs UPPER_CASE naming',
            'Rename to "ADAPTER_{upper_name}".',
        ),
        "utility_not_static": (
            "utility must be @staticmethod or @classmethod",
            "Utilities must be stateless.",
        ),
        "class_prefix": (
            'class name missing project prefix "{expected}"',
            'Rename to start with "{expected}".',
        ),
        "cross_strenum": ("StrEnum in wrong layer", "Move to constants (c.*)."),
        "cross_protocol": ("Protocol in wrong layer", "Move to protocols (p.*)."),
        "nested_mro": (
            'must be nested inside a "{expected}*" container class',
            'Wrap in a container whose name starts with "{expected}".',
        ),
        "proto_inner_kind": (
            "inner class must be Protocol / namespace / ABC",
            "Declare a Protocol subclass, namespace holder, or nominal contract.",
        ),
        "proto_not_runtime": (
            "Protocol must be @runtime_checkable",
            "Decorate the Protocol with @runtime_checkable.",
        ),
        "no_accessor_methods": (
            'accessor method "{name}" FORBIDDEN (AGENTS.md §3.1)',
            'Rename "{name}" to a domain verb ({suggestion}) or expose as a field/@u.computed_field.',
        ),
        "settings_inheritance": (
            '"{name}" must inherit FlextSettings (AGENTS.md §2.6)',
            "Add FlextSettings to the MRO; remove BaseModel/BaseSettings bases.",
        ),
        "cast_outside_core": (
            "cast() call in {file} is outside flext-core (AGENTS.md §3.2)",
            "Replace cast() with FlextResult narrowing or explicit isinstance().",
        ),
        "model_rebuild_call": (
            "model_rebuild() invocation in {file} (AGENTS.md §3.4)",
            "Resolve forward refs via proper imports / __future__ annotations.",
        ),
        "pass_through_wrapper": (
            'pass-through wrapper "{name}" in {file} (AGENTS.md §3.5)',
            "Inline the wrapper at every call site and delete the function.",
        ),
        "private_attr_probe": (
            '{probe}(obj, "{name}") probes private attribute in {file}',
            "Refactor the consumer to use the public surface (AGENTS.md §3.6).",
        ),
        "no_core_tests_namespace": (
            'deprecated namespace "{symbol}" in {file}:{line}',
            "Use flat test namespace access (c/p/t/m/u.Tests.*) with no Core intermediary.",
        ),
        "no_wrapper_root_alias_import": (
            'wrapper alias import must use root package in {file}:{line}: "{statement}"',
            "Use `from tests|examples|scripts import c, p, t, m, u` (no submodule alias imports).",
        ),
        "no_concrete_namespace_import": (
            "bare Flext* class import FORBIDDEN (R1, R3)",
            "Import alias (t, m, c, u, p) from parent; use in class bases.",
        ),
        "no_pydantic_consumer_import": (
            "bare pydantic import FORBIDDEN (R2)",
            "Use u.Field(), m.BaseModel, m.ConfigDict, m.TypeAdapter, u.model_validator, u.field_validator, u.computed_field, u.PrivateAttr from parent facade.",
        ),
        "facade_base_is_alias_or_peer": (
            "facade class base must be alias, alias-base, or peer concrete class (R4, R5)",
            "Use class Base(t): or class Base(FlextProjectServiceBase): for Pattern A; class Base(t, FlextPeerXxx): for Pattern B.",
        ),
        "alias_first_multi_parent": (
            "multi-parent facade must have alias or alias-base first in MRO (R5)",
            "Order bases: alias or alias-base first (t or FlextProjectServiceBase), then concrete peer (FlextPeerXxx).",
        ),
        "alias_rebound_at_module_end": (
            "module must rebind alias at end (R6)",
            "Add {rebind_form} as final statement (e.g., t = FlextxxxTypes).",
        ),
        "no_redundant_inner_namespace": (
            "redundant inner namespace re-inheritance (R8)",
            "Remove empty inner class — MRO already exposes it from parent.",
        ),
        "no_self_root_import_in_core_files": (
            "same-package root import in canonical file (R7)",
            "Import alias from parent package, not own package.",
        ),
        "sibling_models_type_checking": (
            "sibling models/* import used only in annotation must be TYPE_CHECKING (R9)",
            "Wrap annotation-only imports under `if TYPE_CHECKING:`.",
        ),
        "utilities_explicit_class_when_self_ref": (
            "utilities.py with self-referencing method must use explicit class base (R10)",
            "Use class FlextXxxUtilities(FlextParentUtilities, FlextPeerUtilities): for parent (not alias).",
        ),
        "loc_cap": (
            "module {file} has {loc} logical LOC > cap {cap} (AGENTS.md §3.1)",
            "Decompose into focused submodules under the same package.",
        ),
        "library_abstraction": (
            "import of {lib} outside its owner {owner} (AGENTS.md §2.7)",
            "Route through the owner facade (u.Observability.* / u.Cli.* / u.Infra.*).",
        ),
        "deprecated_typealias_syntax": (
            "'X: TypeAlias = ...' deprecated in {file}:{line} (AGENTS.md §3.5)",
            "Use PEP 695 'type X = ...' syntax.",
        ),
        "nested_layer_misplacement": (
            "{qn} nested in wrong facade family (AGENTS.md §2.2)",
            "Move declaration to its canonical _models/_protocols/_typings/ tree.",
        ),
        "cross_project_duplicate": (
            "{qn} duplicated across {owners} (AGENTS.md §2.3, §3.5)",
            "Move the symbol to the highest project in hierarchy and re-export.",
        ),
    })
    """Legacy: problem/fix text indexed by tag. Use m.EnforcementCatalog for new code."""

    ENFORCEMENT_RECURSIVE_TAGS: Final[frozenset[str]] = frozenset({
        "const_mutable",
    })
    """Tags that must recurse into inner namespace classes during scanning."""

    ENFORCEMENT_NAMESPACE_TARGET_TAGS: Final[frozenset[str]] = frozenset({
        "alias_first_multi_parent",
        "alias_rebound_at_module_end",
        "cast_outside_core",
        "cross_project_duplicate",
        "deprecated_typealias_syntax",
        "facade_base_is_alias_or_peer",
        "library_abstraction",
        "loc_cap",
        "model_rebuild_call",
        "nested_layer_misplacement",
        "no_concrete_namespace_import",
        "no_core_tests_namespace",
        "no_pydantic_consumer_import",
        "no_redundant_inner_namespace",
        "no_self_root_import_in_core_files",
        "no_wrapper_root_alias_import",
        "pass_through_wrapper",
        "private_attr_probe",
        "settings_inheritance",
        "sibling_models_type_checking",
        "utilities_explicit_class_when_self_ref",
    })
    """NAMESPACE tags that use simple class-target dispatch (yield qn, (target,))."""

    ENFORCEMENT_CANONICAL_FILES: Final[frozenset[str]] = frozenset({
        "constants.py",
        "models.py",
        "protocols.py",
        "typings.py",
        "utilities.py",
    })
    """The five canonical facade files per project (AGENTS.md §2.2)."""

    ENFORCEMENT_LIBRARY_OWNERS: Final[Mapping[str, str]] = MappingProxyType({
        "pydantic": "flext-core",
        "pydantic_settings": "flext-core",
        "pydantic_core": "flext-core",
        "dependency_injector": "flext-core",
        "returns": "flext-core",
        "structlog": "flext-observability",
        "rich": "flext-cli",
        "rope": "flext-infra",
        "orjson": "flext-cli",
        "yaml": "flext-cli",
        "pyyaml": "flext-cli",
    })
    """SSOT mapping: external library → owning FLEXT abstraction project (§2.7).

    Every consumer accesses these libraries via the owning project's facades
    (``c/m/p/t/u``), never via a bare top-level import. The runtime
    LIBRARY_IMPORT predicate (``m.Enforcement.LibraryImportParams``) and the
    rope-based source-level tier-whitelist validator both source their data
    from this mapping. Adding a new abstracted library = one entry here, no
    parallel list elsewhere.
    """

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

    # Compact data-table for the FLEXT_INFRA_DETECTOR family. Each row maps to a
    # single ``EnforcementRuleSpec`` built by ``u.build_canonical_catalog()`` —
    # collapsing 14 verbose constructor calls into a tabular SSOT (Pydantic v2
    # batch construction via comprehension; no per-rule boilerplate).
    # Row layout: (id, severity, violation_field, agents_md_anchor, skills,
    # match_missing, description).
    INFRA_DETECTOR_ROWS: Final[
        tuple[tuple[str, str, str, str, tuple[str, ...], bool, str], ...]
    ] = (
        (
            "ENFORCE-001",
            "HIGH",
            "loose_objects",
            "2-architecture-law",
            ("flext-mro-namespace-rules",),
            False,
            "Loose object detected at module level — every public symbol must be nested inside its facade family.",
        ),
        (
            "ENFORCE-002",
            "HIGH",
            "import_violations",
            "",
            ("flext-import-rules",),
            False,
            "Import alias source is wrong — alias imported from a non-canonical module.",
        ),
        (
            "ENFORCE-003",
            "HIGH",
            "namespace_source_violations",
            "",
            ("flext-import-rules", "flext-mro-namespace-rules"),
            False,
            "Namespace source violation — canonical alias imported from a project that does not own that slot.",
        ),
        (
            "ENFORCE-004",
            "MEDIUM",
            "internal_import_violations",
            "",
            ("flext-import-rules",),
            False,
            "Internal (private) module import reaches outside its owning package boundary.",
        ),
        (
            "ENFORCE-005",
            "MEDIUM",
            "manual_protocol_violations",
            "",
            ("flext-patterns",),
            False,
            "Manual Protocol class declared outside protocols.py / _protocols/ tree.",
        ),
        (
            "ENFORCE-006",
            "CRITICAL",
            "cyclic_imports",
            "",
            ("flext-import-rules",),
            False,
            "Cyclic import detected between modules.",
        ),
        (
            "ENFORCE-007",
            "HIGH",
            "runtime_alias_violations",
            "",
            ("flext-mro-namespace-rules",),
            False,
            "Runtime alias (c/p/t/m/u/r/s/x/d/e/h) rebound in a module that should not own it.",
        ),
        (
            "ENFORCE-008",
            "MEDIUM",
            "future_violations",
            "",
            ("rules-examples",),
            False,
            "Python module missing `from __future__ import annotations`.",
        ),
        (
            "ENFORCE-009",
            "MEDIUM",
            "manual_typing_violations",
            "",
            ("flext-type-system",),
            False,
            "Manual typing alias declared outside typings.py / _typings/ tree.",
        ),
        (
            "ENFORCE-010",
            "LOW",
            "compatibility_alias_violations",
            "",
            ("flext-refactoring-workflow",),
            False,
            "Backwards-compatibility alias retained after refactor — should be removed.",
        ),
        (
            "ENFORCE-011",
            "HIGH",
            "class_placement_violations",
            "",
            ("flext-mro-namespace-rules",),
            False,
            "Class placed in the wrong facade layer (e.g. Protocol in models.py).",
        ),
        (
            "ENFORCE-012",
            "CRITICAL",
            "mro_completeness_violations",
            "",
            ("flext-mro-namespace-rules",),
            False,
            "MRO composition incomplete — facade does not compose all its domain mixin trees.",
        ),
        (
            "ENFORCE-013",
            "HIGH",
            "parse_failures",
            "",
            (),
            False,
            "Source file failed to parse during enforcement.",
        ),
        (
            "ENFORCE-014",
            "CRITICAL",
            "facade_statuses",
            "",
            ("flext-mro-namespace-rules",),
            True,
            "Canonical facade family is missing (no constants.py / models.py / typings.py / protocols.py / utilities.py).",
        ),
    )

    SKILL_POINTER_ROWS: Final[
        tuple[tuple[str, str, str, str, str, tuple[str, ...], str], ...]
    ] = (
        (
            "ENFORCE-034",
            "HIGH",
            "flext-patterns",
            "no-accessor-methods",
            "3-code-law",
            ("flext-patterns",),
            "Accessor method (get_*, set_*) forbidden — expose as field or @u.computed_field (AGENTS.md §3.1).",
        ),
        (
            "ENFORCE-035",
            "HIGH",
            "lib-pydantic-settings",
            "settings-baseline",
            "2-architecture-law",
            ("lib-pydantic-settings",),
            "Settings models must inherit FlextSettings, not BaseModel or BaseSettings (AGENTS.md §2.6).",
        ),
        (
            "ENFORCE-036",
            "MEDIUM",
            "pydantic-v2-governance",
            "",
            "0-quick-reference-must-read",
            ("pydantic-v2-governance",),
            "Never call `model_rebuild()` as a fix strategy — resolve forward refs via proper imports/annotations.",
        ),
        (
            "ENFORCE-037",
            "HIGH",
            "lib-pydantic-settings",
            "",
            "0-quick-reference-must-read",
            ("lib-pydantic-settings", "flext-constants-discipline"),
            "No `os.environ` / `os.getenv` in src/ — use settings + constants contracts.",
        ),
        (
            "ENFORCE-038",
            "HIGH",
            "flext-mro-namespace-rules",
            "",
            "0-quick-reference-must-read",
            ("flext-mro-namespace-rules",),
            "Never flatten organic namespace paths — preserve `m.TargetOracle.ExecuteResult` etc., don't rebind to `m.ExecuteResult`.",
        ),
    )

    RUFF_ROWS: Final[tuple[tuple[str, str, str, tuple[str, ...], str], ...]] = (
        (
            "ENFORCE-023",
            "HIGH",
            "ANN401",
            ("flext-strict-typing",),
            "Dynamic Any usage (ruff ANN401) — enforced at lint time by `make lint`; listed here for cross-reference.",
        ),
        (
            "ENFORCE-024",
            "MEDIUM",
            "PGH003",
            ("flext-strict-typing",),
            "Missing specific rule code on pyright/pygrep suppressions (ruff PGH003).",
        ),
        (
            "ENFORCE-025",
            "MEDIUM",
            "TID252",
            ("flext-import-rules",),
            "Relative import (ruff TID252) — prefer absolute imports.",
        ),
    )

    TESTS_VALIDATOR_ROWS: Final[
        tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...], str], ...]
    ] = (
        (
            "ENFORCE-015",
            "HIGH",
            "imports",
            (
                "IMPORT-001",
                "IMPORT-002",
                "IMPORT-003",
                "IMPORT-004",
                "IMPORT-006",
            ),
            ("flext-import-rules",),
            "Import discipline violation — lazy imports, TYPE_CHECKING misuse, sys.path manipulation, or non-root flext-* imports.",
        ),
        (
            "ENFORCE-016",
            "HIGH",
            "types",
            ("TYPE-001", "TYPE-002", "TYPE-003"),
            ("flext-strict-typing", "flext-type-system"),
            "Type-system violation — Any/object/legacy typing or type: ignore bypass.",
        ),
        (
            "ENFORCE-017",
            "HIGH",
            "bypass",
            ("BYPASS-001", "BYPASS-002", "BYPASS-003"),
            ("flext-patterns",),
            "Bypass pattern — noqa, pragma: no cover (unapproved), or exception swallowing.",
        ),
        (
            "ENFORCE-018",
            "CRITICAL",
            "layer",
            ("LAYER-001",),
            ("flext-architecture-layers",),
            "Layer violation — lower layer importing an upper layer.",
        ),
        (
            "ENFORCE-019",
            "MEDIUM",
            "tests",
            ("TEST-001", "TEST-002", "TEST-003"),
            ("testing-patterns",),
            "Test pattern violation — monkeypatch, Mock/MagicMock, or @patch usage.",
        ),
        (
            "ENFORCE-020",
            "HIGH",
            "validate_config",
            ("CONFIG-001", "CONFIG-003", "CONFIG-004", "CONFIG-005"),
            ("flext-development-workflow",),
            "pyproject.toml deviation — mypy ignore_errors, unapproved ruff ignores, or incomplete type strictness.",
        ),
        (
            "ENFORCE-021",
            "MEDIUM",
            "markdown",
            ("MD-001", "MD-002", "MD-003", "MD-004"),
            ("testing-patterns",),
            "Markdown code block validation — syntax, forbidden typings, missing future annotations, object as type.",
        ),
    )

    AST_GREP_ROWS: Final[tuple[tuple[str, str, str, str, str], ...]] = (
        (
            "ENFORCE-026",
            "HIGH",
            "flext-patterns",
            "ban-bare-except",
            "Bare `except:` clause swallows all exceptions including SystemExit.",
        ),
        (
            "ENFORCE-027",
            "MEDIUM",
            "flext-patterns",
            "ban-print-in-src",
            "`print()` call in source code — use structured logging.",
        ),
        (
            "ENFORCE-028",
            "HIGH",
            "flext-patterns",
            "no-breakpoint",
            "`breakpoint()` / pdb left in code.",
        ),
        (
            "ENFORCE-029",
            "MEDIUM",
            "flext-patterns",
            "ban-open-no-encoding",
            "`open()` without explicit encoding.",
        ),
        (
            "ENFORCE-030",
            "HIGH",
            "flext-strict-typing",
            "ban-dict-type-annotation",
            "`dict` in type annotation — prefer Mapping / MutableMapping / TypedDict.",
        ),
        (
            "ENFORCE-031",
            "HIGH",
            "flext-strict-typing",
            "ban-typing-dict-attribute",
            "`typing.Dict` attribute usage — use collections.abc.Mapping family.",
        ),
        (
            "ENFORCE-032",
            "HIGH",
            "flext-strict-typing",
            "ban-typing-dict-import",
            "`from typing import Dict` — banned in favor of dict / Mapping.",
        ),
        (
            "ENFORCE-033",
            "HIGH",
            "flext-patterns",
            "no-hardcoded-version-string",
            "Hardcoded `__version__` string — use importlib.metadata.",
        ),
    )

    BEARTYPE_ROWS: Final[
        tuple[tuple[str, str, str, str, tuple[str, ...], str], ...]
    ] = (
        (
            "ENFORCE-039",
            "HIGH",
            "cast_outside_core",
            "3-2-types-and-contracts",
            ("flext-strict-typing", "flext-patterns"),
            "cast() call outside flext-core result internals violates AGENTS.md §3.2 (Strict Types).",
        ),
        (
            "ENFORCE-041",
            "HIGH",
            "model_rebuild_call",
            "3-4-tools-and-modules",
            ("flext-patterns",),
            "model_rebuild() call indicates unresolved forward refs and violates AGENTS.md §3.4 (Tools/Modules/Env).",
        ),
        (
            "ENFORCE-042",
            "HIGH",
            "settings_inheritance",
            "2-6-settings-law",
            ("flext-patterns", "lib-pydantic-settings"),
            "Settings class missing FlextSettings base or wrong env_prefix violates AGENTS.md §2.6 (Settings Law). Reuses the existing check_settings_inheritance hook — no new detection code per SSOT/DRY.",
        ),
        (
            "ENFORCE-043",
            "MEDIUM",
            "pass_through_wrapper",
            "3-5-integrity",
            ("flext-refactoring-workflow",),
            "Pass-through wrapper (single-statement return delegating to another callable with identical args) violates AGENTS.md §3.5.",
        ),
        (
            "ENFORCE-044",
            "HIGH",
            "private_attr_probe",
            "3-6-test-standardization",
            ("flext-strict-typing", "flext-patterns"),
            "hasattr/getattr/setattr probing of private attributes (single-underscore names) violates AGENTS.md §3.6.",
        ),
        (
            "ENFORCE-045",
            "HIGH",
            "no_pydantic_consumer_import",
            "2-7-library-abstraction-boundaries",
            ("flext-import-rules", "pydantic-v2-patterns"),
            "Direct 'from pydantic import ...' in a consumer project (outside its own '_' base pyramid) violates AGENTS.md §2.7 (Library Abstraction Boundaries) + §3.1 (Pydantic v2 Mastery — facade-only access).",
        ),
        (
            "ENFORCE-046",
            "HIGH",
            "no_concrete_namespace_import",
            "4-import-law",
            ("flext-import-rules", "flext-mro-namespace-rules"),
            "Canonical facade files (constants/models/protocols/typings/utilities) must import only c/m/p/t/u aliases from parent — never bare FlextXxx concrete classes (unless Pattern-B peer). Violates AGENTS.md §4 (Import Law).",
        ),
        (
            "ENFORCE-047",
            "HIGH",
            "facade_base_is_alias_or_peer",
            "2-2-facades-namespaces-naming-patterns",
            ("flext-mro-namespace-rules", "flext-import-rules"),
            "Facade-class first base must be an alias (c/m/p/t/u) or a Pattern-B peer FlextXxx — never an arbitrary Flext* concrete class. Violates AGENTS.md §2.2 (One Facade Rule + Pattern-A/B).",
        ),
        (
            "ENFORCE-048",
            "MEDIUM",
            "no_redundant_inner_namespace",
            "2-3-mro-inheritance-namespace-composition",
            ("flext-mro-namespace-rules",),
            "Inner namespace class with empty body that re-inherits from outer (e.g. 'class Cli(FlextCliTypes): pass') is redundant — parent already exposes the namespace. Violates AGENTS.md §2.3 (Single Root Nested Namespace).",
        ),
        (
            "ENFORCE-049",
            "HIGH",
            "alias_first_multi_parent",
            "2-2-facades-namespaces-naming-patterns",
            ("flext-mro-namespace-rules",),
            "Multi-parent facade class must list canonical alias (c/m/p/t/u) as FIRST base — required for C3 MRO linearization. Violates AGENTS.md §2.2 (Pattern-B facade ordering).",
        ),
        (
            "ENFORCE-050",
            "MEDIUM",
            "alias_rebound_at_module_end",
            "4-import-law",
            ("flext-import-rules", "flext-mro-namespace-rules"),
            "Canonical facade module must rebind its alias at end-of-file (e.g. 't = FlextXxxTypes') — establishes the public contract surface. Violates AGENTS.md §4 (Aliases — assigned once at module bottom).",
        ),
        (
            "ENFORCE-051",
            "HIGH",
            "no_self_root_import_in_core_files",
            "4-import-law",
            ("flext-import-rules",),
            "Canonical facade files must NOT import c/m/p/t/u from their own package — must import from the parent MRO package to avoid lazy-load circular initialization. Violates AGENTS.md §4 (No Same-Project Cross-Facade Runtime Imports).",
        ),
        (
            "ENFORCE-052",
            "HIGH",
            "sibling_models_type_checking",
            "4-import-law",
            ("flext-import-rules",),
            "Sibling _models/* imports referenced only in annotations must live under 'if TYPE_CHECKING:' to avoid circular runtime imports. Violates AGENTS.md §4 (Circular Import Resolution — TYPE_CHECKING for annotation-only siblings).",
        ),
        (
            "ENFORCE-053",
            "HIGH",
            "utilities_explicit_class_when_self_ref",
            "2-3-mro-inheritance-namespace-composition",
            ("flext-mro-namespace-rules",),
            "Multi-parent utilities.py facade must list explicit PARENT class as first base (not alias 'u') to allow pyrefly to resolve the MRO when 'u' is rebound to the local class. Violates AGENTS.md §2.3 (MRO Cascade).",
        ),
        (
            "ENFORCE-054",
            "HIGH",
            "no_core_tests_namespace",
            "0-quick-reference-must-read",
            ("flext-mro-namespace-rules", "flext-import-rules"),
            "Deprecated test namespace path '.Core.Tests' is forbidden in tests/examples/scripts. Use flat c/p/t/m/u.Tests.* access only.",
        ),
        (
            "ENFORCE-055",
            "HIGH",
            "no_wrapper_root_alias_import",
            "0-quick-reference-must-read",
            ("flext-import-rules", "flext-mro-namespace-rules", "rules-scripts"),
            "Wrapper alias imports in tests/examples/scripts must come from wrapper root package (`from tests|examples|scripts import ...`). Submodule alias imports are forbidden outside `__init__.py`.",
        ),
    )
