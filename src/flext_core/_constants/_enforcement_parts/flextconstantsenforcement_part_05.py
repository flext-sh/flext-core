"""Rule text enforcement constants for FlextConstantsEnforcement."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from flext_core._typings.base import FlextTypingBase as t


class FlextConstantsEnforcementRuleText:
    """Legacy problem/fix text indexed by enforcement tag."""

    ENFORCEMENT_RULES_TEXT: Final[Mapping[str, t.StrPair]] = MappingProxyType({
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
        "compatibility_alias_import": (
            'non-canonical compatibility alias import "{name}" from {module} in {file}',
            "Use the canonical alias `{alias}`: `from {module} import {alias}`.",
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
        "no_module_compat_alias": (
            "module-level compat alias '{alias} = {target}' in {file} (AGENTS.md §2.4)",
            "Delete the alias and use the canonical class path at every call site.",
        ),
        "one_class_per_module": (
            "module {file} declares {count} top-level classes > cap {cap} (NS-000)",
            "Keep exactly one top-level class per module; absorb extras as MRO mixins.",
        ),
        "no_private_module_bypass": (
            "import '{import}' reaches private tree {origin} from {file} (AGENTS.md §4)",
            "Import via the owning facade (constants/models/protocols/typings/utilities).",
        ),
        "forbid_deep_namespace": (
            "nested namespace depth > 2 at {qn} (AGENTS.md §2.3, single nesting)",
            "Flatten: prefix-merge the inner class into its parent domain class.",
        ),
    })
    """Legacy: problem/fix text indexed by tag. Use m.EnforcementCatalog for new code."""


__all__: list[str] = ["FlextConstantsEnforcementRuleText"]
