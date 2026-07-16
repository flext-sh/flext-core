"""Namespace/import rule-text enforcement constants (extracted for LOC cap)."""

from __future__ import annotations


from flext_core._typings.base import FlextTypingBase as t


NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT: dict[str, t.StrPair] = {
    "no_core_tests_namespace": (
        'deprecated namespace "{symbol}" in {file}:{line}',
        "Use flat test namespace access (c/p/t/m/u.Tests.*) with no Core intermediary.",
    ),
    "no_wrapper_root_alias_import": (
        'wrapper alias import must use root package in {file}:{line}: "{statement}"',
        "Use `from tests|examples|scripts import c, p, t, m, p, u` (no submodule alias imports).",
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
}

__all__: list[str] = ["NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT"]
