"""Beartype enforcement catalog rows."""

from __future__ import annotations

from typing import Final


class FlextConstantsEnforcementCatalogBeartypeRows:
    """Beartype hook rows for the enforcement catalog."""

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


__all__ = ["FlextConstantsEnforcementCatalogBeartypeRows"]
