"""Rule metadata enforcement constants for FlextConstantsEnforcement."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class FlextConstantsEnforcementRules:
    """Rule categories, AST hooks, and tag dispatch metadata."""

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
            "compatibility_alias_import": EnforcementCategory.NAMESPACE,
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
            "forbid_deep_namespace": EnforcementCategory.NAMESPACE,
            "nested_layer_misplacement": EnforcementCategory.NAMESPACE,
            "nested_mro": EnforcementCategory.NAMESPACE,
            "no_accessor_methods": EnforcementCategory.NAMESPACE,
            "no_any": EnforcementCategory.FIELD,
            "no_bare_collection": EnforcementCategory.FIELD,
            "no_concrete_namespace_import": EnforcementCategory.NAMESPACE,
            "no_core_tests_namespace": EnforcementCategory.NAMESPACE,
            "no_inline_union": EnforcementCategory.FIELD,
            "no_module_compat_alias": EnforcementCategory.NAMESPACE,
            "no_mutable_default": EnforcementCategory.FIELD,
            "no_private_module_bypass": EnforcementCategory.NAMESPACE,
            "no_pydantic_consumer_import": EnforcementCategory.NAMESPACE,
            "one_class_per_module": EnforcementCategory.NAMESPACE,
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


__all__: list[str] = ["FlextConstantsEnforcementRules"]
