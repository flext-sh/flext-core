"""Predicate parameter models for enforcement rules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (  # noqa: TC003 — Pydantic field resolution needs runtime Mapping/Sequence
    Mapping,
    Sequence,
)
from typing import Literal

from pydantic import Field

from ._base import EnforcementModelBase, FlextModelsEnforcementBase


class FlextModelsEnforcementParams(FlextModelsEnforcementBase):
    """Predicate-specific payload models used by beartype enforcement rules."""

    class FieldShapeParams(EnforcementModelBase):
        """Parameters for FIELD_SHAPE predicate."""

        kind: Literal["field_shape"] = "field_shape"
        forbid_any: bool = False
        forbid_bare_collection: bool = False
        forbid_mutable_default: bool = False
        forbid_raw_default_factory: bool = False
        forbid_str_none_empty: bool = False
        forbid_inline_union: bool = False
        require_description: bool = False
        max_union_arms: int = 2

    class ModelConfigParams(EnforcementModelBase):
        """Parameters for MODEL_CONFIG predicate."""

        kind: Literal["model_config"] = "model_config"
        forbid_v1_config: bool = False
        require_extra_forbid: bool = False
        allowed_extra_values: Sequence[str] = ()
        require_frozen_for_value_objects: bool = False

    class LooseSymbolParams(EnforcementModelBase):
        """Parameters for LOOSE_SYMBOL predicate."""

        kind: Literal["loose_symbol"] = "loose_symbol"
        allowed_prefixes: Sequence[str] = ()
        require_future_annotations: bool = False
        required_canonical_files: Sequence[str] = ()
        require_settings_base: bool = False

    class ImportBlacklistParams(EnforcementModelBase):
        """Parameters for IMPORT_BLACKLIST predicate."""

        kind: Literal["import_blacklist"] = "import_blacklist"
        forbidden_modules: Sequence[str] = ()
        forbidden_symbols: Sequence[str] = ()
        private_package_only: bool = False
        detect_cycles: bool = False

    class ForeignCanonicalAliasImportParams(EnforcementModelBase):
        """Parameters for FOREIGN_CANONICAL_ALIAS_IMPORT predicate."""

        kind: Literal["foreign_canonical_alias_import"] = (
            "foreign_canonical_alias_import"
        )
        project_alias_owners: Mapping[str, Sequence[str]] = Field(default_factory=dict)

    class ClassPlacementParams(EnforcementModelBase):
        """Parameters for CLASS_PLACEMENT predicate."""

        kind: Literal["class_placement"] = "class_placement"
        forbidden_bases: frozenset[str] = frozenset()
        canonical_path_fragment: str = ""
        check_nested: bool = False
        max_nested_class_depth: int = 0

    class LocCapParams(EnforcementModelBase):
        """Parameters for LOC_CAP predicate."""

        kind: Literal["loc_cap"] = "loc_cap"
        max_logical_loc: int = 200
        max_top_level_classes: int = 0

    class WrapperParams(EnforcementModelBase):
        """Parameters for WRAPPER predicate."""

        kind: Literal["wrapper"] = "wrapper"

    class AliasRebindParams(EnforcementModelBase):
        """Parameters for ALIAS_REBIND predicate."""

        kind: Literal["alias_rebind"] = "alias_rebind"
        canonical_files: Sequence[str] = ()
        alias_names: Sequence[str] = ()
        expected_form: str = ""

    class CompatibilityAliasParams(EnforcementModelBase):
        """Parameters for COMPATIBILITY_ALIAS predicate."""

        kind: Literal["compatibility_alias"] = "compatibility_alias"
        alias_renames: Mapping[str, str] = Field(default_factory=dict)
        project_alias_owners: Mapping[str, Sequence[str]] = Field(default_factory=dict)

    class LibraryImportParams(EnforcementModelBase):
        """Parameters for LIBRARY_IMPORT predicate."""

        kind: Literal["library_import"] = "library_import"
        library_owners: Mapping[str, str] = Field(default_factory=dict)

    class DuplicateSymbolParams(EnforcementModelBase):
        """Parameters for DUPLICATE_SYMBOL predicate."""

        kind: Literal["duplicate_symbol"] = "duplicate_symbol"
        hierarchy: Sequence[str] = ()
        symbol_kinds: frozenset[str] = frozenset()

    class DeprecatedSyntaxParams(EnforcementModelBase):
        """Parameters for DEPRECATED_SYNTAX predicate."""

        kind: Literal["deprecated_syntax"] = "deprecated_syntax"
        ast_shape: str = ""

    class MethodShapeParams(EnforcementModelBase):
        """Parameters for METHOD_SHAPE predicate."""

        kind: Literal["method_shape"] = "method_shape"
        forbidden_prefixes: Sequence[str] = ()
        require_static_or_classmethod: bool = False
        max_params: int = 0

    class AttrShapeParams(EnforcementModelBase):
        """Parameters for ATTR_SHAPE predicate."""

        kind: Literal["attr_shape"] = "attr_shape"
        forbid_mutable_value: bool = False
        require_uppercase_name: bool = False
        forbid_any_in_alias: bool = False
        require_typeadapter_naming: bool = False

    class ClassVarConstantParams(EnforcementModelBase):
        """Parameters for CLASSVAR_CONSTANT predicate."""

        kind: Literal["classvar_constant"] = "classvar_constant"
        detect_implicit_constants: bool = True
        """Also flag UPPER_CASE attributes that look like constants but lack ClassVar."""

    class ProtocolTreeParams(EnforcementModelBase):
        """Parameters for PROTOCOL_TREE predicate."""

        kind: Literal["protocol_tree"] = "protocol_tree"
        require_inner_kind_protocol_or_namespace: bool = False
        require_runtime_checkable: bool = False

    class MroShapeParams(EnforcementModelBase):
        """Parameters for MRO_SHAPE predicate."""

        kind: Literal["mro_shape"] = "mro_shape"
        require_alias_first: bool = False
        forbid_redundant_inner: bool = False
        require_explicit_class_when_self_ref: bool = False


__all__: list[str] = ["FlextModelsEnforcementParams"]
