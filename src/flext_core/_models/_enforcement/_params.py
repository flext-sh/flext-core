"""Predicate parameter models for enforcement rules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Discriminator, Field

from flext_core._constants import FlextConstantsEnforcement
from flext_core._models._enforcement._base import (
    FlextModelsEnforcementBase,
    FlextModelsEnforcementModelBase,
)
from flext_core._typings.base import FlextTypingBase

type EnforcementPredicateParams = (
    FlextModelsEnforcementParams.FieldShapeParams
    | FlextModelsEnforcementParams.ModelConfigParams
    | FlextModelsEnforcementParams.LooseSymbolParams
    | FlextModelsEnforcementParams.ImportBlacklistParams
    | FlextModelsEnforcementParams.ClassPlacementParams
    | FlextModelsEnforcementParams.LocCapParams
    | FlextModelsEnforcementParams.AliasRebindParams
    | FlextModelsEnforcementParams.CompatibilityAliasParams
    | FlextModelsEnforcementParams.LibraryImportParams
    | FlextModelsEnforcementParams.DuplicateSymbolParams
    | FlextModelsEnforcementParams.DeprecatedSyntaxParams
    | FlextModelsEnforcementParams.MethodShapeParams
    | FlextModelsEnforcementParams.AttrShapeParams
    | FlextModelsEnforcementParams.ClassVarConstantParams
    | FlextModelsEnforcementParams.ProtocolTreeParams
    | FlextModelsEnforcementParams.MroShapeParams
)


class FlextModelsEnforcementParams(FlextModelsEnforcementBase):
    """Predicate-specific payload models used by beartype enforcement rules."""

    class FieldShapeParams(FlextModelsEnforcementModelBase):
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

    class ModelConfigParams(FlextModelsEnforcementModelBase):
        """Parameters for MODEL_CONFIG predicate."""

        kind: Literal["model_config"] = "model_config"
        forbid_v1_config: bool = False
        require_extra_forbid: bool = False
        allowed_extra_values: FlextTypingBase.StrSequence = ()
        require_frozen_for_value_objects: bool = False

    class LooseSymbolParams(FlextModelsEnforcementModelBase):
        """Parameters for LOOSE_SYMBOL predicate."""

        kind: Literal["loose_symbol"] = "loose_symbol"
        allowed_prefixes: FlextTypingBase.StrSequence = ()
        require_future_annotations: bool = False
        required_canonical_files: FlextTypingBase.StrSequence = ()
        require_settings_base: bool = False

    class ImportBlacklistParams(FlextModelsEnforcementModelBase):
        """Parameters for IMPORT_BLACKLIST predicate."""

        kind: Literal["import_blacklist"] = "import_blacklist"
        forbidden_modules: FlextTypingBase.StrSequence = ()
        forbidden_symbols: FlextTypingBase.StrSequence = ()
        private_package_only: bool = False
        detect_cycles: bool = False

    class ClassPlacementParams(FlextModelsEnforcementModelBase):
        """Parameters for CLASS_PLACEMENT predicate."""

        kind: Literal["class_placement"] = "class_placement"
        forbidden_bases: frozenset[str] = frozenset()
        canonical_path_fragment: str = ""
        check_nested: bool = False
        max_nested_class_depth: int = 0

    class LocCapParams(FlextModelsEnforcementModelBase):
        """Parameters for the LOC_CAP predicate (top-level-class census).

        The module-LOC ceiling moved out of this predicate entirely (operator
        2026-08-07): it read source text, so flext-infra's tokei gate owns it.
        Only the class census remains here, hence the single knob.
        """

        kind: Literal["loc_cap"] = "loc_cap"
        max_top_level_classes: int = 0

    class AliasRebindParams(FlextModelsEnforcementModelBase):
        """Parameters for ALIAS_REBIND predicate."""

        kind: Literal["alias_rebind"] = "alias_rebind"
        canonical_files: FlextTypingBase.StrSequence = ()
        alias_names: FlextTypingBase.StrSequence = ()
        expected_form: str = ""

    class CompatibilityAliasParams(FlextModelsEnforcementModelBase):
        """Parameters for COMPATIBILITY_ALIAS predicate."""

        kind: Literal["compatibility_alias"] = "compatibility_alias"
        alias_renames: FlextTypingBase.StrMapping = Field(default_factory=dict)

    class LibraryImportParams(FlextModelsEnforcementModelBase):
        """Parameters for the LIBRARY_IMPORT predicate.

        Owners: ``c.ENFORCEMENT_LIBRARY_OWNERS``.
        """

        kind: Literal["library_import"] = "library_import"

    class DuplicateSymbolParams(FlextModelsEnforcementModelBase):
        """Parameters for DUPLICATE_SYMBOL predicate."""

        kind: Literal["duplicate_symbol"] = "duplicate_symbol"
        hierarchy: FlextTypingBase.StrSequence = ()
        symbol_kinds: frozenset[str] = frozenset()

    class DeprecatedSyntaxParams(FlextModelsEnforcementModelBase):
        """Parameters for DEPRECATED_SYNTAX predicate."""

        kind: Literal["deprecated_syntax"] = "deprecated_syntax"
        ast_shape: str = ""

    class MethodShapeParams(FlextModelsEnforcementModelBase):
        """Parameters for METHOD_SHAPE predicate."""

        kind: Literal["method_shape"] = "method_shape"
        forbidden_prefixes: FlextTypingBase.StrMapping = Field(default_factory=dict)
        """Forbidden name prefix → the replacement it suggests."""
        require_static_or_classmethod: bool = False

    class AttrShapeParams(FlextModelsEnforcementModelBase):
        """Parameters for ATTR_SHAPE predicate."""

        kind: Literal["attr_shape"] = "attr_shape"
        forbid_mutable_value: bool = False
        require_uppercase_name: bool = False
        forbid_any_in_alias: bool = False
        require_typeadapter_naming: bool = False

    class ClassVarConstantParams(FlextModelsEnforcementModelBase):
        """Parameters for CLASSVAR_CONSTANT predicate."""

        kind: Literal["classvar_constant"] = "classvar_constant"
        detect_implicit_constants: bool = True
        """Also flag UPPER_CASE attributes that look like constants.

        Flagged when they lack ``ClassVar``.
        """

    class ProtocolTreeParams(FlextModelsEnforcementModelBase):
        """Parameters for PROTOCOL_TREE predicate."""

        kind: Literal["protocol_tree"] = "protocol_tree"
        require_inner_kind_protocol_or_namespace: bool = False
        require_runtime_checkable: bool = False

    class MroShapeParams(FlextModelsEnforcementModelBase):
        """Parameters for MRO_SHAPE predicate."""

        kind: Literal["mro_shape"] = "mro_shape"
        require_alias_first: bool = False
        forbid_redundant_inner: bool = False
        require_explicit_class_when_self_ref: bool = False

    class EnforcementPredicateSpec(FlextModelsEnforcementModelBase):
        """One runtime rule binding validated from the predicate package data."""

        predicate: FlextConstantsEnforcement.EnforcementPredicateKind
        params: Annotated[EnforcementPredicateParams, Discriminator("kind")]


__all__: list[str] = ["FlextModelsEnforcementParams"]
