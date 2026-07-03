"""Runtime enforcement engine MRO part."""

from __future__ import annotations

from types import MappingProxyType

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t

from .enforcement_part_06 import EXTENDED_PREDICATE_BINDINGS


def _bindings() -> t.MappingKV[str, tuple[c.EnforcementPredicateKind, mp.BaseModel]]:
    """Build the tag → (predicate_kind, params) dispatch mapping (1 row = 1 rule)."""
    pk = c.EnforcementPredicateKind
    fp = me.FieldShapeParams
    mc = me.ModelConfigParams
    asp = me.AttrShapeParams
    msp = me.MethodShapeParams
    cpp = me.ClassPlacementParams
    ptp = me.ProtocolTreeParams
    msh = me.MroShapeParams
    lsp = me.LooseSymbolParams
    wp = me.WrapperParams
    iblp = me.ImportBlacklistParams
    arp = me.AliasRebindParams
    cap = me.CompatibilityAliasParams
    dsp = me.DeprecatedSyntaxParams
    return MappingProxyType({
        "no_any": (pk.FIELD_SHAPE, fp(forbid_any=True)),
        "no_bare_collection": (pk.FIELD_SHAPE, fp(forbid_bare_collection=True)),
        "no_mutable_default": (pk.FIELD_SHAPE, fp(forbid_mutable_default=True)),
        "no_raw_collections_field_default": (
            pk.FIELD_SHAPE,
            fp(forbid_raw_default_factory=True),
        ),
        "no_str_none_empty": (pk.FIELD_SHAPE, fp(forbid_str_none_empty=True)),
        "no_inline_union": (
            pk.FIELD_SHAPE,
            fp(forbid_inline_union=True, max_union_arms=c.ENFORCEMENT_INLINE_UNION_MAX),
        ),
        "missing_description": (pk.FIELD_SHAPE, fp(require_description=True)),
        "no_v1_config": (pk.MODEL_CONFIG, mc(forbid_v1_config=True)),
        "extra_missing": (pk.MODEL_CONFIG, mc(require_extra_forbid=True)),
        "extra_wrong": (pk.MODEL_CONFIG, mc(allowed_extra_values=())),
        "value_not_frozen": (
            pk.MODEL_CONFIG,
            mc(require_frozen_for_value_objects=True),
        ),
        "const_mutable": (pk.ATTR_SHAPE, asp(forbid_mutable_value=True)),
        "const_lowercase": (pk.ATTR_SHAPE, asp(require_uppercase_name=True)),
        "alias_any": (pk.ATTR_SHAPE, asp(forbid_any_in_alias=True)),
        "typeadapter_name": (pk.ATTR_SHAPE, asp(require_typeadapter_naming=True)),
        "utility_not_static": (
            pk.METHOD_SHAPE,
            msp(require_static_or_classmethod=True),
        ),
        "class_prefix": (pk.CLASS_PLACEMENT, cpp()),
        "cross_strenum": (
            pk.CLASS_PLACEMENT,
            cpp(forbidden_bases=frozenset({"StrEnum"})),
        ),
        "cross_protocol": (
            pk.CLASS_PLACEMENT,
            cpp(forbidden_bases=frozenset({"Protocol"})),
        ),
        "nested_mro": (pk.CLASS_PLACEMENT, cpp(check_nested=True)),
        "proto_inner_kind": (
            pk.PROTOCOL_TREE,
            ptp(require_inner_kind_protocol_or_namespace=True),
        ),
        "proto_not_runtime": (
            pk.PROTOCOL_TREE,
            ptp(require_runtime_checkable=True),
        ),
        "no_accessor_methods": (
            pk.METHOD_SHAPE,
            msp(forbidden_prefixes=("get_", "set_", "is_")),
        ),
        "settings_inheritance": (
            pk.LOOSE_SYMBOL,
            lsp(require_settings_base=True),
        ),
        "cast_outside_core": (
            pk.DEPRECATED_SYNTAX,
            dsp(ast_shape="cast_outside_core"),
        ),
        "model_rebuild_call": (
            pk.DEPRECATED_SYNTAX,
            dsp(ast_shape="model_rebuild_call"),
        ),
        "pass_through_wrapper": (pk.WRAPPER, wp()),
        "private_attr_probe": (
            pk.DEPRECATED_SYNTAX,
            dsp(ast_shape="private_attr_probe"),
        ),
        "no_core_tests_namespace": (
            pk.DEPRECATED_SYNTAX,
            dsp(ast_shape="no_core_tests_namespace"),
        ),
        "no_wrapper_root_alias_import": (
            pk.DEPRECATED_SYNTAX,
            dsp(ast_shape="no_wrapper_root_alias_import"),
        ),
        "compatibility_alias_import": (
            pk.COMPATIBILITY_ALIAS,
            cap(alias_renames=c.ENFORCEMENT_COMPATIBILITY_ALIAS_RENAMES),
        ),
        "no_concrete_namespace_import": (pk.IMPORT_BLACKLIST, iblp()),
        "no_pydantic_consumer_import": (
            pk.IMPORT_BLACKLIST,
            iblp(
                forbidden_modules=("pydantic",),
                forbidden_symbols=(
                    "BaseModel",
                    "Field",
                    "ConfigDict",
                    "TypeAdapter",
                    "field_validator",
                    "model_validator",
                    "computed_field",
                    "PrivateAttr",
                    "AfterValidator",
                    "BeforeValidator",
                ),
            ),
        ),
        "facade_base_is_alias_or_peer": (
            pk.MRO_SHAPE,
            msh(require_alias_first=True),
        ),
        "alias_first_multi_parent": (
            pk.MRO_SHAPE,
            msh(require_alias_first=True),
        ),
        "alias_rebound_at_module_end": (
            pk.ALIAS_REBIND,
            arp(expected_form="rebound_at_module_end"),
        ),
        "no_redundant_inner_namespace": (
            pk.MRO_SHAPE,
            msh(forbid_redundant_inner=True),
        ),
        "no_self_root_import_in_core_files": (
            pk.ALIAS_REBIND,
            arp(expected_form="no_self_root_import_in_core_files"),
        ),
        "sibling_models_type_checking": (
            pk.ALIAS_REBIND,
            arp(expected_form="sibling_models_type_checking"),
        ),
        "utilities_explicit_class_when_self_ref": (
            pk.MRO_SHAPE,
            msh(require_explicit_class_when_self_ref=True),
        ),
        **EXTENDED_PREDICATE_BINDINGS,
    })


PREDICATE_BINDINGS: t.MappingKV[
    str,
    tuple[c.EnforcementPredicateKind, mp.BaseModel],
] = _bindings()


__all__: list[str] = ["PREDICATE_BINDINGS"]
