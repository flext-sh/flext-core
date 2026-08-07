"""Runtime enforcement predicate bindings — extended / architecture rules."""

from __future__ import annotations

from types import MappingProxyType

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t


def _extended_bindings() -> t.MappingKV[
    str, tuple[c.EnforcementPredicateKind, mp.BaseModel]
]:
    """Map extended tags to (predicate_kind, params) dispatch entries."""
    pk = c.EnforcementPredicateKind
    dsp = me.DeprecatedSyntaxParams
    cpp = me.ClassPlacementParams
    arp = me.AliasRebindParams
    iblp = me.ImportBlacklistParams
    return MappingProxyType({
        # --- Phase 3 data-only additions ---
        # Why (operator 2026-08-07): the "loc_cap" tag is GONE from this runtime
        # catalog. It measured module LOC by reading source text
        # (inspect.getsourcelines), so it was a static check wearing a runtime
        # costume, and the census re-ran it once per class of the same module —
        # reporting one file up to 8 times. flext-infra's tokei gate is the one
        # owner of the module-LOC ceiling. "one_class_per_module" below stays:
        # it is a different rule that happens to share this predicate.
        "library_abstraction": (
            pk.LIBRARY_IMPORT,
            me.LibraryImportParams(library_owners=c.ENFORCEMENT_LIBRARY_OWNERS),
        ),
        "deprecated_typealias_syntax": (
            pk.DEPRECATED_SYNTAX,
            dsp(ast_shape="AnnAssign[TypeAlias]"),
        ),
        "nested_layer_misplacement": (
            pk.CLASS_PLACEMENT,
            cpp(
                forbidden_bases=frozenset({"BaseModel", "RootModel", "Protocol"}),
                canonical_path_fragment="_models/",
                check_nested=True,
            ),
        ),
        "cross_project_duplicate": (
            pk.DUPLICATE_SYMBOL,
            me.DuplicateSymbolParams(
                hierarchy=("flext-core", "flext-cli", "flext-infra"),
                symbol_kinds=frozenset({
                    "StrEnum",
                    "Protocol",
                    "TypeAlias",
                    "BaseModel",
                    "frozenset_const",
                }),
            ),
        ),
        "no_module_compat_alias": (
            pk.MODULE_ALIAS,
            arp(expected_form="no_module_compat_alias"),
        ),
        # Why: top-level-class census only. The LOC ceiling is no longer part of
        # this predicate — flext-infra's tokei gate owns it.
        "one_class_per_module": (
            pk.LOC_CAP,
            me.LocCapParams(max_top_level_classes=1),
        ),
        "no_private_module_bypass": (
            pk.IMPORT_BLACKLIST,
            iblp(private_package_only=True),
        ),
        "forbid_deep_namespace": (pk.CLASS_PLACEMENT, cpp(max_nested_class_depth=2)),
        # --- Smell rules (JSON-loaded thresholds) ---
        "smell_function_parameters": (
            pk.METHOD_SHAPE,
            me.MethodShapeParams(max_params=c.SMELL_THRESHOLDS["params"]),
        ),
        # --- Constants discipline ---
        "classvar_constant_outside_constants": (
            pk.CLASSVAR_CONSTANT,
            me.ClassVarConstantParams(detect_implicit_constants=True),
        ),
        "foreign_canonical_alias_import": (
            pk.FOREIGN_CANONICAL_ALIAS_IMPORT,
            me.ForeignCanonicalAliasImportParams(
                project_alias_owners=c.ENFORCEMENT_PROJECT_ALIAS_OWNERS
            ),
        ),
    })


EXTENDED_PREDICATE_BINDINGS: t.MappingKV[
    str, tuple[c.EnforcementPredicateKind, mp.BaseModel]
] = _extended_bindings()


__all__: list[str] = ["EXTENDED_PREDICATE_BINDINGS"]
