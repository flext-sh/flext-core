"""Structural contracts for the cross-package enforcement catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flext_core import t

    from .base import FlextProtocolsBase


class FlextProtocolsEnforcement:
    """Read-only contracts implemented by enforcement catalog models."""

    @runtime_checkable
    class InfraDetectorSource(Protocol):
        """Namespace detector source metadata."""

        @property
        def kind(self) -> Literal["flext_infra_detector"]: ...

        @property
        def violation_field(self) -> str: ...

        @property
        def match_missing(self) -> bool: ...

    @runtime_checkable
    class TestsValidatorSource(Protocol):
        """Flext-tests validator source metadata."""

        @property
        def kind(self) -> Literal["flext_tests_validator"]: ...

        @property
        def method(self) -> str: ...

        @property
        def rule_ids(self) -> t.StrSequence: ...

    @runtime_checkable
    class RuntimeWarningSource(Protocol):
        """Runtime warning source metadata."""

        @property
        def kind(self) -> Literal["runtime_warning"]: ...

        @property
        def category(self) -> str: ...

    @runtime_checkable
    class BeartypeSource(Protocol):
        """Beartype predicate source metadata."""

        @property
        def kind(self) -> Literal["beartype"]: ...

        @property
        def predicate_kind(self) -> FlextProtocolsBase.AttributeProbe: ...

    @runtime_checkable
    class RuffSource(Protocol):
        """Ruff rule source metadata."""

        @property
        def kind(self) -> Literal["ruff"]: ...

        @property
        def rule_code(self) -> str: ...

    @runtime_checkable
    class SkillPointerSource(Protocol):
        """Narrative skill-pointer source metadata."""

        @property
        def kind(self) -> Literal["skill_pointer"]: ...

        @property
        def skill(self) -> str: ...

        @property
        def anchor(self) -> str: ...

    @runtime_checkable
    class CodeSmellSource(Protocol):
        """Code-smell detector source metadata."""

        @property
        def kind(self) -> Literal["code_smell"]: ...

        @property
        def smell_tag(self) -> str: ...

    type EnforcementRuleSource = (
        InfraDetectorSource
        | TestsValidatorSource
        | RuntimeWarningSource
        | BeartypeSource
        | RuffSource
        | SkillPointerSource
        | CodeSmellSource
    )

    @runtime_checkable
    class EnforcementFixAction(Protocol):
        """Actionable fix metadata for one rule."""

        @property
        def kind(self) -> Literal["gate", "transformer", "rope", "manual"]: ...

        @property
        def target(self) -> str: ...

        @property
        def params(self) -> t.JsonMapping: ...

        @property
        def safe(self) -> bool: ...

    @runtime_checkable
    class EnforcementRuleSpec(Protocol):
        """Single rule consumed by infra and test enforcement dispatchers."""

        @property
        def id(self) -> str: ...

        @property
        def description(self) -> str: ...

        @property
        def severity(self) -> str: ...

        @property
        def source(self) -> FlextProtocolsEnforcement.EnforcementRuleSource: ...

        @property
        def enabled(self) -> bool: ...

        @property
        def promote_to_error_when_strict(self) -> bool: ...

        @property
        def fix_action(
            self,
        ) -> FlextProtocolsEnforcement.EnforcementFixAction | None: ...

    @runtime_checkable
    class EnforcementViolation(Protocol):
        """Single enforcement violation emitted by the canonical model."""

        @property
        def qualname(self) -> str: ...

        @property
        def layer(self) -> str: ...

        @property
        def severity(self) -> str: ...

        @property
        def message(self) -> str: ...

        @property
        def rule_id(self) -> str: ...

        @property
        def agents_md_anchor(self) -> str: ...

        @property
        def file_path(self) -> str: ...

        @property
        def line_number(self) -> int: ...

    @runtime_checkable
    class EnforcementCatalog(Protocol):
        """Catalog operations consumed by enforcement selection flows."""

        @property
        def version(self) -> int: ...

        @property
        def rules(
            self,
        ) -> tuple[FlextProtocolsEnforcement.EnforcementRuleSpec, ...]: ...

        def enabled_rules(
            self,
        ) -> tuple[FlextProtocolsEnforcement.EnforcementRuleSpec, ...]: ...

    @runtime_checkable
    class FieldShapeParams(Protocol):
        """Parameters for FIELD_SHAPE predicate."""

        kind: Literal["field_shape"]
        forbid_any: bool
        forbid_bare_collection: bool
        forbid_mutable_default: bool
        forbid_raw_default_factory: bool
        forbid_str_none_empty: bool
        forbid_inline_union: bool
        require_description: bool
        max_union_arms: int

    @runtime_checkable
    class ModelConfigParams(Protocol):
        """Parameters for MODEL_CONFIG predicate."""

        kind: Literal["model_config"]
        forbid_v1_config: bool
        require_extra_forbid: bool
        allowed_extra_values: t.StrSequence
        require_frozen_for_value_objects: bool

    @runtime_checkable
    class AttrShapeParams(Protocol):
        """Parameters for ATTR_SHAPE predicate."""

        kind: Literal["attr_shape"]
        forbid_mutable_value: bool
        require_uppercase_name: bool
        forbid_any_in_alias: bool
        require_typeadapter_naming: bool

    @runtime_checkable
    class MethodShapeParams(Protocol):
        """Parameters for METHOD_SHAPE predicate."""

        kind: Literal["method_shape"]
        forbidden_prefixes: t.StrSequence
        require_static_or_classmethod: bool
        max_params: int

    @runtime_checkable
    class ClassPlacementParams(Protocol):
        """Parameters for CLASS_PLACEMENT predicate."""

        kind: Literal["class_placement"]
        forbidden_bases: frozenset[str]
        canonical_path_fragment: str
        check_nested: bool
        max_nested_class_depth: int

    @runtime_checkable
    class ProtocolTreeParams(Protocol):
        """Parameters for PROTOCOL_TREE predicate."""

        kind: Literal["protocol_tree"]
        require_inner_kind_protocol_or_namespace: bool
        require_runtime_checkable: bool

    @runtime_checkable
    class MroShapeParams(Protocol):
        """Parameters for MRO_SHAPE predicate."""

        kind: Literal["mro_shape"]
        require_alias_first: bool
        forbid_redundant_inner: bool
        require_explicit_class_when_self_ref: bool

    @runtime_checkable
    class LooseSymbolParams(Protocol):
        """Parameters for LOOSE_SYMBOL predicate."""

        kind: Literal["loose_symbol"]
        allowed_prefixes: t.StrSequence
        require_future_annotations: bool
        required_canonical_files: t.StrSequence
        require_settings_base: bool

    @runtime_checkable
    class WrapperParams(Protocol):
        """Parameters for WRAPPER predicate."""

        kind: Literal["wrapper"]

    @runtime_checkable
    class ImportBlacklistParams(Protocol):
        """Parameters for IMPORT_BLACKLIST predicate."""

        kind: Literal["import_blacklist"]
        forbidden_modules: t.StrSequence
        forbidden_symbols: t.StrSequence
        private_package_only: bool
        detect_cycles: bool

    @runtime_checkable
    class AliasRebindParams(Protocol):
        """Parameters for ALIAS_REBIND predicate."""

        kind: Literal["alias_rebind"]
        canonical_files: t.StrSequence
        alias_names: t.StrSequence
        expected_form: str

    @runtime_checkable
    class CompatibilityAliasParams(Protocol):
        """Parameters for COMPATIBILITY_ALIAS predicate."""

        kind: Literal["compatibility_alias"]
        alias_renames: t.StrMapping
        project_alias_owners: t.StrSequenceMapping

    @runtime_checkable
    class DeprecatedSyntaxParams(Protocol):
        """Parameters for DEPRECATED_SYNTAX predicate."""

        kind: Literal["deprecated_syntax"]
        ast_shape: str

    @runtime_checkable
    class LocCapParams(Protocol):
        """Parameters for LOC_CAP predicate."""

        kind: Literal["loc_cap"]
        max_logical_loc: int
        max_top_level_classes: int

    @runtime_checkable
    class LibraryImportParams(Protocol):
        """Parameters for LIBRARY_IMPORT predicate."""

        kind: Literal["library_import"]
        library_owners: t.StrMapping

    @runtime_checkable
    class DuplicateSymbolParams(Protocol):
        """Parameters for DUPLICATE_SYMBOL predicate."""

        kind: Literal["duplicate_symbol"]
        hierarchy: t.StrSequence
        symbol_kinds: frozenset[str]

    @runtime_checkable
    class ClassVarConstantParams(Protocol):
        """Parameters for CLASSVAR_CONSTANT predicate."""

        kind: Literal["classvar_constant"]
        detect_implicit_constants: bool

    @runtime_checkable
    class ForeignCanonicalAliasImportParams(Protocol):
        """Parameters for FOREIGN_CANONICAL_ALIAS_IMPORT predicate."""

        kind: Literal["foreign_canonical_alias_import"]
        project_alias_owners: t.StrSequenceMapping


__all__: tuple[str, ...] = ("FlextProtocolsEnforcement",)
