"""Pydantic v2 data models for enforcement violation reporting.

Typed containers (``m.Enforcement.*``) that enforcement checks return
instead of raw string lists. This eliminates ``t.StrSequence`` conversions
at call sites and lets callers narrow by severity / layer / qualname
without reparsing formatted messages.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal

from pydantic import Discriminator, Field, model_validator

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.annotateds import FlextTypesAnnotateds as ta
from flext_core._typings.base import FlextTypingBase as t


class FlextModelsEnforcement:
    """Namespace for enforcement violation data models.

    Exposed via MRO on the ``m`` facade. Members include violation containers
    (``Violation`` / ``Report``) plus the cross-layer rule catalog types
    (``EnforcementRuleSeverity`` / ``EnforcementSourceKind`` /
    ``EnforcementRuleSpec`` / ``EnforcementCatalog``) that index detectors,
    validators, runtime warnings, ruff codes, ast-grep rules, and skill
    pointers under stable ``ENFORCE-NNN`` identifiers.
    """

    class _EnforcementSourceBase(mp.BaseModel):
        """Frozen, extra-forbid base for every enforcement model.

        Shared by source-discriminator classes (`EnforcementXxxSource`) and
        by the violation / report / catalog / rule-spec containers below.
        """

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

    class Violation(_EnforcementSourceBase):
        """Single enforcement violation located at ``qualname``."""

        qualname: Annotated[
            str,
            Field(description="Qualified name of the violating class/attribute."),
        ]
        layer: Annotated[
            str,
            Field(description="Layer where the violation was detected."),
        ]
        severity: Annotated[
            str,
            Field(description='Severity label (e.g. "HARD rules", "best practices").'),
        ]
        message: Annotated[
            str,
            Field(description="Human-readable violation message."),
        ]

    class Report(_EnforcementSourceBase):
        """Aggregated violation report returned by a check or runner."""

        violations: Annotated[
            Sequence[FlextModelsEnforcement.Violation],
            Field(
                default_factory=list,
                description="Violations detected by the check.",
            ),
        ] = Field(default_factory=tuple)

        @property
        def messages(self) -> t.StrSequence:
            """Return plain messages for backwards-compatible emission."""
            return [v.message for v in self.violations]

        @property
        def empty(self) -> bool:
            """True when no violations were recorded."""
            return not self.violations

        def __len__(self) -> int:
            """Expose violation count for ``len(report)``."""
            return len(self.violations)

        def __bool__(self) -> bool:
            """Truthy when violations exist."""
            return bool(self.violations)

        def __getitem__(self, index: int) -> str:
            """Return the nth message for ``report[i]`` access."""
            return self.messages[index]

        def __contains__(self, fragment: t.Scalar | None) -> bool:
            """``"Any" in report`` — search message text."""
            if not isinstance(fragment, str):
                return False
            return any(fragment in m for m in self.messages)

    @unique
    class EnforcementRuleSeverity(StrEnum):
        """Severity scale for catalog rules.

        Matches ``c.Tests.ValidatorSeverity`` semantics (CRITICAL/HIGH/MEDIUM/LOW)
        so adapters can map either way without reshaping. Distinct from
        ``c.EnforcementSeverity`` (HARD rules / best practices / namespace
        rules), which describes Pydantic-governance violations.
        """

        CRITICAL = "CRITICAL"
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"

    @unique
    class EnforcementSourceKind(StrEnum):
        """Addressable origin layer for a catalog rule."""

        FLEXT_INFRA_DETECTOR = "flext_infra_detector"
        FLEXT_TESTS_VALIDATOR = "flext_tests_validator"
        RUNTIME_WARNING = "runtime_warning"
        BEARTYPE = "beartype"
        MINIMAL_AST = "minimal_ast"
        RUFF = "ruff"
        AST_GREP = "ast_grep"
        SKILL_POINTER = "skill_pointer"

    class EnforcementInfraDetectorSource(_EnforcementSourceBase):
        """Rule backed by a ``FlextInfraNamespaceEnforcer`` detector field."""

        kind: Literal["flext_infra_detector"] = "flext_infra_detector"
        violation_field: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Attribute name on m.Infra.ProjectEnforcementReport — e.g. "
                    "'loose_objects', 'cyclic_imports', 'facade_statuses'."
                ),
            ),
        ]
        match_missing: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "When True the adapter flags entries where 'exists' is "
                    "False (used with 'facade_statuses' to detect missing "
                    "facades); when False the field is treated as a violation "
                    "list already."
                ),
            ),
        ]

    class EnforcementTestsValidatorSource(_EnforcementSourceBase):
        """Rule backed by a ``FlextTestsValidator`` (``tv.*``) classmethod."""

        kind: Literal["flext_tests_validator"] = "flext_tests_validator"
        method: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Public classmethod on FlextTestsValidator — one of "
                    "'imports', 'types', 'bypass', 'layer', 'tests', "
                    "'validate_config', 'markdown'."
                ),
            ),
        ]
        rule_ids: Annotated[
            tuple[str, ...],
            Field(
                default_factory=tuple,
                description=(
                    "Filter within the validator's output — e.g. ('IMPORT-001',"
                    " 'IMPORT-002'). Empty tuple means accept every rule the "
                    "validator emits."
                ),
            ),
        ]

    class EnforcementRuntimeWarningSource(_EnforcementSourceBase):
        """Rule backed by a ``warnings``-category raised at runtime."""

        kind: Literal["runtime_warning"] = "runtime_warning"
        category: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Dotted path to the warning class — e.g. "
                    "'flext_core._constants.enforcement.FlextMroViolation'."
                ),
            ),
        ]

    class EnforcementBeartypeSource(_EnforcementSourceBase):
        """Rule delegated to a ``FlextUtilitiesBeartypeEngine`` hook."""

        kind: Literal["beartype"] = "beartype"
        hook: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Hook name on FlextUtilitiesBeartypeEngine — e.g. "
                    "'check_model_rebuild_call'."
                ),
            ),
        ]

    class EnforcementMinimalAstSource(_EnforcementSourceBase):
        """Rule detected via minimal AST inspection of an existing source path.

        Used for textual constructs that vanish post-import (alias rebinds,
        flat-facade roots, direct-library imports). Dispatchers MUST honour
        the skip-on-missing-source contract from AGENTS.md §3.8.
        """

        kind: Literal["minimal_ast"] = "minimal_ast"
        pattern: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "AST-grep / regex pattern fragment that locates the violating"
                    " construct in module source."
                ),
            ),
        ]
        require_source: Annotated[
            bool,
            Field(
                default=True,
                description=(
                    "When True, dispatchers must skip rules whose source path"
                    " is unavailable instead of recording a missed match."
                ),
            ),
        ]

    class EnforcementRuffSource(_EnforcementSourceBase):
        """Rule delegated to ``ruff`` (documentation-only in dispatcher)."""

        kind: Literal["ruff"] = "ruff"
        rule_code: Annotated[
            ta.NonEmptyStr,
            Field(description="Ruff rule code — e.g. 'ANN401', 'PGH003', 'TID252'."),
        ]

    class EnforcementAstGrepSource(_EnforcementSourceBase):
        """Rule delegated to ``ast-grep`` via ``sgconfig.yml``."""

        kind: Literal["ast_grep"] = "ast_grep"
        skill: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Skill directory under .agents/skills — e.g. 'flext-patterns'."
                ),
            ),
        ]
        rule_id: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Rule id within .agents/skills/<skill>/rules.yml::rules[].id."
                ),
            ),
        ]

    class EnforcementSkillPointerSource(_EnforcementSourceBase):
        """Rule that is narrative skill content only (no automation)."""

        kind: Literal["skill_pointer"] = "skill_pointer"
        skill: Annotated[
            ta.NonEmptyStr,
            Field(description="Skill directory under .agents/skills."),
        ]
        anchor: Annotated[
            str,
            Field(
                default="",
                description="Optional in-file anchor (heading slug) inside SKILL.md.",
            ),
        ]

    class FieldShapeParams(_EnforcementSourceBase):
        """Parameters for FIELD_SHAPE predicate (Pydantic field annotations)."""

        kind: Literal["field_shape"] = "field_shape"
        forbid_any: Annotated[
            bool,
            mp.Field(default=False, description="Reject Any in field annotation."),
        ]
        forbid_bare_collection: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Reject bare list/dict/set/frozenset annotations.",
            ),
        ]
        forbid_mutable_default: Annotated[
            bool,
            mp.Field(default=False, description="Reject mutable default values."),
        ]
        forbid_raw_default_factory: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Reject default_factory=list/dict/set on read-only fields.",
            ),
        ]
        forbid_str_none_empty: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Reject 'str | None' fields with default=''.",
            ),
        ]
        forbid_inline_union: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Reject inline unions wider than max_union_arms.",
            ),
        ]
        require_description: Annotated[
            bool,
            mp.Field(default=False, description="Require Field(description=...)."),
        ]
        max_union_arms: Annotated[
            int,
            mp.Field(
                default=2,
                ge=2,
                le=8,
                description="Maximum allowed inline-union arity.",
            ),
        ]

    class ModelConfigParams(_EnforcementSourceBase):
        """Parameters for MODEL_CONFIG predicate (Pydantic model_config + frozen)."""

        kind: Literal["model_config"] = "model_config"
        forbid_v1_config: Annotated[
            bool,
            mp.Field(default=False, description="Reject 'class Config' (v1 syntax)."),
        ]
        require_extra_forbid: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Require model_config.extra == 'forbid'.",
            ),
        ]
        allowed_extra_values: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Permitted ConfigDict.extra values when not 'forbid'.",
            ),
        ]
        require_frozen_for_value_objects: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Require frozen=True on ValueObject subclasses.",
            ),
        ]

    class LooseSymbolParams(_EnforcementSourceBase):
        """Parameters for LOOSE_SYMBOL predicate (top-level non-facade symbols)."""

        kind: Literal["loose_symbol"] = "loose_symbol"
        allowed_prefixes: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Class/function-name prefixes that are facade-canonical.",
            ),
        ]
        require_future_annotations: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Require 'from __future__ import annotations' header.",
            ),
        ]
        required_canonical_files: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Canonical filenames a package must expose.",
            ),
        ]

    class ImportBlacklistParams(_EnforcementSourceBase):
        """Parameters for IMPORT_BLACKLIST predicate (forbidden import statements)."""

        kind: Literal["import_blacklist"] = "import_blacklist"
        forbidden_modules: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Module roots that may not be imported in scope.",
            ),
        ]
        forbidden_symbols: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Specific symbol names that may not be imported.",
            ),
        ]
        private_package_only: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Restrict to private '_*' package boundary checks.",
            ),
        ]
        detect_cycles: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Whether the visitor must build the import graph.",
            ),
        ]

    class ClassPlacementParams(_EnforcementSourceBase):
        """Parameters for CLASS_PLACEMENT predicate (class in canonical layer file)."""

        kind: Literal["class_placement"] = "class_placement"
        forbidden_bases: Annotated[
            frozenset[str],
            mp.Field(
                default_factory=frozenset,
                description="Base-class names that force layer placement.",
            ),
        ]
        canonical_path_fragment: Annotated[
            str,
            mp.Field(
                default="",
                description="Path fragment that the source file MUST contain.",
            ),
        ]
        check_nested: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Also flag classes whose qualname is module-root.",
            ),
        ]

    class LocCapParams(_EnforcementSourceBase):
        """Parameters for LOC_CAP predicate (module logical-LOC ceiling, §3.1)."""

        kind: Literal["loc_cap"] = "loc_cap"
        max_logical_loc: Annotated[
            int,
            mp.Field(
                default=200,
                ge=50,
                le=2000,
                description="Maximum non-blank/non-comment line count per module.",
            ),
        ]

    class WrapperParams(_EnforcementSourceBase):
        """Parameters for WRAPPER predicate (pass-through wrappers — no payload)."""

        kind: Literal["wrapper"] = "wrapper"

    class AliasRebindParams(_EnforcementSourceBase):
        """Parameters for ALIAS_REBIND predicate (canonical alias rebind at EOF)."""

        kind: Literal["alias_rebind"] = "alias_rebind"
        canonical_files: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Filenames where the alias rebind is required.",
            ),
        ]
        alias_names: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Canonical short alias names (c/m/p/t/u).",
            ),
        ]
        expected_form: Annotated[
            str,
            mp.Field(
                default="",
                description="Expected rebind form (e.g. 't = FlextXxxTypes').",
            ),
        ]

    class LibraryImportParams(_EnforcementSourceBase):
        """Parameters for LIBRARY_IMPORT predicate (AGENTS.md §2.7 abstraction owners)."""

        kind: Literal["library_import"] = "library_import"
        library_owners: Annotated[
            t.StrMapping,
            mp.Field(
                default_factory=dict,
                description="Mapping library_root → owning project (e.g. pydantic→flext-core).",
            ),
        ]

    class DuplicateSymbolParams(_EnforcementSourceBase):
        """Parameters for DUPLICATE_SYMBOL predicate (cross-project SSOT hierarchy)."""

        kind: Literal["duplicate_symbol"] = "duplicate_symbol"
        hierarchy: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Project precedence (root-most first).",
            ),
        ]
        symbol_kinds: Annotated[
            frozenset[str],
            mp.Field(
                default_factory=frozenset,
                description="Symbol kinds eligible for duplicate detection.",
            ),
        ]

    class DeprecatedSyntaxParams(_EnforcementSourceBase):
        """Parameters for DEPRECATED_SYNTAX predicate (legacy AST shapes)."""

        kind: Literal["deprecated_syntax"] = "deprecated_syntax"
        ast_shape: Annotated[
            ta.NonEmptyStr,
            mp.Field(description="AST shape selector (e.g. 'AnnAssign[TypeAlias]')."),
        ]

    class MethodShapeParams(_EnforcementSourceBase):
        """Parameters for METHOD_SHAPE predicate (forbidden prefixes + decorator shape)."""

        kind: Literal["method_shape"] = "method_shape"
        forbidden_prefixes: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple,
                description="Method-name prefixes that constitute accessors.",
            ),
        ]
        require_static_or_classmethod: Annotated[
            bool,
            mp.Field(
                default=False,
                description=(
                    "Require @staticmethod / @classmethod (utility-tier rule)."
                ),
            ),
        ]

    class AttrShapeParams(_EnforcementSourceBase):
        """Parameters for ATTR_SHAPE predicate (class-attribute value/name checks)."""

        kind: Literal["attr_shape"] = "attr_shape"
        forbid_mutable_value: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Reject attribute values whose runtime type is mutable.",
            ),
        ]
        require_uppercase_name: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Require constant-style UPPER_CASE attribute names.",
            ),
        ]
        forbid_any_in_alias: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Reject Any inside type-alias attribute values.",
            ),
        ]
        require_typeadapter_naming: Annotated[
            bool,
            mp.Field(
                default=False,
                description=(
                    "Require ADAPTER_/UPPER_CASE name on TypeAdapter attributes."
                ),
            ),
        ]

    class ProtocolTreeParams(_EnforcementSourceBase):
        """Parameters for PROTOCOL_TREE predicate (Protocol nesting rules)."""

        kind: Literal["protocol_tree"] = "protocol_tree"
        require_inner_kind_protocol_or_namespace: Annotated[
            bool,
            mp.Field(
                default=False,
                description=("Require inner classes to be Protocol/namespace/ABC."),
            ),
        ]
        require_runtime_checkable: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Require @runtime_checkable on Protocol classes.",
            ),
        ]

    class MroShapeParams(_EnforcementSourceBase):
        """Parameters for MRO_SHAPE predicate (facade MRO ordering)."""

        kind: Literal["mro_shape"] = "mro_shape"
        require_alias_first: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Require canonical alias as first base class.",
            ),
        ]
        forbid_redundant_inner: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Reject empty inner classes that re-inherit outer.",
            ),
        ]
        require_explicit_class_when_self_ref: Annotated[
            bool,
            mp.Field(
                default=False,
                description="Require explicit parent class in utilities.py multi-parent.",
            ),
        ]

    class EnforcementRuleTarget(_EnforcementSourceBase):
        """Single dispatch target — per-class or per-module — passed to ``apply()``.

        Populated by :class:`FlextUtilitiesEnforcement` for runtime hooks
        and by the workspace walker for parse-only flows. Visitors read
        only the fields they require — string fields default to ``""``.
        """

        file_path: Annotated[
            str,
            mp.Field(default="", description="Absolute path to the source file."),
        ]
        module_qualname: Annotated[
            str,
            mp.Field(default="", description="Dotted import path of the module."),
        ]
        owning_project: Annotated[
            str,
            mp.Field(default="", description="Project root (e.g. 'flext-core')."),
        ]

    class EnforcementRuleSpec(_EnforcementSourceBase):
        """Single rule entry in the enforcement catalog."""

        id: Annotated[
            str,
            Field(
                pattern=r"^ENFORCE-\d{3}$",
                description="Stable catalog identifier (ENFORCE-NNN).",
            ),
        ]
        description: Annotated[
            ta.NonEmptyStr,
            Field(description="Human-readable rule summary."),
        ]
        severity: Annotated[
            FlextModelsEnforcement.EnforcementRuleSeverity,
            Field(description="Catalog severity — CRITICAL/HIGH/MEDIUM/LOW."),
        ]
        source: Annotated[
            (
                FlextModelsEnforcement.EnforcementInfraDetectorSource
                | FlextModelsEnforcement.EnforcementTestsValidatorSource
                | FlextModelsEnforcement.EnforcementRuntimeWarningSource
                | FlextModelsEnforcement.EnforcementBeartypeSource
                | FlextModelsEnforcement.EnforcementMinimalAstSource
                | FlextModelsEnforcement.EnforcementRuffSource
                | FlextModelsEnforcement.EnforcementAstGrepSource
                | FlextModelsEnforcement.EnforcementSkillPointerSource
            ),
            Discriminator("kind"),
            Field(description="Addressable origin of the rule."),
        ]
        agents_md_anchor: Annotated[
            str,
            Field(
                default="",
                description="Optional AGENTS.md anchor (heading slug).",
            ),
        ]
        skills: Annotated[
            tuple[str, ...],
            Field(
                default_factory=tuple,
                description="Skill directories under .agents/skills that describe this rule.",
            ),
        ]
        enabled: Annotated[
            bool,
            Field(
                default=True,
                description=(
                    "Whether the dispatcher runs this rule. Skill-pointer/ruff/"
                    "ast-grep rules stay enabled=True for indexing but the "
                    "dispatcher skips them (documentation-only)."
                ),
            ),
        ]
        promote_to_error_when_strict: Annotated[
            bool,
            Field(
                default=True,
                description=(
                    "When --flext-enforce-strict is set, violations of this "
                    "rule become pytest failures instead of warnings."
                ),
            ),
        ]
        notes: Annotated[
            str,
            Field(default="", description="Free-form implementation notes."),
        ]

    class EnforcementCatalog(_EnforcementSourceBase):
        """Frozen catalog of all enforcement rules (SSOT)."""

        version: Annotated[
            int,
            Field(
                default=1,
                description="Schema version; bumped only on breaking shape changes.",
            ),
        ]
        rules: Annotated[
            tuple[FlextModelsEnforcement.EnforcementRuleSpec, ...],
            Field(description="All catalog rules in registration order."),
        ]

        @model_validator(mode="after")
        def _check_unique_ids(self) -> FlextModelsEnforcement.EnforcementCatalog:
            seen: set[str] = set()
            for rule in self.rules:
                if rule.id in seen:
                    msg = f"duplicate rule id in catalog: {rule.id!r}"
                    raise ValueError(msg)
                seen.add(rule.id)
            return self

        def by_id(
            self, rule_id: str
        ) -> FlextModelsEnforcement.EnforcementRuleSpec | None:
            """Return the rule with ``rule_id`` or ``None`` if absent."""
            for rule in self.rules:
                if rule.id == rule_id:
                    return rule
            return None

        def enabled_rules(
            self,
        ) -> tuple[FlextModelsEnforcement.EnforcementRuleSpec, ...]:
            """Return only the rules with ``enabled=True``."""
            return tuple(r for r in self.rules if r.enabled)

        def by_kind(
            self,
            kind: FlextModelsEnforcement.EnforcementSourceKind,
        ) -> tuple[FlextModelsEnforcement.EnforcementRuleSpec, ...]:
            """Filter rules by source kind."""
            return tuple(r for r in self.rules if r.source.kind == kind.value)


__all__: list[str] = ["FlextModelsEnforcement"]
