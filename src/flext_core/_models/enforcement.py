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

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.pydantic import FlextModelsPydantic as mp
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
        """Single enforcement violation located at qualname."""

        qualname: str
        layer: str
        severity: str
        message: str
        file_path: str = ""
        line_number: int = 0

    class Report(_EnforcementSourceBase):
        """Aggregated violation report returned by a check or runner."""

        violations: Sequence[FlextModelsEnforcement.Violation] = ()

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
        violation_field: str
        match_missing: bool = False

    class EnforcementTestsValidatorSource(_EnforcementSourceBase):
        """Rule backed by a ``FlextTestsValidator`` (``tv.*``) classmethod."""

        kind: Literal["flext_tests_validator"] = "flext_tests_validator"
        method: str
        rule_ids: tuple[str, ...] = ()

    class EnforcementRuntimeWarningSource(_EnforcementSourceBase):
        """Rule backed by a ``warnings``-category raised at runtime."""

        kind: Literal["runtime_warning"] = "runtime_warning"
        category: str

    class EnforcementBeartypeSource(_EnforcementSourceBase):
        """Rule dispatched via predicate kind; params in c._PREDICATE_BINDINGS."""

        kind: Literal["beartype"] = "beartype"
        predicate_kind: c.EnforcementPredicateKind

    class EnforcementMinimalAstSource(_EnforcementSourceBase):
        """Rule detected via minimal AST inspection; skips if source unavailable."""

        kind: Literal["minimal_ast"] = "minimal_ast"
        pattern: str = Field(min_length=1)
        require_source: bool = True

    class EnforcementRuffSource(_EnforcementSourceBase):
        """Rule delegated to ruff."""

        kind: Literal["ruff"] = "ruff"
        rule_code: str

    class EnforcementAstGrepSource(_EnforcementSourceBase):
        """Rule delegated to ast-grep via sgconfig.yml."""

        kind: Literal["ast_grep"] = "ast_grep"
        skill: str
        rule_id: str

    class EnforcementSkillPointerSource(_EnforcementSourceBase):
        """Rule as narrative skill content only (no automation)."""

        kind: Literal["skill_pointer"] = "skill_pointer"
        skill: str
        anchor: str = ""

    class FieldShapeParams(_EnforcementSourceBase):
        """Parameters for FIELD_SHAPE predicate (Pydantic field annotations)."""

        kind: Literal["field_shape"] = "field_shape"
        forbid_any: bool = False
        forbid_bare_collection: bool = False
        forbid_mutable_default: bool = False
        forbid_raw_default_factory: bool = False
        forbid_str_none_empty: bool = False
        forbid_inline_union: bool = False
        require_description: bool = False
        max_union_arms: int = 2

    class ModelConfigParams(_EnforcementSourceBase):
        """Parameters for MODEL_CONFIG predicate (Pydantic model_config + frozen)."""

        kind: Literal["model_config"] = "model_config"
        forbid_v1_config: bool = False
        require_extra_forbid: bool = False
        allowed_extra_values: t.StrSequence = ()
        require_frozen_for_value_objects: bool = False

    class LooseSymbolParams(_EnforcementSourceBase):
        """Parameters for LOOSE_SYMBOL predicate (top-level non-facade symbols)."""

        kind: Literal["loose_symbol"] = "loose_symbol"
        allowed_prefixes: t.StrSequence = ()
        require_future_annotations: bool = False
        required_canonical_files: t.StrSequence = ()
        require_settings_base: bool = False

    class ImportBlacklistParams(_EnforcementSourceBase):
        """Parameters for IMPORT_BLACKLIST predicate (forbidden import statements)."""

        kind: Literal["import_blacklist"] = "import_blacklist"
        forbidden_modules: t.StrSequence = ()
        forbidden_symbols: t.StrSequence = ()
        private_package_only: bool = False
        detect_cycles: bool = False

    class ClassPlacementParams(_EnforcementSourceBase):
        """Parameters for CLASS_PLACEMENT predicate (class in canonical layer file)."""

        kind: Literal["class_placement"] = "class_placement"
        forbidden_bases: frozenset[str] = frozenset()
        canonical_path_fragment: str = ""
        check_nested: bool = False

    class LocCapParams(_EnforcementSourceBase):
        """Parameters for LOC_CAP predicate (module logical-LOC ceiling, §3.1)."""

        kind: Literal["loc_cap"] = "loc_cap"
        max_logical_loc: int = 200

    class WrapperParams(_EnforcementSourceBase):
        """Parameters for WRAPPER predicate (pass-through wrappers — no payload)."""

        kind: Literal["wrapper"] = "wrapper"

    class AliasRebindParams(_EnforcementSourceBase):
        """Parameters for ALIAS_REBIND predicate (canonical alias rebind at EOF)."""

        kind: Literal["alias_rebind"] = "alias_rebind"
        canonical_files: t.StrSequence = ()
        alias_names: t.StrSequence = ()
        expected_form: str = ""

    class LibraryImportParams(_EnforcementSourceBase):
        """Parameters for LIBRARY_IMPORT predicate (AGENTS.md §2.7 abstraction owners)."""

        kind: Literal["library_import"] = "library_import"
        library_owners: t.StrMapping = Field(default_factory=dict)

    class DuplicateSymbolParams(_EnforcementSourceBase):
        """Parameters for DUPLICATE_SYMBOL predicate (cross-project SSOT hierarchy)."""

        kind: Literal["duplicate_symbol"] = "duplicate_symbol"
        hierarchy: t.StrSequence = ()
        symbol_kinds: frozenset[str] = frozenset()

    class DeprecatedSyntaxParams(_EnforcementSourceBase):
        """Parameters for DEPRECATED_SYNTAX predicate (legacy AST shapes)."""

        kind: Literal["deprecated_syntax"] = "deprecated_syntax"
        ast_shape: str = ""

    class MethodShapeParams(_EnforcementSourceBase):
        """Parameters for METHOD_SHAPE predicate (forbidden prefixes + decorator shape)."""

        kind: Literal["method_shape"] = "method_shape"
        forbidden_prefixes: t.StrSequence = ()
        require_static_or_classmethod: bool = False

    class AttrShapeParams(_EnforcementSourceBase):
        """Parameters for ATTR_SHAPE predicate (class-attribute value/name checks)."""

        kind: Literal["attr_shape"] = "attr_shape"
        forbid_mutable_value: bool = False
        require_uppercase_name: bool = False
        forbid_any_in_alias: bool = False
        require_typeadapter_naming: bool = False

    class ProtocolTreeParams(_EnforcementSourceBase):
        """Parameters for PROTOCOL_TREE predicate (Protocol nesting rules)."""

        kind: Literal["protocol_tree"] = "protocol_tree"
        require_inner_kind_protocol_or_namespace: bool = False
        require_runtime_checkable: bool = False

    class MroShapeParams(_EnforcementSourceBase):
        """Parameters for MRO_SHAPE predicate (facade MRO ordering)."""

        kind: Literal["mro_shape"] = "mro_shape"
        require_alias_first: bool = False
        forbid_redundant_inner: bool = False
        require_explicit_class_when_self_ref: bool = False

    class EnforcementRuleTarget(_EnforcementSourceBase):
        """Single dispatch target passed to apply()."""

        file_path: str = ""
        module_qualname: str = ""
        owning_project: str = ""

    class EnforcementRuleSpec(_EnforcementSourceBase):
        """Single rule entry in the enforcement catalog."""

        id: Annotated[str, Field(pattern=r"^ENFORCE-\d{3}$")]
        description: str
        severity: FlextModelsEnforcement.EnforcementRuleSeverity
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
        ]
        agents_md_anchor: str = ""
        skills: tuple[str, ...] = ()
        enabled: bool = True
        promote_to_error_when_strict: bool = True
        notes: str = ""

    class EnforcementCatalog(_EnforcementSourceBase):
        """Frozen catalog of all enforcement rules (SSOT)."""

        version: int = 1
        rules: tuple[FlextModelsEnforcement.EnforcementRuleSpec, ...] = ()

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
