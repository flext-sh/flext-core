"""Structural contracts for the cross-package enforcement catalog."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

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


__all__: tuple[str, ...] = ("FlextProtocolsEnforcement",)
