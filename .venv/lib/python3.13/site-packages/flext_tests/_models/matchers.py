"""Matcher parameter models for flext-tests."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar

from flext_infra import m, u
from flext_tests import p, t


class FlextTestsMatchersModelsMixin:
    """Matcher model group (result, that, scope, and chain parameters)."""

    class MatchRule(m.Value):
        """One nominal matcher rule parsed from a scalar, type, predicate, or mapping."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True, arbitrary_types_allowed=True, populate_by_name=True
        )

        eq: Annotated[
            t.Tests.MatcherEqTarget | None,
            m.SkipValidation,
            u.Field(description="Expected equality value."),
        ] = None
        ne: Annotated[
            t.Tests.MatcherEqTarget | None,
            m.SkipValidation,
            u.Field(description="Expected inequality value."),
        ] = None
        is_: Annotated[
            type[object] | tuple[type[object], ...] | None,
            m.SkipValidation,
            u.Field(
                validation_alias=t.AliasChoices("is_", "is"),
                description="Expected runtime type.",
            ),
        ] = None
        has: Annotated[
            t.Tests.ContainmentSpec | None,
            m.SkipValidation,
            u.Field(description="Required contained value."),
        ] = None
        lacks: Annotated[
            t.Tests.ExclusionSpec | None,
            m.SkipValidation,
            u.Field(
                validation_alias=t.AliasChoices("lacks", "excludes"),
                description="Forbidden contained value.",
            ),
        ] = None
        none: Annotated[bool | None, u.Field(description="Expected null state.")] = None
        empty: Annotated[bool | None, u.Field(description="Expected empty state.")] = (
            None
        )
        gt: Annotated[
            t.Tests.ComparableScalar | None,
            u.Field(description="Exclusive lower bound."),
        ] = None
        gte: Annotated[
            t.Tests.ComparableScalar | None,
            u.Field(description="Inclusive lower bound."),
        ] = None
        lt: Annotated[
            t.Tests.ComparableScalar | None,
            u.Field(description="Exclusive upper bound."),
        ] = None
        lte: Annotated[
            t.Tests.ComparableScalar | None,
            u.Field(description="Inclusive upper bound."),
        ] = None
        starts: Annotated[
            str | None, u.Field(description="Required string prefix.")
        ] = None
        ends: Annotated[str | None, u.Field(description="Required string suffix.")] = (
            None
        )
        match: Annotated[
            t.Infra.RegexPattern | None,
            m.SkipValidation,
            u.Field(description="Required regular expression."),
        ] = None
        len: Annotated[
            t.Tests.LengthSpec | None, u.Field(description="Required length.")
        ] = None
        where: Annotated[
            t.Tests.PredicateSpec | None,
            m.SkipValidation,
            u.Field(description="Predicate applied to the subject."),
        ] = None
        msg: Annotated[
            str | None, u.Field(description="Inherited assertion message.")
        ] = None

        @classmethod
        def parse(cls, value: object) -> FlextTestsMatchersModelsMixin.MatchRule:
            """Parse one public matcher rule into its nominal representation."""
            if isinstance(value, cls):
                return value
            if isinstance(value, Mapping):
                rule_keys = frozenset({*cls.model_fields, "is", "excludes"})
                if value and set(value).issubset(rule_keys):
                    return cls.model_validate(value)
                return cls(eq=value)
            if isinstance(value, type) or (
                isinstance(value, tuple)
                and all(isinstance(item, type) for item in value)
            ):
                return cls(is_=value)
            if callable(value):
                return cls(where=value)
            return cls(
                eq=t.Tests.TESTOBJECT_SERIALIZABLE_ADAPTER.validate_python(value)
            )

        @classmethod
        def parse_rule_fields(cls, value: object) -> object:
            """Parse paths, items, and attribute rule collections before validation."""
            if value is None:
                return None
            if isinstance(value, Mapping):
                return {key: cls.parse(rule) for key, rule in value.items()}
            if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                return [cls.parse(rule) for rule in value]
            return value

    class OkParams(m.Value):
        """Matcher parameters for successful result assertions."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            populate_by_name=True, arbitrary_types_allowed=True
        )

        eq: Annotated[
            t.Tests.MatcherEqTarget | None, u.Field(description="Expected value.")
        ] = None
        ne: Annotated[
            t.Tests.MatcherEqTarget | None, u.Field(description="Value must not equal.")
        ] = None
        is_: Annotated[
            type[object] | tuple[type[object], ...] | None,
            u.Field(
                validation_alias=t.AliasChoices("is_", "is"),
                description="Runtime type check.",
            ),
        ] = None
        none: Annotated[bool | None, u.Field(description="None check.")] = None
        empty: Annotated[bool | None, u.Field(description="Empty check.")] = None
        gt: Annotated[
            t.Tests.ComparableScalar | None, u.Field(description="Greater than.")
        ] = None
        gte: Annotated[
            t.Tests.ComparableScalar | None,
            u.Field(description="Greater than or equal."),
        ] = None
        lt: Annotated[
            t.Tests.ComparableScalar | None, u.Field(description="Less than.")
        ] = None
        lte: Annotated[
            t.Tests.ComparableScalar | None, u.Field(description="Less than or equal.")
        ] = None
        has: Annotated[
            t.Tests.ContainmentSpec | None,
            m.SkipValidation,
            u.Field(description="Unified containment check."),
        ] = None
        lacks: Annotated[
            t.Tests.ExclusionSpec | None,
            u.Field(description="Unified non-containment check."),
        ] = None
        starts: Annotated[
            str | None, u.Field(description="String starts with prefix.")
        ] = None
        ends: Annotated[str | None, u.Field(description="String ends with suffix.")] = (
            None
        )
        match: Annotated[
            t.Infra.RegexPattern | None, u.Field(description="Compiled regex pattern.")
        ] = None
        len: Annotated[
            t.Tests.LengthSpec | None, u.Field(description="Length spec.")
        ] = None
        deep: Annotated[
            t.Tests.DeepSpec | None, u.Field(description="Deep structural matching.")
        ] = None
        path: Annotated[
            t.Tests.PathSpec | None,
            u.Field(description="Extract nested value via dot notation."),
        ] = None
        paths: Annotated[
            Mapping[str, FlextTestsMatchersModelsMixin.MatchRule] | None,
            u.Field(description="Multiple path-based assertions."),
        ] = None
        items: Annotated[
            Sequence[FlextTestsMatchersModelsMixin.MatchRule]
            | Mapping[str | int, FlextTestsMatchersModelsMixin.MatchRule]
            | None,
            u.Field(description="Sequence item assertions by selector."),
        ] = None
        attrs_match: Annotated[
            Mapping[str, FlextTestsMatchersModelsMixin.MatchRule] | None,
            u.Field(description="Attribute assertions by attribute path."),
        ] = None
        where: Annotated[
            t.Tests.PredicateSpec | None,
            u.Field(description="Custom predicate function."),
        ] = None
        msg: Annotated[str | None, u.Field(description="Custom error message.")] = None

        @u.field_validator("paths", "items", "attrs_match", mode="before")
        @classmethod
        def parse_rules(cls, value: object) -> object:
            """Parse public rule collections into nominal rules."""
            return FlextTestsMatchersModelsMixin.MatchRule.parse_rule_fields(value)

    class FailParams(m.Value):
        """Matcher parameters for failure result assertions."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(populate_by_name=True)

        msg: Annotated[str | None, u.Field(description="Custom error message.")] = None
        has: Annotated[
            t.Tests.ExclusionSpec | None,
            u.Field(
                validation_alias=t.AliasChoices("has", "contains"),
                description="Error contains substrings.",
            ),
        ] = None
        lacks: Annotated[
            t.Tests.ExclusionSpec | None,
            u.Field(
                validation_alias=t.AliasChoices("lacks", "excludes"),
                description="Error does not contain substrings.",
            ),
        ] = None
        starts: Annotated[
            str | None, u.Field(description="Error starts with prefix.")
        ] = None
        ends: Annotated[str | None, u.Field(description="Error ends with suffix.")] = (
            None
        )
        match: Annotated[
            t.Infra.RegexPattern | None,
            u.Field(description="Error matches compiled regex."),
        ] = None
        code: Annotated[str | None, u.Field(description="Error code equals.")] = None
        code_has: Annotated[
            t.Tests.ErrorCodeSpec | None,
            u.Field(description="Error code contains substrings."),
        ] = None
        data: Annotated[
            t.Tests.ErrorDataSpec | None,
            u.Field(description="Error data contains key-value pairs."),
        ] = None

    class ThatParams(m.Value):
        """Generic matcher parameters for value assertions."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            populate_by_name=True, arbitrary_types_allowed=True
        )

        msg: Annotated[str | None, u.Field(description="Message.")] = None
        eq: Annotated[
            t.Tests.TestobjectSerializable | None, u.Field(description="Equals.")
        ] = None
        ne: Annotated[
            t.Tests.TestobjectSerializable | None, u.Field(description="Not equals.")
        ] = None
        is_: Annotated[
            type[object] | tuple[type[object], ...] | None,
            u.Field(validation_alias=t.AliasChoices("is_", "is"), description="Type."),
        ] = None
        not_: Annotated[
            type[object] | tuple[type[object], ...] | None,
            u.Field(
                validation_alias=t.AliasChoices("not_", "not"), description="Not type."
            ),
        ] = None
        none: Annotated[bool | None, u.Field(description="None check.")] = None
        empty: Annotated[bool | None, u.Field(description="Empty check.")] = None
        gt: Annotated[
            t.Tests.ComparableScalar | None, u.Field(description="Greater than.")
        ] = None
        gte: Annotated[
            t.Tests.ComparableScalar | None, u.Field(description="Greater/equal.")
        ] = None
        lt: Annotated[
            t.Tests.ComparableScalar | None, u.Field(description="Less than.")
        ] = None
        lte: Annotated[
            t.Tests.ComparableScalar | None, u.Field(description="Less/equal.")
        ] = None
        len: Annotated[
            t.Tests.LengthSpec | None,
            u.Field(
                validation_alias=t.AliasChoices("len", "length"), description="Length."
            ),
        ] = None
        length_gt: Annotated[int | None, u.Field(description="Length >.")] = None
        length_gte: Annotated[int | None, u.Field(description="Length >=.")] = None
        length_lt: Annotated[int | None, u.Field(description="Length <.")] = None
        length_lte: Annotated[int | None, u.Field(description="Length <=.")] = None
        has: Annotated[
            t.Tests.ContainmentSpec | None,
            m.SkipValidation,
            u.Field(
                validation_alias=t.AliasChoices("has", "contains"),
                description="Contains.",
            ),
        ] = None
        lacks: Annotated[
            t.Tests.ExclusionSpec | None,
            m.SkipValidation,
            u.Field(
                validation_alias=t.AliasChoices("lacks", "excludes"),
                description="Lacks.",
            ),
        ] = None
        starts: Annotated[str | None, u.Field(description="Prefix.")] = None
        ends: Annotated[str | None, u.Field(description="Suffix.")] = None
        match: Annotated[t.Infra.RegexPattern | None, u.Field(description="Regex.")] = (
            None
        )
        first: Annotated[
            t.Tests.TestobjectSerializable | None, u.Field(description="First item.")
        ] = None
        last: Annotated[
            t.Tests.TestobjectSerializable | None, u.Field(description="Last item.")
        ] = None
        all_: Annotated[
            t.Tests.SequencePredicate | None,
            m.SkipValidation,
            u.Field(validation_alias=t.AliasChoices("all_", "all"), description="All."),
        ] = None
        any_: Annotated[
            t.Tests.SequencePredicate | None,
            m.SkipValidation,
            u.Field(validation_alias=t.AliasChoices("any_", "any"), description="Any."),
        ] = None
        sorted: Annotated[t.Tests.SortKey | None, u.Field(description="Sort key.")] = (
            None
        )
        unique: Annotated[bool | None, u.Field(description="Unique.")] = None
        keys: Annotated[t.Tests.KeySpec | None, u.Field(description="Keys.")] = None
        lacks_keys: Annotated[
            t.Tests.KeySpec | None, u.Field(description="No keys.")
        ] = None
        values: Annotated[
            t.SequenceOf[t.Tests.TestobjectSerializable] | None,
            u.Field(description="Values."),
        ] = None
        kv: Annotated[
            t.Tests.KeyValueSpec | None, u.Field(description="Key-values.")
        ] = None
        attrs: Annotated[
            t.Tests.AttributeSpec | None, u.Field(description="Attrs.")
        ] = None
        methods: Annotated[
            t.Tests.AttributeSpec | None, u.Field(description="Methods.")
        ] = None
        attr_eq: Annotated[
            t.Tests.AttributeValueSpec | None, u.Field(description="Attr equals.")
        ] = None
        ok: Annotated[bool | None, u.Field(description="Result ok.")] = None
        error: Annotated[
            str | t.StrSequence | None, u.Field(description="Result error.")
        ] = None
        deep: Annotated[t.Tests.DeepSpec | None, u.Field(description="Deep spec.")] = (
            None
        )
        paths: Annotated[
            Mapping[str, FlextTestsMatchersModelsMixin.MatchRule] | None,
            u.Field(description="Paths."),
        ] = None
        items: Annotated[
            Sequence[FlextTestsMatchersModelsMixin.MatchRule]
            | Mapping[str | int, FlextTestsMatchersModelsMixin.MatchRule]
            | None,
            u.Field(description="Items."),
        ] = None
        attrs_match: Annotated[
            Mapping[str, FlextTestsMatchersModelsMixin.MatchRule] | None,
            u.Field(description="Attr rules."),
        ] = None
        where: Annotated[
            t.Tests.PredicateSpec | None,
            m.SkipValidation,
            u.Field(description="Predicate."),
        ] = None

        @u.field_validator("paths", "items", "attrs_match", mode="before")
        @classmethod
        def parse_rules(cls, value: object) -> object:
            """Parse public rule collections into nominal rules."""
            return FlextTestsMatchersModelsMixin.MatchRule.parse_rule_fields(value)

        @u.model_validator(mode="after")
        def normalize_legacy_parameters(
            self,
        ) -> FlextTestsMatchersModelsMixin.ThatParams:
            """Normalize legacy aliases into canonical matcher fields."""
            updates: MutableMapping[str, t.Tests.TestobjectSerializable] = {}
            if self.error is not None and self.has is None:
                updates["has"] = self.error
            if self.len is None and any(
                v is not None
                for v in (
                    self.length_gt,
                    self.length_gte,
                    self.length_lt,
                    self.length_lte,
                )
            ):
                min_len = 0
                max_len = sys.maxsize
                if self.length_gt is not None:
                    min_len = self.length_gt + 1
                if self.length_gte is not None:
                    min_len = max(min_len, self.length_gte)
                if self.length_lt is not None:
                    max_len = self.length_lt - 1
                if self.length_lte is not None:
                    max_len = min(max_len, self.length_lte)
                updates["len"] = (min_len, max_len)
            if updates:
                payload = self.model_dump()
                payload.update(updates)
                return FlextTestsMatchersModelsMixin.ThatParams(**payload)
            return self

    class ScopeParams(m.Value):
        """Parameters for temporary test scope configuration."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(populate_by_name=True)

        settings: Annotated[
            t.MappingKV[str, t.Tests.TestobjectSerializable] | None,
            u.Field(description="Initial configuration values."),
        ] = None
        container: Annotated[
            t.MappingKV[str, t.Tests.TestobjectSerializable] | None,
            u.Field(description="Initial container/service mappings."),
        ] = None
        context: Annotated[
            t.MappingKV[str, t.Tests.TestobjectSerializable] | None,
            u.Field(description="Initial context values."),
        ] = None
        cleanup: Annotated[
            t.Tests.CleanupSpec | None, u.Field(description="Cleanup functions.")
        ] = None
        env: Annotated[
            t.Tests.EnvironmentSpec | None,
            u.Field(description="Temporary environment variables."),
        ] = None
        cwd: Annotated[
            Path | str | None, u.Field(description="Temporary working directory.")
        ] = None

        @u.field_validator("cwd", mode="before")
        @classmethod
        def convert_cwd(cls, value: Path | str | None) -> Path | str | None:
            """Convert string cwd to Path."""
            if isinstance(value, str):
                return Path(value)
            return value

    class DeepMatchResult(m.Value):
        """Structured output for deep-match comparisons."""

        path: Annotated[str, u.Field(description="Path where matching occurred.")]
        expected: Annotated[
            t.Tests.TestobjectSerializable
            | Callable[[t.Tests.TestobjectSerializable], bool]
            | None,
            u.Field(description="Expected value or predicate."),
        ]
        actual: Annotated[
            t.Tests.TestobjectSerializable | None,
            u.Field(description="Actual value found."),
        ] = None
        matched: Annotated[bool, u.Field(description="Whether match succeeded.")]
        reason: Annotated[str, u.Field(description="Reason for match failure.")] = ""

    class Validate:
        """Centralized TypeAdapters for test data validation."""

        DICT_ADAPTER: ClassVar[
            m.TypeAdapter[Mapping[str, t.Tests.TestobjectSerializable]]
        ] = t.Tests.TESTOBJECT_SERIALIZABLE_MAPPING_ADAPTER
        LIST_ADAPTER: ClassVar[
            m.TypeAdapter[Sequence[t.Tests.TestobjectSerializable]]
        ] = t.Tests.TESTOBJECT_SERIALIZABLE_SEQUENCE_ADAPTER

    class Chain[TResult](m.Value):
        """Container for chained result assertions."""

        result: Annotated[
            p.Result[TResult], u.Field(description="Result being chained.")
        ]

    class TestScope(m.ArbitraryTypesModel):
        """Scope container for test configuration and runtime state."""

        settings: Annotated[
            t.MappingKV[str, t.Tests.TestobjectSerializable],
            u.Field(description="Configuration dictionary."),
        ] = u.Field(
            default_factory=lambda: MappingProxyType(
                dict[str, t.Tests.TestobjectSerializable]()
            )
        )
        container: Annotated[
            t.MappingKV[str, t.Tests.TestobjectSerializable],
            u.Field(description="Container/service mappings."),
        ] = u.Field(
            default_factory=lambda: MappingProxyType(
                dict[str, t.Tests.TestobjectSerializable]()
            )
        )
        context: Annotated[
            t.MappingKV[str, t.Tests.TestobjectSerializable],
            u.Field(description="Context values."),
        ] = u.Field(
            default_factory=lambda: MappingProxyType(
                dict[str, t.Tests.TestobjectSerializable]()
            )
        )


__all__: list[str] = ["FlextTestsMatchersModelsMixin"]
