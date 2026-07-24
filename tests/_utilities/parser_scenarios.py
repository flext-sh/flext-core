"""Parser scenario helpers for flext-core tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_tests import u
from tests.constants import c
from tests.models import m

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestsFlextUtilitiesParserScenariosMixin:
    """Parser scenario helpers."""

    class ParserScenarios:
        """Centralized parser scenarios - single source of truth."""

        PUBLIC_PARSE_CASES: ClassVar[Sequence[m.Tests.PublicParseCase]] = [
            m.Tests.PublicParseCase(
                name="string-to-int",
                input_value="42",
                target=int,
                should_succeed=True,
                expected_value=42,
                description="Public parse coerces numeric string into int",
            ),
            m.Tests.PublicParseCase(
                name="int-to-str",
                input_value=42,
                target=str,
                should_succeed=True,
                expected_value="42",
                description="Public parse coerces int into string",
            ),
            m.Tests.PublicParseCase(
                name="string-to-float",
                input_value="2.2",
                target=float,
                should_succeed=True,
                expected_value=2.2,
                description="Public parse coerces numeric string into float",
            ),
            m.Tests.PublicParseCase(
                name="string-to-bool",
                input_value="true",
                target=bool,
                should_succeed=True,
                expected_value=True,
                description="Public parse coerces truthy string into bool",
            ),
            m.Tests.PublicParseCase(
                name="none-uses-default",
                input_value=None,
                target=int,
                options=u.ParseOptions(default=7),
                should_succeed=True,
                expected_value=7,
                description="Public parse returns default when value is None",
            ),
            m.Tests.PublicParseCase(
                name="invalid-uses-default-factory",
                input_value="x",
                target=int,
                options=u.ParseOptions(default_factory=lambda: 9),
                should_succeed=True,
                expected_value=9,
                description="Public parse returns default_factory output on failure",
            ),
            m.Tests.PublicParseCase(
                name="enum-exact",
                input_value="inactive",
                target=c.Tests.STATUS_ENUM,
                should_succeed=True,
                expected_value=c.Tests.STATUS_INACTIVE,
                description="Public parse resolves StrEnum exact values",
            ),
            m.Tests.PublicParseCase(
                name="enum-case-insensitive",
                input_value="INACTIVE",
                target=c.Tests.STATUS_ENUM,
                options=u.ParseOptions(case_insensitive=True),
                should_succeed=True,
                expected_value=c.Tests.STATUS_INACTIVE,
                description="Public parse resolves StrEnum values case-insensitively",
            ),
            m.Tests.PublicParseCase(
                name="model-from-mapping",
                input_value={"name": "parsed", "value": 3},
                target=m.Tests.SampleModel,
                should_succeed=True,
                expected_data={"name": "parsed", "value": 3},
                description="Public parse materializes canonical test model from mapping",
            ),
            m.Tests.PublicParseCase(
                name="invalid-int-fails",
                input_value="x",
                target=int,
                should_succeed=False,
                description="Public parse fails for non-numeric int input",
            ),
            m.Tests.PublicParseCase(
                name="invalid-enum-fails",
                input_value="missing",
                target=c.Tests.STATUS_ENUM,
                should_succeed=False,
                description="Public parse fails for unknown enum values",
            ),
            m.Tests.PublicParseCase(
                name="invalid-model-shape-fails",
                input_value={"bad": "shape"},
                target=m.Tests.SampleModel,
                should_succeed=False,
                description="Public parse fails for invalid model payloads",
            ),
            m.Tests.PublicParseCase(
                name="invalid-bool-field-context",
                input_value="maybe",
                target=bool,
                options=u.ParseOptions(field_name="flag"),
                should_succeed=False,
                error_contains="flag",
                description="Public parse includes field context on bool failure",
            ),
        ]

        LDIF_PARSE_SCENARIOS: ClassVar[Sequence[m.Tests.ParserScenario]] = [
            m.Tests.ParserScenario(
                name="parse_simple_dn",
                parser_method="parse",
                input_data="dn: cn=test,dc=example,dc=com",
                should_succeed=True,
                description="Simple DN parsing",
            ),
            m.Tests.ParserScenario(
                name="parse_with_attributes",
                parser_method="parse",
                input_data="dn: cn=test,dc=example,dc=com\nobjectClass: person\ncn: test",
                should_succeed=True,
                description="DN with attributes",
            ),
            m.Tests.ParserScenario(
                name="parse_invalid_dn",
                parser_method="parse",
                input_data="invalid",
                should_succeed=False,
                error_contains="invalid",
                description="Invalid DN format",
            ),
        ]


__all__: list[str] = ["TestsFlextUtilitiesParserScenariosMixin"]
