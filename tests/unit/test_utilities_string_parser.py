"""Comprehensive tests for u - 100% coverage target.

**Modules Tested:**
- flext_core._utilities.string_parser.u

**Test Scope:**
- parse_delimited: All options, validators, legacy params, edge cases, error handling
- split_on_char_with_escape: Basic, custom escape, edge cases, error handling
- normalize_whitespace: Basic, custom patterns, edge cases, error handling
- apply_regex_pipeline: Basic, multiple patterns, edge cases,
  error handling, invalid inputs
- get_object_key: Types, classes, functions, instances, dicts, objects with attributes

**Coverage Target:** 100% code coverage with all edge cases and error paths tested.

This module uses Python 3.13 advanced features (dataclasses with slots, type hints),
factory patterns for test case generation, and nested class organization for
maximum code reuse and maintainability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from flext_core import m, r, t
from flext_tests import t as tests_t, tm, u
from tests.constants import (
    TestsFlextConstants,
)
from tests.models import m as tm
from tests.utilities import (
    FlextCoreTestsUtilities,
)

_object = tests_t.Tests.object


class StringParserTestFactory:
    """Factory for generating test cases with edge cases."""

    @staticmethod
    def parse_delimited_cases() -> list[tm.ParseDelimitedCase]:
        """Generate comprehensive parse_delimited test cases."""
        return [
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.BASIC_LIST,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                description="basic",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.WITH_SPACES,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                description="with spaces",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.EMPTY,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=[],
                description="empty string",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.WITH_SPACES,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                options=tm.ParseOptions(strip=True, remove_empty=True),
                description="with options",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.WITH_SPACES,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", " b", " c"],
                options=tm.ParseOptions(strip=False, remove_empty=True),
                description="options no strip",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.WITH_EMPTY,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "", "c"],
                options=tm.ParseOptions(strip=True, remove_empty=False),
                description="options no remove empty",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.BASIC_LIST,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                options=tm.ParseOptions(
                    strip=True,
                    remove_empty=True,
                    validator=lambda s: len(s) > 0,
                ),
                description="validator success",
            ),
            tm.ParseDelimitedCase(
                text="a,b",
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=[],
                options=tm.ParseOptions(
                    strip=True,
                    remove_empty=True,
                    validator=lambda s: len(s) > 1,
                ),
                description="validator filters components",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.WITH_SPACES,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                strip=True,
                remove_empty=True,
                use_legacy=True,
                description="legacy params",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.BASIC_LIST,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                validator=lambda s: "x" not in s,
                use_legacy=True,
                description="legacy validator",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.BASIC_LIST,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                description="no spaces",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.EXCESSIVE_SPACES,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                description="excessive spaces",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.LEADING_TRAILING,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                description="leading/trailing delimiters",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.SINGLE_CHAR,
                delimiter=TestsFlextConstants.Delimiters.COMMA,
                expected=["a"],
                description="single component",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.BASIC_LIST,
                delimiter="",
                expected=None,
                expected_error=TestsFlextConstants.TestErrors.DELIMITER_EMPTY,
                description="empty delimiter",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.BASIC_LIST,
                delimiter=",,",
                expected=None,
                expected_error=TestsFlextConstants.TestErrors.DELIMITER_MULTI,
                description="multi-char delimiter",
            ),
            tm.ParseDelimitedCase(
                text=TestsFlextConstants.Strings.BASIC_LIST,
                delimiter=" ",
                expected=None,
                expected_error=TestsFlextConstants.TestErrors.DELIMITER_WHITESPACE,
                description="whitespace delimiter",
            ),
        ]

    @staticmethod
    def split_escape_cases() -> list[tm.SplitEscapeCase]:
        """Generate comprehensive split_on_char_with_escape test cases."""
        return [
            tm.SplitEscapeCase(
                text=TestsFlextConstants.Strings.BASIC_LIST,
                split_char=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                description="basic",
            ),
            tm.SplitEscapeCase(
                text="a\\,b,c",
                split_char=TestsFlextConstants.Delimiters.COMMA,
                expected=["a,b", "c"],
                description="escaped delimiter",
            ),
            tm.SplitEscapeCase(
                text=TestsFlextConstants.Strings.EMPTY,
                split_char=TestsFlextConstants.Delimiters.COMMA,
                expected=[""],
                description="empty string",
            ),
            tm.SplitEscapeCase(
                text="a#b,c",
                split_char=TestsFlextConstants.Delimiters.COMMA,
                escape_char=TestsFlextConstants.EscapeChars.HASH,
                expected=["ab", "c"],
                description="custom escape char",
            ),
            tm.SplitEscapeCase(
                text="a,b\\",
                split_char=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b\\"],
                description="escape at end",
            ),
            tm.SplitEscapeCase(
                text="a\\\\,b",
                split_char=TestsFlextConstants.Delimiters.COMMA,
                expected=["a\\", "b"],
                description="escaped escape char",
            ),
            tm.SplitEscapeCase(
                text="a,b,c,d",
                split_char=TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c", "d"],
                description="multiple components",
            ),
            tm.SplitEscapeCase(
                text="\\,a,b",
                split_char=TestsFlextConstants.Delimiters.COMMA,
                expected=[",a", "b"],
                description="escaped at start",
            ),
            tm.SplitEscapeCase(
                text="a,b",
                split_char="",
                expected_error=TestsFlextConstants.TestErrors.SPLIT_EMPTY,
                description="empty split char",
            ),
            tm.SplitEscapeCase(
                text="a,b",
                split_char=TestsFlextConstants.Delimiters.COMMA,
                escape_char="",
                expected_error=TestsFlextConstants.TestErrors.ESCAPE_EMPTY,
                description="empty escape char",
            ),
            tm.SplitEscapeCase(
                text="a,b",
                split_char=TestsFlextConstants.Delimiters.COMMA,
                escape_char=TestsFlextConstants.Delimiters.COMMA,
                expected_error=TestsFlextConstants.TestErrors.SPLIT_ESCAPE_SAME,
                description="same split and escape",
            ),
        ]

    @staticmethod
    def normalize_whitespace_cases() -> list[tm.NormalizeWhitespaceCase]:
        """Generate comprehensive normalize_whitespace test cases."""
        return [
            tm.NormalizeWhitespaceCase(
                text="  hello   world  ",
                expected="hello world",
                description="basic",
            ),
            tm.NormalizeWhitespaceCase(
                text=TestsFlextConstants.Strings.EMPTY,
                expected=TestsFlextConstants.Strings.EMPTY,
                description="empty string",
            ),
            tm.NormalizeWhitespaceCase(
                text="hello---world",
                pattern=TestsFlextConstants.Patterns.DASH,
                replacement=TestsFlextConstants.Replacements.DASH,
                expected="hello-world",
                description="custom pattern",
            ),
            tm.NormalizeWhitespaceCase(
                text="hello   world",
                replacement=TestsFlextConstants.Replacements.UNDERSCORE,
                expected="hello_world",
                description="custom replacement",
            ),
            tm.NormalizeWhitespaceCase(
                text="hello\t\n\rworld",
                expected="hello world",
                description="various whitespace",
            ),
            tm.NormalizeWhitespaceCase(
                text="   ",
                expected="",
                description="only whitespace",
            ),
            tm.NormalizeWhitespaceCase(
                text=TestsFlextConstants.Strings.SINGLE_CHAR,
                expected=TestsFlextConstants.Strings.SINGLE_CHAR,
                description="single char",
            ),
            tm.NormalizeWhitespaceCase(
                text="hello\n\n\nworld",
                expected="hello world",
                description="multiple newlines",
            ),
        ]

    @staticmethod
    def regex_pipeline_cases() -> list[tm.RegexPipelineCase]:
        """Generate comprehensive apply_regex_pipeline test cases."""
        return [
            tm.RegexPipelineCase(
                text="hello   world",
                patterns=[
                    (
                        TestsFlextConstants.Patterns.WHITESPACE,
                        TestsFlextConstants.Replacements.SPACE,
                    ),
                    ("=", "="),
                ],
                expected="hello world",
                description="basic",
            ),
            tm.RegexPipelineCase(
                text=TestsFlextConstants.Strings.EMPTY,
                patterns=[
                    (
                        TestsFlextConstants.Patterns.WHITESPACE,
                        TestsFlextConstants.Replacements.SPACE,
                    ),
                ],
                expected=TestsFlextConstants.Strings.EMPTY,
                description="empty string",
            ),
            tm.RegexPipelineCase(
                text="cn = REDACTED_LDAP_BIND_PASSWORD , ou = users",
                patterns=[
                    (
                        TestsFlextConstants.Patterns.EQUALS_SPACE,
                        TestsFlextConstants.Replacements.EQUALS,
                    ),
                    (
                        TestsFlextConstants.Patterns.COMMA_SPACE,
                        TestsFlextConstants.Replacements.COMMA,
                    ),
                    (
                        TestsFlextConstants.Patterns.WHITESPACE,
                        TestsFlextConstants.Replacements.SPACE,
                    ),
                ],
                expected="cn= REDACTED_LDAP_BIND_PASSWORD ,ou= users",
                description="multiple patterns",
            ),
            tm.RegexPipelineCase(
                text="test",
                patterns=[],
                expected="test",
                description="empty patterns",
            ),
            tm.RegexPipelineCase(
                text="a=b=c",
                patterns=[("=", ":"), (":", "=")],
                expected="a=b=c",
                description="pattern chaining",
            ),
        ]

    @staticmethod
    def object_key_cases() -> list[tm.ObjectKeyCase]:
        """Generate comprehensive get_object_key test cases (object only)."""
        return [
            tm.ObjectKeyCase(
                obj={},
                expected_exact="dict",
                description="instance",
            ),
            tm.ObjectKeyCase(
                obj="test",
                expected_exact="test",
                description="string",
            ),
            tm.ObjectKeyCase(
                obj={"name": "DictName"},
                expected_exact="DictName",
                description="dict with name",
            ),
            tm.ObjectKeyCase(
                obj={"id": "DictId"},
                expected_exact="DictId",
                description="dict with id",
            ),
        ]


class TestuStringParser:
    """Comprehensive tests for u using nested organization."""

    @pytest.fixture
    def parser(self) -> u:
        """Create parser instance."""
        return u()

    class TestParseDelimited:
        """Test parse_delimited method."""

        @pytest.mark.parametrize(
            "case",
            StringParserTestFactory.parse_delimited_cases(),
        )
        def test_parse_delimited(
            self,
            parser: u,
            case: tm.ParseDelimitedCase,
        ) -> None:
            """Test parse_delimited with parametrized cases."""

            def operation() -> r[list[str]]:
                """Execute parse_delimited based on case configuration."""
                if case.use_legacy:
                    options = m.ParseOptions(
                        strip=case.strip or True,
                        remove_empty=case.remove_empty or True,
                        validator=case.validator,
                    )
                    return parser.parse_delimited(
                        case.text,
                        case.delimiter,
                        options=options,
                    )
                if case.options:
                    return parser.parse_delimited(
                        case.text,
                        case.delimiter,
                        options=case.options,
                    )
                return parser.parse_delimited(case.text, case.delimiter)

            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast("Callable[[], r[_object]]", operation),
                expected_value=case.expected,
                expected_error=case.expected_error,
                description=case.description,
            )

        def test_exception_handling(self, parser: u) -> None:
            """Test parsing exception handling with bad object."""
            bad_obj = FlextCoreTestsUtilities.Tests.CoreBadObjects.create_for_split()
            bad_str = cast("str", cast("object", bad_obj))
            result = parser.parse_delimited(
                bad_str,
                TestsFlextConstants.Delimiters.COMMA,
            )
            assert result.is_failure
            assert TestsFlextConstants.TestErrors.FAILED_PARSE in (result.error or "")

    class TestSplitWithEscape:
        """Test split_on_char_with_escape method."""

        @pytest.mark.parametrize("case", StringParserTestFactory.split_escape_cases())
        def test_split_with_escape(
            self,
            parser: u,
            case: tm.SplitEscapeCase,
        ) -> None:
            """Test split_on_char_with_escape with parametrized cases."""
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[tests_t.Tests.object]]",
                    lambda: parser.split_on_char_with_escape(
                        case.text,
                        case.split_char,
                        escape_char=case.escape_char,
                    ),
                ),
                expected_value=case.expected,
                expected_error=case.expected_error,
                description=case.description,
            )

        def test_exception_handling(self, parser: u) -> None:
            """Test split exception handling with bad object."""
            bad_obj = FlextCoreTestsUtilities.Tests.CoreBadObjects.create_for_index()
            bad_str = cast("str", cast("object", bad_obj))
            result = parser.split_on_char_with_escape(
                bad_str,
                TestsFlextConstants.Delimiters.COMMA,
            )
            assert result.is_failure
            assert TestsFlextConstants.TestErrors.FAILED_SPLIT in (result.error or "")

    class TestNormalizeWhitespace:
        """Test normalize_whitespace method."""

        @pytest.mark.parametrize(
            "case",
            StringParserTestFactory.normalize_whitespace_cases(),
        )
        def test_normalize_whitespace(
            self,
            parser: u,
            case: tm.NormalizeWhitespaceCase,
        ) -> None:
            """Test normalize_whitespace with parametrized cases."""
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[tests_t.Tests.object]]",
                    lambda: parser.normalize_whitespace(
                        case.text,
                        pattern=case.pattern,
                        replacement=case.replacement,
                    ),
                ),
                expected_value=case.expected,
                expected_error=case.expected_error,
                description=case.description,
            )

        def test_exception_handling(self, parser: u) -> None:
            """Test normalization exception handling with bad object."""
            bad_obj = FlextCoreTestsUtilities.Tests.CoreBadObjects.create_for_str()
            bad_str = cast("str", cast("object", bad_obj))
            result = parser.normalize_whitespace(bad_str)
            assert result.is_failure
            assert TestsFlextConstants.TestErrors.FAILED_NORMALIZE in (
                result.error or ""
            )

    class TestRegexPipeline:
        """Test apply_regex_pipeline method."""

        @pytest.mark.parametrize("case", StringParserTestFactory.regex_pipeline_cases())
        def test_apply_regex_pipeline(
            self,
            parser: u,
            case: tm.RegexPipelineCase,
        ) -> None:
            """Test apply_regex_pipeline with parametrized cases."""
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[tests_t.Tests.object]]",
                    lambda: parser.apply_regex_pipeline(case.text, case.patterns),
                ),
                expected_value=case.expected,
                expected_error=case.expected_error,
                description=case.description,
            )

        def test_exception_handling(self, parser: u) -> None:
            """Test pipeline exception handling."""
            invalid_pattern = cast("tuple[str, str]", (None, "replacement"))
            patterns: list[tuple[str, str] | tuple[str, str, int]] = [invalid_pattern]
            result = parser.apply_regex_pipeline("test", patterns)
            assert result.is_failure
            assert TestsFlextConstants.TestErrors.FAILED_PIPELINE in (
                result.error or ""
            )

        def test_invalid_pattern(self, parser: u) -> None:
            """Test pipeline with invalid regex pattern."""
            patterns: list[tuple[str, str] | tuple[str, str, int]] = [
                ("[invalid", "replacement"),
            ]
            result = parser.apply_regex_pipeline("test", patterns)
            assert result.is_failure
            assert TestsFlextConstants.TestErrors.INVALID_REGEX in (result.error or "")

        def test_none_text(self, parser: u) -> None:
            """Test pipeline with None text."""
            text = cast("str", cast("object", None))
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[tests_t.Tests.object]]",
                    lambda: parser.apply_regex_pipeline(
                        text,
                        [
                            (
                                TestsFlextConstants.Patterns.WHITESPACE,
                                TestsFlextConstants.Replacements.SPACE,
                            ),
                        ],
                    ),
                ),
                expected_error="None",
                description="none text",
            )

        def test_invalid_text_type(self, parser: u) -> None:
            """Test pipeline with invalid text type."""
            text = cast("str", cast("object", 123))
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[tests_t.Tests.object]]",
                    lambda: parser.apply_regex_pipeline(
                        text,
                        [
                            (
                                TestsFlextConstants.Patterns.WHITESPACE,
                                TestsFlextConstants.Replacements.SPACE,
                            ),
                        ],
                    ),
                ),
                expected_error="str",
                description="invalid text type",
            )

        def test_empty_patterns(self, parser: u) -> None:
            """Test pipeline with empty patterns list."""
            patterns: list[tuple[str, str] | tuple[str, str, int]] = []
            result = parser.apply_regex_pipeline("test", patterns)
            u.Tests.Result.assert_success_with_value(result, "test")

    class TestGetObjectKey:
        """Test get_object_key method."""

        @pytest.mark.parametrize("case", StringParserTestFactory.object_key_cases())
        def test_get_object_key(
            self,
            parser: u,
            case: tm.ObjectKeyCase,
        ) -> None:
            """Test get_object_key with parametrized cases."""
            key = parser.get_object_key(
                cast("t.TypeHintSpecifier | t.NormalizedValue", case.obj)
            )
            assert isinstance(key, str), f"Key must be string for: {case.description}"
            if case.expected_exact:
                assert key == case.expected_exact, (
                    f"Exact match failed for: {case.description}"
                )
            elif case.expected_contains:
                for expected in case.expected_contains:
                    assert expected in key, (
                        f"Contains check failed for: {case.description}"
                    )
