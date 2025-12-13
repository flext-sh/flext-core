"""Comprehensive tests for u.Parser - 100% coverage target.

**Modules Tested:**
- flext_core._utilities.string_parser.u.Parser

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

from flext_core import (
    FlextModels as core_m,
    FlextResult as r,
    t,
)
from flext_tests import u
from tests.constants import TestsFlextConstants
from tests.models import TestsFlextModels
from tests.utilities import TestsFlextUtilities

# Test case dataclasses from tests.models
NormalizeWhitespaceCase = TestsFlextModels.NormalizeWhitespaceCase
ObjectKeyCase = TestsFlextModels.ObjectKeyCase
ParseDelimitedCase = TestsFlextModels.ParseDelimitedCase
ParseOptions = TestsFlextModels.ParseOptions
RegexPipelineCase = TestsFlextModels.RegexPipelineCase
SplitEscapeCase = TestsFlextModels.SplitEscapeCase


class StringParserTestFactory:
    """Factory for generating test cases with edge cases."""

    @staticmethod
    def parse_delimited_cases() -> list[ParseDelimitedCase]:
        """Generate comprehensive parse_delimited test cases."""
        return [
            # Basic cases
            ParseDelimitedCase(
                TestsFlextConstants.Strings.BASIC_LIST,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                description="basic",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.WITH_SPACES,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                description="with spaces",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.EMPTY,
                TestsFlextConstants.Delimiters.COMMA,
                [],
                description="empty string",
            ),
            # Options cases
            ParseDelimitedCase(
                TestsFlextConstants.Strings.WITH_SPACES,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                options=ParseOptions(strip=True, remove_empty=True),
                description="with options",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.WITH_SPACES,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", " b", " c"],
                options=ParseOptions(strip=False, remove_empty=True),
                description="options no strip",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.WITH_EMPTY,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "", "c"],
                options=ParseOptions(strip=True, remove_empty=False),
                description="options no remove empty",
            ),
            # Validator cases (validator filters, doesn't fail)
            ParseDelimitedCase(
                TestsFlextConstants.Strings.BASIC_LIST,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                options=ParseOptions(
                    strip=True,
                    remove_empty=True,
                    validator=lambda s: len(s) > 0,
                ),
                description="validator success",
            ),
            ParseDelimitedCase(
                "a,b",
                TestsFlextConstants.Delimiters.COMMA,
                [],  # Validator filters out single-char components
                options=ParseOptions(
                    strip=True,
                    remove_empty=True,
                    validator=lambda s: len(s) > 1,  # Filter single char
                ),
                description="validator filters components",
            ),
            # Legacy parameter cases
            ParseDelimitedCase(
                TestsFlextConstants.Strings.WITH_SPACES,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                strip=True,
                remove_empty=True,
                use_legacy=True,
                description="legacy params",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.BASIC_LIST,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                validator=lambda s: "x" not in s,
                use_legacy=True,
                description="legacy validator",
            ),
            # Edge cases
            ParseDelimitedCase(
                TestsFlextConstants.Strings.BASIC_LIST,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                description="no spaces",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.EXCESSIVE_SPACES,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                description="excessive spaces",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.LEADING_TRAILING,
                TestsFlextConstants.Delimiters.COMMA,
                ["a", "b", "c"],
                description="leading/trailing delimiters",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.SINGLE_CHAR,
                TestsFlextConstants.Delimiters.COMMA,
                ["a"],
                description="single component",
            ),
            # Invalid delimiter cases
            ParseDelimitedCase(
                TestsFlextConstants.Strings.BASIC_LIST,
                "",
                None,
                expected_error=TestsFlextConstants.TestErrors.DELIMITER_EMPTY,
                description="empty delimiter",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.BASIC_LIST,
                ",,",
                None,
                expected_error=TestsFlextConstants.TestErrors.DELIMITER_MULTI,
                description="multi-char delimiter",
            ),
            ParseDelimitedCase(
                TestsFlextConstants.Strings.BASIC_LIST,
                " ",
                None,
                expected_error=TestsFlextConstants.TestErrors.DELIMITER_WHITESPACE,
                description="whitespace delimiter",
            ),
        ]

    @staticmethod
    def split_escape_cases() -> list[SplitEscapeCase]:
        """Generate comprehensive split_on_char_with_escape test cases."""
        # Note: split_with_escape removes escape char, so "a\\,b" becomes "a,b"
        return [
            # Basic cases
            SplitEscapeCase(
                TestsFlextConstants.Strings.BASIC_LIST,
                TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c"],
                description="basic",
            ),
            SplitEscapeCase(
                "a\\,b,c",
                TestsFlextConstants.Delimiters.COMMA,
                expected=["a,b", "c"],  # Escape char removed
                description="escaped delimiter",
            ),
            SplitEscapeCase(
                TestsFlextConstants.Strings.EMPTY,
                TestsFlextConstants.Delimiters.COMMA,
                expected=[""],
                description="empty string",
            ),
            # Custom escape char
            SplitEscapeCase(
                "a#b,c",
                TestsFlextConstants.Delimiters.COMMA,
                escape_char=TestsFlextConstants.EscapeChars.HASH,
                expected=["ab", "c"],  # Escape char removed, next char is literal
                description="custom escape char",
            ),
            # Edge cases
            SplitEscapeCase(
                "a,b\\",
                TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b\\"],  # Escape at end is treated as regular char
                description="escape at end",
            ),
            SplitEscapeCase(
                "a\\\\,b",
                TestsFlextConstants.Delimiters.COMMA,
                expected=["a\\", "b"],  # Escaped escape becomes single escape
                description="escaped escape char",
            ),
            SplitEscapeCase(
                "a,b,c,d",
                TestsFlextConstants.Delimiters.COMMA,
                expected=["a", "b", "c", "d"],
                description="multiple components",
            ),
            SplitEscapeCase(
                "\\,a,b",
                TestsFlextConstants.Delimiters.COMMA,
                expected=[",a", "b"],  # Escape char removed
                description="escaped at start",
            ),
            # Invalid cases
            SplitEscapeCase(
                "a,b",
                "",
                expected_error=TestsFlextConstants.TestErrors.SPLIT_EMPTY,
                description="empty split char",
            ),
            SplitEscapeCase(
                "a,b",
                TestsFlextConstants.Delimiters.COMMA,
                escape_char="",
                expected_error=TestsFlextConstants.TestErrors.ESCAPE_EMPTY,
                description="empty escape char",
            ),
            SplitEscapeCase(
                "a,b",
                TestsFlextConstants.Delimiters.COMMA,
                escape_char=TestsFlextConstants.Delimiters.COMMA,
                expected_error=TestsFlextConstants.TestErrors.SPLIT_ESCAPE_SAME,
                description="same split and escape",
            ),
        ]

    @staticmethod
    def normalize_whitespace_cases() -> list[NormalizeWhitespaceCase]:
        """Generate comprehensive normalize_whitespace test cases."""
        return [
            # Basic cases
            NormalizeWhitespaceCase(
                "  hello   world  ",
                expected="hello world",
                description="basic",
            ),
            NormalizeWhitespaceCase(
                TestsFlextConstants.Strings.EMPTY,
                expected=TestsFlextConstants.Strings.EMPTY,
                description="empty string",
            ),
            # Custom pattern
            NormalizeWhitespaceCase(
                "hello---world",
                pattern=TestsFlextConstants.Patterns.DASH,
                replacement=TestsFlextConstants.Replacements.DASH,
                expected="hello-world",
                description="custom pattern",
            ),
            # Custom replacement
            NormalizeWhitespaceCase(
                "hello   world",
                replacement=TestsFlextConstants.Replacements.UNDERSCORE,
                expected="hello_world",
                description="custom replacement",
            ),
            # Edge cases
            NormalizeWhitespaceCase(
                "hello\t\n\rworld",
                expected="hello world",
                description="various whitespace",
            ),
            NormalizeWhitespaceCase(
                "   ",
                expected="",
                description="only whitespace",
            ),
            NormalizeWhitespaceCase(
                TestsFlextConstants.Strings.SINGLE_CHAR,
                expected=TestsFlextConstants.Strings.SINGLE_CHAR,
                description="single char",
            ),
            NormalizeWhitespaceCase(
                "hello\n\n\nworld",
                expected="hello world",
                description="multiple newlines",
            ),
        ]

    @staticmethod
    def regex_pipeline_cases() -> list[RegexPipelineCase]:
        """Generate comprehensive apply_regex_pipeline test cases."""
        return [
            # Basic cases
            RegexPipelineCase(
                "hello   world",
                [
                    (
                        TestsFlextConstants.Patterns.WHITESPACE,
                        TestsFlextConstants.Replacements.SPACE,
                    ),
                    (r"=", "="),
                ],
                expected="hello world",
                description="basic",
            ),
            RegexPipelineCase(
                TestsFlextConstants.Strings.EMPTY,
                [
                    (
                        TestsFlextConstants.Patterns.WHITESPACE,
                        TestsFlextConstants.Replacements.SPACE,
                    ),
                ],
                expected=TestsFlextConstants.Strings.EMPTY,
                description="empty string",
            ),
            # Multiple patterns - note: patterns apply sequentially
            RegexPipelineCase(
                "cn = admin , ou = users",
                [
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
                expected="cn= admin ,ou= users",
                description="multiple patterns",
            ),
            # Edge cases
            RegexPipelineCase(
                "test",
                [],
                expected="test",
                description="empty patterns",
            ),
            RegexPipelineCase(
                "a=b=c",
                [(r"=", ":"), (r":", "=")],
                expected="a=b=c",
                description="pattern chaining",
            ),
        ]

    @staticmethod
    def object_key_cases() -> list[ObjectKeyCase]:
        """Generate comprehensive get_object_key test cases."""

        class TestClass:
            pass

        def test_function() -> None:
            pass

        class NoStr:
            def __str__(self) -> str:
                msg = "Cannot convert to string"
                raise TypeError(msg)

        class WithName:
            name = "TestName"

        class WithId:
            id = "TestId"

        return [
            ObjectKeyCase(int, expected_exact="int", description="type"),
            ObjectKeyCase(TestClass, expected_exact="TestClass", description="class"),
            ObjectKeyCase(
                test_function,
                expected_exact="test_function",
                description="function",
            ),
            ObjectKeyCase(
                {},  # Empty dict - valid t.GeneralValueType, tests dict instance behavior
                expected_exact="dict",
                description="instance",
            ),
            ObjectKeyCase("test", expected_exact="test", description="string"),
            ObjectKeyCase(
                NoStr(),
                expected_contains=["NoStr"],
                description="no str method",
            ),
            ObjectKeyCase(
                WithName(),
                expected_exact="TestName",
                description="with name attr",
            ),
            ObjectKeyCase(
                WithId(),
                expected_exact="TestId",
                description="with id attr",
            ),
            ObjectKeyCase(
                {"name": "DictName"},
                expected_exact="DictName",
                description="dict with name",
            ),
            ObjectKeyCase(
                {"id": "DictId"},
                expected_exact="DictId",
                description="dict with id",
            ),
        ]


class TestuStringParser:
    """Comprehensive tests for u.Parser using nested organization."""

    @pytest.fixture
    def parser(self) -> u.Parser:
        """Create parser instance."""
        return u.Parser()

    class TestParseDelimited:
        """Test parse_delimited method."""

        @pytest.mark.parametrize(
            "case",
            StringParserTestFactory.parse_delimited_cases(),
        )
        def test_parse_delimited(
            self,
            parser: u.Parser,
            case: ParseDelimitedCase,
        ) -> None:
            """Test parse_delimited with parametrized cases."""

            def operation() -> r[list[str]]:
                """Execute parse_delimited based on case configuration."""
                if case.use_legacy:
                    # Legacy params not supported, use options instead
                    options = core_m.CollectionsParseOptions(
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
                cast("Callable[[], r[object]]", operation),
                expected_value=case.expected,
                expected_error=case.expected_error,
                description=case.description,
            )

        def test_exception_handling(self, parser: u.Parser) -> None:
            """Test parsing exception handling with bad object."""
            # Intentionally pass bad object that fails on split() to test error handling
            bad_obj = TestsFlextUtilities.CoreBadObjects.create_for_split()
            # Runtime: parser receives bad object that fails on split()
            # Type checker: cast to str to test runtime error handling
            bad_str = cast("str", bad_obj)
            result = parser.parse_delimited(
                bad_str,
                TestsFlextConstants.Delimiters.COMMA,
            )
            # Type narrowing: assert_failure expects r[t.GeneralValueType], r[list[str]] is compatible
            TestsFlextUtilities.CoreAssertions.assert_failure(
                cast("r[t.GeneralValueType]", result),
                TestsFlextConstants.TestErrors.FAILED_PARSE,
                "exception handling",
            )

    class TestSplitWithEscape:
        """Test split_on_char_with_escape method."""

        @pytest.mark.parametrize("case", StringParserTestFactory.split_escape_cases())
        def test_split_with_escape(
            self,
            parser: u.Parser,
            case: SplitEscapeCase,
        ) -> None:
            """Test split_on_char_with_escape with parametrized cases."""
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[object]]",
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

        def test_exception_handling(self, parser: u.Parser) -> None:
            """Test split exception handling with bad object."""
            # Intentionally pass bad object that fails on __getitem__()
            # to test error handling
            bad_obj = TestsFlextUtilities.CoreBadObjects.create_for_index()
            # Runtime: parser receives bad object that fails on indexing
            # Type checker: cast to str to test runtime error handling
            bad_str = cast("str", bad_obj)
            result = parser.split_on_char_with_escape(
                bad_str,
                TestsFlextConstants.Delimiters.COMMA,
            )
            # Type narrowing: assert_failure expects r[t.GeneralValueType], r[list[str]] is compatible
            TestsFlextUtilities.CoreAssertions.assert_failure(
                cast("r[t.GeneralValueType]", result),
                TestsFlextConstants.TestErrors.FAILED_SPLIT,
                "exception handling",
            )

    class TestNormalizeWhitespace:
        """Test normalize_whitespace method."""

        @pytest.mark.parametrize(
            "case",
            StringParserTestFactory.normalize_whitespace_cases(),
        )
        def test_normalize_whitespace(
            self,
            parser: u.Parser,
            case: NormalizeWhitespaceCase,
        ) -> None:
            """Test normalize_whitespace with parametrized cases."""
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[object]]",
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

        def test_exception_handling(self, parser: u.Parser) -> None:
            """Test normalization exception handling with bad object."""
            # Intentionally pass bad object that fails on str() to test error handling
            bad_obj = TestsFlextUtilities.CoreBadObjects.create_for_str()
            # Runtime: parser receives bad object that fails on str conversion
            # Type checker: cast to str to test runtime error handling
            bad_str = cast("str", bad_obj)
            result = parser.normalize_whitespace(bad_str)
            # Type narrowing: assert_failure expects r[t.GeneralValueType], r[str] is compatible
            TestsFlextUtilities.CoreAssertions.assert_failure(
                cast("r[t.GeneralValueType]", result),
                TestsFlextConstants.TestErrors.FAILED_NORMALIZE,
                "exception handling",
            )

    class TestRegexPipeline:
        """Test apply_regex_pipeline method."""

        @pytest.mark.parametrize("case", StringParserTestFactory.regex_pipeline_cases())
        def test_apply_regex_pipeline(
            self,
            parser: u.Parser,
            case: RegexPipelineCase,
        ) -> None:
            """Test apply_regex_pipeline with parametrized cases."""
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[object]]",
                    lambda: parser.apply_regex_pipeline(case.text, case.patterns),
                ),
                expected_value=case.expected,
                expected_error=case.expected_error,
                description=case.description,
            )

        def test_exception_handling(self, parser: u.Parser) -> None:
            """Test pipeline exception handling."""
            # Intentionally create invalid pattern with None for error testing
            # Type checker: cast to bypass type checking for runtime error testing
            invalid_pattern = cast("tuple[str, str]", (None, "replacement"))
            patterns: list[tuple[str, str] | tuple[str, str, int]] = [
                invalid_pattern,
            ]  # Runtime allows this
            result = parser.apply_regex_pipeline("test", patterns)
            # Type narrowing: assert_failure expects r[t.GeneralValueType], r[str] is compatible
            TestsFlextUtilities.CoreAssertions.assert_failure(
                cast("r[t.GeneralValueType]", result),
                TestsFlextConstants.TestErrors.FAILED_PIPELINE,
                "exception handling",
            )

        def test_invalid_pattern(self, parser: u.Parser) -> None:
            """Test pipeline with invalid regex pattern."""
            patterns: list[tuple[str, str] | tuple[str, str, int]] = [
                (r"[invalid", "replacement"),
            ]
            result = parser.apply_regex_pipeline("test", patterns)
            # Type narrowing: assert_failure expects r[t.GeneralValueType], r[str] is compatible
            TestsFlextUtilities.CoreAssertions.assert_failure(
                cast("r[t.GeneralValueType]", result),
                TestsFlextConstants.TestErrors.INVALID_REGEX,
                "invalid pattern",
            )

        def test_none_text(self, parser: u.Parser) -> None:
            """Test pipeline with None text."""
            # Intentionally pass None for text to test error handling
            # Type checker: cast to str to test runtime error handling
            text = cast("str", None)
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[object]]",
                    lambda: parser.apply_regex_pipeline(
                        text,  # Runtime: None, triggers error handling
                        [
                            (
                                TestsFlextConstants.Patterns.WHITESPACE,
                                TestsFlextConstants.Replacements.SPACE,
                            ),
                        ],
                    ),
                ),
                expected_error="None",  # Expects failure with "None" in error
                description="none text",
            )

        def test_invalid_text_type(self, parser: u.Parser) -> None:
            """Test pipeline with invalid text type."""
            # Intentionally pass int for text to test error handling
            # Type checker: cast to str to test runtime error handling
            text = cast("str", 123)
            u.Tests.ParserHelpers.execute_and_assert_parser_result(
                cast(
                    "Callable[[], r[object]]",
                    lambda: parser.apply_regex_pipeline(
                        text,  # Runtime: 123, triggers error handling
                        [
                            (
                                TestsFlextConstants.Patterns.WHITESPACE,
                                TestsFlextConstants.Replacements.SPACE,
                            ),
                        ],
                    ),
                ),
                expected_error="str",  # Expects failure with "str" in error
                description="invalid text type",
            )

        def test_empty_patterns(self, parser: u.Parser) -> None:
            """Test pipeline with empty patterns list."""
            patterns: list[tuple[str, str] | tuple[str, str, int]] = []
            result = parser.apply_regex_pipeline("test", patterns)
            # Empty pattern list should result in unchanged text
            u.Tests.Result.assert_success_with_value(
                result,
                "test",
            )

    class TestGetObjectKey:
        """Test get_object_key method."""

        @pytest.mark.parametrize("case", StringParserTestFactory.object_key_cases())
        def test_get_object_key(
            self,
            parser: u.Parser,
            case: ObjectKeyCase,
        ) -> None:
            """Test get_object_key with parametrized cases."""
            key = parser.get_object_key(cast("t.GeneralValueType", case.obj))

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
