"""Tests for flext_infra.patterns — regex pattern definitions."""

from __future__ import annotations

import re

from flext_infra.patterns import FlextInfraPatterns


class TestFlextInfraPatternsTooling:
    """Tests for tooling-related regex patterns."""

    def test_mypy_hint_pattern_matches_valid_hint(self) -> None:
        text = 'note: Hint: "python3 -m pip install types-requests"'
        match = FlextInfraPatterns.MYPY_HINT_RE.search(text)
        assert match is not None
        assert match.group(1) == "types-requests"

    def test_mypy_hint_pattern_captures_package_name(self) -> None:
        text = 'note: Hint: "python3 -m pip install mypy-extensions"'
        match = FlextInfraPatterns.MYPY_HINT_RE.search(text)
        assert match.group(1) == "mypy-extensions"

    def test_mypy_hint_pattern_no_match_without_hint(self) -> None:
        text = "some other mypy output"
        match = FlextInfraPatterns.MYPY_HINT_RE.search(text)
        assert match is None

    def test_mypy_stub_pattern_matches_missing_stubs(self) -> None:
        text = 'Library stubs not installed for "requests"'
        match = FlextInfraPatterns.MYPY_STUB_RE.search(text)
        assert match is not None
        assert match.group(1) == "requests"

    def test_mypy_stub_pattern_captures_library_name(self) -> None:
        text = 'Library stubs not installed for "django"'
        match = FlextInfraPatterns.MYPY_STUB_RE.search(text)
        assert match.group(1) == "django"

    def test_mypy_stub_pattern_no_match_without_message(self) -> None:
        text = "stubs are installed"
        match = FlextInfraPatterns.MYPY_STUB_RE.search(text)
        assert match is None

    def test_internal_prefixes_contains_flext(self) -> None:
        assert "flext_" in FlextInfraPatterns.INTERNAL_PREFIXES

    def test_internal_prefixes_is_tuple(self) -> None:
        assert isinstance(FlextInfraPatterns.INTERNAL_PREFIXES, tuple)


class TestFlextInfraPatternsMarkdown:
    """Tests for markdown-related regex patterns."""

    def test_markdown_link_pattern_matches_link(self) -> None:
        text = "[Click here](https://example.com)"
        match = FlextInfraPatterns.MARKDOWN_LINK_RE.search(text)
        assert match is not None
        assert match.group(1) == "Click here"
        assert match.group(2) == "https://example.com"

    def test_markdown_link_pattern_captures_text_and_url(self) -> None:
        text = "[Documentation](./docs/README.md)"
        match = FlextInfraPatterns.MARKDOWN_LINK_RE.search(text)
        assert match.group(1) == "Documentation"
        assert match.group(2) == "./docs/README.md"

    def test_markdown_link_pattern_multiple_matches(self) -> None:
        text = "[Link1](url1) and [Link2](url2)"
        matches = FlextInfraPatterns.MARKDOWN_LINK_RE.findall(text)
        assert len(matches) == 2
        assert matches[0] == ("Link1", "url1")
        assert matches[1] == ("Link2", "url2")

    def test_markdown_link_url_pattern_captures_url_only(self) -> None:
        text = "[Click here](https://example.com)"
        match = FlextInfraPatterns.MARKDOWN_LINK_URL_RE.search(text)
        assert match is not None
        assert match.group(1) == "https://example.com"

    def test_markdown_link_url_pattern_ignores_text(self) -> None:
        text = "[Some text](path/to/file.md)"
        match = FlextInfraPatterns.MARKDOWN_LINK_URL_RE.search(text)
        assert match.group(1) == "path/to/file.md"

    def test_heading_pattern_matches_h1(self) -> None:
        text = "# Main Title"
        match = FlextInfraPatterns.HEADING_RE.search(text)
        assert match is not None
        assert match.group(1) == "Main Title"

    def test_heading_pattern_matches_h6(self) -> None:
        text = "###### Small Heading"
        match = FlextInfraPatterns.HEADING_RE.search(text)
        assert match is not None
        assert match.group(1) == "Small Heading"

    def test_heading_pattern_multiline(self) -> None:
        text = "# Title 1\nSome content\n## Title 2"
        matches = FlextInfraPatterns.HEADING_RE.findall(text)
        assert len(matches) == 2
        assert "Title 1" in matches
        assert "Title 2" in matches

    def test_heading_h2_h3_pattern_matches_h2(self) -> None:
        text = "## Section Title"
        match = FlextInfraPatterns.HEADING_H2_H3_RE.search(text)
        assert match is not None
        assert match.group(1) == "##"
        assert match.group(2) == "Section Title"

    def test_heading_h2_h3_pattern_matches_h3(self) -> None:
        text = "### Subsection"
        match = FlextInfraPatterns.HEADING_H2_H3_RE.search(text)
        assert match is not None
        assert match.group(1) == "###"
        assert match.group(2) == "Subsection"

    def test_heading_h2_h3_pattern_ignores_h1(self) -> None:
        text = "# Title"
        match = FlextInfraPatterns.HEADING_H2_H3_RE.search(text)
        assert match is None

    def test_heading_h2_h3_pattern_ignores_h4(self) -> None:
        text = "#### Deep Heading"
        match = FlextInfraPatterns.HEADING_H2_H3_RE.search(text)
        assert match is None

    def test_anchor_link_pattern_matches_anchor(self) -> None:
        text = "[Go to section](#section-name)"
        match = FlextInfraPatterns.ANCHOR_LINK_RE.search(text)
        assert match is not None
        assert match.group(1) == "Go to section"
        assert match.group(2) == "section-name"

    def test_anchor_link_pattern_captures_text_and_anchor(self) -> None:
        text = "[Back to top](#top)"
        match = FlextInfraPatterns.ANCHOR_LINK_RE.search(text)
        assert match.group(1) == "Back to top"
        assert match.group(2) == "top"

    def test_inline_code_pattern_matches_backticks(self) -> None:
        text = "Use `function_name()` in your code"
        match = FlextInfraPatterns.INLINE_CODE_RE.search(text)
        assert match is not None
        assert match.group(0) == "`function_name()`"

    def test_inline_code_pattern_multiple_matches(self) -> None:
        text = "Use `foo()` and `bar()` together"
        matches = FlextInfraPatterns.INLINE_CODE_RE.findall(text)
        assert len(matches) == 2
        assert "`foo()`" in matches
        assert "`bar()`" in matches

    def test_inline_code_pattern_empty_code(self) -> None:
        text = "Empty code: ``"
        match = FlextInfraPatterns.INLINE_CODE_RE.search(text)
        assert match is not None
        assert match.group(0) == "``"


class TestFlextInfraPatternsPatternTypes:
    """Tests for pattern type consistency."""

    def test_all_patterns_are_compiled_regex(self) -> None:
        patterns = [
            FlextInfraPatterns.MYPY_HINT_RE,
            FlextInfraPatterns.MYPY_STUB_RE,
            FlextInfraPatterns.MARKDOWN_LINK_RE,
            FlextInfraPatterns.MARKDOWN_LINK_URL_RE,
            FlextInfraPatterns.HEADING_RE,
            FlextInfraPatterns.HEADING_H2_H3_RE,
            FlextInfraPatterns.ANCHOR_LINK_RE,
            FlextInfraPatterns.INLINE_CODE_RE,
        ]
        for pattern in patterns:
            assert isinstance(pattern, re.Pattern)

    def test_patterns_are_string_patterns(self) -> None:
        patterns = [
            FlextInfraPatterns.MYPY_HINT_RE,
            FlextInfraPatterns.MYPY_STUB_RE,
            FlextInfraPatterns.MARKDOWN_LINK_RE,
        ]
        for pattern in patterns:
            # Check that pattern is Pattern[str]
            assert hasattr(pattern, "pattern")
            assert isinstance(pattern.pattern, str)


class TestFlextInfraPatternsEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_markdown_link_with_special_chars_in_url(self) -> None:
        text = "[Link](https://example.com/path?query=value&other=123)"
        match = FlextInfraPatterns.MARKDOWN_LINK_RE.search(text)
        assert match is not None
        assert "query=value" in match.group(2)

    def test_heading_with_trailing_whitespace(self) -> None:
        text = "## Title   "
        match = FlextInfraPatterns.HEADING_H2_H3_RE.search(text)
        assert match is not None
        assert match.group(2) == "Title"

    def test_inline_code_with_special_chars(self) -> None:
        text = "Code: `foo_bar-baz.txt`"
        match = FlextInfraPatterns.INLINE_CODE_RE.search(text)
        assert match is not None
        assert "foo_bar-baz.txt" in match.group(0)

    def test_anchor_link_with_hyphens(self) -> None:
        text = "[Section](#my-section-name)"
        match = FlextInfraPatterns.ANCHOR_LINK_RE.search(text)
        assert match is not None
        assert match.group(2) == "my-section-name"
