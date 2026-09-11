"""Pytest plugin for automatic markdown Python code block validation.

Provides the markdown docs option that validates Python code blocks
in .md files against project linting rules.

Usage in any project's conftest.py::

    pytest_plugins = ["flext_tests.conftest_plugin"]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, override

import pytest

from flext_tests import c, u
from flext_tests._validator.markdown import FlextValidatorMarkdown

if TYPE_CHECKING:
    from pathlib import Path


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the markdown docs option if pytest-markdown-docs is not installed.

    Idempotent: ``conftest_plugin`` may register options during ``pytest_addoption``
    and again via historic hooks when the module is registered in configure.
    """
    if find_spec("pytest_markdown_docs") is not None:
        return
    group = parser.getgroup("markdown", "Markdown code block validation")
    try:
        group.addoption(
            c.Tests.VALIDATOR_MD_OPTION_DOCS,
            action="store_true",
            default=False,
            help="Validate Python code blocks in .md files",
        )
    except ValueError:
        # Option already registered by the early conftest_plugin path or a peer.
        return


class MarkdownCodeBlockItem(pytest.Item):
    """Pytest item representing a markdown file to validate."""

    def __init__(self, name: str, parent: pytest.Collector, md_path: Path) -> None:
        super().__init__(name, parent)
        self.md_path = md_path

    @override
    def runtest(self) -> None:
        """Run markdown code block validation."""
        result = FlextValidatorMarkdown.markdown([self.md_path])
        if result.failure:
            msg = f"Markdown validation failed: {result.error}"
            raise MarkdownValidationError(msg)
        scan = result.value
        if scan.violations:
            detail_lines = [
                f"  {v.rule_id} at line {v.line_number}: {v.description}"
                for v in scan.violations
            ]
            msg = "\n".join([
                f"Found {len(scan.violations)} violation(s) in {self.md_path}:",
                *detail_lines,
            ])
            raise MarkdownValidationError(msg)

    @override
    def repr_failure(
        self, excinfo: pytest.ExceptionInfo[BaseException], style: str | None = None
    ) -> str:
        """Represent test failure."""
        _ = style
        return str(excinfo.value)

    @override
    def reportinfo(self) -> tuple[Path, int | None, str]:
        """Report test info."""
        return self.md_path, None, f"markdown-check: {self.md_path.name}"


class MarkdownCodeBlockCollector(pytest.File):
    """Pytest collector for markdown files."""

    @override
    def collect(self) -> list[MarkdownCodeBlockItem]:
        """Collect markdown file as a test item."""
        return [
            MarkdownCodeBlockItem.from_parent(
                self, name=self.path.name, md_path=self.path
            )
        ]


def pytest_collect_file(
    parent: pytest.Collector, file_path: Path
) -> MarkdownCodeBlockCollector | None:
    """Collect .md files when the markdown docs option is enabled."""
    if not parent.config.getoption(c.Tests.VALIDATOR_MD_OPTION_DOCS, default=False):
        return None
    if file_path.suffix == ".md" and file_path.stat().st_size > 0:
        content = u.Cli.files_read_text(file_path).unwrap()
        if c.Tests.VALIDATOR_MD_PYTHON_BLOCK_RE.search(content):
            return MarkdownCodeBlockCollector.from_parent(parent, path=file_path)
    return None


class MarkdownValidationError(Exception):
    """Raised when markdown code block validation fails."""


__all__: list[str] = [
    "MarkdownCodeBlockCollector",
    "MarkdownCodeBlockItem",
    "MarkdownValidationError",
    "pytest_addoption",
    "pytest_collect_file",
]
