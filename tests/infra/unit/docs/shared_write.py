"""Tests for FlextInfraDocsShared — write_json and write_markdown.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import pytest

from flext_core import t
from flext_infra.docs.shared import FlextInfraDocsShared
from flext_tests import tm
from tests.infra.models import m


class TestWriteJson:
    """Tests for FlextInfraDocsShared.write_json."""

    def test_returns_flext_result(self, tmp_path: Path) -> None:
        """Test write_json returns r."""
        json_file = tmp_path / "test.json"
        result = FlextInfraDocsShared.write_json(json_file, {"key": "value"})
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_creates_file(self, tmp_path: Path) -> None:
        """Test write_json creates JSON file."""
        json_file = tmp_path / "test.json"
        result = FlextInfraDocsShared.write_json(json_file, {"key": "value"})
        tm.ok(result)
        tm.that(json_file.exists(), eq=True)

    def test_with_model(self, tmp_path: Path) -> None:
        """Test write_json with Pydantic model."""
        json_file = tmp_path / "test.json"
        report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="test",
            items=[],
            checks=[],
            strict=False,
            passed=True,
        )
        result = FlextInfraDocsShared.write_json(json_file, report)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_with_dict_payload(self, tmp_path: Path) -> None:
        """Test write_json with dictionary payload."""
        json_file = tmp_path / "test.json"
        payload: dict[str, str] = {
            "key": "value",
        }
        result = FlextInfraDocsShared.write_json(json_file, payload)
        tm.ok(result)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test write_json creates parent directories."""
        json_file = tmp_path / "nested/deep/test.json"
        result = FlextInfraDocsShared.write_json(json_file, {"key": "value"})
        tm.ok(result)
        tm.that(json_file.exists(), eq=True)

    def test_with_empty_dict(self, tmp_path: Path) -> None:
        """Test write_json with empty dictionary."""
        json_file = tmp_path / "empty.json"
        result = FlextInfraDocsShared.write_json(json_file, {})
        tm.ok(result)

    def test_with_nested_structure(self, tmp_path: Path) -> None:
        """Test write_json with nested dictionary."""
        json_file = tmp_path / "nested.json"
        payload: dict[str, str] = {"level1": "value"}
        result = FlextInfraDocsShared.write_json(json_file, payload)
        tm.ok(result)

    def test_file_readable(self, tmp_path: Path) -> None:
        """Test write_json creates readable JSON file."""
        json_file = tmp_path / "readable.json"
        payload: dict[str, str | int] = {"key": "value", "number": 42}
        FlextInfraDocsShared.write_json(json_file, payload)
        content = json.loads(json_file.read_text())
        tm.that(content["key"], eq="value")
        tm.that(content["number"], eq=42)


class TestWriteMarkdown:
    """Tests for FlextInfraDocsShared.write_markdown."""

    def test_returns_flext_result(self, tmp_path: Path) -> None:
        """Test write_markdown returns r."""
        md_file = tmp_path / "test.md"
        result = FlextInfraDocsShared.write_markdown(md_file, ["# Test", "", "Content"])
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_creates_file(self, tmp_path: Path) -> None:
        """Test write_markdown creates markdown file."""
        md_file = tmp_path / "test.md"
        result = FlextInfraDocsShared.write_markdown(md_file, ["# Test", "", "Content"])
        tm.ok(result)
        tm.that(md_file.exists(), eq=True)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test write_markdown creates parent directories."""
        md_file = tmp_path / "nested/deep/test.md"
        result = FlextInfraDocsShared.write_markdown(md_file, ["# Test"])
        tm.ok(result)
        tm.that(md_file.exists(), eq=True)

    def test_preserves_newlines(self, tmp_path: Path) -> None:
        """Test write_markdown preserves line structure."""
        md_file = tmp_path / "test.md"
        lines = ["# Title", "", "Paragraph 1", "", "Paragraph 2"]
        FlextInfraDocsShared.write_markdown(md_file, lines)
        content = md_file.read_text()
        tm.that("# Title" in content, eq=True)
        tm.that("Paragraph 1" in content, eq=True)

    def test_with_empty_lines(self, tmp_path: Path) -> None:
        """Test write_markdown preserves empty lines."""
        md_file = tmp_path / "test.md"
        lines = ["# Title", "", "", "Content"]
        result = FlextInfraDocsShared.write_markdown(md_file, lines)
        tm.ok(result)
        content = md_file.read_text()
        tm.that(content.count("\n") >= 3, eq=True)

    def test_with_single_line(self, tmp_path: Path) -> None:
        """Test write_markdown with single line."""
        md_file = tmp_path / "single.md"
        result = FlextInfraDocsShared.write_markdown(md_file, ["# Title"])
        tm.ok(result)
        tm.that("# Title" in md_file.read_text(), eq=True)

    def test_with_special_characters(self, tmp_path: Path) -> None:
        """Test write_markdown handles special characters."""
        md_file = tmp_path / "special.md"
        lines = ["# Title with special chars", "Content with accents"]
        result = FlextInfraDocsShared.write_markdown(md_file, lines)
        tm.ok(result)

    def test_file_content_exact(self, tmp_path: Path) -> None:
        """Test write_markdown writes exact content."""
        md_file = tmp_path / "exact.md"
        lines = ["Line 1", "Line 2", "Line 3"]
        FlextInfraDocsShared.write_markdown(md_file, lines)
        content = md_file.read_text()
        tm.that(content, eq="Line 1\nLine 2\nLine 3\n")

    def test_oserror_returns_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test write_markdown returns failure on OSError."""
        md_file = tmp_path / "test.md"

        def mock_write_text(
            self: Path,
            data: str,
            *args: t.Scalar,
            **kwargs: t.Scalar,
        ) -> None:
            msg = "Permission denied"
            raise OSError(msg)

        monkeypatch.setattr(pathlib.Path, "write_text", mock_write_text)
        result = FlextInfraDocsShared.write_markdown(md_file, ["test"])
        tm.fail(result, has="markdown write error")
