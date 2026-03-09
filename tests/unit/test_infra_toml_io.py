"""Tests for FlextInfraUtilitiesToml.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from unittest.mock import patch

import tomlkit
from tomlkit.items import Table

from flext_infra import FlextInfraUtilitiesToml


class TestFlextInfraTomlService:
    """Test suite for FlextInfraUtilitiesToml."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """read() returns TOMLDocument for valid file."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[section]\nkey = "value"\nnumber = 42\n', encoding="utf-8"
        )

        service = FlextInfraUtilitiesToml()
        doc = service.read(toml_file)

        assert doc is not None
        section = doc["section"]
        assert isinstance(section, Mapping)
        assert section["key"] == "value"
        assert section["number"] == 42

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """read() returns None when file does not exist."""
        toml_file = tmp_path / "missing.toml"
        service = FlextInfraUtilitiesToml()

        assert service.read(toml_file) is None

    def test_read_invalid_toml(self, tmp_path: Path) -> None:
        """read() returns None for invalid TOML."""
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("[invalid\nkey = value", encoding="utf-8")
        service = FlextInfraUtilitiesToml()

        assert service.read(toml_file) is None

    def test_read_document_existing_file(self, tmp_path: Path) -> None:
        """read_document() returns successful FlextResult for valid file."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[section]\nkey = "value"  # comment\n', encoding="utf-8")
        service = FlextInfraUtilitiesToml()

        result = service.read_document(toml_file)

        assert result.is_success
        doc = result.value
        section = doc["section"]
        assert isinstance(section, Mapping)
        assert section["key"] == "value"

    def test_read_document_nonexistent_file(self, tmp_path: Path) -> None:
        """read_document() returns failure for missing file."""
        toml_file = tmp_path / "missing.toml"
        service = FlextInfraUtilitiesToml()

        result = service.read_document(toml_file)

        assert result.is_failure
        assert isinstance(result.error, str)
        assert "failed to read TOML" in result.error

    def test_read_document_invalid_toml(self, tmp_path: Path) -> None:
        """read_document() returns failure for invalid TOML."""
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("[invalid\nkey = value", encoding="utf-8")
        service = FlextInfraUtilitiesToml()

        result = service.read_document(toml_file)

        assert result.is_failure

    def test_write_document(self, tmp_path: Path) -> None:
        """write_document() writes TOMLDocument and returns success."""
        toml_file = tmp_path / "doc.toml"
        service = FlextInfraUtilitiesToml()
        doc = tomlkit.document()
        doc["section"] = {"key": "value"}

        result = service.write_document(toml_file, doc)

        assert result.is_success
        assert toml_file.exists()

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """write_document() creates parent directories as needed."""
        toml_file = tmp_path / "nested" / "deep" / "file.toml"
        service = FlextInfraUtilitiesToml()
        doc = tomlkit.document()
        doc["key"] = "value"

        result = service.write_document(toml_file, doc)

        assert result.is_success
        assert toml_file.exists()

    def test_write_preserves_formatting(self, tmp_path: Path) -> None:
        """write_document() preserves tomlkit formatting/comments."""
        toml_file = tmp_path / "formatted.toml"
        service = FlextInfraUtilitiesToml()
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Configuration file"))
        doc["section"] = {"key": "value"}

        result = service.write_document(toml_file, doc)

        assert result.is_success
        content = toml_file.read_text(encoding="utf-8")
        assert "Configuration file" in content

    def test_write_permission_error(self, tmp_path: Path) -> None:
        """write_document() returns failure when write raises OSError."""
        toml_file = tmp_path / "readonly.toml"
        service = FlextInfraUtilitiesToml()
        doc = tomlkit.document()
        doc["key"] = "value"

        with patch("pathlib.Path.write_text", side_effect=OSError("permission denied")):
            result = service.write_document(toml_file, doc)

        assert result.is_failure
        assert isinstance(result.error, str)
        assert "TOML write error" in result.error

    def test_update_section(self, tmp_path: Path) -> None:
        """read_document()/write_document() round-trip updates content."""
        toml_file = tmp_path / "update.toml"
        toml_file.write_text('[section]\nkey = "old"\n', encoding="utf-8")
        service = FlextInfraUtilitiesToml()

        read_result = service.read_document(toml_file)
        assert read_result.is_success
        doc = read_result.value
        section = doc["section"]
        assert isinstance(section, MutableMapping)
        section["key"] = "new"

        write_result = service.write_document(toml_file, doc)
        assert write_result.is_success

        verify_doc = service.read(toml_file)
        assert verify_doc is not None
        verify_section = verify_doc["section"]
        assert isinstance(verify_section, Mapping)
        assert verify_section["key"] == "new"

    def test_array_creates_multiline(self) -> None:
        """array() creates a multiline TOML array."""
        arr = FlextInfraUtilitiesToml.array(["a", "b", "c"])

        assert list(arr) == ["a", "b", "c"]

    def test_ensure_table_reuses_existing(self) -> None:
        """ensure_table() returns existing table instance when present."""
        parent = tomlkit.table()
        existing = tomlkit.table()
        existing["key"] = "value"
        parent["section"] = existing

        table = FlextInfraUtilitiesToml.ensure_table(parent, "section")

        assert isinstance(table, Table)
        assert table["key"] == "value"

    def test_as_toml_mapping_and_get_helpers(self) -> None:
        """as_toml_mapping() and get() normalize container values."""
        mapping = {"key": "value"}
        assert FlextInfraUtilitiesToml.as_toml_mapping(mapping) == mapping
        assert FlextInfraUtilitiesToml.as_toml_mapping("bad") is None

        doc = tomlkit.document()
        doc["a"] = 1
        doc["b"] = [1, 2]
        assert FlextInfraUtilitiesToml.get(doc, "a") == 1
        assert FlextInfraUtilitiesToml.get(doc, "b") == [1, 2]
        assert FlextInfraUtilitiesToml.get(doc, 123) is None
