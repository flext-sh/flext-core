"""Tests for FlextInfraTomlService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import tomlkit
from flext_infra import FlextInfraTomlService


class TestFlextInfraTomlService:
    """Test suite for FlextInfraTomlService."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """Test reading an existing TOML file."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[section]\nkey = "value"\nnumber = 42\n', encoding="utf-8"
        )
        service = FlextInfraTomlService()

        result = service.read(toml_file)

        assert result.is_success
        assert result.value["section"]["key"] == "value"
        assert result.value["section"]["number"] == 42

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading a nonexistent file returns empty dict."""
        toml_file = tmp_path / "missing.toml"
        service = FlextInfraTomlService()

        result = service.read(toml_file)

        assert result.is_success
        assert result.value == {}

    def test_read_invalid_toml(self, tmp_path: Path) -> None:
        """Test reading invalid TOML returns failure."""
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("[invalid\nkey = value", encoding="utf-8")
        service = FlextInfraTomlService()

        result = service.read(toml_file)

        assert result.is_failure
        assert "TOML read error" in result.error

    def test_read_document_existing_file(self, tmp_path: Path) -> None:
        """Test reading TOML as document preserves formatting."""
        toml_file = tmp_path / "test.toml"
        content = '[section]\nkey = "value"  # comment\n'
        toml_file.write_text(content, encoding="utf-8")
        service = FlextInfraTomlService()

        result = service.read_document(toml_file)

        assert result.is_success
        doc = result.value
        assert isinstance(doc, tomlkit.TOMLDocument)
        assert doc["section"]["key"] == "value"

    def test_read_document_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading nonexistent file as document returns failure."""
        toml_file = tmp_path / "missing.toml"
        service = FlextInfraTomlService()

        result = service.read_document(toml_file)

        assert result.is_failure
        assert "file not found" in result.error

    def test_read_document_invalid_toml(self, tmp_path: Path) -> None:
        """Test reading invalid TOML as document returns failure."""
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("[invalid\nkey = value", encoding="utf-8")
        service = FlextInfraTomlService()

        result = service.read_document(toml_file)

        assert result.is_failure

    def test_write_dict_payload(self, tmp_path: Path) -> None:
        """Test writing a dict payload to TOML file."""
        toml_file = tmp_path / "output.toml"
        service = FlextInfraTomlService()
        payload = {"section": {"key": "value", "number": 42}}

        result = service.write(toml_file, payload)

        assert result.is_success
        assert toml_file.exists()
        content = toml_file.read_text(encoding="utf-8")
        assert "[section]" in content
        assert "key" in content

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test write creates parent directories."""
        toml_file = tmp_path / "nested" / "deep" / "file.toml"
        service = FlextInfraTomlService()
        payload = {"key": "value"}

        result = service.write(toml_file, payload)

        assert result.is_success
        assert toml_file.exists()

    def test_write_document(self, tmp_path: Path) -> None:
        """Test writing a tomlkit document."""
        toml_file = tmp_path / "doc.toml"
        service = FlextInfraTomlService()
        doc = tomlkit.document()
        doc["section"] = {"key": "value"}

        result = service.write(toml_file, doc)

        assert result.is_success
        assert toml_file.exists()

    def test_write_preserves_formatting(self, tmp_path: Path) -> None:
        """Test write preserves formatting when using document."""
        toml_file = tmp_path / "formatted.toml"
        service = FlextInfraTomlService()
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Configuration file"))
        doc["section"] = {"key": "value"}

        result = service.write(toml_file, doc)

        assert result.is_success
        content = toml_file.read_text(encoding="utf-8")
        assert "Configuration file" in content

    def test_write_permission_error(self, tmp_path: Path) -> None:
        """Test write failure on permission error."""
        toml_file = tmp_path / "readonly.toml"
        toml_file.write_text("[section]\n", encoding="utf-8")
        toml_file.chmod(0o444)  # Read-only
        service = FlextInfraTomlService()

        try:
            result = service.write(toml_file, {"key": "value"})
            assert result.is_failure
        finally:
            toml_file.chmod(0o644)  # Restore permissions for cleanup

    def test_update_section(self, tmp_path: Path) -> None:
        """Test updating a section in TOML file."""
        toml_file = tmp_path / "update.toml"
        toml_file.write_text('[section]\nkey = "old"\n', encoding="utf-8")
        service = FlextInfraTomlService()

        # Read document
        read_result = service.read_document(toml_file)
        assert read_result.is_success
        doc = read_result.value

        # Update
        doc["section"]["key"] = "new"

        # Write back
        write_result = service.write(toml_file, doc)
        assert write_result.is_success

        # Verify
        verify_result = service.read(toml_file)
        assert verify_result.is_success
        assert verify_result.value["section"]["key"] == "new"
