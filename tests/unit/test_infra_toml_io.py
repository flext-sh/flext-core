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

    def test_value_differs_with_lists(self) -> None:
        """Test value_differs compares lists as strings."""
        current = [1, 2, 3]
        expected = [1, 2, 3]
        assert not FlextInfraTomlService.value_differs(current, expected)

        expected_diff = [1, 2, 4]
        assert FlextInfraTomlService.value_differs(current, expected_diff)

    def test_value_differs_with_scalars(self) -> None:
        """Test value_differs with scalar values."""
        assert not FlextInfraTomlService.value_differs("same", "same")
        assert FlextInfraTomlService.value_differs("a", "b")
        assert not FlextInfraTomlService.value_differs(42, 42)
        assert FlextInfraTomlService.value_differs(42, 43)

    def test_build_table_with_nested_mapping(self, tmp_path: Path) -> None:
        """Test build_table creates nested tomlkit tables."""
        service = FlextInfraTomlService()
        data = {
            "section": {"key": "value", "nested": {"deep": "value"}},
            "simple": "scalar",
        }
        table = service.build_table(data)
        assert isinstance(table, tomlkit.items.Table)
        assert table["section"]["key"] == "value"
        assert table["simple"] == "scalar"

    def test_sync_mapping_adds_new_keys(self, tmp_path: Path) -> None:
        """Test sync_mapping adds missing keys to target."""
        service = FlextInfraTomlService()
        target = {}
        canonical = {"new_key": "new_value"}
        added = []
        updated = []
        removed = []

        service.sync_mapping(
            target,
            canonical,
            prune_extras=False,
            prefix="",
            added=added,
            updated=updated,
            removed=removed,
        )

        assert target["new_key"] == "new_value"
        assert "new_key" in added

    def test_sync_mapping_updates_changed_values(self, tmp_path: Path) -> None:
        """Test sync_mapping updates changed values."""
        service = FlextInfraTomlService()
        target = {"key": "old_value"}
        canonical = {"key": "new_value"}
        added = []
        updated = []
        removed = []

        service.sync_mapping(
            target,
            canonical,
            prune_extras=False,
            prefix="",
            added=added,
            updated=updated,
            removed=removed,
        )

        assert target["key"] == "new_value"
        assert "key" in updated

    def test_sync_mapping_prunes_extras(self, tmp_path: Path) -> None:
        """Test sync_mapping removes extra keys when prune_extras=True."""
        service = FlextInfraTomlService()
        target = {"keep": "value", "remove": "extra"}
        canonical = {"keep": "value"}
        added = []
        updated = []
        removed = []

        service.sync_mapping(
            target,
            canonical,
            prune_extras=True,
            prefix="",
            added=added,
            updated=updated,
            removed=removed,
        )

        assert "remove" not in target
        assert "remove" in removed

    def test_sync_mapping_nested_with_prefix(self, tmp_path: Path) -> None:
        """Test sync_mapping with nested mappings and prefix."""
        service = FlextInfraTomlService()
        target = {"section": {"key": "old"}}
        canonical = {"section": {"key": "new"}}
        added = []
        updated = []
        removed = []

        service.sync_mapping(
            target,
            canonical,
            prune_extras=False,
            prefix="config",
            added=added,
            updated=updated,
            removed=removed,
        )

        assert target["section"]["key"] == "new"
        assert "config.section.key" in updated

    def test_sync_mapping_skips_prune_when_false(self, tmp_path: Path) -> None:
        """Test sync_mapping skips pruning when prune_extras=False."""
        service = FlextInfraTomlService()
        target = {"keep": "value", "extra": "stays"}
        canonical = {"keep": "value"}
        added = []
        updated = []
        removed = []

        service.sync_mapping(
            target,
            canonical,
            prune_extras=False,
            prefix="",
            added=added,
            updated=updated,
            removed=removed,
        )

        assert "extra" in target
        assert len(removed) == 0

    def test_execute_returns_success(self) -> None:
        """Test execute() returns FlextResult[bool] with True."""
        service = FlextInfraTomlService()
        result = service.execute()
        assert result.is_success
        assert result.value is True

    def test_build_table_with_nested_mapping_dict(self) -> None:
        """Test build_table handles nested mappings."""
        from flext_infra.toml_io import FlextInfraTomlService  # noqa: PLC0415

        service = FlextInfraTomlService()
        nested = {"key": {"nested": "value"}}
        result = service.build_table(nested)
        assert result is not None

    def test_sync_mapping_replaces_scalar_with_mapping(self) -> None:
        """Test sync_mapping replaces scalar with mapping (lines 192-194).

        When canonical has a nested mapping but target has a scalar value,
        the scalar should be replaced with a new table.
        """
        service = FlextInfraTomlService()
        target = {"section": "scalar_value"}
        canonical = {"section": {"nested": "value"}}
        added = []
        updated = []
        removed = []

        service.sync_mapping(
            target,
            canonical,
            prune_extras=False,
            prefix="",
            added=added,
            updated=updated,
            removed=removed,
        )

        # The scalar should be replaced with a mapping
        assert isinstance(target["section"], dict)
        assert "section" in added
