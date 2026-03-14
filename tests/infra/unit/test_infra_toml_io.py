"""Tests for FlextInfraUtilitiesToml.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import stat
from pathlib import Path

import tomlkit
from tomlkit.items import Table

from flext_infra import FlextInfraUtilitiesToml
from flext_tests import tm


class TestFlextInfraTomlRead:
    """Tests for TOML read operations."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[section]\nkey = "value"\nnumber = 42\n', encoding="utf-8"
        )
        service = FlextInfraUtilitiesToml()
        doc = service.read(toml_file)
        tm.that(doc, none=False)
        if doc is None:
            msg = "expected parsed TOML document"
            raise AssertionError(msg)
        section_obj = doc["section"]
        assert isinstance(section_obj, Table)
        section = section_obj
        if doc is None:
            msg = "expected parsed TOML document"
            raise AssertionError(msg)
        section_obj = doc["section"]
        assert isinstance(section_obj, Table)
        section = section_obj
        tm.that(section["key"], eq="value")
        tm.that(section["number"], eq=42)

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "missing.toml"
        service = FlextInfraUtilitiesToml()
        tm.that(service.read(toml_file), none=True)

    def test_read_invalid_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("[invalid\nkey = value", encoding="utf-8")
        service = FlextInfraUtilitiesToml()
        tm.that(service.read(toml_file), none=True)


class TestFlextInfraTomlDocument:
    """Tests for TOML document read/write operations."""

    def test_read_document_existing_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[section]\nkey = "value"  # comment\n', encoding="utf-8")
        service = FlextInfraUtilitiesToml()
        result = service.read_document(toml_file)
        tm.ok(result)
        doc = result.value
        section_obj = doc["section"]
        assert isinstance(section_obj, Table)
        section = section_obj
        section_obj = doc["section"]
        assert isinstance(section_obj, Table)
        section = section_obj
        tm.that(section["key"], eq="value")

    def test_read_document_nonexistent_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "missing.toml"
        service = FlextInfraUtilitiesToml()
        result = service.read_document(toml_file)
        tm.fail(result, has="failed to read TOML")

    def test_read_document_invalid_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("[invalid\nkey = value", encoding="utf-8")
        service = FlextInfraUtilitiesToml()
        tm.fail(service.read_document(toml_file))

    def test_write_document(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "doc.toml"
        service = FlextInfraUtilitiesToml()
        doc = tomlkit.document()
        doc["section"] = {"key": "value"}
        result = service.write_document(toml_file, doc)
        tm.ok(result)
        tm.that(toml_file.exists(), eq=True)

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "nested" / "deep" / "file.toml"
        service = FlextInfraUtilitiesToml()
        doc = tomlkit.document()
        doc["key"] = "value"
        tm.ok(service.write_document(toml_file, doc))
        tm.that(toml_file.exists(), eq=True)

    def test_write_preserves_formatting(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "formatted.toml"
        service = FlextInfraUtilitiesToml()
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Configuration file"))
        doc["section"] = {"key": "value"}
        tm.ok(service.write_document(toml_file, doc))
        content = toml_file.read_text(encoding="utf-8")
        tm.that(content, has="Configuration file")

    def test_write_permission_error(self, tmp_path: Path) -> None:
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        toml_file = readonly_dir / "test.toml"
        Path(readonly_dir).chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            service = FlextInfraUtilitiesToml()
            doc = tomlkit.document()
            doc["key"] = "value"
            result = service.write_document(toml_file, doc)
            tm.fail(result, has="TOML write error")
        finally:
            Path(readonly_dir).chmod(stat.S_IRWXU)

    def test_update_section(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "update.toml"
        toml_file.write_text('[section]\nkey = "old"\n', encoding="utf-8")
        service = FlextInfraUtilitiesToml()
        read_result = service.read_document(toml_file)
        tm.ok(read_result)
        doc = read_result.value
        section_obj = doc["section"]
        assert isinstance(section_obj, Table)
        section = section_obj
        section_obj = doc["section"]
        assert isinstance(section_obj, Table)
        section = section_obj
        section["key"] = "new"
        tm.ok(service.write_document(toml_file, doc))
        verify_doc = service.read(toml_file)
        tm.that(verify_doc, none=False)
        if verify_doc is None:
            msg = "expected persisted TOML document"
            raise AssertionError(msg)
        verify_section_obj = verify_doc["section"]
        assert isinstance(verify_section_obj, Table)
        verify_section = verify_section_obj
        if verify_doc is None:
            msg = "expected persisted TOML document"
            raise AssertionError(msg)
        verify_section_obj = verify_doc["section"]
        assert isinstance(verify_section_obj, Table)
        verify_section = verify_section_obj
        tm.that(verify_section["key"], eq="new")


class TestFlextInfraTomlHelpers:
    """Tests for TOML helper methods."""

    def test_array_creates_multiline(self) -> None:
        arr = FlextInfraUtilitiesToml.array(["a", "b", "c"])
        arr_text = arr.as_string()
        tm.that(arr_text, has='"a"')
        tm.that(arr_text, has='"b"')
        tm.that(arr_text, has='"c"')

    def test_ensure_table_reuses_existing(self) -> None:
        parent = tomlkit.table()
        existing = tomlkit.table()
        existing["key"] = "value"
        parent["section"] = existing
        table = FlextInfraUtilitiesToml.ensure_table(parent, "section")
        tm.that(table["key"], eq="value")

    def test_as_toml_mapping_and_get_helpers(self) -> None:
        mapping = {"key": "value"}
        tm.that(FlextInfraUtilitiesToml.as_toml_mapping(mapping), eq=mapping)
        tm.that(FlextInfraUtilitiesToml.as_toml_mapping("bad"), none=True)
        doc = tomlkit.document()
        doc["a"] = 1
        doc["b"] = [1, 2]
        tm.that(FlextInfraUtilitiesToml.get(doc, "a"), eq=1)
        tm.that(FlextInfraUtilitiesToml.get(doc, "b"), eq=[1, 2])
        tm.that(FlextInfraUtilitiesToml.get(doc, 123), none=True)
