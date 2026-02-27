"""Generic model tests with full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
"""

from __future__ import annotations

# pyright: reportMissingImports=false
from flext_core import c, m, r, t


def test_operation_progress_start_operation_sets_runtime_fields() -> None:
    op = m.Operation()

    op.start_operation(name="sync", estimated_total=5)

    assert op.operation_name == "sync"
    assert op.estimated_total == 5
    assert op.start_time is not None
    assert op.last_update == op.start_time


def test_conversion_add_converted_and_error_metadata_append_paths() -> None:
    conv = m.Conversion()

    conv.add_converted("ok-1")
    conv.add_error("bad-1", item="item-a")
    conv.add_error("bad-2", item="item-b")
    conv.add_error("bad-3")

    assert conv.converted == ["ok-1"]
    assert conv.errors == ["bad-1", "bad-2", "bad-3"]
    assert conv.metadata.root["failed_items"] == ["item-a", "item-b"]


def test_conversion_add_warning_metadata_append_paths() -> None:
    conv = m.Conversion()

    conv.add_warning("warn-1", item="item-a")
    conv.add_warning("warn-2", item="item-b")
    conv.add_warning("warn-3")

    assert conv.warnings == ["warn-1", "warn-2", "warn-3"]
    assert conv.metadata.root["warning_items"] == ["item-a", "item-b"]


def test_conversion_add_skipped_skip_reason_upsert_paths() -> None:
    conv = m.Conversion()

    conv.add_skipped("item-a", reason="empty")
    conv.add_skipped("item-b", reason="invalid")
    conv.add_skipped("item-c")

    assert conv.skipped == ["item-a", "item-b", "item-c"]
    assert conv.metadata.root["skip_reasons"] == {
        "item-a": "empty",
        "item-b": "invalid",
    }


def test_conversion_start_and_complete_methods() -> None:
    conv = m.Conversion()

    conv.start_conversion(
        source_format="xml", target_format="json", total_input_count=99
    )
    conv.complete_conversion()

    assert conv.source_format == "xml"
    assert conv.target_format == "json"
    assert conv.total_input_count == 99
    assert conv.start_time is not None
    assert conv.end_time is not None
    assert conv.end_time >= conv.start_time


def test_canonical_aliases_are_available() -> None:
    result = r[str].ok("ok")
    value = t.Dict(root={"k": "v"})
    assert result.value == "ok"
    assert value.root == {"k": "v"}
    assert c.Performance.RECENT_THRESHOLD_SECONDS > 0
