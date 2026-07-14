"""Public behavioral tests for the ``FlextRuntime`` facade."""

from __future__ import annotations

from pathlib import Path

import flext_core
from flext_core import m
from flext_core.runtime import FlextRuntime
from flext_tests import tm


class TestsFlextRuntime:
    """Verify observable runtime normalization contracts."""

    class TestsFacade:
        """Public facade identity scenarios."""

        def test_exports_runtime_facade(self) -> None:
            """Expose the runtime facade from the public package root."""
            tm.that(flext_core.FlextRuntime, is_=FlextRuntime)

    class TestsNormalization:
        """Runtime payload normalization scenarios."""

        def test_normalizes_nested_mapping(self) -> None:
            """Retain nested JSON-compatible mapping structure."""
            normalized = FlextRuntime.normalize_to_container({
                "nested": {"count": 1},
                "items": ["x", 2],
            })

            tm.that(normalized, eq={"nested": {"count": 1}, "items": ["x", 2]})

        def test_normalizes_config_map_to_json_dict(self) -> None:
            """Materialize a ConfigMap root as a JSON dictionary."""
            normalized = FlextRuntime.normalize_to_container(
                m.ConfigMap(root={"items": ["x", 2]})
            )

            tm.that(normalized, eq={"items": ["x", 2]})

        def test_normalizes_path_and_sequence_members(self) -> None:
            """Convert path members while preserving sequence values."""
            normalized = FlextRuntime.normalize_to_container([Path("runtime-value"), 2])

            tm.that(normalized, eq=["runtime-value", 2])

        def test_normalizes_mapping_to_concrete_dict(self) -> None:
            """Return a concrete JSON dictionary for a mapping input."""
            normalized = FlextRuntime.normalize_to_json_mapping({
                "path": Path("runtime")
            })

            tm.that(normalized, eq={"path": "runtime"})
