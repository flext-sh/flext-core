"""Tests for workspace checker gate resolution and CSV parsing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.check.services import FlextInfraWorkspaceChecker
from flext_tests import tm


class TestWorkspaceCheckerResolveGates:
    """Test FlextInfraWorkspaceChecker.resolve_gates."""

    def test_resolve_gates_type_maps_to_pyrefly(self) -> None:
        result = FlextInfraWorkspaceChecker.resolve_gates(["type"])
        tm.ok(result)
        tm.that("pyrefly" in result.value, eq=True)

    def test_resolve_gates_skips_empty_strings(self) -> None:
        result = FlextInfraWorkspaceChecker.resolve_gates(["lint", "", "format"])
        tm.ok(result)
        tm.that("" not in result.value, eq=True)

    def test_resolve_gates_deduplicates_entries(self) -> None:
        result = FlextInfraWorkspaceChecker.resolve_gates([
            "lint",
            "lint",
            "format",
            "lint",
        ])
        tm.ok(result)
        tm.that(result.value.count("lint"), eq=1)

    def test_resolve_gates_invalid_gate_fails(self) -> None:
        result = FlextInfraWorkspaceChecker.resolve_gates(["invalid"])
        tm.fail(result, has="unknown gate")

    def test_resolve_gates_all_valid_types(self) -> None:
        gates = [
            "lint",
            "format",
            "type",
            "mypy",
            "pyright",
            "security",
            "markdown",
            "go",
        ]
        result = FlextInfraWorkspaceChecker.resolve_gates(gates)
        tm.ok(result)
        tm.that(len(result.value) > 0, eq=True)


class TestWorkspaceCheckerParseGateCSV:
    """Test FlextInfraWorkspaceChecker.parse_gate_csv."""

    def test_parse_gate_csv_simple(self) -> None:
        result = FlextInfraWorkspaceChecker.parse_gate_csv("lint,format,type")
        tm.that(result, eq=["lint", "format", "type"])

    def test_parse_gate_csv_with_spaces(self) -> None:
        result = FlextInfraWorkspaceChecker.parse_gate_csv("lint , format , type")
        tm.that(result, eq=["lint", "format", "type"])

    def test_parse_gate_csv_empty_entries(self) -> None:
        result = FlextInfraWorkspaceChecker.parse_gate_csv("lint,,format")
        tm.that("" not in result, eq=True)
