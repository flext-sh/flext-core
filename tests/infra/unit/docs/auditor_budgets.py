"""Tests for FlextInfraDocAuditor — load_audit_budgets.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_tests import tm


class TestLoadAuditBudgets:
    def test_no_config(self, tmp_path: Path) -> None:
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=None)
        tm.that(by_scope, eq={})

    def test_with_config(self, tmp_path: Path) -> None:
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "docs_validation": {
                "audit_gate": {
                    "max_issues_default": 5,
                    "max_issues_by_scope": {"test-project": 3},
                }
            }
        }
        (arch_dir / "architecture_config.json").write_text(json.dumps(config_data))
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=5)
        tm.that(by_scope.get("test-project"), eq=3)

    def test_invalid_json(self, tmp_path: Path) -> None:
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        (arch_dir / "architecture_config.json").write_text("{invalid json}")
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=None)
        tm.that(by_scope, eq={})

    def test_float_values(self, tmp_path: Path) -> None:
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "docs_validation": {
                "audit_gate": {
                    "max_issues_default": 5.5,
                    "max_issues_by_scope": {"test-project": 3.7},
                }
            }
        }
        (arch_dir / "architecture_config.json").write_text(json.dumps(config_data))
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=5)
        tm.that(by_scope.get("test-project"), eq=3)

    def test_no_default_budget(self, tmp_path: Path) -> None:
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "docs_validation": {
                "audit_gate": {"max_issues_by_scope": {"test-project": 3}}
            }
        }
        (arch_dir / "architecture_config.json").write_text(json.dumps(config_data))
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=None)
        tm.that(by_scope.get("test-project"), eq=3)
