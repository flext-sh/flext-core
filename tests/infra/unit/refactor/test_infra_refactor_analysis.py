"""Unit tests for violation analysis, impact maps, and main CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from flext_infra import m

try:
    from flext_infra.refactor import (
        FlextInfraRefactorEngine,
        FlextInfraRefactorViolationAnalyzer,
    )
except ImportError as exc:
    pytest.skip(f"refactor package unavailable: {exc}", allow_module_level=True)


def test_build_impact_map_extracts_rename_entries() -> None:
    result = m.Infra.Refactor.Result(
        file_path=Path("/tmp/flext-core/src/module.py"),
        success=True,
        modified=True,
        changes=[
            "Renamed imported symbol: LegacyRemovalRule -> FlextInfraRefactorLegacyRemovalRule (local=LegacyRemovalRule)",
        ],
        refactored_code="",
    )
    impact_map = FlextInfraRefactorEngine.build_impact_map([result])
    assert len(impact_map) == 1
    assert impact_map[0]["kind"] == "rename"
    assert impact_map[0]["old"] == "LegacyRemovalRule"
    assert impact_map[0]["new"] == "FlextInfraRefactorLegacyRemovalRule"


def test_build_impact_map_extracts_signature_entries() -> None:
    result = m.Infra.Refactor.Result(
        file_path=Path("/tmp/flext-core/src/module.py"),
        success=True,
        modified=True,
        changes=[
            "[run-signature-v2] Removed keyword: legacy",
            '[run-signature-v2] Added keyword: mode="modern"',
        ],
        refactored_code="",
    )
    impact_map = FlextInfraRefactorEngine.build_impact_map([result])
    kinds = {entry["kind"] for entry in impact_map}
    assert "signature_remove" in kinds
    assert "signature_add" in kinds


def test_violation_analysis_counts_massive_patterns(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        'refactor_engine:\n  project_scan_dirs: ["src"]\n',
        encoding="utf-8",
    )
    project_root = tmp_path / "project"
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)
    target_file = src_dir / "sample.py"
    target_file.write_text(
        'from typing import Mapping, cast\nfrom flext_core.models import User\nfrom flext_core import t\n\ndef f(data: dict[str, t.Container]) -> dict[str, t.Container]:\n    value = cast("m.ConfigMap", data)\n    return value\n',
        encoding="utf-8",
    )
    engine = FlextInfraRefactorEngine(config_path=config_path)
    _ = engine.load_config()
    files = engine.collect_workspace_files(project_root)
    result = FlextInfraRefactorViolationAnalyzer.analyze_files(files)
    totals = result.totals
    assert "container_invariance" in totals
    assert "redundant_cast" in totals
    assert "direct_submodule_import" in totals
    assert totals["container_invariance"] >= 2
    assert totals["redundant_cast"] >= 1
    assert totals["direct_submodule_import"] >= 1


def test_violation_analyzer_skips_non_utf8_files(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.py"
    file_path.write_bytes(b"\x80\x81\x82")
    result = FlextInfraRefactorViolationAnalyzer.analyze_files([file_path])
    assert result.files_scanned == 1
    assert result.totals == {}


def test_main_analyze_violations_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        'refactor_engine:\n  project_scan_dirs: ["src"]\n',
        encoding="utf-8",
    )
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    target_file = src_dir / "sample.py"
    target_file.write_text("import os\n", encoding="utf-8")
    original = target_file.read_text(encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refactor-engine",
            "--file",
            str(target_file),
            "--analyze-violations",
            "--config",
            str(config_path),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        FlextInfraRefactorEngine.main()
    assert exc_info.value.code == 0
    assert target_file.read_text(encoding="utf-8") == original


def test_main_analyze_violations_writes_json_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        'refactor_engine:\n  project_scan_dirs: ["src"]\n',
        encoding="utf-8",
    )
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    target_file = src_dir / "sample.py"
    target_file.write_text("import os\n", encoding="utf-8")
    report_path = tmp_path / "reports" / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refactor-engine",
            "--file",
            str(target_file),
            "--analyze-violations",
            "--analysis-output",
            str(report_path),
            "--config",
            str(config_path),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        FlextInfraRefactorEngine.main()
    assert exc_info.value.code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["files_scanned"] == 1
    assert payload["totals"] == {}
