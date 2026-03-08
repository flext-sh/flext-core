"""CLI workflow tests for model centralization automation."""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.refactor import __main__ as refactor_main


def test_centralize_pydantic_cli_outputs_extended_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    module_dir = workspace / "src" / "sample_pkg"
    module_dir.mkdir(parents=True)
    (module_dir / "service.py").write_text(
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n"
        "from typing import TypeAlias\n\n"
        "class SamplePayload(BaseModel):\n"
        "    value: str\n\n"
        "PayloadMap: TypeAlias = dict[str, str]\n",
        encoding="utf-8",
    )
    run_code = refactor_main._run_centralize_pydantic(
        argv=["--workspace", str(workspace), "--dry-run", "--normalize-remaining"]
    )
    captured = capsys.readouterr()
    assert run_code == 0
    assert "detected_model_violations=" in captured.out
    assert "detected_alias_violations=" in captured.out
    assert "created_model_files=" in captured.out


def test_ultrawork_models_cli_runs_dry_run_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "sample-project"
    module_dir = project / "src" / "sample_pkg"
    module_dir.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\n", encoding="utf-8"
    )
    (project / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (module_dir / "service.py").write_text(
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n\n"
        "class SamplePayload(BaseModel):\n"
        "    value: str\n",
        encoding="utf-8",
    )
    run_code = refactor_main._run_ultrawork_models(
        argv=[
            "--workspace",
            str(workspace),
            "--dry-run-copy-workspace",
            "--normalize-remaining",
        ]
    )
    captured = capsys.readouterr()
    assert run_code == 0
    assert "namespace_loose_objects=" in captured.out
    assert "mro_remaining_violations=" in captured.out
