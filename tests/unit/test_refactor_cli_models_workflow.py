"""CLI workflow tests for refactor model automation."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from flext_infra.refactor import __main__ as refactor_main


def test_centralize_pydantic_cli_outputs_extended_metrics(tmp_path: Path) -> None:
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
    buffer = StringIO()
    with redirect_stdout(buffer):
        run_code = refactor_main._run_centralize_pydantic(
            argv=[
                "--workspace",
                str(workspace),
                "--dry-run",
                "--normalize-remaining",
            ]
        )
    captured = buffer.getvalue()
    assert run_code == 0
    assert "detected_model_violations=" in captured
    assert "detected_alias_violations=" in captured
    assert "created_model_files=" in captured
    assert "created_typings_files=" in captured


def test_ultrawork_models_cli_runs_dry_run_copy(tmp_path: Path) -> None:
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
    buffer = StringIO()
    with redirect_stdout(buffer):
        run_code = refactor_main._run_ultrawork_models(
            argv=[
                "--workspace",
                str(workspace),
                "--dry-run-copy-workspace",
                "--normalize-remaining",
            ]
        )
    captured = buffer.getvalue()
    assert run_code == 0
    assert "namespace_loose_objects=" in captured
    assert "mro_remaining_violations=" in captured
    assert "namespace_runtime_alias_violations=" in captured
    assert "namespace_manual_protocols=" in captured
    assert "namespace_manual_typing_aliases=" in captured


def test_namespace_enforce_cli_fails_on_manual_protocol_violation(
    tmp_path: Path,
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
        "from typing import Protocol\n\n"
        "class ExternalProtocol(Protocol):\n"
        "    def call(self) -> str:\n"
        "        ...\n",
        encoding="utf-8",
    )
    buffer = StringIO()
    with redirect_stdout(buffer):
        run_code = refactor_main._run_namespace_enforce(
            argv=[
                "--workspace",
                str(workspace),
                "--dry-run",
            ]
        )
    captured = buffer.getvalue()
    assert run_code == 1
    assert "Manual protocols: 1" in captured
