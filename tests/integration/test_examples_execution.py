"""Execute public flext-core example scripts against their golden files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestExamplesExecution:
    """Execute public flext-core example scripts against their golden files."""

    @pytest.mark.parametrize(
        ("example_name", "module_name", "script_path"),
        [
            (
                "ex_01_flext_result",
                "examples.ex_01_flext_result",
                REPO_ROOT / "examples" / "ex_01_flext_result.py",
            ),
            (
                "ex_02_flext_settings",
                "examples.ex_02_flext_settings",
                REPO_ROOT / "examples" / "ex_02_flext_settings.py",
            ),
            (
                "ex_03_flext_logger",
                "examples.ex_03_flext_logger",
                REPO_ROOT / "examples" / "ex_03_flext_logger.py",
            ),
            (
                "ex_04_flext_dispatcher",
                "examples.ex_04_flext_dispatcher",
                REPO_ROOT / "examples" / "ex_04_flext_dispatcher.py",
            ),
            (
                "ex_11_flext_service",
                "examples.ex_11_flext_service",
                REPO_ROOT / "examples" / "ex_11_flext_service.py",
            ),
        ],
    )
    def test_public_example_scripts_match_golden_files(
        self,
        example_name: str,
        module_name: str,
        script_path: Path,
    ) -> None:
        """Examples must pass via real script execution and keep golden outputs stable."""
        actual_path = script_path.with_suffix(".actual")
        actual_path.unlink(missing_ok=True)

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        assert result.returncode == 0, result.stderr or result.stdout
        assert f"PASS: {example_name}" in result.stdout
        assert script_path.with_suffix(".expected").exists()
        assert not actual_path.exists()


__all__: list[str] = ["TestExamplesExecution"]
