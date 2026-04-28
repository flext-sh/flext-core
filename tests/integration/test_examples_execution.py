"""Execute public flext-core example scripts against their golden files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import t
from tests.constants import TestsFlextCoreConstants


class TestsFlextCoreExamplesExecution:
    """Execute public flext-core example scripts against their golden files."""

    @pytest.mark.parametrize(
        ("example_name", "module_name", "script_name"),
        TestsFlextCoreConstants.Tests.ExamplesExecution.PUBLIC_EXAMPLES,
    )
    def test_public_example_scripts_match_golden_files(
        self,
        example_name: str,
        module_name: str,
        script_name: str,
    ) -> None:
        """Examples must pass via real script execution and keep golden outputs stable."""
        script_path = (
            Path(__file__)
            .resolve()
            .parents[TestsFlextCoreConstants.Tests.Paths.REPO_ROOT_PARENT_DEPTH]
            / TestsFlextCoreConstants.Tests.Paths.EXAMPLES_DIR
            / script_name
        )
        actual_path = script_path.with_suffix(".actual")
        actual_path.unlink(missing_ok=True)

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=Path(__file__)
            .resolve()
            .parents[TestsFlextCoreConstants.Tests.Paths.REPO_ROOT_PARENT_DEPTH],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        assert result.returncode == 0, result.stderr or result.stdout
        assert f"PASS: {example_name}" in result.stdout
        assert script_path.with_suffix(".expected").exists()
        assert not actual_path.exists()


__all__: t.MutableSequenceOf[str] = ["TestsFlextCoreExamplesExecution"]
