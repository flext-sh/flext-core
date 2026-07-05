"""Behavioral tests for the public flext-core example scripts.

Each example script is a self-verifying harness: run as ``python -m examples.<name>``
it exercises the public API of one flext-core primitive, compares every audited
value against a committed golden file, and reports the outcome on stdout. These
tests assert only that observable contract -- process exit status, the announced
``PASS`` marker, the reported check count, the absence of failure/traceback
markers, and the golden-file artifacts -- never any harness internals.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.constants import c

if TYPE_CHECKING:
    from tests.typings import t

_CHECK_COUNT_RE = re.compile(r"\((\d+) checks\)")


class TestsFlextExamplesExecution:
    """Assert the observable execution contract of every public example script."""

    @staticmethod
    def _repo_root() -> Path:
        """Return the flext-core repository root that hosts ``examples/``."""
        return Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]

    @staticmethod
    def _run_example(
        module_name: str, repo_root: Path
    ) -> subprocess.CompletedProcess[str]:
        """Execute an example via ``python -m`` in a clean environment."""
        env: t.MutableStrMapping = dict(os.environ)
        env.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    @pytest.mark.parametrize(
        ("example_name", "module_name", "script_name"),
        c.Tests.PUBLIC_EXAMPLES,
    )
    def test_public_example_scripts_match_golden_files(
        self,
        example_name: str,
        module_name: str,
        script_name: str,
    ) -> None:
        """A public example runs to completion and matches its golden file.

        Observable public contract asserted here:
        - the process exits successfully (return code ``0``);
        - stdout announces ``PASS: <name>`` (golden comparison succeeded);
        - stdout reports a positive audited-check count (the API was exercised);
        - stdout carries no ``FAIL`` marker (no golden mismatch);
        - stderr carries no ``Traceback`` (no unhandled exception escaped);
        - the committed golden ``.expected`` fixture exists;
        - no ``.actual`` diff artifact is left behind (clean pass).
        """
        repo_root = self._repo_root()
        script_path = repo_root / c.Tests.EXAMPLES_DIR / script_name
        actual_path = script_path.with_suffix(".actual")
        expected_path = script_path.with_suffix(".expected")
        actual_path.unlink(missing_ok=True)

        result = self._run_example(module_name, repo_root)

        diagnostic = result.stderr or result.stdout
        assert result.returncode == 0, diagnostic
        assert f"PASS: {example_name}" in result.stdout
        assert "FAIL" not in result.stdout, result.stdout
        assert "Traceback" not in result.stderr, result.stderr

        check_match = _CHECK_COUNT_RE.search(result.stdout)
        assert check_match is not None, result.stdout
        assert int(check_match.group(1)) > 0, result.stdout

        assert expected_path.exists()
        assert not actual_path.exists()


__all__: t.MutableSequenceOf[str] = ["TestsFlextExamplesExecution"]
