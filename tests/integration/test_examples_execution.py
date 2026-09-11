"""Behavioral tests for the public flext-core example scripts.

Each example script is a self-verifying harness: run as ``python -m examples.<name>``
it exercises the public API of one flext-core primitive, compares every audited
value against a committed golden file, and reports the outcome on stdout. These
tests assert only that observable contract -- process exit status, the announced
``PASS`` marker, the reported check count, the absence of failure/traceback
markers, and the golden-file artifacts -- never any harness internals.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

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
    async def _run_example(module_name: str, repo_root: Path) -> tuple[int, str, str]:
        """Execute an example via ``python -m`` in a clean environment."""
        env: t.MutableStrMapping = dict(os.environ)
        env.pop("PYTHONPATH", None)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            module_name,
            cwd=repo_root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return (tm.not_none(process.returncode), stdout.decode(), stderr.decode())

    @pytest.mark.parametrize(
        ("example_name", "module_name", "script_name"), c.Tests.PUBLIC_EXAMPLES
    )
    def test_public_example_scripts_match_golden_files(
        self, example_name: str, module_name: str, script_name: str
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

        returncode, stdout, stderr = asyncio.run(
            self._run_example(module_name, repo_root)
        )

        tm.that(returncode, eq=0)
        tm.that(stdout, has=f"PASS: {example_name}")
        tm.that(stdout, lacks="FAIL")
        tm.that(stderr, lacks="Traceback")

        check_match = tm.not_none(_CHECK_COUNT_RE.search(stdout))
        tm.that(int(check_match.group(1)), gt=0)

        tm.that(expected_path.exists(), eq=True)
        tm.that(actual_path.exists(), eq=False)


__all__: t.MutableSequenceOf[str] = ["TestsFlextExamplesExecution"]
